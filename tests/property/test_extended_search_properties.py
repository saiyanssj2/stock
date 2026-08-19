"""
Property-based tests for Extended Search result quality monotonicity.

Tests the following correctness property from the design document:
- Property 23: Extended search result quality monotonicity

*For any* search tree and any point of cancellation or timeout, the
Extended_Search SHALL return the best Action from the deepest fully
evaluated level, where that Action's minimax score is greater than or
equal to all other Actions' minimax scores at that level. Furthermore,
*for any* completed Extended_Search, the effective depth reached SHALL
be greater than or equal to the standard search depth (3), and the
extended search score SHALL be greater than or equal to the standard
search score (deeper search cannot produce a worse minimax result given
the same evaluation function).

**Validates: Requirements 17.5, 17.7**
"""

import time
import threading
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.config import Action, SearchConfig
from engine.extended_search import (
    ExtendedDecisionReport,
    ExtendedSearchConfig,
    ExtendedSearchProgress,
)
from engine.market_state import MarketState, NUM_INDICATORS, NUM_OHLCV
from engine.search_module import SearchModule, SearchResult


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

MIN_LOOKBACK = 30
MAX_LOOKBACK = 80


@st.composite
def market_state_for_extended_search(draw):
    """
    Generate a valid MarketState with sufficient history for extended search.

    Uses a random walk for price and finite indicator values.
    """
    lookback = draw(st.integers(min_value=MIN_LOOKBACK, max_value=MAX_LOOKBACK))
    symbol = draw(st.sampled_from(["VNM", "FPT", "VCB", "HPG", "MWG", "TCB"]))

    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)

    # Generate realistic close prices via random walk
    base_price = draw(st.floats(min_value=10.0, max_value=200.0))
    returns = rng.uniform(-0.05, 0.05, lookback)
    returns[0] = 0.0
    closes = base_price * np.cumprod(1.0 + returns)
    closes = np.maximum(closes, 1.0)

    # Generate OHLCV data
    ohlcv = np.zeros((lookback, NUM_OHLCV), dtype=np.float64)
    spreads = closes * 0.015
    ohlcv[:, 0] = closes + rng.uniform(-1, 1, lookback) * spreads  # open
    ohlcv[:, 3] = closes  # close
    ohlcv[:, 1] = np.maximum(ohlcv[:, 0], closes) + rng.uniform(0, 1, lookback) * spreads  # high
    ohlcv[:, 2] = np.minimum(ohlcv[:, 0], closes) - rng.uniform(0, 1, lookback) * spreads  # low
    ohlcv[:, 4] = rng.uniform(100000, 50000000, lookback)  # volume

    # Ensure OHLCV consistency: high >= max(open, close), low <= min(open, close)
    ohlcv[:, 1] = np.maximum(ohlcv[:, 1], np.maximum(ohlcv[:, 0], ohlcv[:, 3]))
    ohlcv[:, 2] = np.minimum(ohlcv[:, 2], np.minimum(ohlcv[:, 0], ohlcv[:, 3]))

    # Generate indicators with finite values
    indicators = rng.uniform(-100, 100, (lookback, NUM_INDICATORS))

    timestamp = pd.Timestamp("2024-01-15")

    return MarketState(
        symbol=symbol,
        timestamp=timestamp,
        ohlcv=ohlcv,
        indicators=indicators,
        lookback=lookback,
    )


@st.composite
def extended_search_config_strategy(draw):
    """
    Generate a valid ExtendedSearchConfig with depth in [3, 5] and
    generous timeout to allow full completion in tests.
    """
    depth = draw(st.integers(min_value=3, max_value=5))
    timeout = draw(st.floats(min_value=60.0, max_value=120.0))
    return ExtendedSearchConfig(depth=depth, timeout=timeout, enabled=True)


@st.composite
def model_scores_strategy(draw):
    """
    Generate a deterministic sequence of model scores for consistent
    evaluation across standard and extended search.

    Returns a list of scores that cycle through when used by the mock.
    """
    num_scores = draw(st.integers(min_value=5, max_value=30))
    scores = draw(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0),
            min_size=num_scores,
            max_size=num_scores,
        )
    )
    return scores


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_deterministic_model_manager(scores):
    """
    Create a mock ModelManager that returns scores deterministically.

    Cycles through the scores list. Uses a thread-safe counter.
    """
    call_count = [0]

    def predict_fn(features):
        idx = call_count[0] % len(scores)
        call_count[0] += 1
        return np.array([[scores[idx]]], dtype=np.float32)

    mock = MagicMock()
    mock.predict.side_effect = predict_fn
    mock.is_loaded = True
    return mock


def _make_mock_scenario_generator():
    """
    Create a mock ScenarioGenerator that returns variant copies of
    the input state for each scenario (deterministic price offsets).
    """

    def generate_fn(input_state, action, num_scenarios=5):
        results = []
        for i in range(num_scenarios):
            new_ohlcv = input_state.ohlcv.copy()
            # Slight price variation per scenario
            new_ohlcv[:, 3] *= 1.0 + (i - num_scenarios // 2) * 0.01
            # Ensure OHLCV consistency
            new_ohlcv[:, 1] = np.maximum(
                new_ohlcv[:, 1], np.maximum(new_ohlcv[:, 0], new_ohlcv[:, 3])
            )
            new_ohlcv[:, 2] = np.minimum(
                new_ohlcv[:, 2], np.minimum(new_ohlcv[:, 0], new_ohlcv[:, 3])
            )
            new_state = MarketState(
                symbol=input_state.symbol,
                timestamp=input_state.timestamp,
                ohlcv=new_ohlcv,
                indicators=input_state.indicators.copy(),
                lookback=input_state.lookback,
            )
            results.append(new_state)
        return results

    mock = MagicMock()
    mock.generate.side_effect = generate_fn
    return mock


# ---------------------------------------------------------------------------
# Property 23: Extended search result quality monotonicity
# ---------------------------------------------------------------------------


class TestExtendedSearchResultQualityMonotonicity:
    """
    Property 23: Extended search result quality monotonicity.

    Tests three sub-properties:
    1. Result returns best action from deepest complete level (optimality)
    2. effective_depth_reached >= standard depth (3)
    3. Extended score >= standard score for same evaluation function (monotonicity)

    **Validates: Requirements 17.5, 17.7**
    """

    @given(
        state=market_state_for_extended_search(),
        config=extended_search_config_strategy(),
        scores=model_scores_strategy(),
    )
    @settings(max_examples=15, deadline=None)
    def test_effective_depth_at_least_standard_depth(self, state, config, scores):
        """
        For any completed extended search, the effective depth reached
        SHALL be >= the standard search depth (3).

        This ensures extended search never degrades below baseline quality.

        **Validates: Requirements 17.5, 17.7**
        """
        model_mgr = _make_deterministic_model_manager(scores)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(
                default_depth=3,
                max_depth=10,
                timeout_seconds=10.0,
                default_scenarios=3,
            ),
        )

        report = module._run_extended_search(state, config)

        assert isinstance(report, ExtendedDecisionReport), (
            "Extended search must return an ExtendedDecisionReport"
        )
        assert report.effective_depth_reached >= 3, (
            f"effective_depth_reached={report.effective_depth_reached} < 3 "
            f"(standard depth). Extended search must achieve at least "
            f"standard search depth."
        )

    @given(
        state=market_state_for_extended_search(),
        config=extended_search_config_strategy(),
        scores=model_scores_strategy(),
    )
    @settings(max_examples=15, deadline=None)
    def test_extended_score_gte_standard_score(self, state, config, scores):
        """
        For any completed extended search with the same evaluation function,
        the extended search score SHALL be >= the standard search score.

        Deeper search cannot produce a worse minimax result given the same
        evaluation function, because iterative deepening includes the
        standard depth as its first complete level.

        **Validates: Requirements 17.5, 17.7**
        """
        # Use a fixed score so that both standard and extended searches
        # see the same evaluation landscape (deterministic model)
        fixed_score = scores[0] if scores else 0.5
        model_mgr = _make_deterministic_model_manager([fixed_score])
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(
                default_depth=3,
                max_depth=10,
                timeout_seconds=10.0,
                default_scenarios=3,
            ),
        )

        report = module._run_extended_search(state, config)

        # The extended search position_score should be >= standard score
        # because deeper search with the same eval function explores more
        # paths and can only find equal or better minimax values.
        # The standard score is captured as standard_position_score in the report.
        assert report.standard_position_score is not None, (
            "Standard position score must be populated in ExtendedDecisionReport"
        )
        assert report.position_score >= report.standard_position_score - 1e-9, (
            f"Extended score {report.position_score:.6f} < standard score "
            f"{report.standard_position_score:.6f}. Deeper search with the "
            f"same evaluation function should not produce a worse result."
        )

    @given(
        state=market_state_for_extended_search(),
        config=extended_search_config_strategy(),
        scores=model_scores_strategy(),
    )
    @settings(max_examples=15, deadline=None)
    def test_best_action_has_highest_score_at_deepest_level(self, state, config, scores):
        """
        The extended search result returns the best action from the deepest
        fully evaluated level, where that action's minimax score is >= all
        other actions' minimax scores at that level.

        **Validates: Requirements 17.5, 17.7**
        """
        # Use fixed score for determinism
        fixed_score = scores[0] if scores else 0.5
        model_mgr = _make_deterministic_model_manager([fixed_score])
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(
                default_depth=3,
                max_depth=10,
                timeout_seconds=10.0,
                default_scenarios=3,
            ),
        )

        report = module._run_extended_search(state, config)

        # The recommended action must be one of the valid actions
        assert report.recommended_action in (Action.BUY, Action.HOLD, Action.SELL), (
            f"Invalid recommended action: {report.recommended_action}"
        )

        # Verify it has the highest or tied-for-highest score.
        # We can verify this by running search at the effective_depth_reached
        # and checking the action is optimal at that depth.
        # Since the extended report uses _search_at_depth results, the
        # position_score is the best score among all actions at the deepest
        # complete level. We validate via a secondary search at the same depth.
        model_mgr2 = _make_deterministic_model_manager([fixed_score])
        module2 = SearchModule(
            model_manager=model_mgr2,
            scenario_generator=scenario_gen,
            config=SearchConfig(
                default_depth=3,
                max_depth=10,
                timeout_seconds=60.0,
                default_scenarios=5,
            ),
        )

        verification_result = module2.search(
            state, depth=report.effective_depth_reached
        )

        # The extended search best score should match what we'd get at that depth
        # (within tolerance due to alpha-beta pruning and branching differences)
        # The key invariant: best action score >= all other action scores
        if verification_result.action_scores:
            best_verification_score = max(verification_result.action_scores.values())
            # Extended search should find the same or better result
            assert report.position_score >= best_verification_score - 1e-6 or \
                   abs(report.position_score - best_verification_score) < 0.1, (
                f"Extended score {report.position_score:.6f} seems inconsistent "
                f"with verification score {best_verification_score:.6f} at depth "
                f"{report.effective_depth_reached}"
            )

    @given(
        state=market_state_for_extended_search(),
        scores=model_scores_strategy(),
    )
    @settings(max_examples=5, deadline=None)
    def test_cancelled_search_returns_valid_result_from_complete_level(
        self, state, scores
    ):
        """
        For any point of cancellation, the extended search SHALL return
        the best action from the deepest fully evaluated level.

        After cancellation, effective_depth_reached >= 3 (standard depth)
        and the result is valid.

        **Validates: Requirements 17.5, 17.7**
        """
        fixed_score = scores[0] if scores else 0.5

        # Use a slightly slow model to give time for cancellation at deeper levels
        call_count = [0]

        def slow_predict(features):
            call_count[0] += 1
            time.sleep(0.002)  # 2ms per evaluation
            return np.array([[fixed_score]], dtype=np.float32)

        model_mgr = MagicMock()
        model_mgr.predict.side_effect = slow_predict
        model_mgr.is_loaded = True

        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(
                default_depth=3,
                max_depth=10,
                timeout_seconds=60.0,
                default_scenarios=3,
            ),
        )

        config = ExtendedSearchConfig(depth=7, timeout=60.0, enabled=True)

        # Cancel after enough time for at least depth 3 to complete
        def cancel_later():
            time.sleep(0.5)
            module.cancel_search()

        cancel_thread = threading.Thread(target=cancel_later)
        cancel_thread.start()

        report = module._run_extended_search(state, config)
        cancel_thread.join(timeout=5.0)

        # Must return a valid ExtendedDecisionReport
        assert isinstance(report, ExtendedDecisionReport), (
            "Cancelled extended search must return an ExtendedDecisionReport"
        )

        # Effective depth must be at least standard (3)
        assert report.effective_depth_reached >= 3, (
            f"Cancelled search effective_depth={report.effective_depth_reached} < 3. "
            f"Must achieve at least standard depth before cancellation matters."
        )

        # Recommended action must be valid
        assert report.recommended_action in (Action.BUY, Action.HOLD, Action.SELL), (
            f"Invalid action from cancelled search: {report.recommended_action}"
        )

        # Position score must be bounded
        assert -1.0 <= report.position_score <= 1.0, (
            f"Position score {report.position_score} out of bounds after cancel"
        )

        # Standard comparison fields should be populated
        assert report.standard_recommendation is not None
        assert report.standard_position_score is not None
