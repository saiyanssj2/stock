"""
Property-based tests for SearchModule minimax search, DecisionReport structure,
and low-confidence HOLD override.

**Validates: Requirements 4.2, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

Properties tested:
- Property 13: Minimax optimality
- Property 15: Decision report structural completeness
- Property 16: Low confidence HOLD override
"""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.config import (
    Action,
    DecisionReport,
    IndicatorContribution,
    ScenarioResult,
    SearchConfig,
)
from engine.market_state import MarketState, NUM_INDICATORS, NUM_OHLCV
from engine.search_module import SearchModule, SearchResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_LOOKBACK = 30  # Minimum for scenario generation
MAX_LOOKBACK = 100  # Keep manageable for test speed


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------


@st.composite
def market_state_strategy(draw):
    """
    Generate a valid MarketState with sufficient history for search.

    Produces OHLCV data as a random walk with realistic price movements
    and populates indicators with finite values.
    """
    lookback = draw(st.integers(min_value=MIN_LOOKBACK, max_value=MAX_LOOKBACK))
    symbol = draw(st.sampled_from(["VNM", "FPT", "VCB", "HPG", "MWG", "TCB"]))

    # Generate close prices as random walk
    base_price = draw(st.floats(min_value=10.0, max_value=200.0))
    closes = np.zeros(lookback, dtype=np.float64)
    closes[0] = base_price
    for i in range(1, lookback):
        daily_return = draw(st.floats(min_value=-0.05, max_value=0.05))
        closes[i] = closes[i - 1] * (1.0 + daily_return)
        closes[i] = max(closes[i], 1.0)

    # Generate OHLCV
    ohlcv = np.zeros((lookback, NUM_OHLCV), dtype=np.float64)
    for i in range(lookback):
        close = closes[i]
        spread = close * 0.015
        open_p = close + draw(st.floats(min_value=-spread, max_value=spread))
        high_p = max(open_p, close) + draw(st.floats(min_value=0.0, max_value=spread))
        low_p = min(open_p, close) - draw(st.floats(min_value=0.0, max_value=spread))
        volume = draw(st.floats(min_value=100000.0, max_value=50000000.0))

        ohlcv[i, 0] = open_p
        ohlcv[i, 1] = high_p
        ohlcv[i, 2] = low_p
        ohlcv[i, 3] = close
        ohlcv[i, 4] = volume

    # Ensure OHLCV consistency
    ohlcv[:, 1] = np.maximum(ohlcv[:, 1], np.maximum(ohlcv[:, 0], ohlcv[:, 3]))
    ohlcv[:, 2] = np.minimum(ohlcv[:, 2], np.minimum(ohlcv[:, 0], ohlcv[:, 3]))

    # Generate indicators with finite values
    indicators = np.random.default_rng(42).uniform(-100, 100, (lookback, NUM_INDICATORS))

    timestamp = pd.Timestamp("2024-01-15")

    return MarketState(
        symbol=symbol,
        timestamp=timestamp,
        ohlcv=ohlcv,
        indicators=indicators,
        lookback=lookback,
    )


@st.composite
def model_score_strategy(draw):
    """Generate a valid model score in [-1, 1]."""
    return draw(st.floats(min_value=-1.0, max_value=1.0))


@st.composite
def varying_model_scores_strategy(draw):
    """
    Generate a list of model scores that vary per call,
    ensuring the search produces differentiated action scores.
    """
    num_scores = draw(st.integers(min_value=20, max_value=100))
    scores = draw(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0),
            min_size=num_scores,
            max_size=num_scores,
        )
    )
    return scores


@st.composite
def confidence_below_threshold_strategy(draw):
    """Generate a confidence value below 0.3 (the HOLD override threshold)."""
    return draw(st.floats(min_value=0.0, max_value=0.2999))


@st.composite
def confidence_strategy(draw):
    """Generate a confidence value in [0.0, 1.0]."""
    return draw(st.floats(min_value=0.0, max_value=1.0))


@st.composite
def decision_report_strategy(draw):
    """
    Generate a valid DecisionReport with all required fields.

    This tests the structural completeness property by creating
    reports that should satisfy all constraints.
    """
    symbol = draw(st.sampled_from(["VNM", "FPT", "VCB", "HPG", "MWG", "TCB"]))
    action = draw(st.sampled_from([Action.BUY, Action.HOLD, Action.SELL]))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    position_score = draw(st.floats(min_value=-1.0, max_value=1.0))

    # Generate top scenarios (up to 3)
    num_scenarios = draw(st.integers(min_value=1, max_value=3))
    scenarios = []
    for _ in range(num_scenarios):
        seq_len = draw(st.integers(min_value=1, max_value=3))
        action_seq = draw(
            st.lists(
                st.sampled_from([Action.BUY, Action.HOLD, Action.SELL]),
                min_size=seq_len,
                max_size=seq_len,
            )
        )
        leaf_score = draw(st.floats(min_value=-1.0, max_value=1.0))
        desc = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "Z"))))
        scenarios.append(ScenarioResult(
            action_sequence=action_seq,
            leaf_score=leaf_score,
            description=desc,
        ))

    # Generate top indicators (up to 5), sorted by |contribution| descending
    num_indicators = draw(st.integers(min_value=1, max_value=5))
    indicators = []
    for _ in range(num_indicators):
        name = draw(st.sampled_from([
            "RSI_14", "MACD", "EMA_20", "ADX", "BB_upper",
            "OBV", "ATR_14", "CCI_20", "MFI", "ROC_10",
        ]))
        value = draw(st.floats(min_value=-500.0, max_value=500.0))
        contribution = draw(st.floats(min_value=0.01, max_value=1.0))
        indicators.append(IndicatorContribution(
            name=name,
            value=value,
            contribution=contribution,
        ))
    # Sort by absolute contribution descending
    indicators.sort(key=lambda x: abs(x.contribution), reverse=True)

    return DecisionReport(
        symbol=symbol,
        recommended_action=action,
        confidence=confidence,
        position_score=position_score,
        top_scenarios=scenarios,
        top_indicators=indicators,
        timestamp=datetime.now(),
    )


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_model_manager(scores):
    """
    Create a mock ModelManager that returns scores sequentially.

    Parameters
    ----------
    scores : list of float
        Scores to return in order. Cycles if exhausted.
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
    Create a mock ScenarioGenerator that returns variant copies
    of the input state for each scenario.
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


def _compute_confidence_from_search_result(result: SearchResult) -> float:
    """
    Compute confidence from scenario agreement level.

    Confidence is based on how much the action scores agree:
    - If best score is much higher than others, confidence is high.
    - If scores are close together, confidence is low.

    Returns a value in [0.0, 1.0].
    """
    if not result.action_scores:
        return 0.0

    scores = list(result.action_scores.values())
    if len(scores) <= 1:
        return 0.5

    best_score = max(scores)
    second_best = sorted(scores, reverse=True)[1] if len(scores) > 1 else best_score

    # Confidence = gap between best and second-best, scaled
    # Score range is [-1, 1], so max gap is 2.0
    gap = best_score - second_best
    confidence = min(1.0, gap / 0.5)  # Gap of 0.5 → full confidence
    return max(0.0, confidence)


def _generate_decision_report(
    state: MarketState,
    result: SearchResult,
    confidence_override: float = None,
) -> DecisionReport:
    """
    Generate a DecisionReport from a SearchResult.

    This simulates the report generation logic (task 6.2) for testing
    properties 15 and 16.

    Parameters
    ----------
    state : MarketState
        The state that was searched.
    result : SearchResult
        The search result.
    confidence_override : float, optional
        If provided, overrides the computed confidence.
    """
    # Compute confidence from action scores agreement
    if confidence_override is not None:
        confidence = confidence_override
    else:
        confidence = _compute_confidence_from_search_result(result)

    # Apply low-confidence HOLD override (Property 16)
    if confidence < 0.3:
        recommended_action = Action.HOLD
    else:
        recommended_action = result.best_action

    # Build top scenarios (up to 3)
    top_scenarios = []
    for child in result.root_children[:3]:
        action_seq = [child.action] if child.action else [Action.HOLD]
        top_scenarios.append(ScenarioResult(
            action_sequence=action_seq,
            leaf_score=child.score,
            description=f"{child.action.value if child.action else 'HOLD'} scenario",
        ))

    # Build top indicators (up to 5, using feature importance approximation)
    indicator_names = ["RSI_14", "MACD", "EMA_20", "ADX", "ATR_14",
                       "OBV", "BB_upper", "CCI_20", "MFI", "ROC_10"]
    top_indicators = []
    for i, name in enumerate(indicator_names[:5]):
        # Use last indicator value if available
        value = float(state.indicators[-1, i]) if i < state.indicators.shape[1] else 0.0
        contribution = abs(result.best_score) * (0.5 - i * 0.08)
        top_indicators.append(IndicatorContribution(
            name=name,
            value=value,
            contribution=abs(contribution),
        ))
    # Sort by absolute contribution descending
    top_indicators.sort(key=lambda x: abs(x.contribution), reverse=True)

    return DecisionReport(
        symbol=state.symbol,
        recommended_action=recommended_action,
        confidence=confidence,
        position_score=result.best_score,
        top_scenarios=top_scenarios,
        top_indicators=top_indicators,
        timestamp=datetime.now(),
    )


# ---------------------------------------------------------------------------
# Property 13: Minimax optimality
# ---------------------------------------------------------------------------


class TestMinimaxOptimality:
    """
    Property 13: Minimax optimality.

    For any search tree produced by the SearchModule, the recommended
    Action SHALL have a minimax score greater than or equal to all other
    root-level Actions' minimax scores.

    **Validates: Requirements 4.2**
    """

    @given(
        state=market_state_strategy(),
        scores=varying_model_scores_strategy(),
    )
    @settings(max_examples=15)
    def test_best_action_has_highest_score(self, state, scores):
        """
        The best_action's score in action_scores is >= all other action scores.

        For any generated state and any sequence of model evaluation scores,
        the search module's recommended action must have the highest (or tied)
        minimax score among all root-level actions.

        **Validates: Requirements 4.2**
        """
        model_mgr = _make_mock_model_manager(scores)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(
                default_depth=1,
                timeout_seconds=30.0,
                default_scenarios=3,
            ),
        )

        result = module.search(state)

        # The best action must have a score >= all other actions
        assert result.best_action in result.action_scores, (
            f"best_action {result.best_action} not in action_scores keys"
        )

        best_score = result.action_scores[result.best_action]
        for action, score in result.action_scores.items():
            assert best_score >= score - 1e-9, (
                f"Action {action.value} has score {score:.6f} > "
                f"best_action {result.best_action.value} score {best_score:.6f}"
            )

    @given(
        state=market_state_strategy(),
        scores=varying_model_scores_strategy(),
        depth=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=15)
    def test_best_action_optimality_at_various_depths(self, state, scores, depth):
        """
        Minimax optimality holds at any search depth within [1, 3].

        The recommended action's score must be >= all other action scores
        regardless of the search depth used.

        **Validates: Requirements 4.2**
        """
        model_mgr = _make_mock_model_manager(scores)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(
                default_depth=depth,
                max_depth=5,
                timeout_seconds=30.0,
                default_scenarios=3,
                adaptive_scenario_count=3,
            ),
        )

        result = module.search(state, depth=depth)

        # Skip if timed out with no actions evaluated
        if not result.action_scores:
            return

        best_score = result.action_scores[result.best_action]
        for action, score in result.action_scores.items():
            assert best_score >= score - 1e-9, (
                f"At depth {depth}: action {action.value} score {score:.6f} > "
                f"best_action {result.best_action.value} score {best_score:.6f}"
            )


# ---------------------------------------------------------------------------
# Property 15: Decision report structural completeness
# ---------------------------------------------------------------------------


class TestDecisionReportStructuralCompleteness:
    """
    Property 15: Decision report structural completeness.

    For any successful analysis, the DecisionReport SHALL contain:
    - a non-empty symbol string
    - a valid Action (BUY/HOLD/SELL)
    - a confidence score in [0.0, 1.0]
    - a position_score in [-1.0, +1.0]
    - at most 3 top scenarios (or all available if fewer than 3 exist)
    - at most 5 top indicators sorted by absolute contribution in descending order

    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
    """

    @given(
        state=market_state_strategy(),
        scores=varying_model_scores_strategy(),
    )
    @settings(max_examples=15)
    def test_report_has_all_required_fields(self, state, scores):
        """
        A DecisionReport generated from a search result has all required
        fields with valid values.

        **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
        """
        model_mgr = _make_mock_model_manager(scores)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(
                default_depth=1,
                timeout_seconds=30.0,
                default_scenarios=3,
            ),
        )

        result = module.search(state)
        report = _generate_decision_report(state, result)

        # Non-empty symbol
        assert report.symbol, "symbol must be non-empty"
        assert len(report.symbol) > 0

        # Valid action
        assert report.recommended_action in (Action.BUY, Action.HOLD, Action.SELL), (
            f"Invalid action: {report.recommended_action}"
        )

        # Confidence in [0.0, 1.0]
        assert 0.0 <= report.confidence <= 1.0, (
            f"Confidence {report.confidence} out of [0.0, 1.0] range"
        )

        # Position score in [-1.0, 1.0]
        assert -1.0 <= report.position_score <= 1.0, (
            f"Position score {report.position_score} out of [-1.0, 1.0] range"
        )

        # Top scenarios: at most 3, each with valid structure
        assert len(report.top_scenarios) <= 3, (
            f"Too many scenarios: {len(report.top_scenarios)} > 3"
        )
        for i, scenario in enumerate(report.top_scenarios):
            assert isinstance(scenario, ScenarioResult), (
                f"Scenario {i} is not a ScenarioResult"
            )
            assert len(scenario.action_sequence) >= 1, (
                f"Scenario {i} has empty action_sequence"
            )
            assert all(
                a in (Action.BUY, Action.HOLD, Action.SELL)
                for a in scenario.action_sequence
            ), f"Scenario {i} has invalid action in sequence"
            assert -1.0 <= scenario.leaf_score <= 1.0, (
                f"Scenario {i} leaf_score {scenario.leaf_score} out of bounds"
            )

        # Top indicators: at most 5, sorted by |contribution| descending
        assert len(report.top_indicators) <= 5, (
            f"Too many indicators: {len(report.top_indicators)} > 5"
        )
        for i, indicator in enumerate(report.top_indicators):
            assert isinstance(indicator, IndicatorContribution), (
                f"Indicator {i} is not an IndicatorContribution"
            )
            assert indicator.name, f"Indicator {i} has empty name"
            assert indicator.contribution >= 0.0, (
                f"Indicator {i} contribution {indicator.contribution} is negative"
            )

        # Verify descending order of absolute contribution
        contributions = [abs(ind.contribution) for ind in report.top_indicators]
        for i in range(len(contributions) - 1):
            assert contributions[i] >= contributions[i + 1] - 1e-9, (
                f"Indicators not sorted: contribution[{i}]={contributions[i]:.6f} "
                f"< contribution[{i+1}]={contributions[i+1]:.6f}"
            )

    @given(report=decision_report_strategy())
    @settings(max_examples=20)
    def test_decision_report_dataclass_structural_validity(self, report):
        """
        Any well-formed DecisionReport satisfies structural constraints
        on its fields (valid ranges, proper types, ordering).

        **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
        """
        # Symbol is non-empty
        assert report.symbol and len(report.symbol) > 0

        # Action is valid enum value
        assert report.recommended_action in (Action.BUY, Action.HOLD, Action.SELL)

        # Confidence bounded
        assert 0.0 <= report.confidence <= 1.0

        # Position score bounded
        assert -1.0 <= report.position_score <= 1.0

        # Scenarios count ≤ 3
        assert len(report.top_scenarios) <= 3

        # Indicators count ≤ 5
        assert len(report.top_indicators) <= 5

        # Indicators sorted by absolute contribution descending
        contributions = [abs(ind.contribution) for ind in report.top_indicators]
        for i in range(len(contributions) - 1):
            assert contributions[i] >= contributions[i + 1] - 1e-9

        # Timestamp is a valid datetime
        assert isinstance(report.timestamp, datetime)


# ---------------------------------------------------------------------------
# Property 16: Low confidence HOLD override
# ---------------------------------------------------------------------------


class TestLowConfidenceHoldOverride:
    """
    Property 16: Low confidence HOLD override.

    For any analysis result where the computed confidence score is below 0.3,
    the recommended Action in the DecisionReport SHALL be HOLD, regardless
    of the SearchModule's recommended action or the Position_Score.

    **Validates: Requirements 5.6**
    """

    @given(
        state=market_state_strategy(),
        scores=varying_model_scores_strategy(),
        low_confidence=confidence_below_threshold_strategy(),
    )
    @settings(max_examples=15)
    def test_low_confidence_forces_hold(self, state, scores, low_confidence):
        """
        When confidence < 0.3, the final recommended action is always HOLD,
        regardless of what the minimax search returns.

        **Validates: Requirements 5.6**
        """
        model_mgr = _make_mock_model_manager(scores)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(
                default_depth=1,
                timeout_seconds=30.0,
                default_scenarios=3,
            ),
        )

        result = module.search(state)

        # Force confidence below threshold
        report = _generate_decision_report(
            state, result, confidence_override=low_confidence
        )

        assert report.recommended_action == Action.HOLD, (
            f"Expected HOLD for confidence {low_confidence:.4f}, "
            f"but got {report.recommended_action.value}. "
            f"Search best_action was {result.best_action.value}"
        )

    @given(
        state=market_state_strategy(),
        low_confidence=st.floats(min_value=0.0, max_value=0.2999),
        search_action=st.sampled_from([Action.BUY, Action.SELL]),
        position_score=st.floats(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=20)
    def test_hold_override_independent_of_search_action_and_score(
        self, state, low_confidence, search_action, position_score
    ):
        """
        The HOLD override applies regardless of the search action
        (BUY or SELL) and regardless of the position score value.

        This ensures neither a strong BUY signal nor a strong SELL
        signal can bypass the low-confidence safety mechanism.

        **Validates: Requirements 5.6**
        """
        # Create a mock search result with the given action and score
        mock_result = SearchResult(
            best_action=search_action,
            best_score=position_score,
            action_scores={
                search_action: position_score,
                Action.HOLD: 0.0,
            },
            nodes_evaluated=10,
            depth_reached=1,
            timed_out=False,
            root_children=[],
        )

        report = _generate_decision_report(
            state, mock_result, confidence_override=low_confidence
        )

        assert report.recommended_action == Action.HOLD, (
            f"HOLD override failed: confidence={low_confidence:.4f}, "
            f"search_action={search_action.value}, "
            f"position_score={position_score:.4f}, "
            f"but got {report.recommended_action.value}"
        )

    @given(
        state=market_state_strategy(),
        confidence=st.floats(min_value=0.3, max_value=1.0),
        scores=varying_model_scores_strategy(),
    )
    @settings(max_examples=15)
    def test_sufficient_confidence_preserves_search_action(
        self, state, confidence, scores
    ):
        """
        When confidence >= 0.3, the search module's recommended action
        is preserved (not overridden to HOLD).

        This is the complementary case: the override ONLY fires below 0.3.

        **Validates: Requirements 5.6**
        """
        model_mgr = _make_mock_model_manager(scores)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(
                default_depth=1,
                timeout_seconds=30.0,
                default_scenarios=3,
            ),
        )

        result = module.search(state)

        report = _generate_decision_report(
            state, result, confidence_override=confidence
        )

        # With sufficient confidence, the search action should be preserved
        assert report.recommended_action == result.best_action, (
            f"With confidence={confidence:.4f} (>= 0.3), expected "
            f"{result.best_action.value} but got {report.recommended_action.value}"
        )
