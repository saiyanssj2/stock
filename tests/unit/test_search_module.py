"""
Unit tests for the SearchModule (Minimax + Alpha-Beta pruning).

Tests cover:
- SearchNode dataclass creation
- Basic search functionality
- Alpha-Beta pruning correctness
- Timeout handling and graceful degradation
- Adaptive branching factor
- Depth control
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from engine.config import Action, DataError, SearchConfig
from engine.market_state import MarketState
from engine.search_module import SearchModule, SearchNode, SearchResult, _forward_fill_2d


# ==============================================================================
# Fixtures
# ==============================================================================


def _make_market_state(
    symbol: str = "FPT",
    lookback: int = 60,
    close_price: float = 100.0,
) -> MarketState:
    """Create a minimal valid MarketState for testing."""
    import pandas as pd

    ohlcv = np.zeros((lookback, 5), dtype=np.float64)
    # Set reasonable OHLCV data
    ohlcv[:, 0] = close_price * 0.99  # open
    ohlcv[:, 1] = close_price * 1.01  # high
    ohlcv[:, 2] = close_price * 0.98  # low
    ohlcv[:, 3] = close_price  # close
    ohlcv[:, 4] = 1_000_000  # volume

    from engine.market_state import NUM_INDICATORS

    indicators = np.random.default_rng(42).random((lookback, NUM_INDICATORS)) * 100

    return MarketState(
        symbol=symbol,
        timestamp=pd.Timestamp("2024-01-15"),
        ohlcv=ohlcv,
        indicators=indicators,
        lookback=lookback,
    )


def _make_mock_model_manager(score: float = 0.5):
    """Create a mock ModelManager that returns a fixed score."""
    mock = MagicMock()
    mock.predict.return_value = np.array([[score]], dtype=np.float32)
    mock.is_loaded = True
    return mock


def _make_mock_scenario_generator(state: MarketState = None):
    """Create a mock ScenarioGenerator that returns copies of the input state."""

    def generate_fn(input_state, action, num_scenarios=5):
        # Return num_scenarios copies (with slight variation)
        results = []
        for i in range(num_scenarios):
            new_ohlcv = input_state.ohlcv.copy()
            new_ohlcv[:, 3] *= 1.0 + (i - num_scenarios // 2) * 0.01
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


# ==============================================================================
# SearchNode Tests
# ==============================================================================


class TestSearchNode:
    """Tests for the SearchNode dataclass."""

    def test_create_basic_node(self):
        """SearchNode can be created with required fields."""
        state = _make_market_state()
        node = SearchNode(state=state, action=Action.BUY, score=0.5, depth=0)
        assert node.action == Action.BUY
        assert node.score == 0.5
        assert node.depth == 0
        assert node.children == []

    def test_create_node_with_children(self):
        """SearchNode can have child nodes."""
        state = _make_market_state()
        child = SearchNode(state=state, action=Action.HOLD, score=0.3, depth=1)
        parent = SearchNode(
            state=state, action=Action.BUY, score=0.5, depth=0, children=[child]
        )
        assert len(parent.children) == 1
        assert parent.children[0].action == Action.HOLD

    def test_default_values(self):
        """SearchNode has sensible defaults."""
        state = _make_market_state()
        node = SearchNode(state=state)
        assert node.action is None
        assert node.score == 0.0
        assert node.children == []
        assert node.depth == 0


# ==============================================================================
# SearchModule Basic Tests
# ==============================================================================


class TestSearchModuleBasic:
    """Basic functionality tests for SearchModule."""

    def test_search_returns_result(self):
        """Search returns a valid SearchResult."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.3)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=1, timeout_seconds=10.0),
        )

        result = module.search(state)

        assert isinstance(result, SearchResult)
        assert result.best_action in [Action.BUY, Action.HOLD, Action.SELL]
        assert -1.0 <= result.best_score <= 1.0
        assert result.nodes_evaluated > 0

    def test_search_uses_model_for_evaluation(self):
        """Search calls model manager predict for leaf evaluation."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.7)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=1, timeout_seconds=10.0),
        )

        result = module.search(state)

        assert model_mgr.predict.called
        assert result.nodes_evaluated > 0

    def test_search_evaluates_all_actions(self):
        """Search evaluates BUY, HOLD, and SELL at root level."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.5)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=1, timeout_seconds=10.0),
        )

        result = module.search(state)

        # Should have scores for all 3 actions
        assert len(result.action_scores) == 3
        assert Action.BUY in result.action_scores
        assert Action.HOLD in result.action_scores
        assert Action.SELL in result.action_scores

    def test_best_action_has_highest_score(self):
        """The recommended action has the highest minimax score."""
        state = _make_market_state()

        # Model returns different scores based on input to create differentiation
        call_count = [0]
        scores = [0.8, 0.3, 0.1, 0.6, 0.2, 0.9, 0.4, 0.5, 0.7, 0.2, 0.1, 0.3, 0.5, 0.6, 0.4]

        def varying_predict(features):
            idx = call_count[0] % len(scores)
            call_count[0] += 1
            return np.array([[scores[idx]]], dtype=np.float32)

        model_mgr = MagicMock()
        model_mgr.predict.side_effect = varying_predict
        model_mgr.is_loaded = True

        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=1, timeout_seconds=10.0),
        )

        result = module.search(state)

        # Best action should have the highest score among evaluated actions
        best_score = result.action_scores[result.best_action]
        for action, score in result.action_scores.items():
            assert best_score >= score


# ==============================================================================
# Depth Control Tests
# ==============================================================================


class TestSearchDepthControl:
    """Tests for depth configuration and control."""

    def test_default_depth(self):
        """Search uses default depth of 3 from config."""
        config = SearchConfig(default_depth=3)
        module = SearchModule(
            model_manager=_make_mock_model_manager(),
            scenario_generator=_make_mock_scenario_generator(),
            config=config,
        )
        assert module.config.default_depth == 3

    def test_max_depth_clamped(self):
        """Depth is clamped to max_depth."""
        state = _make_market_state()
        config = SearchConfig(default_depth=3, max_depth=5, timeout_seconds=30.0)
        model_mgr = _make_mock_model_manager(score=0.5)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=config,
        )

        # Requesting depth > max should be clamped
        result = module.search(state, depth=10)
        # Should complete without error (clamped to max 5)
        assert isinstance(result, SearchResult)

    def test_depth_1_evaluates_leaves_directly(self):
        """Depth 1 evaluates scenarios directly without recursion."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.6)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=1, timeout_seconds=10.0),
        )

        result = module.search(state, depth=1)
        # At depth 1: 3 actions × 5 scenarios = 15 leaf evaluations (approx)
        assert result.nodes_evaluated > 0
        assert result.nodes_evaluated <= 25  # Allow some flexibility

    def test_depth_override(self):
        """Search respects depth parameter override."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.5)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=3, timeout_seconds=30.0),
        )

        # Depth 1 should evaluate fewer nodes than depth 2
        result_d1 = module.search(state, depth=1)
        result_d2 = module.search(state, depth=2)

        assert result_d2.nodes_evaluated >= result_d1.nodes_evaluated


# ==============================================================================
# Timeout Tests
# ==============================================================================


class TestSearchTimeout:
    """Tests for timeout handling and graceful degradation."""

    def test_timeout_returns_partial_result(self):
        """On timeout, search returns best action from evaluated portion."""
        state = _make_market_state()

        # Create a slow model to trigger timeout
        def slow_predict(features):
            time.sleep(0.05)  # 50ms per evaluation
            return np.array([[0.5]], dtype=np.float32)

        model_mgr = MagicMock()
        model_mgr.predict.side_effect = slow_predict
        model_mgr.is_loaded = True

        scenario_gen = _make_mock_scenario_generator()

        # Very short timeout
        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(
                default_depth=5, timeout_seconds=0.2, default_scenarios=5
            ),
        )

        result = module.search(state)

        # Should complete (possibly with timeout)
        assert isinstance(result, SearchResult)
        assert result.best_action in [Action.BUY, Action.HOLD, Action.SELL]

    def test_timeout_flag_set(self):
        """The timed_out flag is set when search exceeds timeout."""
        state = _make_market_state()

        def slow_predict(features):
            time.sleep(0.1)
            return np.array([[0.5]], dtype=np.float32)

        model_mgr = MagicMock()
        model_mgr.predict.side_effect = slow_predict
        model_mgr.is_loaded = True

        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(
                default_depth=5, timeout_seconds=0.15, default_scenarios=5
            ),
        )

        result = module.search(state)

        # With very short timeout and deep search, should timeout
        assert result.timed_out is True

    def test_fast_search_no_timeout(self):
        """A fast search does not trigger timeout."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.5)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=1, timeout_seconds=10.0),
        )

        result = module.search(state)

        assert result.timed_out is False


# ==============================================================================
# Adaptive Branching Tests
# ==============================================================================


class TestAdaptiveBranching:
    """Tests for adaptive branching factor reduction."""

    def test_branching_reduced_at_deep_levels(self):
        """Branching factor is reduced when depth > threshold."""
        config = SearchConfig(
            default_depth=4,
            adaptive_depth_threshold=3,
            adaptive_scenario_count=3,
            default_scenarios=5,
        )
        module = SearchModule(
            model_manager=_make_mock_model_manager(),
            scenario_generator=_make_mock_scenario_generator(),
            config=config,
        )

        # At shallow depth with max_depth=4 (> threshold=3)
        factor_shallow = module._get_branching_factor(current_depth=0, max_depth=4)
        # At deep depth
        factor_deep = module._get_branching_factor(current_depth=3, max_depth=4)

        assert factor_deep == 3  # Reduced
        assert factor_shallow == 5  # Normal

    def test_standard_branching_at_default_depth(self):
        """Standard branching when max_depth <= threshold."""
        config = SearchConfig(
            default_depth=3,
            adaptive_depth_threshold=3,
            adaptive_scenario_count=3,
            default_scenarios=5,
        )
        module = SearchModule(
            model_manager=_make_mock_model_manager(),
            scenario_generator=_make_mock_scenario_generator(),
            config=config,
        )

        # At depth=3 (not exceeding threshold), shallow levels get full branching
        factor = module._get_branching_factor(current_depth=0, max_depth=3)
        assert factor == 5


# ==============================================================================
# Scenario Generation Error Handling
# ==============================================================================


class TestSearchErrorHandling:
    """Tests for error handling in the search module."""

    def test_scenario_generation_failure_uses_neutral_score(self):
        """If scenario generation fails for an action, it gets score 0.0."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.7)

        # Scenario generator that fails for BUY
        def failing_generate(input_state, action, num_scenarios=5):
            if action == Action.BUY:
                raise DataError("Insufficient history")
            results = []
            for i in range(num_scenarios):
                results.append(
                    MarketState(
                        symbol=input_state.symbol,
                        timestamp=input_state.timestamp,
                        ohlcv=input_state.ohlcv.copy(),
                        indicators=input_state.indicators.copy(),
                        lookback=input_state.lookback,
                    )
                )
            return results

        scenario_gen = MagicMock()
        scenario_gen.generate.side_effect = failing_generate

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=1, timeout_seconds=10.0),
        )

        result = module.search(state)

        # BUY should have score 0.0 due to failure
        assert result.action_scores[Action.BUY] == 0.0
        # Other actions should have valid scores
        assert Action.HOLD in result.action_scores
        assert Action.SELL in result.action_scores

    def test_model_evaluation_failure_returns_zero(self):
        """If model prediction fails, leaf evaluation returns 0.0."""
        state = _make_market_state()

        model_mgr = MagicMock()
        model_mgr.predict.side_effect = RuntimeError("CUDA error")
        model_mgr.is_loaded = True

        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=1, timeout_seconds=10.0),
        )

        result = module.search(state)

        # Should not crash, returns neutral result
        assert isinstance(result, SearchResult)
        assert result.best_action in [Action.BUY, Action.HOLD, Action.SELL]


# ==============================================================================
# Helper Function Tests
# ==============================================================================


class TestForwardFill:
    """Tests for the _forward_fill_2d helper."""

    def test_no_nans(self):
        """No-op when there are no NaNs."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = _forward_fill_2d(arr)
        np.testing.assert_array_equal(result, arr)

    def test_fills_nans_forward(self):
        """NaN values are forward-filled from previous valid values."""
        arr = np.array([[1.0, 2.0], [np.nan, 4.0], [np.nan, np.nan]])
        result = _forward_fill_2d(arr)
        expected = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_leading_nans_not_filled(self):
        """Leading NaN values remain NaN (no backward-fill), but subsequent NaN are filled."""
        arr = np.array([[np.nan, 2.0], [1.0, np.nan], [np.nan, 4.0]])
        result = _forward_fill_2d(arr)
        # col 0: NaN at idx 0 has no prior value → remains NaN
        assert np.isnan(result[0, 0])
        # col 0: idx 1 is valid (1.0)
        assert result[1, 0] == 1.0
        # col 0: idx 2 NaN is forward-filled from idx 1 → 1.0
        assert result[2, 0] == 1.0
        # col 1: idx 0 is valid (2.0), idx 1 NaN forward-filled → 2.0
        assert result[1, 1] == 2.0
        # col 1: idx 2 is valid (4.0)
        assert result[2, 1] == 4.0

    def test_all_nans(self):
        """All-NaN columns remain NaN."""
        arr = np.array([[np.nan], [np.nan], [np.nan]])
        result = _forward_fill_2d(arr)
        assert np.all(np.isnan(result))
