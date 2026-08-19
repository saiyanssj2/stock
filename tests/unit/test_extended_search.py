"""
Unit tests for the Extended Search functionality in SearchModule.

Tests cover:
- Extended search configuration validation
- _run_extended_search() iterative deepening
- Standard search baseline comparison
- Progress callback reporting
- cancel_search() behavior
- Timeout behavior returning best from deepest complete level
- Branching factor reduction at deeper levels (5 → 4 → 3)
"""

import time
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from engine.config import Action, ConfigError, SearchConfig
from engine.extended_search import (
    ExtendedDecisionReport,
    ExtendedSearchConfig,
    ExtendedSearchProgress,
)
from engine.market_state import MarketState, NUM_INDICATORS
from engine.search_module import SearchModule, SearchResult


# ==============================================================================
# Fixtures
# ==============================================================================


def _make_market_state(
    symbol: str = "FPT",
    lookback: int = 60,
    close_price: float = 100.0,
) -> MarketState:
    """Create a minimal valid MarketState for testing."""
    ohlcv = np.zeros((lookback, 5), dtype=np.float64)
    ohlcv[:, 0] = close_price * 0.99  # open
    ohlcv[:, 1] = close_price * 1.01  # high
    ohlcv[:, 2] = close_price * 0.98  # low
    ohlcv[:, 3] = close_price  # close
    ohlcv[:, 4] = 1_000_000  # volume

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


def _make_mock_scenario_generator():
    """Create a mock ScenarioGenerator that returns copies of the input state."""

    def generate_fn(input_state, action, num_scenarios=5):
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
# ExtendedSearchConfig Tests
# ==============================================================================


class TestExtendedSearchConfig:
    """Tests for ExtendedSearchConfig validation."""

    def test_valid_default_config(self):
        """Default config is valid."""
        config = ExtendedSearchConfig()
        assert config.depth == 10
        assert config.timeout == 1800.0
        assert config.enabled is False

    def test_valid_custom_config(self):
        """Custom valid config works."""
        config = ExtendedSearchConfig(depth=5, timeout=300.0, enabled=True)
        assert config.depth == 5
        assert config.timeout == 300.0
        assert config.enabled is True

    def test_invalid_depth_below_3(self):
        """Depth below 3 raises ConfigError."""
        with pytest.raises(ConfigError):
            ExtendedSearchConfig(depth=2)

    def test_invalid_depth_above_10(self):
        """Depth above 10 raises ConfigError."""
        with pytest.raises(ConfigError):
            ExtendedSearchConfig(depth=11)

    def test_invalid_timeout_below_60(self):
        """Timeout below 60 raises ConfigError."""
        with pytest.raises(ConfigError):
            ExtendedSearchConfig(timeout=59.0)

    def test_invalid_timeout_above_1800(self):
        """Timeout above 1800 raises ConfigError."""
        with pytest.raises(ConfigError):
            ExtendedSearchConfig(timeout=1801.0)


# ==============================================================================
# Extended Search - Iterative Deepening Tests
# ==============================================================================


class TestExtendedSearchIterativeDeepening:
    """Tests for _run_extended_search() iterative deepening."""

    def test_returns_extended_decision_report(self):
        """Extended search returns an ExtendedDecisionReport."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.5)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=3, timeout_seconds=10.0),
        )

        config = ExtendedSearchConfig(depth=4, timeout=60.0, enabled=True)
        report = module._run_extended_search(state, config)

        assert isinstance(report, ExtendedDecisionReport)
        assert report.recommended_action in [Action.BUY, Action.HOLD, Action.SELL]
        assert report.symbol == "FPT"

    def test_effective_depth_at_least_3(self):
        """Extended search achieves at least standard depth 3."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.4)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=3, timeout_seconds=10.0),
        )

        config = ExtendedSearchConfig(depth=5, timeout=120.0, enabled=True)
        report = module._run_extended_search(state, config)

        assert report.effective_depth_reached >= 3

    def test_standard_comparison_fields_populated(self):
        """Standard search comparison fields are populated."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.6)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=3, timeout_seconds=10.0),
        )

        config = ExtendedSearchConfig(depth=4, timeout=60.0, enabled=True)
        report = module._run_extended_search(state, config)

        # Standard search fields should be filled
        assert report.standard_recommendation is not None
        assert report.standard_recommendation in [Action.BUY, Action.HOLD, Action.SELL]
        assert report.standard_confidence is not None
        assert 0.0 <= report.standard_confidence <= 1.0
        assert report.standard_position_score is not None
        assert -1.0 <= report.standard_position_score <= 1.0

    def test_total_nodes_evaluated_tracked(self):
        """Total nodes evaluated is tracked across all depth levels."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.5)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=3, timeout_seconds=10.0),
        )

        config = ExtendedSearchConfig(depth=4, timeout=60.0, enabled=True)
        report = module._run_extended_search(state, config)

        assert report.total_nodes_evaluated > 0

    def test_time_elapsed_tracked(self):
        """Time elapsed is tracked."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.5)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=3, timeout_seconds=10.0),
        )

        config = ExtendedSearchConfig(depth=3, timeout=60.0, enabled=True)
        report = module._run_extended_search(state, config)

        assert report.time_elapsed_seconds > 0.0

    def test_recommendations_agree_field(self):
        """recommendations_agree field correctly reflects agreement."""
        state = _make_market_state()
        # With a fixed model score, both standard and extended should agree
        model_mgr = _make_mock_model_manager(score=0.5)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=3, timeout_seconds=10.0),
        )

        config = ExtendedSearchConfig(depth=3, timeout=60.0, enabled=True)
        report = module._run_extended_search(state, config)

        # With same model and same depth, they should agree
        assert report.recommendations_agree is True


# ==============================================================================
# Progress Callback Tests
# ==============================================================================


class TestExtendedSearchProgressCallback:
    """Tests for progress reporting via callback."""

    def test_progress_callback_called(self):
        """Progress callback is called during search."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.5)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=3, timeout_seconds=10.0),
        )

        progress_reports = []

        def on_progress(progress: ExtendedSearchProgress):
            progress_reports.append(progress)

        config = ExtendedSearchConfig(depth=4, timeout=60.0, enabled=True)
        module._run_extended_search(state, config, progress_callback=on_progress)

        # Should have received at least one progress report
        assert len(progress_reports) > 0

    def test_progress_reports_contain_required_fields(self):
        """Each progress report has all required fields."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.5)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=3, timeout_seconds=10.0),
        )

        progress_reports = []

        def on_progress(progress: ExtendedSearchProgress):
            progress_reports.append(progress)

        config = ExtendedSearchConfig(depth=4, timeout=60.0, enabled=True)
        module._run_extended_search(state, config, progress_callback=on_progress)

        for report in progress_reports:
            assert isinstance(report, ExtendedSearchProgress)
            assert report.current_depth >= 3
            assert report.max_depth == 4
            assert report.nodes_evaluated >= 0
            assert report.elapsed_seconds >= 0.0

    def test_progress_reports_increasing_nodes(self):
        """Nodes evaluated increases across progress reports."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.5)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=3, timeout_seconds=10.0),
        )

        progress_reports = []

        def on_progress(progress: ExtendedSearchProgress):
            progress_reports.append(progress)

        config = ExtendedSearchConfig(depth=5, timeout=120.0, enabled=True)
        module._run_extended_search(state, config, progress_callback=on_progress)

        # Nodes should be non-decreasing across reports
        if len(progress_reports) >= 2:
            for i in range(1, len(progress_reports)):
                assert progress_reports[i].nodes_evaluated >= progress_reports[i - 1].nodes_evaluated


# ==============================================================================
# Cancel Search Tests
# ==============================================================================


class TestCancelSearch:
    """Tests for cancel_search() behavior."""

    def test_cancel_sets_flag(self):
        """cancel_search() sets the _cancel_requested flag."""
        module = SearchModule(
            model_manager=_make_mock_model_manager(),
            scenario_generator=_make_mock_scenario_generator(),
        )

        assert module._cancel_requested is False
        module.cancel_search()
        assert module._cancel_requested is True

    def test_cancel_returns_best_from_deepest_complete_level(self):
        """Cancelled search returns best action from deepest complete level."""
        state = _make_market_state()

        # Create a slow model that gives us time to cancel
        call_count = [0]

        def slow_predict(features):
            call_count[0] += 1
            time.sleep(0.02)  # 20ms per evaluation
            return np.array([[0.5]], dtype=np.float32)

        model_mgr = MagicMock()
        model_mgr.predict.side_effect = slow_predict
        model_mgr.is_loaded = True

        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=3, timeout_seconds=60.0),
        )

        config = ExtendedSearchConfig(depth=8, timeout=120.0, enabled=True)

        # Cancel after a short delay
        def cancel_later():
            time.sleep(0.5)
            module.cancel_search()

        cancel_thread = threading.Thread(target=cancel_later)
        cancel_thread.start()

        report = module._run_extended_search(state, config)

        cancel_thread.join()

        # Should have returned a valid report
        assert isinstance(report, ExtendedDecisionReport)
        assert report.recommended_action in [Action.BUY, Action.HOLD, Action.SELL]
        # Effective depth should be less than requested max (cancelled early)
        assert report.effective_depth_reached < 8

    def test_cancel_flag_reset_on_new_search(self):
        """_cancel_requested is reset at the start of a new extended search."""
        state = _make_market_state()
        model_mgr = _make_mock_model_manager(score=0.5)
        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=3, timeout_seconds=10.0),
        )

        # Set cancel flag
        module._cancel_requested = True

        # Start extended search - should reset the flag internally
        config = ExtendedSearchConfig(depth=3, timeout=60.0, enabled=True)
        report = module._run_extended_search(state, config)

        # Search should complete normally since flag is reset
        assert isinstance(report, ExtendedDecisionReport)


# ==============================================================================
# Timeout Behavior Tests
# ==============================================================================


class TestExtendedSearchTimeout:
    """Tests for timeout behavior in extended search."""

    def test_timeout_returns_best_from_deepest_complete(self):
        """On timeout, returns best from deepest fully evaluated level."""
        state = _make_market_state()

        # Create a slow model to force timeout
        def slow_predict(features):
            time.sleep(0.03)  # 30ms per eval
            return np.array([[0.5]], dtype=np.float32)

        model_mgr = MagicMock()
        model_mgr.predict.side_effect = slow_predict
        model_mgr.is_loaded = True

        scenario_gen = _make_mock_scenario_generator()

        module = SearchModule(
            model_manager=model_mgr,
            scenario_generator=scenario_gen,
            config=SearchConfig(default_depth=3, timeout_seconds=10.0),
        )

        # Short timeout will force early termination
        config = ExtendedSearchConfig(depth=10, timeout=60.0, enabled=True)
        report = module._run_extended_search(state, config)

        assert isinstance(report, ExtendedDecisionReport)
        assert report.effective_depth_reached >= 3  # At least standard depth
        assert report.effective_depth_reached < 10  # Should not complete max depth


# ==============================================================================
# Branching Factor Reduction Tests
# ==============================================================================


class TestExtendedBranchingFactor:
    """Tests for branching factor reduction at deeper levels."""

    def test_branching_factor_at_shallow_depth(self):
        """Branching factor is 5 at shallow levels."""
        module = SearchModule(
            model_manager=_make_mock_model_manager(),
            scenario_generator=_make_mock_scenario_generator(),
        )

        factor = module._get_extended_branching_factor(current_depth=0, max_depth=10)
        assert factor == 5

    def test_branching_factor_at_medium_depth(self):
        """Branching factor is 4 at medium levels (depth 3-4)."""
        module = SearchModule(
            model_manager=_make_mock_model_manager(),
            scenario_generator=_make_mock_scenario_generator(),
        )

        factor = module._get_extended_branching_factor(current_depth=3, max_depth=10)
        assert factor == 4

        factor = module._get_extended_branching_factor(current_depth=4, max_depth=10)
        assert factor == 4

    def test_branching_factor_at_deep_levels(self):
        """Branching factor is 3 at deep levels (depth 5+)."""
        module = SearchModule(
            model_manager=_make_mock_model_manager(),
            scenario_generator=_make_mock_scenario_generator(),
        )

        factor = module._get_extended_branching_factor(current_depth=5, max_depth=10)
        assert factor == 3

        factor = module._get_extended_branching_factor(current_depth=8, max_depth=10)
        assert factor == 3

    def test_branching_reduction_pattern(self):
        """Branching follows the 5 → 4 → 3 pattern as depth increases."""
        module = SearchModule(
            model_manager=_make_mock_model_manager(),
            scenario_generator=_make_mock_scenario_generator(),
        )

        factors = [
            module._get_extended_branching_factor(d, max_depth=10) for d in range(10)
        ]

        # First 3 levels: 5 scenarios
        assert all(f == 5 for f in factors[:3])
        # Next 2 levels: 4 scenarios
        assert all(f == 4 for f in factors[3:5])
        # Remaining levels: 3 scenarios
        assert all(f == 3 for f in factors[5:])
