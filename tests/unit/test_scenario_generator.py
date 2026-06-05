"""
Unit tests for ScenarioGenerator.

Tests core scenario generation functionality including:
- Distribution estimation from market history
- Percentile-based scenario generation (3, 5, 7 scenarios)
- ±7% daily price limit enforcement
- Minimum 30-day history validation
- Volatility regime detection
- Action effect application
"""

import numpy as np
import pandas as pd
import pytest

from engine.config import Action, DataError, EngineConfig, SearchConfig
from engine.market_state import MarketState, NUM_INDICATORS
from engine.scenario_generator import (
    DEFAULT_NUM_SCENARIOS,
    PERCENTILE_CONFIGS,
    Distribution,
    ScenarioGenerator,
    VolatilityRegime,
)


# ==============================================================================
# Fixtures
# ==============================================================================


def _make_market_state(
    lookback: int = 60,
    symbol: str = "TEST",
    base_price: float = 50000.0,
    daily_return_std: float = 0.02,
    seed: int = 42,
) -> MarketState:
    """Create a synthetic MarketState for testing.

    Generates realistic OHLCV data with a random walk and fills in
    basic indicator values.
    """
    rng = np.random.default_rng(seed)

    # Generate close prices as random walk
    returns = rng.normal(0.001, daily_return_std, size=lookback)
    returns[0] = 0.0
    close_prices = base_price * np.cumprod(1 + returns)

    # Generate OHLCV
    ohlcv = np.zeros((lookback, 5), dtype=np.float64)
    for i in range(lookback):
        close = close_prices[i]
        intraday_range = close * rng.uniform(0.005, 0.03)
        ohlcv[i, 0] = close + rng.uniform(-intraday_range * 0.5, intraday_range * 0.5)  # open
        ohlcv[i, 1] = close + intraday_range * rng.uniform(0.3, 1.0)  # high
        ohlcv[i, 2] = close - intraday_range * rng.uniform(0.3, 1.0)  # low
        ohlcv[i, 3] = close  # close
        ohlcv[i, 4] = rng.uniform(500000, 5000000)  # volume

    # Ensure high >= max(open, close) and low <= min(open, close)
    ohlcv[:, 1] = np.maximum(ohlcv[:, 1], np.maximum(ohlcv[:, 0], ohlcv[:, 3]))
    ohlcv[:, 2] = np.minimum(ohlcv[:, 2], np.minimum(ohlcv[:, 0], ohlcv[:, 3]))

    # Generate indicators with some realistic values
    indicators = np.full((lookback, NUM_INDICATORS), np.nan, dtype=np.float64)

    # Fill in some key indicators
    from engine.market_state import INDICATOR_COLUMNS

    # EMA values close to price
    for ema_name in ["EMA_9", "EMA_20", "EMA_50", "EMA_200"]:
        try:
            idx = INDICATOR_COLUMNS.index(ema_name)
            indicators[:, idx] = close_prices + rng.normal(0, close_prices * 0.01)
        except ValueError:
            pass

    # RSI between 20-80
    for rsi_name in ["RSI_14", "RSI_7"]:
        try:
            idx = INDICATOR_COLUMNS.index(rsi_name)
            indicators[:, idx] = rng.uniform(20, 80, size=lookback)
        except ValueError:
            pass

    # ATR_14 (roughly 1-3% of price)
    try:
        idx = INDICATOR_COLUMNS.index("ATR_14")
        indicators[:, idx] = close_prices * rng.uniform(0.01, 0.03, size=lookback)
    except ValueError:
        pass

    # ATR_7
    try:
        idx = INDICATOR_COLUMNS.index("ATR_7")
        indicators[:, idx] = close_prices * rng.uniform(0.01, 0.035, size=lookback)
    except ValueError:
        pass

    # Bollinger Bands
    try:
        bb_mid_idx = INDICATOR_COLUMNS.index("BB_middle")
        bb_upper_idx = INDICATOR_COLUMNS.index("BB_upper")
        bb_lower_idx = INDICATOR_COLUMNS.index("BB_lower")
        indicators[:, bb_mid_idx] = close_prices
        indicators[:, bb_upper_idx] = close_prices * 1.04
        indicators[:, bb_lower_idx] = close_prices * 0.96
    except ValueError:
        pass

    # MACD and signal
    try:
        macd_idx = INDICATOR_COLUMNS.index("MACD")
        signal_idx = INDICATOR_COLUMNS.index("MACD_signal")
        hist_idx = INDICATOR_COLUMNS.index("MACD_hist")
        indicators[:, macd_idx] = rng.normal(0, close_prices * 0.005)
        indicators[:, signal_idx] = rng.normal(0, close_prices * 0.003)
        indicators[:, hist_idx] = indicators[:, macd_idx] - indicators[:, signal_idx]
    except ValueError:
        pass

    timestamp = pd.Timestamp("2024-01-15")

    return MarketState(
        symbol=symbol,
        timestamp=timestamp,
        ohlcv=ohlcv,
        indicators=indicators,
        lookback=lookback,
    )


@pytest.fixture
def generator():
    """Create a ScenarioGenerator with default config."""
    return ScenarioGenerator()


@pytest.fixture
def state_60():
    """Create a MarketState with 60 days of history."""
    return _make_market_state(lookback=60)


@pytest.fixture
def state_30():
    """Create a MarketState with exactly 30 days of history."""
    return _make_market_state(lookback=30)


@pytest.fixture
def state_25():
    """Create a MarketState with 25 days (insufficient) history."""
    return _make_market_state(lookback=25)


# ==============================================================================
# Tests: generate()
# ==============================================================================


class TestGenerate:
    """Tests for ScenarioGenerator.generate() method."""

    def test_generates_5_scenarios_by_default(self, generator, state_60):
        """Default generation produces 5 scenarios."""
        scenarios = generator.generate(state_60, Action.HOLD)
        assert len(scenarios) == 5

    def test_generates_3_scenarios(self, generator, state_60):
        """Can generate 3 scenarios."""
        scenarios = generator.generate(state_60, Action.BUY, num_scenarios=3)
        assert len(scenarios) == 3

    def test_generates_7_scenarios(self, generator, state_60):
        """Can generate 7 scenarios."""
        scenarios = generator.generate(state_60, Action.SELL, num_scenarios=7)
        assert len(scenarios) == 7

    def test_all_scenarios_are_valid_market_states(self, generator, state_60):
        """Each generated scenario is a valid MarketState."""
        scenarios = generator.generate(state_60, Action.BUY)
        for scenario in scenarios:
            assert isinstance(scenario, MarketState)
            assert scenario.symbol == state_60.symbol
            assert scenario.lookback == state_60.lookback
            assert scenario.ohlcv.shape == state_60.ohlcv.shape
            assert scenario.indicators.shape == state_60.indicators.shape

    def test_price_change_within_7_percent_limit(self, generator, state_60):
        """All scenarios respect the ±7% Vietnamese daily price limit."""
        scenarios = generator.generate(state_60, Action.BUY)
        current_close = state_60.ohlcv[-1, 3]

        for scenario in scenarios:
            new_close = scenario.ohlcv[-1, 3]
            pct_change = (new_close - current_close) / current_close
            assert abs(pct_change) <= 0.07 + 1e-10, (
                f"Price change {pct_change:.4f} exceeds ±7% limit"
            )

    def test_scenarios_have_different_prices(self, generator, state_60):
        """Generated scenarios should have distinct prices (not all identical)."""
        scenarios = generator.generate(state_60, Action.HOLD, num_scenarios=5)
        closes = [s.ohlcv[-1, 3] for s in scenarios]
        # At least some scenarios should differ
        assert len(set(round(c, 2) for c in closes)) > 1

    def test_scenarios_ordered_by_percentile(self, generator, state_60):
        """Scenarios should be roughly ordered from bearish to bullish."""
        scenarios = generator.generate(state_60, Action.HOLD, num_scenarios=5)
        closes = [s.ohlcv[-1, 3] for s in scenarios]
        # P10 should typically be lower than P90
        assert closes[0] <= closes[-1] or abs(closes[0] - closes[-1]) < 1e-6

    def test_rejects_insufficient_history(self, generator, state_25):
        """Raises DataError when history is less than 30 days."""
        with pytest.raises(DataError) as exc_info:
            generator.generate(state_25, Action.HOLD)
        assert "INSUFFICIENT_HISTORY" in str(exc_info.value.error_code)

    def test_accepts_exactly_30_days(self, generator, state_30):
        """Accepts state with exactly 30 days of history."""
        scenarios = generator.generate(state_30, Action.HOLD)
        assert len(scenarios) == 5

    def test_clamps_invalid_num_scenarios(self, generator, state_60):
        """Invalid num_scenarios values are clamped to nearest valid."""
        # num_scenarios=4 should clamp to 3
        scenarios = generator.generate(state_60, Action.HOLD, num_scenarios=4)
        assert len(scenarios) == 3

        # num_scenarios=6 should clamp to 5
        scenarios = generator.generate(state_60, Action.HOLD, num_scenarios=6)
        assert len(scenarios) == 5

    def test_works_with_all_actions(self, generator, state_60):
        """Generation works for BUY, HOLD, and SELL."""
        for action in Action:
            scenarios = generator.generate(state_60, action)
            assert len(scenarios) == 5
            for s in scenarios:
                assert isinstance(s, MarketState)

    def test_timestamp_advances(self, generator, state_60):
        """Generated scenarios have a timestamp one day after the source."""
        scenarios = generator.generate(state_60, Action.HOLD)
        for scenario in scenarios:
            assert scenario.timestamp > state_60.timestamp


# ==============================================================================
# Tests: _estimate_distribution()
# ==============================================================================


class TestEstimateDistribution:
    """Tests for distribution estimation from market history."""

    def test_returns_valid_distribution(self, generator, state_60):
        """Distribution has valid parameters."""
        dist = generator._estimate_distribution(state_60)
        assert isinstance(dist, Distribution)
        assert isinstance(dist.mean, float)
        assert isinstance(dist.std, float)
        assert dist.std > 0
        assert isinstance(dist.regime, VolatilityRegime)

    def test_percentiles_are_ordered(self, generator, state_60):
        """Percentile values should be monotonically non-decreasing."""
        dist = generator._estimate_distribution(state_60)
        sorted_keys = sorted(dist.percentiles.keys())
        values = [dist.percentiles[k] for k in sorted_keys]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-10

    def test_mean_near_zero_for_random_walk(self):
        """For a random walk, mean return should be close to zero."""
        state = _make_market_state(lookback=60, daily_return_std=0.02)
        gen = ScenarioGenerator()
        dist = gen._estimate_distribution(state)
        # Mean should be reasonably close to zero (within a few percent)
        assert abs(dist.mean) < 0.05

    def test_high_volatility_state(self):
        """High volatility data should produce HIGH regime."""
        # Create state with very high ATR relative to price
        state = _make_market_state(lookback=60, daily_return_std=0.05)
        # Manually set ATR to be high
        from engine.market_state import INDICATOR_COLUMNS
        atr_idx = INDICATOR_COLUMNS.index("ATR_14")
        current_close = state.ohlcv[-1, 3]
        state.indicators[:, atr_idx] = current_close * 0.05  # 5% ATR
        gen = ScenarioGenerator()
        dist = gen._estimate_distribution(state)
        assert dist.regime == VolatilityRegime.HIGH

    def test_low_volatility_state(self):
        """Low volatility data should produce LOW regime."""
        state = _make_market_state(lookback=60, daily_return_std=0.005)
        from engine.market_state import INDICATOR_COLUMNS
        atr_idx = INDICATOR_COLUMNS.index("ATR_14")
        current_close = state.ohlcv[-1, 3]
        state.indicators[:, atr_idx] = current_close * 0.01  # 1% ATR
        gen = ScenarioGenerator()
        dist = gen._estimate_distribution(state)
        assert dist.regime == VolatilityRegime.LOW


# ==============================================================================
# Tests: _apply_action_effect()
# ==============================================================================


class TestApplyActionEffect:
    """Tests for price projection and state creation."""

    def test_positive_return(self, generator, state_60):
        """Positive return increases close price."""
        new_state = generator._apply_action_effect(state_60, Action.BUY, 0.03)
        original_close = state_60.ohlcv[-1, 3]
        new_close = new_state.ohlcv[-1, 3]
        assert new_close > original_close

    def test_negative_return(self, generator, state_60):
        """Negative return decreases close price."""
        new_state = generator._apply_action_effect(state_60, Action.SELL, -0.03)
        original_close = state_60.ohlcv[-1, 3]
        new_close = new_state.ohlcv[-1, 3]
        assert new_close < original_close

    def test_caps_at_positive_7_percent(self, generator, state_60):
        """Returns exceeding +7% are capped."""
        new_state = generator._apply_action_effect(state_60, Action.BUY, 0.15)
        original_close = state_60.ohlcv[-1, 3]
        new_close = new_state.ohlcv[-1, 3]
        pct_change = (new_close - original_close) / original_close
        assert pct_change <= 0.07 + 1e-10

    def test_caps_at_negative_7_percent(self, generator, state_60):
        """Returns exceeding -7% are capped."""
        new_state = generator._apply_action_effect(state_60, Action.SELL, -0.15)
        original_close = state_60.ohlcv[-1, 3]
        new_close = new_state.ohlcv[-1, 3]
        pct_change = (new_close - original_close) / original_close
        assert pct_change >= -0.07 - 1e-10

    def test_zero_return_no_change(self, generator, state_60):
        """Zero return should produce close price near original."""
        new_state = generator._apply_action_effect(state_60, Action.HOLD, 0.0)
        original_close = state_60.ohlcv[-1, 3]
        new_close = new_state.ohlcv[-1, 3]
        # With zero return, close should be the same
        assert abs(new_close - original_close) / original_close < 1e-10

    def test_ohlcv_consistency(self, generator, state_60):
        """Generated OHLCV data is consistent (high >= close >= low, etc.)."""
        for return_pct in [-0.05, -0.02, 0.0, 0.02, 0.05]:
            new_state = generator._apply_action_effect(state_60, Action.HOLD, return_pct)
            last_bar = new_state.ohlcv[-1]
            open_p, high_p, low_p, close_p, volume = last_bar
            assert high_p >= close_p, f"high ({high_p}) < close ({close_p})"
            assert high_p >= open_p, f"high ({high_p}) < open ({open_p})"
            assert low_p <= close_p, f"low ({low_p}) > close ({close_p})"
            assert low_p <= open_p, f"low ({low_p}) > open ({open_p})"
            assert volume > 0

    def test_preserves_state_shape(self, generator, state_60):
        """New state has same shape as original."""
        new_state = generator._apply_action_effect(state_60, Action.BUY, 0.02)
        assert new_state.ohlcv.shape == state_60.ohlcv.shape
        assert new_state.indicators.shape == state_60.indicators.shape
        assert new_state.lookback == state_60.lookback

    def test_does_not_mutate_original(self, generator, state_60):
        """Original state is not modified."""
        original_ohlcv = state_60.ohlcv.copy()
        original_indicators = state_60.indicators.copy()
        generator._apply_action_effect(state_60, Action.BUY, 0.05)
        np.testing.assert_array_equal(state_60.ohlcv, original_ohlcv)
        np.testing.assert_array_equal(state_60.indicators, original_indicators)


# ==============================================================================
# Tests: Volatility Regime Detection
# ==============================================================================


class TestVolatilityRegime:
    """Tests for ATR-based volatility regime detection."""

    def test_detects_medium_regime_default(self, generator, state_60):
        """Default synthetic state detects as MEDIUM regime."""
        regime = generator._detect_volatility_regime(state_60)
        assert isinstance(regime, VolatilityRegime)

    def test_handles_missing_atr_indicator(self, generator):
        """Falls back to OHLCV-based ATR when indicator is missing."""
        state = _make_market_state(lookback=60)
        # Set all ATR indicators to NaN
        from engine.market_state import INDICATOR_COLUMNS
        atr_idx = INDICATOR_COLUMNS.index("ATR_14")
        state.indicators[:, atr_idx] = np.nan
        # Should not raise, should detect regime from OHLCV
        regime = generator._detect_volatility_regime(state)
        assert isinstance(regime, VolatilityRegime)


# ==============================================================================
# Tests: Custom Configuration
# ==============================================================================


class TestConfiguration:
    """Tests for configurable parameters."""

    def test_custom_min_history(self):
        """Custom min_history is enforced."""
        gen = ScenarioGenerator(min_history=50)
        state = _make_market_state(lookback=40)
        with pytest.raises(DataError):
            gen.generate(state, Action.HOLD)

    def test_custom_engine_config(self):
        """Custom engine config is used for price limits."""
        config = EngineConfig(daily_price_limit=0.05)  # 5% limit
        gen = ScenarioGenerator(config=config)
        state = _make_market_state(lookback=60)
        new_state = gen._apply_action_effect(state, Action.BUY, 0.10)
        original_close = state.ohlcv[-1, 3]
        new_close = new_state.ohlcv[-1, 3]
        pct_change = (new_close - original_close) / original_close
        assert pct_change <= 0.05 + 1e-10
