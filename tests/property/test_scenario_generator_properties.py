"""
Property-based tests for ScenarioGenerator scenario generation validity.

**Validates: Requirements 4.6, 4.7, 4.8**

Properties tested:
- Property 14: Scenario generation validity
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.config import Action, DataError, EngineConfig, SearchConfig
from engine.market_state import MarketState, NUM_INDICATORS, NUM_OHLCV, INDICATOR_COLUMNS
from engine.scenario_generator import (
    PERCENTILE_CONFIGS,
    ScenarioGenerator,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_HISTORY = 30  # Minimum days required for scenario generation
MIN_LOOKBACK = 20
MAX_LOOKBACK = 200


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------


@st.composite
def market_state_with_sufficient_history(draw):
    """
    Generate a valid MarketState with at least 30 days of history.

    Produces realistic OHLCV data with a random walk and populates key
    indicators (ATR_14, EMA, RSI) for proper scenario generation.
    """
    lookback = draw(st.integers(min_value=MIN_HISTORY, max_value=MAX_LOOKBACK))
    symbol = draw(st.sampled_from(["VNM", "FPT", "VCB", "HPG", "MWG", "TCB"]))

    # Generate close prices as random walk with ±7% daily moves (VN market)
    base_price = draw(st.floats(min_value=10.0, max_value=200.0))
    closes = np.zeros(lookback, dtype=np.float64)
    closes[0] = base_price
    for i in range(1, lookback):
        daily_return = draw(st.floats(min_value=-0.065, max_value=0.065))
        closes[i] = closes[i - 1] * (1.0 + daily_return)
        # Ensure price stays positive
        closes[i] = max(closes[i], 1.0)

    # Generate OHLCV from close prices
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

    # Ensure OHLCV consistency: high >= max(open, close), low <= min(open, close)
    ohlcv[:, 1] = np.maximum(ohlcv[:, 1], np.maximum(ohlcv[:, 0], ohlcv[:, 3]))
    ohlcv[:, 2] = np.minimum(ohlcv[:, 2], np.minimum(ohlcv[:, 0], ohlcv[:, 3]))

    # Generate indicators with realistic values for key indicators
    indicators = np.full((lookback, NUM_INDICATORS), np.nan, dtype=np.float64)

    # ATR_14: roughly 1-3% of price (needed for volatility regime detection)
    try:
        atr_idx = INDICATOR_COLUMNS.index("ATR_14")
        indicators[:, atr_idx] = closes * draw(
            st.floats(min_value=0.01, max_value=0.03)
        )
    except ValueError:
        pass

    # EMA values near price
    for ema_name in ["EMA_9", "EMA_20", "EMA_50"]:
        try:
            idx = INDICATOR_COLUMNS.index(ema_name)
            indicators[:, idx] = closes * (1.0 + np.random.uniform(-0.02, 0.02, lookback))
        except ValueError:
            pass

    # RSI between 20-80
    for rsi_name in ["RSI_14", "RSI_7"]:
        try:
            idx = INDICATOR_COLUMNS.index(rsi_name)
            indicators[:, idx] = np.random.uniform(20, 80, size=lookback)
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


@st.composite
def market_state_with_insufficient_history(draw):
    """
    Generate a valid MarketState with fewer than 30 days of history.

    These states should be rejected by ScenarioGenerator.generate().
    """
    lookback = draw(st.integers(min_value=MIN_LOOKBACK, max_value=MIN_HISTORY - 1))
    symbol = draw(st.sampled_from(["VNM", "FPT", "VCB", "HPG"]))

    base_price = draw(st.floats(min_value=10.0, max_value=200.0))
    closes = np.zeros(lookback, dtype=np.float64)
    closes[0] = base_price
    for i in range(1, lookback):
        daily_return = draw(st.floats(min_value=-0.05, max_value=0.05))
        closes[i] = closes[i - 1] * (1.0 + daily_return)
        closes[i] = max(closes[i], 1.0)

    # Build OHLCV
    ohlcv = np.zeros((lookback, NUM_OHLCV), dtype=np.float64)
    for i in range(lookback):
        close = closes[i]
        spread = close * 0.01
        ohlcv[i, 0] = close + draw(st.floats(min_value=-spread, max_value=spread))
        ohlcv[i, 1] = max(ohlcv[i, 0], close) + draw(st.floats(min_value=0.0, max_value=spread))
        ohlcv[i, 2] = min(ohlcv[i, 0], close) - draw(st.floats(min_value=0.0, max_value=spread))
        ohlcv[i, 3] = close
        ohlcv[i, 4] = draw(st.floats(min_value=100000.0, max_value=10000000.0))

    ohlcv[:, 1] = np.maximum(ohlcv[:, 1], np.maximum(ohlcv[:, 0], ohlcv[:, 3]))
    ohlcv[:, 2] = np.minimum(ohlcv[:, 2], np.minimum(ohlcv[:, 0], ohlcv[:, 3]))

    indicators = np.full((lookback, NUM_INDICATORS), np.nan, dtype=np.float64)

    timestamp = pd.Timestamp("2024-01-15")

    return MarketState(
        symbol=symbol,
        timestamp=timestamp,
        ohlcv=ohlcv,
        indicators=indicators,
        lookback=lookback,
    )


# ---------------------------------------------------------------------------
# Property 14: Scenario generation validity
# ---------------------------------------------------------------------------


class TestScenarioGenerationValidity:
    """
    Property 14: Scenario generation validity.

    For any valid MarketState with at least 30 days of history, the
    ScenarioGenerator SHALL produce between 3 and 7 scenarios (inclusive),
    where each scenario's projected price change does not exceed ±7% from
    the current price, and all generated MarketStates have valid structure.
    For any state with fewer than 30 days of history, scenario generation
    SHALL reject the request.

    **Validates: Requirements 4.6, 4.7, 4.8**
    """

    @given(
        state=market_state_with_sufficient_history(),
        num_scenarios=st.sampled_from([3, 5, 7]),
        action=st.sampled_from([Action.BUY, Action.HOLD, Action.SELL]),
    )
    @settings(max_examples=20)
    def test_generates_correct_number_of_scenarios(self, state, num_scenarios, action):
        """
        For any MarketState with ≥30 days history, generate() produces
        exactly the requested number of scenarios (3, 5, or 7).

        **Validates: Requirements 4.7**
        """
        generator = ScenarioGenerator()
        scenarios = generator.generate(state, action, num_scenarios=num_scenarios)

        assert len(scenarios) == num_scenarios, (
            f"Expected {num_scenarios} scenarios, got {len(scenarios)} "
            f"for state with lookback={state.lookback}"
        )
        assert 3 <= len(scenarios) <= 7

    @given(
        state=market_state_with_sufficient_history(),
        action=st.sampled_from([Action.BUY, Action.HOLD, Action.SELL]),
    )
    @settings(max_examples=20)
    def test_price_change_within_7_percent_limit(self, state, action):
        """
        Each scenario's close price differs from the original by at most ±7%.

        The Vietnamese market has a daily price limit of ±7%, and all
        generated scenarios must respect this constraint.

        **Validates: Requirements 4.8**
        """
        generator = ScenarioGenerator()
        scenarios = generator.generate(state, action)

        current_close = state.ohlcv[-1, 3]
        assume(current_close > 0)  # Ensure valid base price

        for i, scenario in enumerate(scenarios):
            new_close = scenario.ohlcv[-1, 3]
            pct_change = (new_close - current_close) / current_close

            assert abs(pct_change) <= 0.07 + 1e-9, (
                f"Scenario {i}: price change {pct_change:.6f} exceeds ±7% limit. "
                f"current_close={current_close:.2f}, new_close={new_close:.2f}"
            )

    @given(state=market_state_with_insufficient_history())
    @settings(max_examples=20)
    def test_rejects_insufficient_history(self, state):
        """
        For MarketState with <30 days history, DataError is raised.

        The ScenarioGenerator requires a minimum of 30 trading days of
        historical data for meaningful distribution estimation.

        **Validates: Requirements 4.6**
        """
        generator = ScenarioGenerator()

        with pytest.raises(DataError) as exc_info:
            generator.generate(state, Action.HOLD)

        assert exc_info.value.error_code == "INSUFFICIENT_HISTORY"

    @given(
        state=market_state_with_sufficient_history(),
        action=st.sampled_from([Action.BUY, Action.HOLD, Action.SELL]),
        num_scenarios=st.sampled_from([3, 5, 7]),
    )
    @settings(max_examples=20)
    def test_all_scenarios_are_valid_market_states(self, state, action, num_scenarios):
        """
        All generated scenarios are valid MarketState objects with correct
        structure: same symbol, same lookback, correct array shapes, and
        advancing timestamp.

        **Validates: Requirements 4.6, 4.7**
        """
        generator = ScenarioGenerator()
        scenarios = generator.generate(state, action, num_scenarios=num_scenarios)

        for i, scenario in enumerate(scenarios):
            # Must be a MarketState instance
            assert isinstance(scenario, MarketState), (
                f"Scenario {i} is not a MarketState instance"
            )

            # Symbol preserved
            assert scenario.symbol == state.symbol, (
                f"Scenario {i}: symbol mismatch "
                f"({scenario.symbol} != {state.symbol})"
            )

            # Lookback preserved
            assert scenario.lookback == state.lookback, (
                f"Scenario {i}: lookback mismatch "
                f"({scenario.lookback} != {state.lookback})"
            )

            # OHLCV shape preserved
            assert scenario.ohlcv.shape == (state.lookback, NUM_OHLCV), (
                f"Scenario {i}: OHLCV shape mismatch "
                f"({scenario.ohlcv.shape} != {(state.lookback, NUM_OHLCV)})"
            )

            # Indicators shape preserved
            assert scenario.indicators.shape == (state.lookback, NUM_INDICATORS), (
                f"Scenario {i}: indicators shape mismatch "
                f"({scenario.indicators.shape} != {(state.lookback, NUM_INDICATORS)})"
            )

            # Timestamp advances (new state is in the future)
            assert scenario.timestamp > state.timestamp, (
                f"Scenario {i}: timestamp did not advance "
                f"({scenario.timestamp} <= {state.timestamp})"
            )

            # OHLCV values should be finite and positive for the last bar
            last_bar = scenario.ohlcv[-1]
            assert np.all(np.isfinite(last_bar)), (
                f"Scenario {i}: last OHLCV bar contains non-finite values"
            )
            assert last_bar[3] > 0, (
                f"Scenario {i}: close price is not positive ({last_bar[3]})"
            )
            assert last_bar[4] > 0, (
                f"Scenario {i}: volume is not positive ({last_bar[4]})"
            )

            # OHLCV consistency: high >= max(open, close), low <= min(open, close)
            open_p, high_p, low_p, close_p, _ = last_bar
            assert high_p >= close_p - 1e-10, (
                f"Scenario {i}: high ({high_p}) < close ({close_p})"
            )
            assert high_p >= open_p - 1e-10, (
                f"Scenario {i}: high ({high_p}) < open ({open_p})"
            )
            assert low_p <= close_p + 1e-10, (
                f"Scenario {i}: low ({low_p}) > close ({close_p})"
            )
            assert low_p <= open_p + 1e-10, (
                f"Scenario {i}: low ({low_p}) > open ({open_p})"
            )
