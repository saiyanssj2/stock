"""
Property-based tests for philosophy-based trading strategies.

**Validates: Requirements 10.1, 10.2, 10.3, 10.4**

Property 18: Philosophy strategy signal validity
- Each strategy produces exactly one Action from {BUY, HOLD, SELL}
- That action is consistent with the strategy's documented threshold rules
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from engine.config import Action
from engine.strategies.wyckoff import WyckoffStrategy
from engine.strategies.technical import TechnicalStrategy
from engine.strategies.momentum import MomentumStrategy
from engine.strategies.mean_reversion import MeanReversionStrategy


# ---------------------------------------------------------------------------
# Hypothesis Strategies for generating valid DataFrames
# ---------------------------------------------------------------------------

VALID_ACTIONS = {Action.BUY, Action.HOLD, Action.SELL}


@st.composite
def strategy_dataframe(draw, min_rows=61, max_rows=120):
    """
    Generate a DataFrame with OHLCV and indicator columns suitable
    for all philosophy strategies.
    """
    num_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    # Generate realistic price series via random walk
    base_price = draw(st.floats(min_value=10.0, max_value=200.0))
    returns = np.array([
        draw(st.floats(min_value=-0.069, max_value=0.069))
        for _ in range(num_rows)
    ])
    returns[0] = 0.0
    closes = base_price * np.cumprod(1.0 + returns)

    # Build OHLCV
    spread_pct = 0.015
    highs = closes * (1.0 + np.abs(np.random.default_rng(42).uniform(0, spread_pct, num_rows)))
    lows = closes * (1.0 - np.abs(np.random.default_rng(43).uniform(0, spread_pct, num_rows)))
    opens = lows + (highs - lows) * np.random.default_rng(44).uniform(0.2, 0.8, num_rows)
    volumes = np.random.default_rng(45).uniform(100000, 50000000, num_rows)

    # Ensure OHLCV integrity
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    # Add indicator columns needed by strategies
    # RSI_14: [0, 100]
    df["RSI_14"] = draw(st.lists(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=num_rows, max_size=num_rows
    ))

    # ADX: [0, 100]
    df["ADX"] = draw(st.lists(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=num_rows, max_size=num_rows
    ))

    # ROC_10: rate of change, can be negative
    df["ROC_10"] = draw(st.lists(
        st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=num_rows, max_size=num_rows
    ))

    # EMA columns
    df["EMA_20"] = closes * (1.0 + np.random.default_rng(46).uniform(-0.05, 0.05, num_rows))
    df["EMA_50"] = closes * (1.0 + np.random.default_rng(47).uniform(-0.08, 0.08, num_rows))

    # MACD
    df["MACD"] = draw(st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=num_rows, max_size=num_rows
    ))
    df["MACD_signal"] = draw(st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=num_rows, max_size=num_rows
    ))
    df["MACD_hist"] = draw(st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=num_rows, max_size=num_rows
    ))

    # Bollinger Bands
    bb_middle = closes.copy()
    bb_width = np.abs(closes) * 0.05
    df["BB_upper"] = bb_middle + bb_width
    df["BB_lower"] = bb_middle - bb_width
    df["BB_middle"] = bb_middle

    # SMA_20
    df["SMA_20"] = closes * (1.0 + np.random.default_rng(48).uniform(-0.03, 0.03, num_rows))

    # OBV (for Wyckoff)
    df["OBV"] = np.cumsum(np.random.default_rng(49).uniform(-1000000, 1000000, num_rows))

    return df


@st.composite
def wyckoff_score_dataframe(draw):
    """
    Generate a DataFrame specifically designed to produce controlled Wyckoff
    composite scores for threshold testing.
    """
    num_rows = draw(st.integers(min_value=61, max_value=80))
    base_price = draw(st.floats(min_value=20.0, max_value=150.0))
    closes = np.full(num_rows, base_price)
    highs = closes * 1.01
    lows = closes * 0.99
    opens = closes.copy()
    volumes = np.full(num_rows, 1000000.0)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "OBV": np.cumsum(np.random.default_rng(50).uniform(-100000, 100000, num_rows)),
    })
    return df


@st.composite
def momentum_controlled_dataframe(draw):
    """
    Generate a DataFrame with controlled ADX, ROC_10, and volume values
    to test momentum strategy threshold rules.
    """
    num_rows = draw(st.integers(min_value=30, max_value=60))
    base_price = draw(st.floats(min_value=20.0, max_value=150.0))

    closes = np.full(num_rows, base_price)
    highs = closes * 1.01
    lows = closes * 0.99
    opens = closes.copy()

    # Draw controlled indicator values
    adx_value = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    roc_value = draw(st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False))
    volume_ratio = draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))

    avg_volume = 1000000.0
    # Set all historical volumes to the average
    volumes = np.full(num_rows, avg_volume)
    # Set current (last) volume to volume_ratio * avg
    volumes[-1] = avg_volume * volume_ratio

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "ADX": np.full(num_rows, adx_value),
        "ROC_10": np.full(num_rows, roc_value),
    })

    return df, adx_value, roc_value, volume_ratio


@st.composite
def mean_reversion_controlled_dataframe(draw):
    """
    Generate a DataFrame with controlled BB, RSI, and SMA values
    to test mean reversion strategy threshold rules.
    """
    num_rows = draw(st.integers(min_value=25, max_value=60))
    base_price = draw(st.floats(min_value=20.0, max_value=150.0))

    # Price positioning relative to indicators
    price_offset = draw(st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False))
    close_price = base_price * (1.0 + price_offset)

    closes = np.full(num_rows, close_price)
    highs = closes * 1.005
    lows = closes * 0.995
    opens = closes.copy()
    volumes = np.full(num_rows, 1000000.0)

    # Controlled indicator values
    rsi_value = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    bb_width = draw(st.floats(min_value=0.01, max_value=0.1, allow_nan=False, allow_infinity=False))
    bb_upper = base_price * (1.0 + bb_width)
    bb_lower = base_price * (1.0 - bb_width)
    sma_20 = base_price

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "RSI_14": np.full(num_rows, rsi_value),
        "BB_upper": np.full(num_rows, bb_upper),
        "BB_lower": np.full(num_rows, bb_lower),
        "SMA_20": np.full(num_rows, sma_20),
    })

    return df, close_price, rsi_value, bb_upper, bb_lower, sma_20


# ---------------------------------------------------------------------------
# Property 18: Philosophy strategy signal validity
# ---------------------------------------------------------------------------


class TestStrategySignalValidity:
    """
    Property 18: Philosophy strategy signal validity.

    Each strategy produces exactly one Action from {BUY, HOLD, SELL}
    consistent with documented threshold rules.

    **Validates: Requirements 10.1, 10.2, 10.3, 10.4**
    """

    # --- Universal validity: all strategies return exactly one valid Action ---

    @given(df=strategy_dataframe())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_wyckoff_returns_valid_action(self, df):
        """
        WyckoffStrategy always returns exactly one Action from {BUY, HOLD, SELL}.

        **Validates: Requirements 10.1**
        """
        strategy = WyckoffStrategy()
        index = len(df) - 1
        result = strategy.generate_signal(df, index)
        assert result in VALID_ACTIONS, f"WyckoffStrategy returned {result}, not a valid Action"

    @given(df=strategy_dataframe())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_technical_returns_valid_action(self, df):
        """
        TechnicalStrategy always returns exactly one Action from {BUY, HOLD, SELL}.

        **Validates: Requirements 10.2**
        """
        strategy = TechnicalStrategy()
        index = len(df) - 1
        result = strategy.generate_signal(df, index)
        assert result in VALID_ACTIONS, f"TechnicalStrategy returned {result}, not a valid Action"

    @given(df=strategy_dataframe())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_momentum_returns_valid_action(self, df):
        """
        MomentumStrategy always returns exactly one Action from {BUY, HOLD, SELL}.

        **Validates: Requirements 10.3**
        """
        strategy = MomentumStrategy()
        index = len(df) - 1
        result = strategy.generate_signal(df, index)
        assert result in VALID_ACTIONS, f"MomentumStrategy returned {result}, not a valid Action"

    @given(df=strategy_dataframe())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_mean_reversion_returns_valid_action(self, df):
        """
        MeanReversionStrategy always returns exactly one Action from {BUY, HOLD, SELL}.

        **Validates: Requirements 10.4**
        """
        strategy = MeanReversionStrategy()
        index = len(df) - 1
        result = strategy.generate_signal(df, index)
        assert result in VALID_ACTIONS, f"MeanReversionStrategy returned {result}, not a valid Action"

    # --- Wyckoff threshold consistency ---

    @given(df=wyckoff_score_dataframe())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_wyckoff_threshold_consistency(self, df):
        """
        WyckoffStrategy: score >= 3 → BUY, score <= -3 → SELL, otherwise HOLD.

        Verifies the threshold logic matches documented rules.

        **Validates: Requirements 10.1**
        """
        strategy = WyckoffStrategy()
        index = len(df) - 1
        score = strategy.compute_composite_score(df, index)
        result = strategy.generate_signal(df, index)

        if score >= 3:
            assert result == Action.BUY, (
                f"Wyckoff score={score} >= 3 but got {result}, expected BUY"
            )
        elif score <= -3:
            assert result == Action.SELL, (
                f"Wyckoff score={score} <= -3 but got {result}, expected SELL"
            )
        else:
            assert result == Action.HOLD, (
                f"Wyckoff score={score} in (-3, 3) but got {result}, expected HOLD"
            )

    # --- Technical threshold consistency ---

    @given(df=strategy_dataframe())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_technical_threshold_consistency(self, df):
        """
        TechnicalStrategy: score >= 5 → BUY, score <= -4 → SELL, otherwise HOLD.

        Verifies the threshold logic matches documented rules.

        **Validates: Requirements 10.2**
        """
        strategy = TechnicalStrategy()
        index = len(df) - 1
        score = strategy._compute_composite_score(df, index)
        result = strategy.generate_signal(df, index)

        if score >= 5:
            assert result == Action.BUY, (
                f"Technical score={score} >= 5 but got {result}, expected BUY"
            )
        elif score <= -4:
            assert result == Action.SELL, (
                f"Technical score={score} <= -4 but got {result}, expected SELL"
            )
        else:
            assert result == Action.HOLD, (
                f"Technical score={score} in (-4, 5) but got {result}, expected HOLD"
            )

    # --- Momentum threshold consistency ---

    @given(data=momentum_controlled_dataframe())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_momentum_threshold_consistency(self, data):
        """
        MomentumStrategy: ADX > 25 AND ROC > 0 AND volume > 1.5x → BUY;
        ADX > 25 AND ROC < 0 → SELL; otherwise HOLD.

        Verifies the threshold logic matches documented rules.

        **Validates: Requirements 10.3**
        """
        df, adx_value, roc_value, volume_ratio = data
        strategy = MomentumStrategy()
        index = len(df) - 1
        result = strategy.generate_signal(df, index)

        if adx_value > 25 and roc_value < 0:
            assert result == Action.SELL, (
                f"ADX={adx_value}>25, ROC={roc_value}<0 but got {result}, expected SELL"
            )
        elif adx_value > 25 and roc_value > 0 and volume_ratio > 1.5:
            assert result == Action.BUY, (
                f"ADX={adx_value}>25, ROC={roc_value}>0, vol_ratio={volume_ratio}>1.5 "
                f"but got {result}, expected BUY"
            )
        elif adx_value <= 25:
            assert result == Action.HOLD, (
                f"ADX={adx_value}<=25 but got {result}, expected HOLD"
            )

    # --- Mean Reversion threshold consistency ---

    @given(data=mean_reversion_controlled_dataframe())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_mean_reversion_buy_conditions(self, data):
        """
        MeanReversionStrategy: price < BB_lower OR RSI < 30 OR price < SMA_20 - 2σ → BUY.

        Verifies the BUY threshold logic matches documented rules.

        **Validates: Requirements 10.4**
        """
        df, close_price, rsi_value, bb_upper, bb_lower, sma_20 = data
        strategy = MeanReversionStrategy()
        index = len(df) - 1
        result = strategy.generate_signal(df, index)

        # Check if any BUY condition is met
        below_bb_lower = close_price < bb_lower
        rsi_oversold = rsi_value < 30
        # std from uniform close prices is ~0, so SMA_20-2σ check is usually close < sma_20
        std_dev = strategy._compute_std(df, index)
        below_sma_2std = (std_dev > 0 and close_price < sma_20 - 2.0 * std_dev)

        if below_bb_lower or rsi_oversold or below_sma_2std:
            assert result == Action.BUY, (
                f"Oversold conditions met (BB_lower={bb_lower}, RSI={rsi_value}, "
                f"close={close_price}, SMA_20={sma_20}, std={std_dev}) "
                f"but got {result}, expected BUY"
            )

    @given(data=mean_reversion_controlled_dataframe())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_mean_reversion_sell_conditions(self, data):
        """
        MeanReversionStrategy: price > BB_upper OR RSI > 70 OR price > SMA_20 + 2σ → SELL.

        Verifies the SELL threshold logic matches documented rules.

        **Validates: Requirements 10.4**
        """
        df, close_price, rsi_value, bb_upper, bb_lower, sma_20 = data
        strategy = MeanReversionStrategy()
        index = len(df) - 1
        result = strategy.generate_signal(df, index)

        # Check if any SELL condition is met (and no BUY condition)
        below_bb_lower = close_price < bb_lower
        rsi_oversold = rsi_value < 30
        std_dev = strategy._compute_std(df, index)
        below_sma_2std = (std_dev > 0 and close_price < sma_20 - 2.0 * std_dev)

        above_bb_upper = close_price > bb_upper
        rsi_overbought = rsi_value > 70
        above_sma_2std = (std_dev > 0 and close_price > sma_20 + 2.0 * std_dev)

        # BUY conditions take priority in the implementation (checked first)
        buy_conditions_met = below_bb_lower or rsi_oversold or below_sma_2std
        sell_conditions_met = above_bb_upper or rsi_overbought or above_sma_2std

        if sell_conditions_met and not buy_conditions_met:
            assert result == Action.SELL, (
                f"Overbought conditions met (BB_upper={bb_upper}, RSI={rsi_value}, "
                f"close={close_price}, SMA_20={sma_20}, std={std_dev}) "
                f"but got {result}, expected SELL"
            )

    @given(data=mean_reversion_controlled_dataframe())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_mean_reversion_hold_conditions(self, data):
        """
        MeanReversionStrategy: no oversold/overbought condition → HOLD.

        Verifies the HOLD condition matches documented rules.

        **Validates: Requirements 10.4**
        """
        df, close_price, rsi_value, bb_upper, bb_lower, sma_20 = data
        strategy = MeanReversionStrategy()
        index = len(df) - 1
        result = strategy.generate_signal(df, index)

        # Check conditions
        std_dev = strategy._compute_std(df, index)
        below_bb_lower = close_price < bb_lower
        rsi_oversold = rsi_value < 30
        below_sma_2std = (std_dev > 0 and close_price < sma_20 - 2.0 * std_dev)
        above_bb_upper = close_price > bb_upper
        rsi_overbought = rsi_value > 70
        above_sma_2std = (std_dev > 0 and close_price > sma_20 + 2.0 * std_dev)

        buy_conditions_met = below_bb_lower or rsi_oversold or below_sma_2std
        sell_conditions_met = above_bb_upper or rsi_overbought or above_sma_2std

        if not buy_conditions_met and not sell_conditions_met:
            assert result == Action.HOLD, (
                f"No extreme conditions (RSI={rsi_value}, close={close_price}, "
                f"BB=[{bb_lower},{bb_upper}], SMA_20={sma_20}) "
                f"but got {result}, expected HOLD"
            )
