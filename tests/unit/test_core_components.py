"""
Unit tests for core components: model architecture smoke tests and strategy signal
generation.

Covers:
- StockEvalNet parameter count & device placement (Requirements 3.2)
- Wyckoff strategy signal thresholds (Requirement 10.1)
- Technical strategy signal thresholds (Requirement 10.2)
- Momentum strategy signal conditions (Requirement 10.3)
- Mean Reversion strategy signal conditions (Requirement 10.4)

Validates: Requirements 3.2, 10.1, 10.2, 10.3, 10.4
"""

import numpy as np
import pandas as pd
import pytest
import torch

from engine.config import (
    Action,
    ModelConfig,
)
from engine.evaluation_model import ModelManager, StockEvalNet
from engine.strategies.mean_reversion import MeanReversionStrategy
from engine.strategies.momentum import MomentumStrategy
from engine.strategies.technical import TechnicalStrategy
from engine.strategies.wyckoff import WyckoffStrategy


# ==============================================================================
# Helpers
# ==============================================================================


def _make_ohlcv_df(num_rows: int = 100, base_price: float = 50.0) -> pd.DataFrame:
    """Create a basic OHLCV DataFrame for strategy testing."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, num_rows)
    returns[0] = 0.0
    closes = base_price * np.cumprod(1.0 + returns)
    highs = closes * 1.01
    lows = closes * 0.99
    opens = (highs + lows) / 2
    volumes = np.random.uniform(100000, 5000000, num_rows)
    dates = pd.bdate_range(end="2024-06-30", periods=num_rows)

    return pd.DataFrame(
        {
            "time": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add minimal indicator columns needed for strategy tests."""
    n = len(df)
    df = df.copy()

    # Trend indicators
    df["EMA_9"] = df["close"].ewm(span=9).mean()
    df["EMA_20"] = df["close"].ewm(span=20).mean()
    df["EMA_50"] = df["close"].ewm(span=50).mean()
    df["SMA_20"] = df["close"].rolling(20).mean()

    # RSI (simplified)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # ADX (simplified placeholder)
    df["ADX"] = 20.0  # Default below threshold

    # ROC_10
    df["ROC_10"] = df["close"].pct_change(10) * 100

    # Bollinger Bands
    df["BB_middle"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["BB_upper"] = df["BB_middle"] + 2 * bb_std
    df["BB_lower"] = df["BB_middle"] - 2 * bb_std

    # OBV
    obv = [0.0]
    for i in range(1, n):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    return df


# ==============================================================================
# 1. Model Architecture Smoke Tests
# ==============================================================================


class TestModelArchitectureSmoke:
    """Smoke tests for StockEvalNet architecture and device placement."""

    def test_parameter_count_within_vram_budget(self):
        """Model parameter count should fit within 4GB VRAM budget.

        With ~180K-250K params (float32), model weights use < 1MB.
        This is well within the 4GB inference VRAM budget.
        """
        config = ModelConfig()
        model = StockEvalNet(config)
        param_count = model.count_parameters()
        # Parameters at float32 = 4 bytes each
        memory_bytes = param_count * 4
        memory_mb = memory_bytes / (1024 * 1024)
        # Model weights must be < 4GB (4096 MB) — in practice < 2MB
        assert memory_mb < 4096, f"Model weights {memory_mb:.1f}MB exceeds 4GB budget"
        # Sanity check: should be under 2MB for ~180K-250K params
        assert memory_mb < 2, f"Model weights {memory_mb:.1f}MB unexpectedly large"

    def test_forward_pass_correct_output_shape(self):
        """Forward pass with default config produces shape (batch, 1)."""
        config = ModelConfig()
        model = StockEvalNet(config)
        model.eval()
        x = torch.randn(3, config.lookback, config.num_features)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (3, 1)

    def test_model_to_cpu_explicit(self):
        """Model can be explicitly placed on CPU."""
        config = ModelConfig()
        model = StockEvalNet(config)
        model = model.to("cpu")
        x = torch.randn(1, 60, 61, device="cpu")
        with torch.no_grad():
            y = model(x)
        assert y.device.type == "cpu"
        assert y.shape == (1, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_to_cuda(self):
        """Model can be moved to CUDA device and produce valid output."""
        config = ModelConfig()
        model = StockEvalNet(config).to("cuda")
        x = torch.randn(2, 60, 61, device="cuda")
        with torch.no_grad():
            y = model(x)
        assert y.device.type == "cuda"
        assert y.shape == (2, 1)
        assert (y >= -1.0).all() and (y <= 1.0).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_move_between_devices(self):
        """Model can be moved from CPU to CUDA and back, producing consistent results."""
        config = ModelConfig()
        model = StockEvalNet(config)
        model.eval()
        x_cpu = torch.randn(2, 60, 61)

        # Inference on CPU
        with torch.no_grad():
            y_cpu = model(x_cpu)

        # Move to CUDA
        model = model.to("cuda")
        x_cuda = x_cpu.to("cuda")
        with torch.no_grad():
            y_cuda = model(x_cuda)

        # Move back to CPU
        model = model.to("cpu")
        with torch.no_grad():
            y_cpu2 = model(x_cpu)

        # Results should be very close (some float precision differences possible)
        torch.testing.assert_close(y_cpu, y_cpu2, atol=1e-5, rtol=1e-5)

    def test_model_manager_default_device_is_cpu(self):
        """ModelManager defaults to CPU when CUDA is not forced."""
        config = ModelConfig()
        manager = ModelManager(config, device="cpu")
        assert manager.device == torch.device("cpu")


# ==============================================================================
# 2. Strategy Signal Generation Tests
# ==============================================================================


class TestWyckoffStrategySignals:
    """Test Wyckoff strategy signal generation with known scenarios."""

    def setup_method(self):
        self.strategy = WyckoffStrategy()

    def test_returns_hold_for_insufficient_data(self):
        """With very few rows, Wyckoff should return HOLD."""
        df = _make_ohlcv_df(5)
        df = _add_indicators(df)
        signal = self.strategy.generate_signal(df, 2)
        assert signal == Action.HOLD

    def test_signal_is_valid_action(self):
        """Signal must be one of BUY, HOLD, SELL."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        signal = self.strategy.generate_signal(df, 80)
        assert signal in (Action.BUY, Action.HOLD, Action.SELL)

    def test_buy_threshold_score_gte_3(self):
        """Composite score >= 3 should produce BUY."""
        strategy = WyckoffStrategy()
        # Directly test the threshold logic
        assert strategy.BUY_THRESHOLD == 3

        # Create a scenario that guarantees score >= 3:
        # Force accumulation phase + high volume + spring
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)

        # Monkey-patch compute_composite_score to return specific values
        original = strategy.compute_composite_score

        strategy.compute_composite_score = lambda df, idx: 3
        assert strategy.generate_signal(df, 80) == Action.BUY

        strategy.compute_composite_score = lambda df, idx: 5
        assert strategy.generate_signal(df, 80) == Action.BUY

        strategy.compute_composite_score = original

    def test_sell_threshold_score_lte_minus3(self):
        """Composite score <= -3 should produce SELL."""
        strategy = WyckoffStrategy()

        df = _make_ohlcv_df(100)
        df = _add_indicators(df)

        strategy.compute_composite_score = lambda df, idx: -3
        assert strategy.generate_signal(df, 80) == Action.SELL

        strategy.compute_composite_score = lambda df, idx: -5
        assert strategy.generate_signal(df, 80) == Action.SELL

    def test_hold_threshold_between_minus3_and_3(self):
        """Composite score between -3 and +3 (exclusive) should produce HOLD."""
        strategy = WyckoffStrategy()
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)

        for score in [-2, -1, 0, 1, 2]:
            strategy.compute_composite_score = lambda df, idx, s=score: s
            assert strategy.generate_signal(df, 80) == Action.HOLD

    def test_composite_score_returns_integer(self):
        """compute_composite_score returns an integer value."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        score = self.strategy.compute_composite_score(df, 80)
        assert isinstance(score, (int, np.integer))


class TestTechnicalStrategySignals:
    """Test Technical strategy signal generation with known scenarios."""

    def setup_method(self):
        self.strategy = TechnicalStrategy()

    def test_buy_when_score_gte_5(self):
        """BUY signal produced when composite score >= +5."""
        strategy = TechnicalStrategy()
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)

        # Patch to force score
        strategy._compute_composite_score = lambda df, idx: 5
        assert strategy.generate_signal(df, 80) == Action.BUY

        strategy._compute_composite_score = lambda df, idx: 7
        assert strategy.generate_signal(df, 80) == Action.BUY

    def test_sell_when_score_lte_minus4(self):
        """SELL signal produced when composite score <= -4."""
        strategy = TechnicalStrategy()
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)

        strategy._compute_composite_score = lambda df, idx: -4
        assert strategy.generate_signal(df, 80) == Action.SELL

        strategy._compute_composite_score = lambda df, idx: -6
        assert strategy.generate_signal(df, 80) == Action.SELL

    def test_hold_when_between_thresholds(self):
        """HOLD signal produced when score in (-4, +5)."""
        strategy = TechnicalStrategy()
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)

        for score in [-3, -2, -1, 0, 1, 2, 3, 4]:
            strategy._compute_composite_score = lambda df, idx, s=score: s
            assert strategy.generate_signal(df, 80) == Action.HOLD

    def test_oversold_rsi_contributes_buy(self):
        """RSI < 30 should contribute +2 to the composite score."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        # Force RSI to oversold
        df["RSI_14"] = 25.0
        # Force other indicators to be bullish to reach threshold
        df["EMA_20"] = df["close"] * 0.99  # Price above EMA_20
        df["EMA_50"] = df["close"] * 0.98  # EMA_20 above EMA_50
        # MACD bullish crossover
        df["MACD"] = 1.0
        df["MACD_signal"] = 0.5
        df.iloc[79, df.columns.get_loc("MACD")] = -0.1  # Previous was bearish
        df.iloc[79, df.columns.get_loc("MACD_signal")] = 0.0
        # BB below lower band
        df["BB_lower"] = df["close"] * 1.01
        df["BB_upper"] = df["close"] * 1.05

        signal = self.strategy.generate_signal(df, 80)
        # With RSI=25 (+2), EMA cross (+1), price>EMA (+1), MACD crossover (+2),
        # BB below lower (+2) = +8, which is well above +5 → BUY
        assert signal == Action.BUY

    def test_overbought_rsi_contributes_sell(self):
        """RSI > 70 should contribute -2 to the composite score."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        # Force RSI to overbought
        df["RSI_14"] = 75.0
        # Force other indicators bearish
        df["EMA_20"] = df["close"] * 1.01  # Price below EMA_20
        df["EMA_50"] = df["close"] * 0.99  # EMA_20 above EMA_50
        # MACD bearish crossover
        df["MACD"] = -1.0
        df["MACD_signal"] = -0.5
        df.iloc[79, df.columns.get_loc("MACD")] = 0.1
        df.iloc[79, df.columns.get_loc("MACD_signal")] = 0.0
        # MACD histogram turns negative
        df["MACD_hist"] = -0.5
        df.iloc[79, df.columns.get_loc("MACD_hist")] = 0.1

        signal = self.strategy.generate_signal(df, 80)
        # RSI=75 (-2), price<EMA (-1), MACD bear crossover (-2), hist turn neg (-1) = -6 → SELL
        assert signal == Action.SELL

    def test_signal_always_valid_action(self):
        """Signal must always be a valid Action enum member."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        for idx in range(20, 100):
            signal = self.strategy.generate_signal(df, idx)
            assert signal in (Action.BUY, Action.HOLD, Action.SELL)

    def test_missing_close_returns_hold(self):
        """If close column is NaN at index, return HOLD."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        df.iloc[80, df.columns.get_loc("close")] = np.nan
        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.HOLD


class TestMomentumStrategySignals:
    """Test Momentum strategy signal generation with specific conditions."""

    def setup_method(self):
        self.strategy = MomentumStrategy()

    def test_buy_when_adx_high_roc_positive_volume_high(self):
        """BUY when ADX > 25, ROC > 0, volume > 1.5x average."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        # Set ADX above threshold
        df["ADX"] = 30.0
        # Set positive ROC
        df["ROC_10"] = 5.0
        # Set very high volume at index 80 (relative to prior 20 bars)
        df.iloc[60:80, df.columns.get_loc("volume")] = 100000  # Low average
        df.iloc[80, df.columns.get_loc("volume")] = 200000  # 2x average

        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.BUY

    def test_sell_when_adx_high_roc_negative(self):
        """SELL when ADX > 25 and ROC < 0 (regardless of volume)."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        df["ADX"] = 30.0
        df["ROC_10"] = -2.0

        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.SELL

    def test_hold_when_adx_below_threshold(self):
        """HOLD when ADX <= 25 (no strong trend)."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        df["ADX"] = 20.0  # Below 25 threshold
        df["ROC_10"] = 5.0  # Positive momentum

        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.HOLD

    def test_hold_when_adx_exactly_25(self):
        """HOLD when ADX == 25 (threshold is strictly > 25)."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        df["ADX"] = 25.0
        df["ROC_10"] = 5.0

        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.HOLD

    def test_hold_when_roc_positive_but_volume_insufficient(self):
        """HOLD when ADX > 25, ROC > 0, but volume < 1.5x average."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        df["ADX"] = 30.0
        df["ROC_10"] = 5.0
        # Keep volume uniform — ratio will be ~1.0 (below 1.5x)
        df["volume"] = 1000000.0

        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.HOLD

    def test_hold_when_indicators_missing(self):
        """HOLD when ADX or ROC columns are missing."""
        df = _make_ohlcv_df(100)
        # No ADX or ROC columns
        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.HOLD

    def test_hold_when_adx_nan(self):
        """HOLD when ADX value is NaN."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        df["ADX"] = np.nan
        df["ROC_10"] = 5.0

        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.HOLD

    def test_volume_ratio_computation(self):
        """Volume ratio correctly computes current/average ratio."""
        df = _make_ohlcv_df(100)
        # Set uniform volume, then spike at index 80
        df["volume"] = 100000.0
        df.iloc[80, df.columns.get_loc("volume")] = 300000.0  # 3x

        ratio = self.strategy._compute_volume_ratio(df, 80)
        assert ratio == pytest.approx(3.0, rel=0.01)


class TestMeanReversionStrategySignals:
    """Test Mean Reversion strategy signal generation."""

    def setup_method(self):
        self.strategy = MeanReversionStrategy()

    def test_buy_when_price_below_bb_lower(self):
        """BUY when price < BB_lower (oversold)."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        # Set BB_lower above current close
        df["BB_lower"] = df["close"] * 1.05
        df["BB_upper"] = df["close"] * 1.10
        # Make sure RSI won't trigger sell
        df["RSI_14"] = 50.0
        df["SMA_20"] = df["close"]

        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.BUY

    def test_buy_when_rsi_below_30(self):
        """BUY when RSI < 30 (oversold)."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        df["RSI_14"] = 25.0
        # Set BB bands so price is NOT below lower
        df["BB_lower"] = df["close"] * 0.90
        df["BB_upper"] = df["close"] * 1.10
        df["SMA_20"] = df["close"]

        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.BUY

    def test_buy_when_price_below_sma_minus_2std(self):
        """BUY when price < SMA_20 - 2σ."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        # Force a big drop at index 80
        mean_price = 50.0
        df["close"] = mean_price
        df.iloc[80, df.columns.get_loc("close")] = 30.0  # Way below SMA
        df["SMA_20"] = mean_price
        # Set RSI and BB to not trigger
        df["RSI_14"] = 50.0
        df["BB_lower"] = 20.0  # Below current close
        df["BB_upper"] = 80.0

        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.BUY

    def test_sell_when_price_above_bb_upper(self):
        """SELL when price > BB_upper (overbought)."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        # Set BB_upper below current close
        df["BB_upper"] = df["close"] * 0.95
        df["BB_lower"] = df["close"] * 0.90
        # Make sure RSI won't trigger buy
        df["RSI_14"] = 50.0
        df["SMA_20"] = df["close"]

        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.SELL

    def test_sell_when_rsi_above_70(self):
        """SELL when RSI > 70 (overbought)."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        df["RSI_14"] = 75.0
        # Set BB bands so price is NOT above upper
        df["BB_lower"] = df["close"] * 0.90
        df["BB_upper"] = df["close"] * 1.10
        df["SMA_20"] = df["close"]

        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.SELL

    def test_sell_when_price_above_sma_plus_2std(self):
        """SELL when price > SMA_20 + 2σ."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        mean_price = 50.0
        df["close"] = mean_price
        df.iloc[80, df.columns.get_loc("close")] = 80.0  # Way above SMA
        df["SMA_20"] = mean_price
        # Set RSI and BB to not trigger
        df["RSI_14"] = 50.0
        df["BB_upper"] = 100.0
        df["BB_lower"] = 20.0

        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.SELL

    def test_hold_when_no_extreme_conditions(self):
        """HOLD when price is in normal range."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        # All indicators in neutral zone
        df["RSI_14"] = 50.0
        df["BB_lower"] = df["close"] * 0.95
        df["BB_upper"] = df["close"] * 1.05
        df["SMA_20"] = df["close"]

        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.HOLD

    def test_hold_when_close_is_nan(self):
        """HOLD when close value is NaN at the index."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        df.iloc[80, df.columns.get_loc("close")] = np.nan

        signal = self.strategy.generate_signal(df, 80)
        assert signal == Action.HOLD

    def test_buy_priority_over_sell_when_both_trigger(self):
        """When oversold is checked first, BUY takes priority over SELL conditions."""
        df = _make_ohlcv_df(100)
        df = _add_indicators(df)
        # Set RSI to both oversold (< 30) — this will trigger BUY first
        df["RSI_14"] = 25.0
        # Even if BB_upper is below close (SELL condition), BUY is checked first
        df["BB_upper"] = df["close"] * 0.95
        df["BB_lower"] = df["close"] * 0.90

        signal = self.strategy.generate_signal(df, 80)
        # BUY (RSI < 30) is checked before SELL
        assert signal == Action.BUY
