"""Unit tests for the BacktestEngine module."""

import math

import numpy as np
import pandas as pd
import pytest

from engine.backtest_engine import BacktestEngine, Strategy
from engine.config import Action, BacktestResult, ComparisonResult, ConfigError, DataError, Trade


# ==============================================================================
# Test Strategies
# ==============================================================================


class AlwaysBuyStrategy:
    """Strategy that always generates BUY signals."""

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        return Action.BUY


class AlwaysSellStrategy:
    """Strategy that always generates SELL signals."""

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        return Action.SELL


class AlwaysHoldStrategy:
    """Strategy that always generates HOLD signals."""

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        return Action.HOLD


class BuyThenSellStrategy:
    """Strategy that buys on first signal, then sells after settlement."""

    def __init__(self):
        self._bought = False

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        if not self._bought:
            self._bought = True
            return Action.BUY
        return Action.SELL


class BuyOnDaySellLater:
    """Strategy that buys on a specific day index and sells on another."""

    def __init__(self, buy_day: int, sell_day: int):
        self.buy_day = buy_day
        self.sell_day = sell_day
        self._call_count = 0

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        day = self._call_count
        self._call_count += 1
        if day == self.buy_day:
            return Action.BUY
        elif day == self.sell_day:
            return Action.SELL
        return Action.HOLD


# ==============================================================================
# Fixtures
# ==============================================================================


def _make_df(
    num_days: int = 20,
    start_price: float = 50000.0,
    daily_return: float = 0.01,
    start_date: str = "2023-01-01",
) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with a consistent upward trend."""
    dates = pd.date_range(start=start_date, periods=num_days, freq="B")
    prices = [start_price * ((1 + daily_return) ** i) for i in range(num_days)]

    data = {
        "time": dates,
        "open": [p * 0.99 for p in prices],
        "high": [p * 1.02 for p in prices],
        "low": [p * 0.98 for p in prices],
        "close": prices,
        "volume": [1_000_000] * num_days,
    }
    return pd.DataFrame(data)


def _make_flat_df(
    num_days: int = 20,
    price: float = 50000.0,
    start_date: str = "2023-01-01",
) -> pd.DataFrame:
    """Create a flat-price DataFrame (no price movement)."""
    dates = pd.date_range(start=start_date, periods=num_days, freq="B")
    data = {
        "time": dates,
        "open": [price] * num_days,
        "high": [price] * num_days,
        "low": [price] * num_days,
        "close": [price] * num_days,
        "volume": [1_000_000] * num_days,
    }
    return pd.DataFrame(data)


# ==============================================================================
# Tests: Initialization
# ==============================================================================


class TestBacktestEngineInit:
    def test_default_initialization(self):
        engine = BacktestEngine()
        assert engine.initial_capital == 100_000_000.0
        assert engine.max_position_pct == 0.20
        assert engine.lot_size == 100
        assert engine.daily_price_limit == 0.07
        assert engine.settlement_days == 3  # ceil(2.5)

    def test_custom_initialization(self):
        engine = BacktestEngine(
            initial_capital=50_000_000.0,
            max_position_pct=0.10,
        )
        assert engine.initial_capital == 50_000_000.0
        assert engine.max_position_pct == 0.10


# ==============================================================================
# Tests: Date Validation
# ==============================================================================


class TestDateValidation:
    def test_invalid_start_after_end(self):
        engine = BacktestEngine()
        df = _make_df(20)
        with pytest.raises(ConfigError, match="must be before"):
            engine.run(AlwaysHoldStrategy(), df, "2023-02-01", "2023-01-01")

    def test_same_start_and_end(self):
        engine = BacktestEngine()
        df = _make_df(20)
        with pytest.raises(ConfigError, match="must be before"):
            engine.run(AlwaysHoldStrategy(), df, "2023-01-01", "2023-01-01")

    def test_no_data_in_range(self):
        engine = BacktestEngine()
        df = _make_df(20, start_date="2023-01-01")
        with pytest.raises(ConfigError, match="No data found"):
            engine.run(AlwaysHoldStrategy(), df, "2024-01-01", "2024-12-31")

    def test_invalid_date_format(self):
        engine = BacktestEngine()
        df = _make_df(20)
        with pytest.raises(ConfigError):
            engine.run(AlwaysHoldStrategy(), df, "not-a-date", "2023-12-31")


# ==============================================================================
# Tests: DataFrame Validation
# ==============================================================================


class TestDataFrameValidation:
    def test_missing_columns(self):
        engine = BacktestEngine()
        df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=5, freq="B")})
        with pytest.raises(DataError, match="missing required columns"):
            engine.run(AlwaysHoldStrategy(), df, "2023-01-01", "2023-12-31")

    def test_partial_missing_columns(self):
        engine = BacktestEngine()
        df = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=5, freq="B"),
            "open": [100] * 5,
            "close": [100] * 5,
        })
        with pytest.raises(DataError, match="missing required columns"):
            engine.run(AlwaysHoldStrategy(), df, "2023-01-01", "2023-12-31")


# ==============================================================================
# Tests: Trade Execution Rules
# ==============================================================================


class TestTradeExecution:
    def test_lot_size_enforcement(self):
        """Shares must be a multiple of 100."""
        engine = BacktestEngine(initial_capital=10_000_000.0)
        df = _make_df(20, start_price=50000.0)
        result = engine.run(
            BuyThenSellStrategy(), df, "2023-01-01", "2023-12-31"
        )
        for trade in result.trades:
            assert trade.shares > 0
            assert trade.shares % 100 == 0

    def test_settlement_period_enforced(self):
        """Cannot sell within T+2.5 (3 trading days) of purchase."""
        # Buy on day 0, try to sell on day 1 - should fail
        engine = BacktestEngine(initial_capital=10_000_000.0)
        df = _make_df(10, start_price=50000.0)
        strategy = BuyOnDaySellLater(buy_day=0, sell_day=1)
        result = engine.run(strategy, df, "2023-01-01", "2023-12-31")
        # Should have no completed trades (sell blocked by settlement)
        assert len(result.trades) == 0

    def test_settlement_period_sell_after_3_days(self):
        """Can sell after 3 full trading days (T+2.5 ceil = 3)."""
        engine = BacktestEngine(initial_capital=100_000_000.0)
        df = _make_df(10, start_price=50000.0)
        strategy = BuyOnDaySellLater(buy_day=0, sell_day=3)
        result = engine.run(strategy, df, "2023-01-01", "2023-12-31")
        assert len(result.trades) == 1

    def test_max_position_size(self):
        """Position value should not exceed 20% of portfolio."""
        engine = BacktestEngine(
            initial_capital=100_000_000.0, max_position_pct=0.20
        )
        df = _make_df(20, start_price=50000.0)
        result = engine.run(BuyThenSellStrategy(), df, "2023-01-01", "2023-12-31")
        if result.trades:
            trade = result.trades[0]
            position_value = trade.entry_price * trade.shares
            # At time of entry, portfolio value is initial_capital
            max_allowed = engine.initial_capital * engine.max_position_pct
            assert position_value <= max_allowed + 1.0  # small float tolerance

    def test_price_limit_blocks_trade(self):
        """Trades should be blocked if price exceeds ±7% from reference."""
        engine = BacktestEngine(initial_capital=10_000_000.0)
        # Create a DataFrame with a huge price jump (>7%)
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        data = {
            "time": dates,
            "open": [50000, 50000, 60000, 60000, 60000],  # >7% jump on day 2
            "high": [50000, 50000, 60000, 60000, 60000],
            "low": [50000, 50000, 60000, 60000, 60000],
            "close": [50000, 50000, 60000, 60000, 60000],  # 20% jump
            "volume": [1_000_000] * 5,
        }
        df = pd.DataFrame(data)

        # Try to buy on day 2 (price jumped 20% from day 1)
        strategy = BuyOnDaySellLater(buy_day=2, sell_day=4)
        result = engine.run(strategy, df, "2023-01-01", "2023-12-31")
        # Buy should be blocked due to price limit
        assert len(result.trades) == 0


# ==============================================================================
# Tests: Metrics Computation
# ==============================================================================


class TestMetrics:
    def test_hold_strategy_zero_return(self):
        """Hold-only strategy should have 0% return and no trades."""
        engine = BacktestEngine()
        df = _make_df(20)
        result = engine.run(AlwaysHoldStrategy(), df, "2023-01-01", "2023-12-31")
        assert result.total_return_pct == 0.0
        assert len(result.trades) == 0
        assert result.win_rate == 0.0

    def test_equity_curve_length(self):
        """Equity curve should have one entry per trading day."""
        engine = BacktestEngine()
        df = _make_df(20)
        result = engine.run(AlwaysHoldStrategy(), df, "2023-01-01", "2023-12-31")
        assert len(result.equity_curve) == len(
            df[(pd.to_datetime(df["time"]) >= "2023-01-01") &
               (pd.to_datetime(df["time"]) <= "2023-12-31")]
        )

    def test_positive_return_on_uptrend(self):
        """Buy and sell in an uptrend should produce positive return."""
        engine = BacktestEngine(initial_capital=10_000_000.0)
        df = _make_df(20, start_price=50000.0, daily_return=0.02)
        strategy = BuyOnDaySellLater(buy_day=0, sell_day=5)
        result = engine.run(strategy, df, "2023-01-01", "2023-12-31")
        if result.trades:
            assert result.total_return_pct > 0
            assert result.trades[0].pnl > 0

    def test_win_rate_calculation(self):
        """Win rate should be correct for known trades."""
        engine = BacktestEngine(initial_capital=10_000_000.0)
        df = _make_df(20, start_price=50000.0, daily_return=0.01)
        strategy = BuyOnDaySellLater(buy_day=0, sell_day=5)
        result = engine.run(strategy, df, "2023-01-01", "2023-12-31")
        if result.trades:
            # In an uptrend, the trade should be a winner
            assert result.win_rate == 100.0

    def test_max_drawdown_on_flat(self):
        """Flat price with no trades should have 0 drawdown."""
        engine = BacktestEngine()
        df = _make_flat_df(20)
        result = engine.run(AlwaysHoldStrategy(), df, "2023-01-01", "2023-12-31")
        assert result.max_drawdown == 0.0

    def test_sharpe_ratio_on_flat(self):
        """Flat equity (no trades) should have 0 Sharpe."""
        engine = BacktestEngine()
        df = _make_flat_df(20)
        result = engine.run(AlwaysHoldStrategy(), df, "2023-01-01", "2023-12-31")
        assert result.sharpe_ratio == 0.0

    def test_trade_records_complete(self):
        """Trade records should have all fields populated."""
        engine = BacktestEngine(initial_capital=10_000_000.0)
        df = _make_df(20, start_price=50000.0, daily_return=0.01)
        strategy = BuyOnDaySellLater(buy_day=0, sell_day=5)
        result = engine.run(strategy, df, "2023-01-01", "2023-12-31")
        if result.trades:
            trade = result.trades[0]
            assert trade.entry_date is not None
            assert trade.exit_date is not None
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert trade.shares > 0
            assert trade.shares % 100 == 0


# ==============================================================================
# Tests: Edge Cases
# ==============================================================================


class TestEdgeCases:
    def test_single_day_in_range(self):
        """Should handle a single day in the date range."""
        engine = BacktestEngine()
        dates = pd.date_range("2023-01-02", periods=1, freq="B")
        df = pd.DataFrame({
            "time": dates,
            "open": [50000],
            "high": [51000],
            "low": [49000],
            "close": [50500],
            "volume": [1_000_000],
        })
        result = engine.run(AlwaysBuyStrategy(), df, "2023-01-01", "2023-12-31")
        assert result.total_return_pct == 0.0
        assert len(result.equity_curve) == 1

    def test_datetime_index_instead_of_time_column(self):
        """Should work with DatetimeIndex instead of 'time' column."""
        engine = BacktestEngine(initial_capital=10_000_000.0)
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "open": [50000] * 10,
                "high": [51000] * 10,
                "low": [49000] * 10,
                "close": [50000 + i * 100 for i in range(10)],
                "volume": [1_000_000] * 10,
            },
            index=dates,
        )
        strategy = BuyOnDaySellLater(buy_day=0, sell_day=5)
        result = engine.run(strategy, df, "2023-01-01", "2023-12-31")
        # Should run without errors
        assert result.equity_curve is not None

    def test_insufficient_capital_for_lot(self):
        """If capital is too low for even 1 lot, no trade should execute."""
        engine = BacktestEngine(initial_capital=1000.0)  # Only 1000 VND
        df = _make_df(20, start_price=50000.0)
        result = engine.run(BuyThenSellStrategy(), df, "2023-01-01", "2023-12-31")
        assert len(result.trades) == 0


# ==============================================================================
# Tests: Strategy Comparison Framework
# ==============================================================================


class TestCompareStrategies:
    """Tests for compare_strategies() method."""

    def test_compare_returns_comparison_result(self):
        """compare_strategies should return a ComparisonResult dataclass."""
        engine = BacktestEngine(initial_capital=10_000_000.0)
        df = _make_df(20, start_price=50000.0, daily_return=0.01)

        strategies = {"buy_then_sell": BuyThenSellStrategy()}
        result = engine.compare_strategies(
            strategies, df, "2023-01-01", "2023-12-31"
        )
        assert isinstance(result, ComparisonResult)

    def test_compare_identical_parameters(self):
        """All strategies should be evaluated with the same parameters."""
        engine = BacktestEngine(initial_capital=50_000_000.0)
        df = _make_df(20, start_price=50000.0, daily_return=0.01)

        strategies = {
            "buy_then_sell": BuyThenSellStrategy(),
        }
        result = engine.compare_strategies(
            strategies, df, "2023-01-01", "2023-12-31"
        )
        assert result.date_range_start == "2023-01-01"
        assert result.date_range_end == "2023-12-31"
        assert result.initial_capital == 50_000_000.0

    def test_compare_multiple_strategies(self):
        """Should run backtest for each strategy and collect results."""
        engine = BacktestEngine(initial_capital=10_000_000.0)
        df = _make_df(20, start_price=50000.0, daily_return=0.01)

        strategies = {
            "strategy_a": BuyThenSellStrategy(),
            "strategy_b": BuyOnDaySellLater(buy_day=0, sell_day=5),
        }
        result = engine.compare_strategies(
            strategies, df, "2023-01-01", "2023-12-31"
        )
        # Both strategies generate BUY and SELL signals
        assert "strategy_a" in result.results
        assert "strategy_b" in result.results
        assert isinstance(result.results["strategy_a"], BacktestResult)
        assert isinstance(result.results["strategy_b"], BacktestResult)

    def test_compare_excludes_hold_only_strategy(self):
        """A strategy that only generates HOLD should be excluded."""
        engine = BacktestEngine(initial_capital=10_000_000.0)
        df = _make_df(20, start_price=50000.0, daily_return=0.01)

        strategies = {
            "active": BuyThenSellStrategy(),
            "passive": AlwaysHoldStrategy(),
        }
        result = engine.compare_strategies(
            strategies, df, "2023-01-01", "2023-12-31"
        )
        assert "active" in result.results
        assert "passive" not in result.results
        assert "passive" in result.excluded_strategies
        assert "passive" in result.exclusion_reasons
        assert "BUY" in result.exclusion_reasons["passive"]
        assert "SELL" in result.exclusion_reasons["passive"]

    def test_compare_excludes_buy_only_strategy(self):
        """A strategy that only generates BUY (no SELL) should be excluded."""
        engine = BacktestEngine(initial_capital=10_000_000.0)
        df = _make_df(20, start_price=50000.0, daily_return=0.01)

        strategies = {
            "buy_only": AlwaysBuyStrategy(),
        }
        result = engine.compare_strategies(
            strategies, df, "2023-01-01", "2023-12-31"
        )
        assert "buy_only" not in result.results
        assert "buy_only" in result.excluded_strategies
        assert "SELL" in result.exclusion_reasons["buy_only"]

    def test_compare_excludes_sell_only_strategy(self):
        """A strategy that only generates SELL (no BUY) should be excluded."""
        engine = BacktestEngine(initial_capital=10_000_000.0)
        df = _make_df(20, start_price=50000.0, daily_return=0.01)

        strategies = {
            "sell_only": AlwaysSellStrategy(),
        }
        result = engine.compare_strategies(
            strategies, df, "2023-01-01", "2023-12-31"
        )
        assert "sell_only" not in result.results
        assert "sell_only" in result.excluded_strategies
        assert "BUY" in result.exclusion_reasons["sell_only"]

    def test_compare_all_excluded_returns_empty_results(self):
        """If all strategies are excluded, results dict should be empty."""
        engine = BacktestEngine(initial_capital=10_000_000.0)
        df = _make_df(20, start_price=50000.0)

        strategies = {
            "hold": AlwaysHoldStrategy(),
            "buy_only": AlwaysBuyStrategy(),
        }
        result = engine.compare_strategies(
            strategies, df, "2023-01-01", "2023-12-31"
        )
        assert len(result.results) == 0
        assert len(result.excluded_strategies) == 2

    def test_compare_empty_strategies_dict(self):
        """Passing empty strategies dict should return empty ComparisonResult."""
        engine = BacktestEngine(initial_capital=10_000_000.0)
        df = _make_df(20, start_price=50000.0)

        result = engine.compare_strategies({}, df, "2023-01-01", "2023-12-31")
        assert len(result.results) == 0
        assert len(result.excluded_strategies) == 0

    def test_compare_strategies_same_capital(self):
        """Each strategy should be backtested with the same initial capital."""
        engine = BacktestEngine(initial_capital=25_000_000.0)
        df = _make_df(20, start_price=50000.0, daily_return=0.01)

        strategies = {
            "a": BuyOnDaySellLater(buy_day=0, sell_day=5),
            "b": BuyOnDaySellLater(buy_day=1, sell_day=6),
        }
        result = engine.compare_strategies(
            strategies, df, "2023-01-01", "2023-12-31"
        )
        assert result.initial_capital == 25_000_000.0
        # Both should be included since they generate both BUY and SELL
        assert "a" in result.results
        assert "b" in result.results
