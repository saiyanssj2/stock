"""
Property-based tests for BacktestEngine trade constraints and strategy comparison fairness.

**Validates: Requirements 9.2, 9.3, 9.4, 10.5**

Properties tested:
- Property 17: Backtest trade constraints (Vietnamese market rules)
- Property 19: Strategy comparison fairness
"""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.backtest_engine import BacktestEngine
from engine.config import Action, BacktestResult, EngineConfig, Trade
from engine.strategies.base import BaseStrategy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOT_SIZE = 100
DAILY_PRICE_LIMIT = 0.07  # ±7%
SETTLEMENT_DAYS = 3  # ceil(2.5)
MAX_POSITION_PCT = 0.20  # 20%


# ---------------------------------------------------------------------------
# Test Strategies
# ---------------------------------------------------------------------------


class ParameterizedStrategy(BaseStrategy):
    """Strategy that returns signals from a predefined list."""

    def __init__(self, signals: list):
        self._signals = signals
        self._call_count = 0

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        if self._call_count < len(self._signals):
            signal = self._signals[self._call_count]
        else:
            signal = Action.HOLD
        self._call_count += 1
        return signal

    @property
    def name(self) -> str:
        return "ParameterizedStrategy"


class FixedSignalStrategy(BaseStrategy):
    """Strategy that always returns the same signal."""

    def __init__(self, signal: Action, strategy_name: str = "FixedSignal"):
        self._signal = signal
        self._name = strategy_name

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        return self._signal

    @property
    def name(self) -> str:
        return self._name


class AlternatingBuySellStrategy(BaseStrategy):
    """Strategy that alternates between BUY and SELL signals."""

    def __init__(self, strategy_name: str = "Alternating"):
        self._state = "buy"
        self._name = strategy_name

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        if self._state == "buy":
            self._state = "sell"
            return Action.BUY
        else:
            self._state = "buy"
            return Action.SELL

    @property
    def name(self) -> str:
        return self._name


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------


@st.composite
def ohlcv_dataframe_strategy(draw, min_days=20, max_days=60):
    """
    Generate a valid OHLCV DataFrame with realistic price movement
    respecting the ±7% daily price limit.
    """
    num_days = draw(st.integers(min_value=min_days, max_value=max_days))

    # Start price between 10 and 200 (realistic Vietnamese stock price in 1000 VND)
    base_price = draw(st.floats(min_value=20.0, max_value=150.0))

    # Generate daily returns within ±7% (Vietnamese market constraint)
    closes = np.zeros(num_days)
    closes[0] = base_price
    for i in range(1, num_days):
        daily_return = draw(st.floats(min_value=-0.065, max_value=0.065))
        closes[i] = closes[i - 1] * (1.0 + daily_return)
        closes[i] = max(closes[i], 1.0)  # Ensure positive price

    # Derive OHLCV ensuring consistency and ±7% limit adherence
    opens = np.zeros(num_days)
    highs = np.zeros(num_days)
    lows = np.zeros(num_days)
    volumes = np.zeros(num_days)

    opens[0] = closes[0]
    for i in range(1, num_days):
        # Open is close of previous day ± small gap (within 7% of prev close)
        gap_pct = draw(st.floats(min_value=-0.02, max_value=0.02))
        opens[i] = closes[i - 1] * (1.0 + gap_pct)

    # For first day, open ~= close
    opens[0] = closes[0] * (1.0 + draw(st.floats(min_value=-0.01, max_value=0.01)))

    for i in range(num_days):
        high_spread = draw(st.floats(min_value=0.001, max_value=0.03))
        low_spread = draw(st.floats(min_value=0.001, max_value=0.03))
        highs[i] = max(opens[i], closes[i]) * (1.0 + high_spread)
        lows[i] = min(opens[i], closes[i]) * (1.0 - low_spread)
        volumes[i] = draw(st.floats(min_value=100000.0, max_value=10000000.0))

    # Ensure OHLCV consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    dates = pd.bdate_range(start="2023-01-02", periods=num_days)

    df = pd.DataFrame({
        "time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    return df


@st.composite
def signal_sequence_strategy(draw, min_len=5, max_len=30):
    """
    Generate a sequence of trading signals.

    Ensures at least one BUY followed by at least one SELL
    (after settlement period) to produce trades.
    """
    length = draw(st.integers(min_value=max(min_len, 10), max_value=max_len))

    # Place a BUY early and a SELL after settlement
    buy_pos = draw(st.integers(min_value=0, max_value=min(3, length - SETTLEMENT_DAYS - 2)))
    sell_pos = draw(st.integers(
        min_value=buy_pos + SETTLEMENT_DAYS,
        max_value=min(buy_pos + SETTLEMENT_DAYS + 10, length - 1),
    ))

    signals = [Action.HOLD] * length
    signals[buy_pos] = Action.BUY
    signals[sell_pos] = Action.SELL

    return signals


@st.composite
def initial_capital_strategy(draw):
    """Generate a realistic initial capital amount (in VND)."""
    return draw(st.floats(min_value=10_000_000.0, max_value=1_000_000_000.0))


@st.composite
def strategy_names_strategy(draw, min_count=2, max_count=5):
    """Generate a list of distinct strategy names."""
    count = draw(st.integers(min_value=min_count, max_value=max_count))
    all_names = ["AI_Engine", "Wyckoff", "Technical", "Momentum", "MeanReversion"]
    return all_names[:count]


# ---------------------------------------------------------------------------
# Property 17: Backtest trade constraints (Vietnamese market rules)
# ---------------------------------------------------------------------------


class TestBacktestTradeConstraints:
    """
    Property 17: Backtest trade constraints.

    For any backtest execution, ALL executed trades SHALL satisfy:
    (a) share quantity is a positive multiple of 100
    (b) executed price does not exceed ±7% from the reference price of that session
    (c) no sell occurs on a position within 2.5 trading days of its purchase
    (d) entry position value does not exceed 20% of portfolio value at time of entry

    **Validates: Requirements 9.2, 9.3, 9.4**
    """

    @given(
        df=ohlcv_dataframe_strategy(min_days=20, max_days=50),
        signals=signal_sequence_strategy(min_len=15, max_len=50),
        capital=initial_capital_strategy(),
    )
    @settings(max_examples=100, suppress_health_check=[])
    def test_lot_size_multiple_of_100(self, df, signals, capital):
        """
        All executed trades have share quantities that are positive multiples of 100.

        **Validates: Requirements 9.2**
        """
        # Ensure signal sequence length matches available data
        assume(len(signals) <= len(df))

        engine = BacktestEngine(initial_capital=capital, max_position_pct=MAX_POSITION_PCT)
        strategy = ParameterizedStrategy(signals[:len(df)])

        start_date = str(df["time"].iloc[0].date())
        end_date = str(df["time"].iloc[-1].date())

        result = engine.run(strategy, df, start_date, end_date)

        for i, trade in enumerate(result.trades):
            assert trade.shares > 0, (
                f"Trade {i}: shares must be positive, got {trade.shares}"
            )
            assert trade.shares % LOT_SIZE == 0, (
                f"Trade {i}: shares {trade.shares} is not a multiple of {LOT_SIZE}"
            )

    @given(
        df=ohlcv_dataframe_strategy(min_days=20, max_days=50),
        signals=signal_sequence_strategy(min_len=15, max_len=50),
        capital=initial_capital_strategy(),
    )
    @settings(max_examples=100, suppress_health_check=[])
    def test_price_within_daily_limit(self, df, signals, capital):
        """
        All executed trade prices do not exceed ±7% from the reference price.

        The reference price is the previous session's closing price. Trades
        executing at prices beyond this limit should be rejected by the engine.

        **Validates: Requirements 9.2**
        """
        assume(len(signals) <= len(df))

        engine = BacktestEngine(initial_capital=capital, max_position_pct=MAX_POSITION_PCT)
        strategy = ParameterizedStrategy(signals[:len(df)])

        start_date = str(df["time"].iloc[0].date())
        end_date = str(df["time"].iloc[-1].date())

        result = engine.run(strategy, df, start_date, end_date)

        # For each trade, verify that the entry and exit prices are within
        # ±7% of their respective reference prices (previous day close)
        for i, trade in enumerate(result.trades):
            # Find entry day in the DataFrame
            entry_mask = pd.to_datetime(df["time"]).dt.date == trade.entry_date.date()
            entry_rows = df[entry_mask]
            if len(entry_rows) > 0:
                entry_idx = entry_rows.index[0]
                # Reference price for entry: previous session close
                if entry_idx > 0:
                    ref_price_entry = float(df.iloc[entry_idx - 1]["close"])
                else:
                    ref_price_entry = float(df.iloc[entry_idx]["open"])

                if ref_price_entry > 0:
                    change_pct = abs(trade.entry_price - ref_price_entry) / ref_price_entry
                    assert change_pct <= DAILY_PRICE_LIMIT + 1e-9, (
                        f"Trade {i} entry: price {trade.entry_price:.2f} exceeds "
                        f"±7% from reference {ref_price_entry:.2f} "
                        f"(change: {change_pct*100:.2f}%)"
                    )

            # Find exit day in the DataFrame
            exit_mask = pd.to_datetime(df["time"]).dt.date == trade.exit_date.date()
            exit_rows = df[exit_mask]
            if len(exit_rows) > 0:
                exit_idx = exit_rows.index[0]
                if exit_idx > 0:
                    ref_price_exit = float(df.iloc[exit_idx - 1]["close"])
                else:
                    ref_price_exit = float(df.iloc[exit_idx]["open"])

                if ref_price_exit > 0:
                    change_pct = abs(trade.exit_price - ref_price_exit) / ref_price_exit
                    assert change_pct <= DAILY_PRICE_LIMIT + 1e-9, (
                        f"Trade {i} exit: price {trade.exit_price:.2f} exceeds "
                        f"±7% from reference {ref_price_exit:.2f} "
                        f"(change: {change_pct*100:.2f}%)"
                    )

    @given(
        df=ohlcv_dataframe_strategy(min_days=20, max_days=50),
        signals=signal_sequence_strategy(min_len=15, max_len=50),
        capital=initial_capital_strategy(),
    )
    @settings(max_examples=100, suppress_health_check=[])
    def test_settlement_period_respected(self, df, signals, capital):
        """
        No sell occurs on a position within T+2.5 (3 trading days) of purchase.

        For each trade, the number of trading days between entry_date and
        exit_date must be >= 3.

        **Validates: Requirements 9.3**
        """
        assume(len(signals) <= len(df))

        engine = BacktestEngine(initial_capital=capital, max_position_pct=MAX_POSITION_PCT)
        strategy = ParameterizedStrategy(signals[:len(df)])

        start_date = str(df["time"].iloc[0].date())
        end_date = str(df["time"].iloc[-1].date())

        result = engine.run(strategy, df, start_date, end_date)

        dates_in_period = pd.to_datetime(df["time"]).values

        for i, trade in enumerate(result.trades):
            # Count trading days between entry and exit
            entry_ts = pd.Timestamp(trade.entry_date)
            exit_ts = pd.Timestamp(trade.exit_date)

            # Count trading days in the DataFrame between entry and exit
            trading_days_between = sum(
                1 for d in dates_in_period
                if pd.Timestamp(d) > entry_ts and pd.Timestamp(d) <= exit_ts
            )

            assert trading_days_between >= SETTLEMENT_DAYS, (
                f"Trade {i}: only {trading_days_between} trading days between "
                f"entry ({trade.entry_date}) and exit ({trade.exit_date}), "
                f"minimum required is {SETTLEMENT_DAYS} (T+2.5 settlement)"
            )

    @given(
        df=ohlcv_dataframe_strategy(min_days=20, max_days=50),
        signals=signal_sequence_strategy(min_len=15, max_len=50),
        capital=initial_capital_strategy(),
    )
    @settings(max_examples=100, suppress_health_check=[])
    def test_max_position_size_constraint(self, df, signals, capital):
        """
        Entry position value does not exceed 20% of portfolio value at time of entry.

        **Validates: Requirements 9.4**
        """
        assume(len(signals) <= len(df))

        engine = BacktestEngine(initial_capital=capital, max_position_pct=MAX_POSITION_PCT)
        strategy = ParameterizedStrategy(signals[:len(df)])

        start_date = str(df["time"].iloc[0].date())
        end_date = str(df["time"].iloc[-1].date())

        result = engine.run(strategy, df, start_date, end_date)

        # For the first trade, portfolio value at entry is approximately initial_capital
        # (may differ slightly due to price movements on equity curve)
        # We use the equity curve to get the portfolio value at trade entry time
        for i, trade in enumerate(result.trades):
            position_value = trade.entry_price * trade.shares
            # Portfolio value at time of entry: use equity curve if available
            if result.equity_curve is not None and len(result.equity_curve) > 0:
                # Find equity value on trade entry date
                entry_ts = pd.Timestamp(trade.entry_date)
                # Get portfolio value from equity curve on or before entry date
                equity_at_entry = None
                for idx in result.equity_curve.index:
                    if pd.Timestamp(idx) <= entry_ts:
                        equity_at_entry = result.equity_curve[idx]
                    else:
                        break

                if equity_at_entry is None:
                    equity_at_entry = capital

                max_allowed = equity_at_entry * MAX_POSITION_PCT
                # Use a tolerance for floating point: position_value <= max_allowed + 1 lot worth
                tolerance = trade.entry_price * LOT_SIZE  # One lot tolerance for rounding
                assert position_value <= max_allowed + tolerance, (
                    f"Trade {i}: position value {position_value:.2f} exceeds "
                    f"20% of portfolio {equity_at_entry:.2f} "
                    f"(max allowed: {max_allowed:.2f})"
                )


# ---------------------------------------------------------------------------
# Property 19: Strategy comparison fairness
# ---------------------------------------------------------------------------


class TestStrategyComparisonFairness:
    """
    Property 19: Strategy comparison fairness.

    For any comparison run, ALL strategies SHALL be evaluated on exactly the
    same date range, same initial capital, same universe of instruments, and
    same position sizing rules.

    **Validates: Requirements 10.5**
    """

    @given(
        df=ohlcv_dataframe_strategy(min_days=20, max_days=50),
        capital=initial_capital_strategy(),
        num_strategies=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=50, suppress_health_check=[])
    def test_all_strategies_same_date_range(self, df, capital, num_strategies):
        """
        All strategies in a comparison are evaluated on identical date ranges.

        When multiple strategies are run through the BacktestEngine with the
        same parameters, the resulting equity curves must cover the same dates.

        **Validates: Requirements 10.5**
        """
        start_date = str(df["time"].iloc[0].date())
        end_date = str(df["time"].iloc[-1].date())

        # Create distinct strategies
        strategies = []
        for i in range(num_strategies):
            if i == 0:
                strategies.append(FixedSignalStrategy(Action.HOLD, f"Strategy_{i}"))
            elif i == 1:
                strategies.append(FixedSignalStrategy(Action.BUY, f"Strategy_{i}"))
            elif i == 2:
                strategies.append(FixedSignalStrategy(Action.SELL, f"Strategy_{i}"))
            else:
                strategies.append(AlternatingBuySellStrategy(f"Strategy_{i}"))

        # Run each strategy with the same engine parameters
        results = []
        for strategy in strategies:
            engine = BacktestEngine(
                initial_capital=capital,
                max_position_pct=MAX_POSITION_PCT,
            )
            result = engine.run(strategy, df, start_date, end_date)
            results.append(result)

        # All results must have equity curves with the same date range
        if len(results) >= 2:
            reference_dates = results[0].equity_curve.index
            for i, result in enumerate(results[1:], start=1):
                assert len(result.equity_curve) == len(reference_dates), (
                    f"Strategy {i}: equity curve has {len(result.equity_curve)} days "
                    f"vs reference {len(reference_dates)} days"
                )
                # Verify the dates are identical
                for j, (ref_date, strat_date) in enumerate(
                    zip(reference_dates, result.equity_curve.index)
                ):
                    assert ref_date == strat_date, (
                        f"Strategy {i}, day {j}: date {strat_date} != "
                        f"reference date {ref_date}"
                    )

    @given(
        df=ohlcv_dataframe_strategy(min_days=20, max_days=50),
        capital=initial_capital_strategy(),
        num_strategies=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=50, suppress_health_check=[])
    def test_all_strategies_same_initial_capital(self, df, capital, num_strategies):
        """
        All strategies start with identical initial capital.

        The first entry in each equity curve must equal the initial capital
        (before any trades are executed on day 1).

        **Validates: Requirements 10.5**
        """
        start_date = str(df["time"].iloc[0].date())
        end_date = str(df["time"].iloc[-1].date())

        strategies = []
        for i in range(num_strategies):
            strategies.append(FixedSignalStrategy(
                Action.HOLD if i % 2 == 0 else Action.BUY,
                f"Strategy_{i}",
            ))

        results = []
        for strategy in strategies:
            engine = BacktestEngine(
                initial_capital=capital,
                max_position_pct=MAX_POSITION_PCT,
            )
            result = engine.run(strategy, df, start_date, end_date)
            results.append(result)

        # All HOLD strategies should start with exactly the initial capital
        for i, result in enumerate(results):
            if result.equity_curve is not None and len(result.equity_curve) > 0:
                first_equity = result.equity_curve.iloc[0]
                # For HOLD strategies (even index), equity should equal capital
                if i % 2 == 0:
                    assert abs(first_equity - capital) < 1.0, (
                        f"Strategy {i} (HOLD): first equity {first_equity:.2f} != "
                        f"initial capital {capital:.2f}"
                    )

    @given(
        df=ohlcv_dataframe_strategy(min_days=20, max_days=50),
        capital=initial_capital_strategy(),
        num_strategies=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=50, suppress_health_check=[])
    def test_all_strategies_same_position_sizing_rules(self, df, capital, num_strategies):
        """
        All strategies are subject to the same position sizing rules
        (same lot size, same max position percentage).

        When multiple strategies produce BUY signals, the resulting trades
        must all respect the same 100-share lot size and 20% max position
        relative to portfolio value at time of entry.

        **Validates: Requirements 10.5**
        """
        start_date = str(df["time"].iloc[0].date())
        end_date = str(df["time"].iloc[-1].date())

        # All strategies will BUY immediately to test position sizing
        strategies = []
        for i in range(num_strategies):
            strategies.append(AlternatingBuySellStrategy(f"Strategy_{i}"))

        results = []
        for strategy in strategies:
            engine = BacktestEngine(
                initial_capital=capital,
                max_position_pct=MAX_POSITION_PCT,
            )
            result = engine.run(strategy, df, start_date, end_date)
            results.append(result)

        # All strategies must produce trades with the same lot size constraints
        # and position sizing rules (using portfolio value at entry time)
        for i, result in enumerate(results):
            for j, trade in enumerate(result.trades):
                # Lot size rule: shares must be multiples of 100
                assert trade.shares % LOT_SIZE == 0, (
                    f"Strategy {i}, trade {j}: shares {trade.shares} "
                    f"not a multiple of {LOT_SIZE}"
                )

                # Max position rule: position <= 20% of portfolio at entry time
                # Use equity curve to find portfolio value at trade entry
                position_value = trade.entry_price * trade.shares
                entry_ts = pd.Timestamp(trade.entry_date)

                portfolio_at_entry = capital  # fallback
                if result.equity_curve is not None and len(result.equity_curve) > 0:
                    for idx in result.equity_curve.index:
                        if pd.Timestamp(idx) <= entry_ts:
                            portfolio_at_entry = result.equity_curve[idx]
                        else:
                            break

                max_allowed = portfolio_at_entry * MAX_POSITION_PCT
                # Tolerance: one lot worth (rounding to lot size can overshoot slightly)
                tolerance = trade.entry_price * LOT_SIZE
                assert position_value <= max_allowed + tolerance, (
                    f"Strategy {i}, trade {j}: position value {position_value:.2f} "
                    f"exceeds 20% of portfolio at entry {portfolio_at_entry:.2f} "
                    f"(max allowed: {max_allowed:.2f})"
                )

    @given(
        df=ohlcv_dataframe_strategy(min_days=20, max_days=50),
        capital=initial_capital_strategy(),
    )
    @settings(max_examples=50, suppress_health_check=[])
    def test_comparison_uses_same_data_for_all_strategies(self, df, capital):
        """
        All strategies receive the exact same DataFrame for signal generation.

        When multiple strategies are run on the same data, each receives
        the same price history (universe of instruments).

        **Validates: Requirements 10.5**
        """
        start_date = str(df["time"].iloc[0].date())
        end_date = str(df["time"].iloc[-1].date())

        # Create strategies that record which data they receive
        class RecordingStrategy(BaseStrategy):
            def __init__(self, name: str):
                self._name = name
                self.received_closes = []

            def generate_signal(self, input_df: pd.DataFrame, index: int) -> Action:
                self.received_closes.append(float(input_df.iloc[index]["close"]))
                return Action.HOLD

            @property
            def name(self) -> str:
                return self._name

        strategies = [RecordingStrategy(f"Recorder_{i}") for i in range(3)]

        for strategy in strategies:
            engine = BacktestEngine(
                initial_capital=capital,
                max_position_pct=MAX_POSITION_PCT,
            )
            engine.run(strategy, df, start_date, end_date)

        # All strategies should have received the same price data
        if strategies[0].received_closes:
            reference_closes = strategies[0].received_closes
            for i, strategy in enumerate(strategies[1:], start=1):
                assert len(strategy.received_closes) == len(reference_closes), (
                    f"Strategy {i}: received {len(strategy.received_closes)} data points "
                    f"vs reference {len(reference_closes)}"
                )
                for j, (ref_close, strat_close) in enumerate(
                    zip(reference_closes, strategy.received_closes)
                ):
                    assert abs(ref_close - strat_close) < 1e-9, (
                        f"Strategy {i}, day {j}: close {strat_close} != "
                        f"reference close {ref_close}"
                    )
