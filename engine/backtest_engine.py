"""
Backtesting framework with Vietnamese market rules.

Simulates trading on historical data with:
- T+2.5 settlement period
- ±7% daily price limits
- 100-share lot size
- 20% maximum single position size

Computes: total return %, annualized return, win rate, max drawdown, Sharpe ratio.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from engine.config import (
    Action,
    BacktestResult,
    ComparisonResult,
    ConfigError,
    DataError,
    EngineConfig,
    Trade,
)


# ==============================================================================
# Strategy Protocol
# ==============================================================================


class Strategy(Protocol):
    """Protocol for trading strategies that generate signals."""

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        """Generate a trading signal (BUY/HOLD/SELL) for the given row index."""
        ...


# ==============================================================================
# Internal Position Tracking
# ==============================================================================


@dataclass
class _Position:
    """Internal representation of an open position."""

    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    entry_day_index: int  # Index in the trading day sequence


# ==============================================================================
# BacktestEngine
# ==============================================================================


class BacktestEngine:
    """
    Simulates trading on historical data with Vietnamese market rules.

    Vietnamese Market Rules Applied:
    - T+2.5 settlement: Cannot sell until 3 trading days after purchase
      (2.5 rounds up to 3 full trading days)
    - ±7% daily price limit: Trades cannot execute at prices beyond ±7%
      from the previous session's closing price
    - 100-share lot size: All trades must be in multiples of 100 shares
    - 20% max position: Single position cannot exceed 20% of portfolio
      value at time of entry

    Parameters
    ----------
    initial_capital : float
        Starting capital in VND (default: 100,000,000).
    max_position_pct : float
        Maximum single position as a fraction of portfolio (default: 0.20).
    config : EngineConfig, optional
        Engine configuration. Uses defaults if not provided.
    """

    def __init__(
        self,
        initial_capital: float = 100_000_000.0,
        max_position_pct: float = 0.20,
        config: Optional[EngineConfig] = None,
    ):
        if config is None:
            config = EngineConfig()

        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.lot_size = config.lot_size
        self.daily_price_limit = config.daily_price_limit
        # T+2.5 rounds up to 3 full trading days before selling is allowed
        self.settlement_days = math.ceil(config.settlement_days)
        self.config = config
        # Price in CSV is in units of 1000 VND (e.g., 76.5 = 76,500 VND).
        # Multiply by this factor to get actual VND for capital calculations.
        self.price_scale = getattr(config, "price_scale", 1000.0)

    def run(
        self,
        strategy: Strategy,
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """
        Run a backtest simulation on historical data.

        Parameters
        ----------
        strategy : Strategy
            Trading strategy that generates BUY/HOLD/SELL signals.
        df : pd.DataFrame
            Historical DataFrame with OHLCV data. Must contain 'time' column
            or a DatetimeIndex, plus 'open', 'high', 'low', 'close', 'volume'.
        start_date : str
            Start date for the backtest period (format: 'YYYY-MM-DD').
        end_date : str
            End date for the backtest period (format: 'YYYY-MM-DD').

        Returns
        -------
        BacktestResult
            Complete backtest results with metrics and trade list.

        Raises
        ------
        ConfigError
            If the date range is invalid (start >= end or no data in range).
        DataError
            If the DataFrame is missing required columns.
        """
        # Validate inputs
        self._validate_dataframe(df)
        df_period = self._filter_date_range(df, start_date, end_date)

        # Initialize state
        cash = self.initial_capital
        position: Optional[_Position] = None
        trades: List[Trade] = []
        equity_values: List[float] = []
        equity_dates: List[pd.Timestamp] = []

        # Iterate through each trading day
        for day_idx in range(len(df_period)):
            row = df_period.iloc[day_idx]
            current_price = float(row["close"])
            current_price_vnd = current_price * self.price_scale
            current_date = self._get_date(df_period, day_idx)

            # Calculate current portfolio value
            portfolio_value = cash
            if position is not None:
                portfolio_value += position.shares * current_price_vnd

            # Record equity curve
            equity_values.append(portfolio_value)
            equity_dates.append(current_date)

            # Get the original DataFrame index for this row to pass to strategy
            original_idx = df_period.index[day_idx]
            # Find the positional index in the original df
            original_pos = df.index.get_loc(original_idx)

            # Generate signal from strategy
            signal = strategy.generate_signal(df, original_pos)

            # Get reference price for price limit check
            ref_price = self._get_reference_price(df_period, day_idx)

            # Execute trade logic
            if signal == Action.BUY and position is None:
                # Check price limit
                if not self._is_within_price_limit(current_price, ref_price):
                    continue

                # Calculate position size respecting 20% max and lot size
                max_investment = portfolio_value * self.max_position_pct
                max_shares_by_capital = int(cash // current_price_vnd)
                max_shares_by_position = int(max_investment // current_price_vnd)
                shares = min(max_shares_by_capital, max_shares_by_position)

                # Round down to lot size
                shares = (shares // self.lot_size) * self.lot_size

                if shares >= self.lot_size:
                    cost = shares * current_price_vnd
                    cash -= cost
                    position = _Position(
                        entry_date=current_date,
                        entry_price=current_price,
                        shares=shares,
                        entry_day_index=day_idx,
                    )

            elif signal == Action.SELL and position is not None:
                # Check T+2.5 settlement (must wait 3 trading days)
                days_held = day_idx - position.entry_day_index
                if days_held < self.settlement_days:
                    continue

                # Check price limit
                if not self._is_within_price_limit(current_price, ref_price):
                    continue

                # Execute sell
                revenue = position.shares * current_price_vnd
                cash += revenue

                entry_price_vnd = position.entry_price * self.price_scale
                pnl = revenue - (position.shares * entry_price_vnd)
                pnl_pct = (
                    (current_price - position.entry_price) / position.entry_price
                ) * 100.0

                trade = Trade(
                    entry_date=position.entry_date,
                    exit_date=current_date,
                    entry_price=position.entry_price,
                    exit_price=current_price,
                    shares=position.shares,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
                trades.append(trade)
                position = None

        # If still holding a position at end, don't force close it
        # (it's not a completed trade)

        # Build equity curve Series
        equity_curve = pd.Series(
            data=equity_values,
            index=pd.DatetimeIndex(equity_dates),
            name="equity",
        )

        # Compute metrics
        metrics = self._compute_metrics(equity_curve, trades, df_period)

        return BacktestResult(
            total_return_pct=metrics["total_return_pct"],
            annualized_return_pct=metrics["annualized_return_pct"],
            win_rate=metrics["win_rate"],
            max_drawdown=metrics["max_drawdown"],
            sharpe_ratio=metrics["sharpe_ratio"],
            equity_curve=equity_curve,
            trades=trades,
        )

    def compare_strategies(
        self,
        strategies: Dict[str, "Strategy"],
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> ComparisonResult:
        """
        Run identical backtests across all strategies and compare results.

        Enforces the same date range, initial capital, universe, and position
        sizing rules for every strategy to ensure a fair comparison.

        Strategies that fail to generate at least one BUY and one SELL signal
        during the backtest period are excluded from results with a notification.

        Parameters
        ----------
        strategies : Dict[str, Strategy]
            Mapping of strategy name to strategy instance.
        df : pd.DataFrame
            Historical DataFrame with OHLCV data.
        start_date : str
            Start date for the backtest period (format: 'YYYY-MM-DD').
        end_date : str
            End date for the backtest period (format: 'YYYY-MM-DD').

        Returns
        -------
        ComparisonResult
            Comparison results with all strategy BacktestResults, excluded
            strategies, and metadata.
        """
        results: Dict[str, BacktestResult] = {}
        excluded_strategies: List[str] = []
        exclusion_reasons: Dict[str, str] = {}

        for name, strategy in strategies.items():
            # Run backtest with signal tracking for identical parameters
            result, has_buy, has_sell = self._run_with_signal_tracking(
                strategy, df, start_date, end_date
            )

            if not has_buy or not has_sell:
                excluded_strategies.append(name)
                missing = []
                if not has_buy:
                    missing.append("BUY")
                if not has_sell:
                    missing.append("SELL")
                exclusion_reasons[name] = (
                    f"Insufficient signal generation: no {' or '.join(missing)} "
                    f"signals during the backtest period"
                )
            else:
                results[name] = result

        return ComparisonResult(
            results=results,
            date_range_start=start_date,
            date_range_end=end_date,
            initial_capital=self.initial_capital,
            excluded_strategies=excluded_strategies,
            exclusion_reasons=exclusion_reasons,
        )

    def _run_with_signal_tracking(
        self,
        strategy: "Strategy",
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> tuple:
        """
        Run a backtest while tracking which signal types were generated.

        This is used by compare_strategies to determine if a strategy should
        be excluded due to insufficient signal generation.

        Returns
        -------
        tuple of (BacktestResult, has_buy: bool, has_sell: bool)
        """
        # Validate inputs
        self._validate_dataframe(df)
        df_period = self._filter_date_range(df, start_date, end_date)

        # Initialize state
        cash = self.initial_capital
        position: Optional[_Position] = None
        trades: List[Trade] = []
        equity_values: List[float] = []
        equity_dates: List[pd.Timestamp] = []
        has_buy = False
        has_sell = False

        # Iterate through each trading day
        for day_idx in range(len(df_period)):
            row = df_period.iloc[day_idx]
            current_price = float(row["close"])
            current_price_vnd = current_price * self.price_scale
            current_date = self._get_date(df_period, day_idx)

            # Calculate current portfolio value
            portfolio_value = cash
            if position is not None:
                portfolio_value += position.shares * current_price_vnd

            # Record equity curve
            equity_values.append(portfolio_value)
            equity_dates.append(current_date)

            # Get the original DataFrame index for this row to pass to strategy
            original_idx = df_period.index[day_idx]
            original_pos = df.index.get_loc(original_idx)

            # Generate signal from strategy
            signal = strategy.generate_signal(df, original_pos)

            # Track signal types
            if signal == Action.BUY:
                has_buy = True
            elif signal == Action.SELL:
                has_sell = True

            # Get reference price for price limit check
            ref_price = self._get_reference_price(df_period, day_idx)

            # Execute trade logic
            if signal == Action.BUY and position is None:
                if not self._is_within_price_limit(current_price, ref_price):
                    continue

                max_investment = portfolio_value * self.max_position_pct
                max_shares_by_capital = int(cash // current_price_vnd)
                max_shares_by_position = int(max_investment // current_price_vnd)
                shares = min(max_shares_by_capital, max_shares_by_position)
                shares = (shares // self.lot_size) * self.lot_size

                if shares >= self.lot_size:
                    cost = shares * current_price_vnd
                    cash -= cost
                    position = _Position(
                        entry_date=current_date,
                        entry_price=current_price,
                        shares=shares,
                        entry_day_index=day_idx,
                    )

            elif signal == Action.SELL and position is not None:
                days_held = day_idx - position.entry_day_index
                if days_held < self.settlement_days:
                    continue

                if not self._is_within_price_limit(current_price, ref_price):
                    continue

                revenue = position.shares * current_price_vnd
                cash += revenue

                entry_price_vnd = position.entry_price * self.price_scale
                pnl = revenue - (position.shares * entry_price_vnd)
                pnl_pct = (
                    (current_price - position.entry_price) / position.entry_price
                ) * 100.0

                trade = Trade(
                    entry_date=position.entry_date,
                    exit_date=current_date,
                    entry_price=position.entry_price,
                    exit_price=current_price,
                    shares=position.shares,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
                trades.append(trade)
                position = None

        # Build equity curve Series
        equity_curve = pd.Series(
            data=equity_values,
            index=pd.DatetimeIndex(equity_dates),
            name="equity",
        )

        # Compute metrics
        metrics = self._compute_metrics(equity_curve, trades, df_period)

        result = BacktestResult(
            total_return_pct=metrics["total_return_pct"],
            annualized_return_pct=metrics["annualized_return_pct"],
            win_rate=metrics["win_rate"],
            max_drawdown=metrics["max_drawdown"],
            sharpe_ratio=metrics["sharpe_ratio"],
            equity_curve=equity_curve,
            trades=trades,
        )

        return result, has_buy, has_sell

    # ==========================================================================
    # Validation
    # ==========================================================================

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that the DataFrame has required columns."""
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise DataError(
                f"DataFrame missing required columns: {missing}",
                error_code="MISSING_COLUMNS",
                details={"missing_columns": missing},
            )

    def _filter_date_range(
        self, df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Filter DataFrame to the specified date range.

        Raises ConfigError if date range is invalid.
        """
        try:
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
        except (ValueError, TypeError) as e:
            raise ConfigError(
                f"Invalid date format: {e}",
                error_code="INVALID_DATE_FORMAT",
                details={"start_date": start_date, "end_date": end_date, "error": str(e)},
            )

        if start >= end:
            raise ConfigError(
                f"Start date ({start_date}) must be before end date ({end_date})",
                error_code="INVALID_DATE_RANGE",
                details={"start_date": start_date, "end_date": end_date},
            )

        # Get dates from DataFrame
        dates = self._get_dates_series(df)

        # Filter to date range
        mask = (dates >= start) & (dates <= end)
        df_period = df.loc[mask].copy()

        if len(df_period) == 0:
            raise ConfigError(
                f"No data found in date range {start_date} to {end_date}",
                error_code="NO_DATA_IN_RANGE",
                details={"start_date": start_date, "end_date": end_date},
            )

        return df_period

    # ==========================================================================
    # Price and Date Helpers
    # ==========================================================================

    def _get_dates_series(self, df: pd.DataFrame) -> pd.Series:
        """Extract dates as a pandas Series from the DataFrame."""
        if "time" in df.columns:
            return pd.to_datetime(df["time"])
        elif isinstance(df.index, pd.DatetimeIndex):
            return df.index.to_series()
        else:
            raise DataError(
                "DataFrame must have a 'time' column or DatetimeIndex",
                error_code="NO_DATE_COLUMN",
            )

    def _get_date(self, df: pd.DataFrame, idx: int) -> pd.Timestamp:
        """Get the date for a specific row index."""
        if "time" in df.columns:
            return pd.Timestamp(df.iloc[idx]["time"])
        elif isinstance(df.index, pd.DatetimeIndex):
            return df.index[idx]
        else:
            return pd.Timestamp.now()

    def _get_reference_price(self, df: pd.DataFrame, day_idx: int) -> float:
        """
        Get the reference price for price limit calculation.

        The reference price is the previous session's closing price.
        For the first day, the current day's open is used.
        """
        if day_idx > 0:
            return float(df.iloc[day_idx - 1]["close"])
        else:
            return float(df.iloc[day_idx]["open"])

    def _is_within_price_limit(
        self, executed_price: float, reference_price: float
    ) -> bool:
        """Check if executed_price is within ±7% of reference_price."""
        if reference_price <= 0:
            return False
        change_pct = abs(executed_price - reference_price) / reference_price
        return change_pct <= self.daily_price_limit

    # ==========================================================================
    # Metrics Computation
    # ==========================================================================

    def _compute_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Trade],
        df_period: pd.DataFrame,
    ) -> dict:
        """
        Compute all backtest performance metrics.

        Returns a dictionary with:
        - total_return_pct
        - annualized_return_pct
        - win_rate
        - max_drawdown
        - sharpe_ratio
        """
        # Total return
        if len(equity_curve) == 0:
            return {
                "total_return_pct": 0.0,
                "annualized_return_pct": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
            }

        initial_value = equity_curve.iloc[0]
        final_value = equity_curve.iloc[-1]

        if initial_value == 0:
            total_return_pct = 0.0
        else:
            total_return_pct = ((final_value - initial_value) / initial_value) * 100.0

        # Annualized return
        num_trading_days = len(equity_curve)
        trading_days_per_year = 252.0

        if num_trading_days > 1 and initial_value > 0:
            total_return_ratio = final_value / initial_value
            years = num_trading_days / trading_days_per_year
            if years > 0 and total_return_ratio > 0:
                annualized_return_pct = (
                    (total_return_ratio ** (1.0 / years)) - 1.0
                ) * 100.0
            else:
                annualized_return_pct = 0.0
        else:
            annualized_return_pct = 0.0

        # Win rate
        if len(trades) > 0:
            winning_trades = sum(1 for t in trades if t.pnl > 0)
            win_rate = (winning_trades / len(trades)) * 100.0
        else:
            win_rate = 0.0

        # Max drawdown
        max_drawdown = self._compute_max_drawdown(equity_curve)

        # Sharpe ratio (risk-free rate = 0%)
        sharpe_ratio = self._compute_sharpe_ratio(equity_curve)

        return {
            "total_return_pct": total_return_pct,
            "annualized_return_pct": annualized_return_pct,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
        }

    def _compute_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Compute maximum drawdown as a positive percentage."""
        if len(equity_curve) < 2:
            return 0.0

        values = equity_curve.values
        peak = values[0]
        max_dd = 0.0

        for value in values:
            if value > peak:
                peak = value
            if peak > 0:
                drawdown = (peak - value) / peak * 100.0
                if drawdown > max_dd:
                    max_dd = drawdown

        return max_dd

    def _compute_sharpe_ratio(self, equity_curve: pd.Series) -> float:
        """Compute annualized Sharpe ratio (risk-free rate = 0%)."""
        if len(equity_curve) < 2:
            return 0.0

        values = equity_curve.values
        # Daily returns
        daily_returns = np.diff(values) / values[:-1]

        if len(daily_returns) == 0:
            return 0.0

        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns, ddof=1)

        if std_return == 0.0 or np.isnan(std_return):
            return 0.0

        # Annualize: multiply by sqrt(252)
        sharpe = (mean_return / std_return) * math.sqrt(252.0)

        return float(sharpe)
