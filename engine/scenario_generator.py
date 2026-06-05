"""
Scenario Generator: percentile-based statistical scenario generation.

Generates plausible future market scenarios for the Minimax search tree
using historical return distributions with volatility regime detection.

Key design decisions:
- Percentile-based sampling ensures scenarios cover the full range of likely outcomes
- ATR-based volatility regime detection adjusts distribution spread
- Vietnamese market ±7% daily price limit enforced on all generated scenarios
- Minimum 30 days of history required for meaningful distribution estimation
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

from engine.config import Action, DataError, EngineConfig, SearchConfig
from engine.market_state import (
    INDICATOR_COLUMNS,
    NUM_INDICATORS,
    OHLCV_COLUMNS,
    MarketState,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Types
# ==============================================================================


class VolatilityRegime(Enum):
    """Volatility regime detected from ATR-based analysis."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Distribution:
    """Estimated return distribution from historical data.

    Attributes
    ----------
    mean : float
        Mean daily return (as a fraction, e.g., 0.01 = 1%).
    std : float
        Standard deviation of daily returns.
    regime : VolatilityRegime
        Detected volatility regime based on ATR.
    percentiles : dict
        Pre-computed percentile values of the return distribution.
    """

    mean: float
    std: float
    regime: VolatilityRegime
    percentiles: dict


# ==============================================================================
# Percentile configurations for different scenario counts
# ==============================================================================

PERCENTILE_CONFIGS = {
    3: [20, 50, 80],
    5: [10, 30, 50, 70, 90],
    7: [5, 15, 30, 50, 70, 85, 95],
}

# Default scenario count
DEFAULT_NUM_SCENARIOS = 5

# ATR-based regime thresholds (ATR / close price ratio)
ATR_REGIME_LOW_THRESHOLD = 0.015  # < 1.5% → low volatility
ATR_REGIME_HIGH_THRESHOLD = 0.035  # > 3.5% → high volatility

# Regime adjustment multipliers for distribution spread
REGIME_MULTIPLIERS = {
    VolatilityRegime.LOW: 0.7,
    VolatilityRegime.MEDIUM: 1.0,
    VolatilityRegime.HIGH: 1.4,
}


# ==============================================================================
# ScenarioGenerator
# ==============================================================================


class ScenarioGenerator:
    """Generates plausible future market scenarios using percentile-based sampling.

    The generator estimates the historical return distribution, detects the
    current volatility regime using ATR, and produces scenarios at strategic
    percentiles of the distribution. All scenarios respect the Vietnamese
    market ±7% daily price limit.

    Parameters
    ----------
    min_history : int
        Minimum number of trading days required for distribution estimation.
        Default is 30 (from SearchConfig.min_history_days).
    config : EngineConfig, optional
        Engine configuration for market constraints.
    search_config : SearchConfig, optional
        Search configuration for scenario parameters.
    """

    def __init__(
        self,
        min_history: int = 30,
        config: Optional[EngineConfig] = None,
        search_config: Optional[SearchConfig] = None,
    ):
        self.min_history = min_history
        self.config = config or EngineConfig()
        self.search_config = search_config or SearchConfig()

    def generate(
        self,
        state: MarketState,
        action: Action,
        num_scenarios: int = DEFAULT_NUM_SCENARIOS,
    ) -> List[MarketState]:
        """Generate future market states after a given action.

        Uses historical return distribution with regime awareness to produce
        scenarios at strategic percentiles. Each scenario represents a
        plausible next-day market state.

        Parameters
        ----------
        state : MarketState
            Current market state with sufficient history.
        action : Action
            The trading action being considered (BUY, HOLD, SELL).
        num_scenarios : int
            Number of scenarios to generate. Must be 3, 5, or 7.
            Default is 5.

        Returns
        -------
        List[MarketState]
            List of generated future MarketState objects (3-7 items).

        Raises
        ------
        DataError
            If the state has fewer than min_history days of data, or if
            num_scenarios is not in {3, 5, 7}.
        """
        # Validate minimum history
        if state.lookback < self.min_history:
            raise DataError(
                f"Insufficient historical data for scenario generation: "
                f"need at least {self.min_history} days, got {state.lookback}",
                error_code="INSUFFICIENT_HISTORY",
                details={
                    "required": self.min_history,
                    "available": state.lookback,
                    "symbol": state.symbol,
                },
            )

        # Validate num_scenarios
        if num_scenarios not in PERCENTILE_CONFIGS:
            # Clamp to nearest valid value
            if num_scenarios < 3:
                num_scenarios = 3
            elif num_scenarios > 7:
                num_scenarios = 7
            elif num_scenarios <= 4:
                num_scenarios = 3
            elif num_scenarios <= 6:
                num_scenarios = 5
            else:
                num_scenarios = 7

        # Estimate return distribution from history
        distribution = self._estimate_distribution(state)

        # Get percentiles for this scenario count
        percentiles = PERCENTILE_CONFIGS[num_scenarios]

        # Generate scenarios at each percentile
        scenarios = []
        for pct in percentiles:
            # Get the return value at this percentile
            return_pct = distribution.percentiles.get(pct, distribution.mean)

            # Apply action effect to create new state
            new_state = self._apply_action_effect(state, action, return_pct)
            scenarios.append(new_state)

        return scenarios

    def _estimate_distribution(self, state: MarketState) -> Distribution:
        """Estimate the return distribution from recent historical data.

        Computes daily returns from the OHLCV close prices in the state,
        detects the current volatility regime using ATR, and returns the
        distribution parameters along with pre-computed percentiles.

        Parameters
        ----------
        state : MarketState
            Market state with OHLCV data of shape (lookback, 5).

        Returns
        -------
        Distribution
            Estimated return distribution with regime awareness.
        """
        # Extract close prices (column index 3 in OHLCV: open, high, low, close, volume)
        close_prices = state.ohlcv[:, 3]

        # Compute daily returns (percentage change)
        # Skip NaN/zero prices
        valid_mask = (close_prices > 0) & ~np.isnan(close_prices)
        valid_prices = close_prices[valid_mask]

        if len(valid_prices) < 2:
            # Not enough valid prices, return neutral distribution
            return Distribution(
                mean=0.0,
                std=0.01,
                regime=VolatilityRegime.MEDIUM,
                percentiles={p: 0.0 for p in [5, 10, 15, 20, 30, 50, 70, 80, 85, 90, 95]},
            )

        # Daily returns as fractions
        daily_returns = np.diff(valid_prices) / valid_prices[:-1]

        # Filter out extreme outliers (beyond ±20% daily, likely data errors)
        daily_returns = daily_returns[np.abs(daily_returns) <= 0.20]

        if len(daily_returns) < 5:
            # Not enough return data after filtering
            return Distribution(
                mean=0.0,
                std=0.01,
                regime=VolatilityRegime.MEDIUM,
                percentiles={p: 0.0 for p in [5, 10, 15, 20, 30, 50, 70, 80, 85, 90, 95]},
            )

        # Compute distribution parameters
        mean_return = float(np.mean(daily_returns))
        std_return = float(np.std(daily_returns, ddof=1))

        # Ensure std is not zero (prevents degenerate scenarios)
        if std_return < 1e-8:
            std_return = 0.01

        # Detect volatility regime using ATR
        regime = self._detect_volatility_regime(state)

        # Apply regime multiplier to adjust the spread
        regime_multiplier = REGIME_MULTIPLIERS[regime]
        adjusted_std = std_return * regime_multiplier

        # Compute percentiles from the actual return data
        all_percentile_levels = [5, 10, 15, 20, 30, 50, 70, 80, 85, 90, 95]
        percentiles = {}
        for p in all_percentile_levels:
            pct_value = float(np.percentile(daily_returns, p))
            # Scale by regime multiplier relative to mean
            deviation = pct_value - mean_return
            adjusted_value = mean_return + deviation * regime_multiplier
            percentiles[p] = adjusted_value

        return Distribution(
            mean=mean_return,
            std=adjusted_std,
            regime=regime,
            percentiles=percentiles,
        )

    def _detect_volatility_regime(self, state: MarketState) -> VolatilityRegime:
        """Detect current volatility regime using ATR-based analysis.

        Uses the ATR_14 indicator if available, otherwise computes a simple
        ATR approximation from the OHLCV data.

        Parameters
        ----------
        state : MarketState
            Market state with OHLCV and indicator data.

        Returns
        -------
        VolatilityRegime
            Detected regime (LOW, MEDIUM, or HIGH).
        """
        # Try to use ATR_14 from indicators
        atr_value = None
        try:
            atr_idx = INDICATOR_COLUMNS.index("ATR_14")
            # Get the most recent ATR_14 value
            atr_col = state.indicators[:, atr_idx]
            valid_atr = atr_col[~np.isnan(atr_col)]
            if len(valid_atr) > 0:
                atr_value = valid_atr[-1]
        except (ValueError, IndexError):
            pass

        # If ATR_14 not available, compute simple ATR from OHLCV
        if atr_value is None or atr_value <= 0:
            high_prices = state.ohlcv[:, 1]
            low_prices = state.ohlcv[:, 2]
            close_prices = state.ohlcv[:, 3]

            # True Range approximation
            valid_mask = (high_prices > 0) & (low_prices > 0) & (close_prices > 0)
            if valid_mask.sum() < 2:
                return VolatilityRegime.MEDIUM

            tr = high_prices[valid_mask] - low_prices[valid_mask]
            atr_value = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))

        # Get current close price for normalization
        close_prices = state.ohlcv[:, 3]
        valid_close = close_prices[close_prices > 0]
        if len(valid_close) == 0:
            return VolatilityRegime.MEDIUM

        current_close = valid_close[-1]

        # ATR as ratio of close price
        atr_ratio = atr_value / current_close

        if atr_ratio < ATR_REGIME_LOW_THRESHOLD:
            return VolatilityRegime.LOW
        elif atr_ratio > ATR_REGIME_HIGH_THRESHOLD:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.MEDIUM

    def _apply_action_effect(
        self,
        state: MarketState,
        action: Action,
        return_pct: float,
    ) -> MarketState:
        """Create a new MarketState with projected price movement.

        Projects the price based on the given return percentage, applies the
        Vietnamese market ±7% daily limit, and recomputes indicators for the
        new price point.

        Parameters
        ----------
        state : MarketState
            Current market state.
        action : Action
            The trading action (affects sentiment adjustment).
        return_pct : float
            Projected return as a fraction (e.g., 0.05 = +5%).

        Returns
        -------
        MarketState
            New market state with projected prices and recomputed indicators.
        """
        # Cap return at ±7% daily limit (Vietnamese market rule)
        daily_limit = self.config.daily_price_limit  # 0.07
        capped_return = np.clip(return_pct, -daily_limit, daily_limit)

        # Get current prices from most recent row
        current_ohlcv = state.ohlcv[-1]  # [open, high, low, close, volume]
        current_close = current_ohlcv[3]
        current_volume = current_ohlcv[4]

        # Project new close price
        new_close = current_close * (1.0 + capped_return)

        # Generate plausible OHLCV for the projected day
        # Use the return to estimate intraday range
        abs_return = abs(capped_return)
        intraday_noise = max(abs_return * 0.5, 0.002)  # minimum 0.2% intraday range

        if capped_return >= 0:
            # Bullish day: open near previous close, close at high end
            new_open = current_close * (1.0 + capped_return * 0.2)
            new_high = new_close * (1.0 + intraday_noise)
            new_low = min(new_open, new_close) * (1.0 - intraday_noise * 0.5)
        else:
            # Bearish day: open near previous close, close at low end
            new_open = current_close * (1.0 + capped_return * 0.2)
            new_low = new_close * (1.0 - intraday_noise)
            new_high = max(new_open, new_close) * (1.0 + intraday_noise * 0.5)

        # Ensure OHLCV consistency: high >= max(open, close), low <= min(open, close)
        new_high = max(new_high, new_open, new_close)
        new_low = min(new_low, new_open, new_close)

        # Volume adjustment based on action and return magnitude
        volume_factor = 1.0 + abs_return * 5.0  # Higher returns tend to have higher volume
        if action == Action.BUY:
            volume_factor *= 1.1  # Slightly higher volume on buy signals
        elif action == Action.SELL:
            volume_factor *= 1.2  # Slightly higher volume on sell signals
        new_volume = current_volume * volume_factor

        # Create new OHLCV array by shifting window: drop first row, append new row
        new_ohlcv = np.zeros_like(state.ohlcv)
        new_ohlcv[:-1] = state.ohlcv[1:]  # Shift existing data left
        new_ohlcv[-1] = [new_open, new_high, new_low, new_close, new_volume]

        # Recompute indicators for the new state
        new_indicators = self._recompute_indicators(
            new_ohlcv, state.indicators
        )

        # Create new timestamp (next trading day)
        new_timestamp = state.timestamp + pd.Timedelta(days=1)

        return MarketState(
            symbol=state.symbol,
            timestamp=new_timestamp,
            ohlcv=new_ohlcv,
            indicators=new_indicators,
            lookback=state.lookback,
        )

    def _recompute_indicators(
        self,
        new_ohlcv: np.ndarray,
        prev_indicators: np.ndarray,
    ) -> np.ndarray:
        """Recompute indicator approximations for the projected state.

        For efficiency in the search tree, we use simplified indicator updates
        rather than running the full analysis.add_indicators() pipeline.
        This provides reasonable approximations for EMA, RSI, ATR, and other
        key indicators while keeping scenario generation fast.

        Parameters
        ----------
        new_ohlcv : np.ndarray
            Updated OHLCV array of shape (lookback, 5).
        prev_indicators : np.ndarray
            Previous indicator values of shape (lookback, num_indicators).

        Returns
        -------
        np.ndarray
            Updated indicator array of shape (lookback, num_indicators).
        """
        # Shift indicators: drop first row, compute approximation for new row
        new_indicators = np.zeros_like(prev_indicators)
        new_indicators[:-1] = prev_indicators[1:]

        # Copy last known valid values as starting point for the new row
        last_valid = prev_indicators[-1].copy()

        # Extract price data for indicator computation
        close_prices = new_ohlcv[:, 3]
        high_prices = new_ohlcv[:, 1]
        low_prices = new_ohlcv[:, 2]
        current_close = close_prices[-1]
        prev_close = close_prices[-2] if len(close_prices) > 1 else current_close

        # Update EMA-based indicators (approximate using EMA formula)
        ema_configs = [
            ("EMA_9", 9),
            ("EMA_20", 20),
            ("EMA_50", 50),
            ("EMA_200", 200),
        ]
        for name, period in ema_configs:
            try:
                idx = INDICATOR_COLUMNS.index(name)
                prev_ema = last_valid[idx]
                if not np.isnan(prev_ema) and prev_ema > 0:
                    alpha = 2.0 / (period + 1)
                    new_ema = alpha * current_close + (1 - alpha) * prev_ema
                    last_valid[idx] = new_ema
            except (ValueError, IndexError):
                pass

        # Update SMA-based indicators
        sma_configs = [("SMA_20", 20), ("SMA_50", 50)]
        for name, period in sma_configs:
            try:
                idx = INDICATOR_COLUMNS.index(name)
                if len(close_prices) >= period:
                    last_valid[idx] = float(np.mean(close_prices[-period:]))
            except (ValueError, IndexError):
                pass

        # Update RSI approximation
        for rsi_name, period in [("RSI_14", 14), ("RSI_7", 7)]:
            try:
                idx = INDICATOR_COLUMNS.index(rsi_name)
                prev_rsi = last_valid[idx]
                if not np.isnan(prev_rsi):
                    # Approximate RSI update
                    change = current_close - prev_close
                    # Smooth update towards new value
                    if change > 0:
                        # Price went up, RSI should increase
                        last_valid[idx] = min(100.0, prev_rsi + (100 - prev_rsi) * 0.1)
                    else:
                        # Price went down, RSI should decrease
                        last_valid[idx] = max(0.0, prev_rsi - prev_rsi * 0.1)
            except (ValueError, IndexError):
                pass

        # Update ATR approximation
        for atr_name, period in [("ATR_14", 14), ("ATR_7", 7)]:
            try:
                idx = INDICATOR_COLUMNS.index(atr_name)
                prev_atr = last_valid[idx]
                if not np.isnan(prev_atr) and prev_atr > 0:
                    # True range for new bar
                    current_high = high_prices[-1]
                    current_low = low_prices[-1]
                    tr = max(
                        current_high - current_low,
                        abs(current_high - prev_close),
                        abs(current_low - prev_close),
                    )
                    # Exponential smoothing
                    alpha = 1.0 / period
                    last_valid[idx] = alpha * tr + (1 - alpha) * prev_atr
            except (ValueError, IndexError):
                pass

        # Update Bollinger Bands approximation
        try:
            bb_mid_idx = INDICATOR_COLUMNS.index("BB_middle")
            bb_upper_idx = INDICATOR_COLUMNS.index("BB_upper")
            bb_lower_idx = INDICATOR_COLUMNS.index("BB_lower")
            if len(close_prices) >= 20:
                sma_20 = float(np.mean(close_prices[-20:]))
                std_20 = float(np.std(close_prices[-20:], ddof=1))
                last_valid[bb_mid_idx] = sma_20
                last_valid[bb_upper_idx] = sma_20 + 2 * std_20
                last_valid[bb_lower_idx] = sma_20 - 2 * std_20

                # BB %B and bandwidth
                bb_pband_idx = INDICATOR_COLUMNS.index("BB_pband")
                bb_wband_idx = INDICATOR_COLUMNS.index("BB_wband")
                bb_range = last_valid[bb_upper_idx] - last_valid[bb_lower_idx]
                if bb_range > 0:
                    last_valid[bb_pband_idx] = (current_close - last_valid[bb_lower_idx]) / bb_range
                    last_valid[bb_wband_idx] = bb_range / sma_20 if sma_20 > 0 else 0.0
        except (ValueError, IndexError):
            pass

        # Update MACD approximation
        try:
            macd_idx = INDICATOR_COLUMNS.index("MACD")
            macd_signal_idx = INDICATOR_COLUMNS.index("MACD_signal")
            macd_hist_idx = INDICATOR_COLUMNS.index("MACD_hist")

            ema_12_idx = None
            ema_26_val = None

            # Use EMA_9 and EMA_20 as proxies for MACD components
            ema_9_idx = INDICATOR_COLUMNS.index("EMA_9")
            ema_20_idx = INDICATOR_COLUMNS.index("EMA_20")

            if not np.isnan(last_valid[ema_9_idx]) and not np.isnan(last_valid[ema_20_idx]):
                # MACD ≈ short EMA - long EMA (approximation)
                new_macd = last_valid[ema_9_idx] - last_valid[ema_20_idx]
                prev_signal = last_valid[macd_signal_idx]

                if not np.isnan(prev_signal):
                    # Signal line: 9-period EMA of MACD
                    signal_alpha = 2.0 / 10.0
                    new_signal = signal_alpha * new_macd + (1 - signal_alpha) * prev_signal
                    last_valid[macd_idx] = new_macd
                    last_valid[macd_signal_idx] = new_signal
                    last_valid[macd_hist_idx] = new_macd - new_signal
        except (ValueError, IndexError):
            pass

        # Set the new row
        new_indicators[-1] = last_valid

        return new_indicators
