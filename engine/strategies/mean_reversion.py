"""
Mean reversion trading strategy.

Generates BUY/HOLD/SELL signals based on the assumption that prices tend to
revert to their mean after extreme deviations.

Signal rules:
- BUY when price < BB_lower OR RSI < 30 OR price < SMA_20 - 2σ
- SELL when price > BB_upper OR RSI > 70 OR price > SMA_20 + 2σ
- HOLD otherwise

Uses Bollinger Bands, RSI, and standard deviation from SMA_20 to detect
overbought/oversold conditions.
"""

import numpy as np
import pandas as pd

from engine.config import Action
from engine.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy.

    Detects overbought/oversold conditions using multiple indicators
    and generates contrarian signals expecting price to revert to mean.

    BUY conditions (any one triggers):
    - Price below lower Bollinger Band
    - RSI_14 below 30 (oversold)
    - Price more than 2 standard deviations below SMA_20

    SELL conditions (any one triggers):
    - Price above upper Bollinger Band
    - RSI_14 above 70 (overbought)
    - Price more than 2 standard deviations above SMA_20

    HOLD when none of the above conditions are met.
    """

    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    STD_MULTIPLIER = 2.0
    SMA_LOOKBACK = 20

    @property
    def name(self) -> str:
        return "MeanReversion"

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        """
        Generate a mean reversion trading signal.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV and indicator columns.
        index : int
            Current row index to evaluate.

        Returns
        -------
        Action
            BUY if oversold conditions detected,
            SELL if overbought conditions detected,
            HOLD otherwise.
        """
        close = self._safe_get(df, index, "close")
        if pd.isna(close):
            return Action.HOLD

        # Check BUY conditions (any one triggers)
        if self._is_oversold(df, index, close):
            return Action.BUY

        # Check SELL conditions (any one triggers)
        if self._is_overbought(df, index, close):
            return Action.SELL

        return Action.HOLD

    def _is_oversold(self, df: pd.DataFrame, idx: int, close: float) -> bool:
        """
        Check if oversold conditions are met.

        Returns True if ANY of the following:
        - Price < BB_lower
        - RSI_14 < 30
        - Price < SMA_20 - 2 * std_dev
        """
        # Condition 1: Price below lower Bollinger Band
        bb_lower = self._safe_get(df, idx, "BB_lower")
        if pd.notna(bb_lower) and close < bb_lower:
            return True

        # Condition 2: RSI below 30
        rsi = self._safe_get(df, idx, "RSI_14")
        if pd.notna(rsi) and rsi < self.RSI_OVERSOLD:
            return True

        # Condition 3: Price < SMA_20 - 2σ
        sma20 = self._safe_get(df, idx, "SMA_20")
        if pd.notna(sma20):
            std_dev = self._compute_std(df, idx)
            if std_dev > 0 and close < sma20 - self.STD_MULTIPLIER * std_dev:
                return True

        return False

    def _is_overbought(self, df: pd.DataFrame, idx: int, close: float) -> bool:
        """
        Check if overbought conditions are met.

        Returns True if ANY of the following:
        - Price > BB_upper
        - RSI_14 > 70
        - Price > SMA_20 + 2 * std_dev
        """
        # Condition 1: Price above upper Bollinger Band
        bb_upper = self._safe_get(df, idx, "BB_upper")
        if pd.notna(bb_upper) and close > bb_upper:
            return True

        # Condition 2: RSI above 70
        rsi = self._safe_get(df, idx, "RSI_14")
        if pd.notna(rsi) and rsi > self.RSI_OVERBOUGHT:
            return True

        # Condition 3: Price > SMA_20 + 2σ
        sma20 = self._safe_get(df, idx, "SMA_20")
        if pd.notna(sma20):
            std_dev = self._compute_std(df, idx)
            if std_dev > 0 and close > sma20 + self.STD_MULTIPLIER * std_dev:
                return True

        return False

    def _compute_std(self, df: pd.DataFrame, idx: int) -> float:
        """
        Compute the standard deviation of close prices over the SMA lookback period.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'close' column.
        idx : int
            Current row index.

        Returns
        -------
        float
            Standard deviation of close prices over the lookback period.
            Returns 0.0 if insufficient data.
        """
        if "close" not in df.columns:
            return 0.0

        start = max(0, idx - self.SMA_LOOKBACK + 1)
        if start >= idx:
            return 0.0

        close_slice = df["close"].iloc[start:idx + 1]
        std = close_slice.std()

        if pd.isna(std):
            return 0.0

        return float(std)

    @staticmethod
    def _safe_get(df: pd.DataFrame, idx: int, col: str):
        """Safely get a value from the DataFrame, returning NaN if unavailable."""
        if col not in df.columns:
            return np.nan
        if idx < 0 or idx >= len(df):
            return np.nan
        return df[col].iloc[idx]
