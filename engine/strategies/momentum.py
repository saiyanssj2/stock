"""
Momentum-based trading strategy.

Generates BUY/HOLD/SELL signals based on trend strength (ADX), rate of change
(ROC), and volume confirmation.

Signal rules:
- BUY when ADX > 25 AND ROC_10 > 0 AND volume > 1.5x 20-period average
- SELL when ADX > 25 AND ROC_10 < 0
- HOLD otherwise

Uses ADX for trend strength confirmation, ROC_10 for price momentum direction,
and volume ratio for participation validation.
"""

import numpy as np
import pandas as pd

from engine.config import Action
from engine.strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.

    Generates signals based on:
    - ADX > 25 confirms a strong trend is present
    - ROC_10 > 0 confirms upward price momentum (BUY condition)
    - ROC_10 < 0 confirms downward price momentum (SELL condition)
    - Volume > 1.5x average confirms participation (BUY only)

    Produces:
    - BUY when ADX > 25 AND ROC_10 > 0 AND volume > 1.5x avg
    - SELL when ADX > 25 AND ROC_10 < 0
    - HOLD otherwise
    """

    ADX_THRESHOLD = 25
    VOLUME_MULTIPLIER = 1.5
    VOLUME_LOOKBACK = 20

    @property
    def name(self) -> str:
        return "Momentum"

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        """
        Generate a momentum-based trading signal.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV and indicator columns.
        index : int
            Current row index to evaluate.

        Returns
        -------
        Action
            BUY if ADX > 25 AND ROC > 0 AND volume > 1.5x avg,
            SELL if ADX > 25 AND ROC < 0,
            HOLD otherwise.
        """
        adx = self._safe_get(df, index, "ADX")
        roc = self._safe_get(df, index, "ROC_10")

        # If key indicators are not available, HOLD
        if pd.isna(adx) or pd.isna(roc):
            return Action.HOLD

        # ADX must confirm strong trend for any signal
        if adx <= self.ADX_THRESHOLD:
            return Action.HOLD

        # SELL: strong trend + negative momentum
        if roc < 0:
            return Action.SELL

        # BUY: strong trend + positive momentum + volume confirmation
        if roc > 0:
            volume_ratio = self._compute_volume_ratio(df, index)
            if volume_ratio > self.VOLUME_MULTIPLIER:
                return Action.BUY

        return Action.HOLD

    def _compute_volume_ratio(self, df: pd.DataFrame, index: int) -> float:
        """
        Compute the current volume relative to the 20-period average.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a 'volume' column.
        index : int
            Current row index.

        Returns
        -------
        float
            Ratio of current volume to 20-period average.
            Returns 0.0 if data is insufficient or invalid.
        """
        if "volume" not in df.columns:
            return 0.0

        current_volume = df["volume"].iloc[index]
        if pd.isna(current_volume) or current_volume <= 0:
            return 0.0

        # Compute 20-period average volume (not including current bar)
        start = max(0, index - self.VOLUME_LOOKBACK)
        if start >= index:
            return 0.0

        vol_slice = df["volume"].iloc[start:index]
        avg_volume = vol_slice.mean()

        if pd.isna(avg_volume) or avg_volume <= 0:
            return 0.0

        return current_volume / avg_volume

    @staticmethod
    def _safe_get(df: pd.DataFrame, idx: int, col: str):
        """Safely get a value from the DataFrame, returning NaN if unavailable."""
        if col not in df.columns:
            return np.nan
        if idx < 0 or idx >= len(df):
            return np.nan
        return df[col].iloc[idx]
