"""
Wyckoff-based trading strategy.

Generates BUY/HOLD/SELL signals based on Wyckoff accumulation/distribution
phase detection. Uses a composite score derived from:
- Wyckoff phase analysis (accumulation, distribution, markup, markdown)
- Volume Spread Analysis (VSA)
- OBV trend analysis
- Spring/Upthrust detection

Signal rules:
- BUY if composite score >= +3
- SELL if composite score <= -3
- HOLD otherwise

Reuses scoring logic from ui_analyze_wyckoff.calc_wyckoff.
"""

import numpy as np
import pandas as pd

from engine.config import Action
from engine.strategies.base import BaseStrategy


class WyckoffStrategy(BaseStrategy):
    """
    Wyckoff accumulation/distribution strategy.

    Computes a composite Wyckoff score based on phase detection, VSA,
    OBV trend, and spring/upthrust patterns. Produces:
    - BUY when score >= +3
    - SELL when score <= -3
    - HOLD otherwise
    """

    BUY_THRESHOLD = 3
    SELL_THRESHOLD = -3

    @property
    def name(self) -> str:
        return "Wyckoff"

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        """
        Generate a Wyckoff-based trading signal.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV and indicator columns.
        index : int
            Current row index to evaluate.

        Returns
        -------
        Action
            BUY if composite score >= +3, SELL if <= -3, HOLD otherwise.
        """
        score = self.compute_composite_score(df, index)

        if score >= self.BUY_THRESHOLD:
            return Action.BUY
        elif score <= self.SELL_THRESHOLD:
            return Action.SELL
        else:
            return Action.HOLD

    def compute_composite_score(self, df: pd.DataFrame, idx: int) -> int:
        """
        Compute the Wyckoff composite score at a given index.

        The score combines:
        - Phase score: Wyckoff market phase detection (-2 to +2)
        - VSA score: Volume Spread Analysis (-2 to +2)
        - OBV score: On-Balance Volume trend (-1 to +1)
        - Spring/Upthrust score: trap detection (-2 to +2)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV and indicator columns.
        idx : int
            Current row index.

        Returns
        -------
        int
            Composite Wyckoff score (typically in range [-7, +7]).
        """
        # Need at least 60 rows of history for meaningful analysis
        lookback_60 = 60
        lookback_20 = 20

        # Determine available lookback window
        start_60 = max(0, idx - lookback_60 + 1)
        start_20 = max(0, idx - lookback_20 + 1)

        # If not enough data for even basic analysis, return 0 (HOLD)
        if idx < 5:
            return 0

        d60 = df.iloc[start_60:idx + 1]
        d20 = df.iloc[start_20:idx + 1]

        current = df.iloc[idx]
        close = current['close']
        high = current['high']
        low = current['low']
        volume = current['volume']

        # Volume average (20-period)
        vol_avg = d20['volume'].mean() if len(d20) > 0 else volume
        vol_ratio = volume / vol_avg if vol_avg > 0 else 1.0

        # Price ranges
        high_60 = d60['high'].max()
        low_60 = d60['low'].min()
        high_20 = d20['high'].max()
        low_20 = d20['low'].min()
        range_60 = high_60 - low_60

        # Spread analysis
        spread = high - low
        spread_avg = (d20['high'] - d20['low']).mean() if len(d20) > 0 else spread
        close_pos = (close - low) / spread if spread > 0 else 0.5

        # OBV analysis
        obv_rising = self._get_obv_rising(df, idx)
        obv_5_change = self._get_obv_change(df, idx, periods=5)

        # Price trend
        price_trend_20 = self._get_price_trend(df, idx, periods=min(20, idx))

        # Volume trend
        vol_trend_up = self._get_vol_trend_up(df, idx)

        # Position relative to range
        near_low = close <= low_60 + range_60 * 0.25 if range_60 > 0 else False
        near_high = close >= high_60 - range_60 * 0.25 if range_60 > 0 else False

        # Spring/Upthrust detection
        spring = (low < low_20 * 0.995 and close > low_20 * 0.995
                  and vol_ratio < 1.2) if low_20 > 0 else False
        upthrust = (high > high_20 * 1.005 and close < high_20 * 1.005
                    and vol_ratio > 1.0) if high_20 > 0 else False

        # Selling Climax
        sc = (near_low and vol_ratio > 2.5
              and spread > spread_avg * 1.5 and close_pos > 0.3)

        wide = spread > spread_avg * 1.3 if spread_avg > 0 else False
        narrow = spread < spread_avg * 0.7 if spread_avg > 0 else False

        # --- VSA Score ---
        vsa_score = self._compute_vsa_score(
            wide, narrow, close_pos, vol_ratio
        )

        # --- Phase Score ---
        phase_score = self._compute_phase_score(
            near_low, near_high, obv_rising, vol_trend_up,
            sc, price_trend_20
        )

        # --- Spring/Upthrust Score ---
        spring_score = 0
        if spring:
            spring_score = 2
        elif upthrust:
            spring_score = -2

        # --- OBV Score ---
        obv_score = self._compute_obv_score(obv_rising, obv_5_change)

        # Total composite score
        total = phase_score + vsa_score + obv_score + spring_score
        return total

    def _compute_vsa_score(
        self, wide: bool, narrow: bool, close_pos: float, vol_ratio: float
    ) -> int:
        """Compute Volume Spread Analysis score (-2 to +2)."""
        if wide and close_pos > 0.6 and vol_ratio > 1.2:
            return 2  # Demand - strong buying
        elif wide and close_pos < 0.4 and vol_ratio > 1.2:
            return -2  # Supply - strong selling
        elif narrow and vol_ratio > 1.3:
            return -1  # No result - resistance
        elif narrow and vol_ratio < 0.7 and close_pos > 0.5:
            return 1  # No supply - drying up
        elif narrow and vol_ratio < 0.7:
            return 0  # No demand/supply
        else:
            return 0  # Normal volume

    def _compute_phase_score(
        self,
        near_low: bool,
        near_high: bool,
        obv_rising: bool,
        vol_trend_up: bool,
        sc: bool,
        price_trend_20: float,
    ) -> int:
        """Compute Wyckoff phase score (-2 to +2)."""
        if near_low and obv_rising and vol_trend_up:
            return 2  # Accumulation Phase C/D
        elif sc:
            return 1  # Selling Climax
        elif near_low and not obv_rising:
            return -1  # Markdown / Phase A
        elif near_high and not obv_rising:
            return -2  # Distribution Phase B/C
        elif near_high and obv_rising:
            return 1  # Markup Phase D
        elif not near_low and not near_high and price_trend_20 > 0 and obv_rising:
            return 1  # Markup forming
        elif not near_low and not near_high and price_trend_20 < 0 and not obv_rising:
            return -1  # Markdown
        else:
            return 0  # Ranging Phase B

    def _compute_obv_score(self, obv_rising: bool, obv_5_change: float) -> int:
        """Compute OBV trend score (-1 to +1)."""
        if obv_rising and obv_5_change > 0:
            return 1
        elif not obv_rising and obv_5_change < 0:
            return -1
        else:
            return 0

    def _get_obv_rising(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if OBV is above its 20-period EMA."""
        if 'OBV' not in df.columns:
            return False

        # Need at least 20 periods for EMA
        start = max(0, idx - 19)
        obv_slice = df['OBV'].iloc[start:idx + 1]

        if len(obv_slice) < 2:
            return False

        obv_ema = obv_slice.ewm(span=min(20, len(obv_slice))).mean().iloc[-1]
        current_obv = df['OBV'].iloc[idx]

        if pd.isna(current_obv) or pd.isna(obv_ema):
            return False

        return current_obv > obv_ema

    def _get_obv_change(self, df: pd.DataFrame, idx: int, periods: int = 5) -> float:
        """Get OBV change over the specified number of periods."""
        if 'OBV' not in df.columns:
            return 0.0

        if idx < periods:
            return 0.0

        current_obv = df['OBV'].iloc[idx]
        past_obv = df['OBV'].iloc[idx - periods]

        if pd.isna(current_obv) or pd.isna(past_obv):
            return 0.0

        return current_obv - past_obv

    def _get_price_trend(self, df: pd.DataFrame, idx: int, periods: int = 20) -> float:
        """Get price trend over the specified number of periods."""
        if idx < periods or periods <= 0:
            return 0.0

        current_close = df['close'].iloc[idx]
        past_close = df['close'].iloc[idx - periods]

        if pd.isna(current_close) or pd.isna(past_close):
            return 0.0

        return current_close - past_close

    def _get_vol_trend_up(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if recent 5-day average volume exceeds 20-day average."""
        if idx < 5:
            return False

        vol_5 = df['volume'].iloc[max(0, idx - 4):idx + 1].mean()
        vol_20 = df['volume'].iloc[max(0, idx - 19):idx + 1].mean()

        if pd.isna(vol_5) or pd.isna(vol_20) or vol_20 == 0:
            return False

        return vol_5 > vol_20
