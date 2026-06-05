"""
Technical indicator composite strategy.

Generates BUY/HOLD/SELL signals based on a composite score from multiple
technical indicators: RSI, MACD, EMA crossover, and Bollinger Bands.

Signal rules:
- BUY if composite score >= +5
- SELL if composite score <= -4
- HOLD otherwise

Scoring logic adapted from ui_analyze_signals.py existing signal analysis.
"""

import numpy as np
import pandas as pd

from engine.config import Action
from engine.strategies.base import BaseStrategy


class TechnicalStrategy(BaseStrategy):
    """
    Technical indicator composite strategy.

    Computes a composite score from RSI, MACD, EMA crossover, and Bollinger
    Bands indicators. Produces:
    - BUY when score >= +5
    - SELL when score <= -4
    - HOLD otherwise
    """

    BUY_THRESHOLD = 5
    SELL_THRESHOLD = -4

    @property
    def name(self) -> str:
        return "Technical"

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        """
        Generate a technical indicator composite signal.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV and indicator columns.
        index : int
            Current row index to evaluate.

        Returns
        -------
        Action
            BUY if composite score >= +5, SELL if <= -4, HOLD otherwise.
        """
        score = self._compute_composite_score(df, index)

        if score >= self.BUY_THRESHOLD:
            return Action.BUY
        elif score <= self.SELL_THRESHOLD:
            return Action.SELL
        else:
            return Action.HOLD

    def _compute_composite_score(self, df: pd.DataFrame, idx: int) -> int:
        """
        Compute the technical composite score at a given index.

        Components:
        - EMA crossover (EMA_20 vs EMA_50): +1/-1
        - Price vs EMA_20: +1/-1
        - RSI_14 signals: -2 to +2
        - MACD crossover/position: -2 to +2
        - MACD histogram direction: -1 to +1
        - Bollinger Bands position: -1 to +2

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV and indicator columns.
        idx : int
            Current row index.

        Returns
        -------
        int
            Composite technical score.
        """
        score = 0

        close = self._safe_get(df, idx, "close")
        if pd.isna(close):
            return 0

        # --- EMA Crossover ---
        ema20 = self._safe_get(df, idx, "EMA_20")
        ema50 = self._safe_get(df, idx, "EMA_50")
        if pd.notna(ema20) and pd.notna(ema50):
            score += 1 if ema20 > ema50 else -1

        # --- Price vs EMA_20 ---
        if pd.notna(ema20):
            score += 1 if close > ema20 else -1

        # --- RSI_14 ---
        rsi = self._safe_get(df, idx, "RSI_14")
        if pd.notna(rsi):
            if rsi < 30:
                score += 2  # Oversold
            elif rsi > 70:
                score += -2  # Overbought
            elif rsi < 45:
                score += -1  # Weak momentum
            elif rsi > 55:
                score += 1  # Good momentum

        # --- MACD ---
        macd = self._safe_get(df, idx, "MACD")
        macd_signal = self._safe_get(df, idx, "MACD_signal")
        if pd.notna(macd) and pd.notna(macd_signal) and idx > 0:
            prev_macd = self._safe_get(df, idx - 1, "MACD")
            prev_macd_signal = self._safe_get(df, idx - 1, "MACD_signal")

            if pd.notna(prev_macd) and pd.notna(prev_macd_signal):
                # Fresh crossover is strongest signal
                if macd > macd_signal and prev_macd <= prev_macd_signal:
                    score += 2  # Bullish crossover
                elif macd < macd_signal and prev_macd >= prev_macd_signal:
                    score += -2  # Bearish crossover
                elif macd > macd_signal:
                    score += 1  # Bullish
                else:
                    score += -1  # Bearish

            # MACD Histogram direction
            macd_hist = self._safe_get(df, idx, "MACD_hist")
            if pd.notna(macd_hist) and idx > 0:
                prev_hist = self._safe_get(df, idx - 1, "MACD_hist")
                if pd.notna(prev_hist):
                    if macd_hist > 0 and prev_hist <= 0:
                        score += 1  # Histogram turned positive
                    elif macd_hist < 0 and prev_hist >= 0:
                        score += -1  # Histogram turned negative

        # --- Bollinger Bands ---
        bb_upper = self._safe_get(df, idx, "BB_upper")
        bb_lower = self._safe_get(df, idx, "BB_lower")
        if pd.notna(bb_upper) and pd.notna(bb_lower) and bb_upper != bb_lower:
            if close < bb_lower:
                score += 2  # Below lower band - oversold
            elif close > bb_upper:
                score += -1  # Above upper band - overbought
            else:
                bb_pct = (close - bb_lower) / (bb_upper - bb_lower)
                if bb_pct < 0.2:
                    score += 1  # Near lower band - support zone

        return score

    @staticmethod
    def _safe_get(df: pd.DataFrame, idx: int, col: str):
        """Safely get a value from the DataFrame, returning NaN if unavailable."""
        if col not in df.columns:
            return np.nan
        if idx < 0 or idx >= len(df):
            return np.nan
        val = df[col].iloc[idx]
        return val
