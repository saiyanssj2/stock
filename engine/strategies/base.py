"""
Base strategy abstract class for philosophy-based trading strategies.

All philosophy strategies (Wyckoff, Technical, Momentum, Mean Reversion) inherit
from BaseStrategy and implement the generate_signal method.
"""

from abc import ABC, abstractmethod

import pandas as pd

from engine.config import Action


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Each strategy receives a DataFrame with OHLCV data and computed technical
    indicators, and produces a trading signal (BUY, HOLD, or SELL) for a given
    index position in the DataFrame.
    """

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        """
        Generate a trading signal for the given index position.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing OHLCV data and technical indicators.
            Expected to have columns: open, high, low, close, volume,
            and indicator columns from analysis.add_indicators().
        index : int
            The current row index in the DataFrame for which to generate
            the signal. The strategy may look back at earlier rows but
            must not look ahead beyond index.

        Returns
        -------
        Action
            One of Action.BUY, Action.HOLD, or Action.SELL.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable name for the strategy."""
        return self.__class__.__name__
