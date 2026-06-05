"""
Philosophy-based trading strategies for comparison with AI engine.

Strategies:
- Wyckoff accumulation/distribution
- Technical indicator composite
- Momentum (ADX + ROC + Volume)
- Mean reversion (Bollinger + RSI)
"""

from engine.strategies.base import BaseStrategy
from engine.strategies.mean_reversion import MeanReversionStrategy
from engine.strategies.momentum import MomentumStrategy
from engine.strategies.technical import TechnicalStrategy
from engine.strategies.wyckoff import WyckoffStrategy

__all__ = [
    "BaseStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "TechnicalStrategy",
    "WyckoffStrategy",
]
