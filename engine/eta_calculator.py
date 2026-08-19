"""ETA Calculator for training session time estimation.

Estimates remaining training time based on observed per-symbol durations
using a running average approach.
"""

from typing import List, Optional


class ETACalculator:
    """Calculates estimated time remaining for a training session."""

    def __init__(self, total_symbols: int):
        """
        Args:
            total_symbols: Total number of symbols in the session.
        """
        self.total_symbols = total_symbols
        self.durations: List[float] = []

    def record_symbol_duration(self, duration_seconds: float) -> None:
        """Record a completed symbol's duration for running average."""
        self.durations.append(duration_seconds)

    @property
    def average_duration(self) -> Optional[float]:
        """Running average seconds per symbol, None if no completions yet."""
        if not self.durations:
            return None
        return sum(self.durations) / len(self.durations)

    def estimate_remaining_seconds(self, remaining_count: int) -> Optional[float]:
        """
        Estimate remaining time.

        Returns:
            Estimated seconds remaining, or None if insufficient data.
        """
        avg = self.average_duration
        if avg is None:
            return None
        return avg * remaining_count

    def format_eta(self, remaining_seconds: Optional[float]) -> str:
        """
        Format ETA for display.

        Returns:
            - "Đang ước tính..." if remaining_seconds is None
            - "Sắp hoàn tất" if < 60 seconds
            - "Xh Ym" if >= 3600 seconds
            - "Ym Zs" otherwise
        """
        if remaining_seconds is None:
            return "Đang ước tính..."

        if remaining_seconds < 60:
            return "Sắp hoàn tất"

        if remaining_seconds >= 3600:
            hours = int(remaining_seconds // 3600)
            minutes = int((remaining_seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

        minutes = int(remaining_seconds // 60)
        seconds = int(remaining_seconds % 60)
        return f"{minutes}m {seconds}s"
