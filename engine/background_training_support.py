"""
Support classes for background training.

Contains:
- GPUMemoryMonitor: GPU memory monitoring for training/inference coordination
- SymbolQueue: Thread-safe queue for incremental training symbol additions

This module is an internal implementation detail. All public APIs are
re-exported from engine.background_training.
"""

import logging
import threading
import time
from queue import Empty, Queue
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


class GPUMemoryMonitor:
    """Monitors GPU memory utilization for training/inference coordination.

    Pauses training when GPU memory exceeds the high threshold (90%),
    resumes when it drops below the low threshold (80%).

    Requirement 7.6: Prioritize inference, pause training if GPU > 90%,
    resume at < 80%.
    """

    def __init__(
        self,
        high_threshold: float = 0.90,
        low_threshold: float = 0.80,
        check_interval: float = 1.0,
    ):
        """Initialize GPU memory monitor.

        Args:
            high_threshold: Pause training if utilization exceeds this (0-1).
            low_threshold: Resume training if utilization drops below this (0-1).
            check_interval: Seconds between memory checks.
        """
        self._high_threshold = high_threshold
        self._low_threshold = low_threshold
        self._check_interval = check_interval
        self._cuda_available = torch.cuda.is_available()

    @property
    def high_threshold(self) -> float:
        """Pause threshold (fraction)."""
        return self._high_threshold

    @property
    def low_threshold(self) -> float:
        """Resume threshold (fraction)."""
        return self._low_threshold

    def get_memory_utilization(self) -> float:
        """Get current GPU memory utilization as a fraction (0-1).

        Returns:
            Memory utilization (allocated / total). Returns 0.0 if CUDA
            is unavailable.
        """
        if not self._cuda_available:
            return 0.0

        try:
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_mem
            if total == 0:
                return 0.0
            return allocated / total
        except Exception:
            return 0.0

    def get_memory_info(self) -> Dict[str, float]:
        """Get detailed GPU memory information.

        Returns:
            Dict with keys: allocated_gb, reserved_gb, total_gb, utilization.
        """
        if not self._cuda_available:
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "total_gb": 0.0,
                "utilization": 0.0,
            }

        try:
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            utilization = self.get_memory_utilization()
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "utilization": utilization,
            }
        except Exception:
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "total_gb": 0.0,
                "utilization": 0.0,
            }

    def should_pause_training(self) -> bool:
        """Check if training should be paused due to high memory.

        Returns:
            True if memory utilization exceeds the high threshold.
        """
        return self.get_memory_utilization() > self._high_threshold

    def can_resume_training(self) -> bool:
        """Check if training can resume after being paused.

        Returns:
            True if memory utilization is below the low threshold.
        """
        return self.get_memory_utilization() < self._low_threshold

    def wait_for_memory(self, timeout: float = 60.0) -> bool:
        """Block until GPU memory drops below the low threshold.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if memory dropped below threshold, False if timed out.
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.can_resume_training():
                return True
            time.sleep(self._check_interval)
        return False


class SymbolQueue:
    """Thread-safe queue for incremental training symbol additions from UI.

    When a user adds additional stock symbols through the UI, they are queued
    here for inclusion in the next incremental training session.

    Requirement 6.11: Queue new symbols without restarting the application.
    """

    def __init__(self):
        """Initialize the symbol queue."""
        self._queue: Queue = Queue()
        self._pending_symbols: List[str] = []
        self._lock = threading.Lock()

    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to the incremental training queue.

        Args:
            symbol: Stock ticker symbol to queue for training.
        """
        with self._lock:
            if symbol not in self._pending_symbols:
                self._pending_symbols.append(symbol)
                self._queue.put(symbol)
                logger.info(f"Symbol '{symbol}' queued for incremental training")

    def add_symbols(self, symbols: List[str]) -> None:
        """Add multiple symbols to the queue.

        Args:
            symbols: List of stock ticker symbols.
        """
        for symbol in symbols:
            self.add_symbol(symbol)

    def get_pending_symbols(self) -> List[str]:
        """Get all pending symbols and clear the queue.

        Returns:
            List of symbols waiting for incremental training.
        """
        with self._lock:
            symbols = list(self._pending_symbols)
            self._pending_symbols.clear()
            # Drain the queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
            return symbols

    def has_pending(self) -> bool:
        """Check if there are symbols waiting for training.

        Returns:
            True if there are pending symbols.
        """
        with self._lock:
            return len(self._pending_symbols) > 0

    @property
    def pending_count(self) -> int:
        """Number of symbols waiting for training."""
        with self._lock:
            return len(self._pending_symbols)
