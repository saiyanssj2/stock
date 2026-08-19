"""
Symbol-level progress tracker for training sessions.

Tracks which symbols have been trained, which are pending, and the current
symbol being processed. Provides overall progress calculation, ETA estimation,
and session checkpoint integration.

Thread-safe: all mutable state is protected by a threading.Lock.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4.1, 4.4, 4.5, 4.6, 5.1, 7.1
"""

import datetime
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from engine.eta_calculator import ETACalculator
from engine.session_checkpoint import SessionCheckpoint, SessionCheckpointManager

logger = logging.getLogger(__name__)


@dataclass
class SymbolResult:
    """Result of training a single symbol.

    Attributes:
        symbol: The stock symbol name (e.g., "VNM", "HPG").
        status: One of "completed", "failed", or "skipped".
        duration_seconds: Wall-clock duration of training in seconds.
        error_message: Error description if status is "failed", None otherwise.
    """

    symbol: str
    status: str  # "completed" | "failed" | "skipped"
    duration_seconds: float
    error_message: Optional[str] = None


class SymbolProgressTracker:
    """Tracks symbol-level progress within a training session.

    Provides:
    - Current symbol tracking with epoch-level granularity
    - Overall progress as float [0.0, 1.0]
    - ETA estimation via ETACalculator
    - Session checkpoint saving via SessionCheckpointManager
    - Heartbeat updates for liveness detection
    - Thread-safe access to all mutable state

    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4.1, 4.4, 4.5, 4.6, 5.1, 7.1
    """

    def __init__(
        self,
        symbols: List[str],
        mode: str,
        total_epochs_per_symbol: int,
        session_state_dict: Optional[Dict] = None,
    ):
        """Initialize the progress tracker.

        Args:
            symbols: Ordered list of symbols to train.
            mode: "full" or "incremental".
            total_epochs_per_symbol: Total epochs configured for each symbol.
            session_state_dict: Reference to st.session_state dict for UI updates.
                If None, session state updates are skipped.
        """
        self._lock = threading.Lock()

        # Immutable config
        self._symbols = list(symbols)
        self._mode = mode
        self._total_epochs_per_symbol = total_epochs_per_symbol
        self._total_count = len(symbols)

        # Mutable state
        self._completed_count = 0
        self._current_symbol: Optional[str] = None
        self._current_epoch = 0
        self._current_total_epochs = total_epochs_per_symbol
        self._current_symbol_start_time: Optional[str] = None
        self._results: List[SymbolResult] = []
        self._heartbeat_timestamp = datetime.datetime.now().isoformat()

        # External references
        self._session_state_dict = session_state_dict

        # Internal components
        self._eta_calculator = ETACalculator(total_symbols=self._total_count)
        self._checkpoint_manager = SessionCheckpointManager()

    def on_symbol_start(self, symbol: str) -> None:
        """Called when training begins for a symbol.

        Updates current symbol, resets epoch state, updates heartbeat,
        and syncs session state.

        Args:
            symbol: The symbol that is starting training.
        """
        with self._lock:
            self._current_symbol = symbol
            self._current_epoch = 0
            self._current_total_epochs = self._total_epochs_per_symbol
            self._current_symbol_start_time = datetime.datetime.now().isoformat()
            self._heartbeat_timestamp = datetime.datetime.now().isoformat()
            self._sync_session_state()

    def on_epoch_complete(self, symbol: str, epoch: int, total_epochs: int) -> None:
        """Called after each epoch completes for the current symbol.

        Updates epoch fraction for progress calculation and heartbeat.

        Args:
            symbol: The symbol currently training.
            epoch: The epoch number that just completed (1-based).
            total_epochs: Total epochs for this symbol.
        """
        with self._lock:
            self._current_epoch = epoch
            self._current_total_epochs = total_epochs
            self._heartbeat_timestamp = datetime.datetime.now().isoformat()
            self._sync_session_state()

    def on_symbol_complete(self, symbol: str, duration_seconds: float) -> None:
        """Called when a symbol's training finishes successfully.

        Increments completed count, records result, updates ETA, saves checkpoint,
        and syncs session state.

        Args:
            symbol: The symbol that completed training.
            duration_seconds: Wall-clock duration in seconds.
        """
        with self._lock:
            self._completed_count += 1
            result = SymbolResult(
                symbol=symbol,
                status="completed",
                duration_seconds=duration_seconds,
            )
            self._results.append(result)

            # Update ETA calculator
            self._eta_calculator.record_symbol_duration(duration_seconds)

            # Reset current epoch state
            self._current_epoch = 0
            self._current_symbol = None
            self._current_symbol_start_time = None
            self._heartbeat_timestamp = datetime.datetime.now().isoformat()

            # Save checkpoint
            self._save_checkpoint()

            # Sync session state
            self._sync_session_state()

    def on_symbol_failed(self, symbol: str, error: str) -> None:
        """Called when a symbol's training fails.

        Records failure with 0.0 epoch fraction, increments completed count,
        and syncs session state.

        Args:
            symbol: The symbol that failed.
            error: Error message describing the failure.
        """
        with self._lock:
            self._completed_count += 1
            result = SymbolResult(
                symbol=symbol,
                status="failed",
                duration_seconds=0.0,
                error_message=error,
            )
            self._results.append(result)

            # Reset current epoch state
            self._current_epoch = 0
            self._current_symbol = None
            self._current_symbol_start_time = None
            self._heartbeat_timestamp = datetime.datetime.now().isoformat()

            # Save checkpoint
            self._save_checkpoint()

            # Sync session state
            self._sync_session_state()

    def update_heartbeat(self) -> None:
        """Update the heartbeat timestamp.

        Called at least every 30 seconds to indicate the training process
        is still active and not hung.
        """
        with self._lock:
            self._heartbeat_timestamp = datetime.datetime.now().isoformat()
            self._sync_session_state()

    @property
    def overall_progress(self) -> float:
        """Overall progress as float in [0.0, 1.0].

        Calculated as: (completed_count + current_epoch/total_epochs) / total_count.
        Returns 0.0 if total_count is 0.
        """
        with self._lock:
            if self._total_count == 0:
                return 0.0
            epoch_fraction = 0.0
            if self._current_total_epochs > 0:
                epoch_fraction = self._current_epoch / self._current_total_epochs
            return (self._completed_count + epoch_fraction) / self._total_count

    @property
    def completed_symbols(self) -> List[SymbolResult]:
        """List of completed symbol results."""
        with self._lock:
            return list(self._results)

    @property
    def pending_symbols(self) -> List[str]:
        """Symbols not yet started.

        Returns the subset of the original symbol list that have not been
        recorded as completed or failed.
        """
        with self._lock:
            completed_names = {r.symbol for r in self._results}
            pending = [s for s in self._symbols if s not in completed_names]
            # Exclude the currently-training symbol
            if self._current_symbol and self._current_symbol in pending:
                pending = [s for s in pending if s != self._current_symbol]
            return pending

    @property
    def current_symbol(self) -> Optional[str]:
        """Currently training symbol name, or None if idle."""
        with self._lock:
            return self._current_symbol

    def get_progress_state(self) -> Dict:
        """Get complete progress state dict for session_state update.

        Returns a dictionary matching the training_progress session state format.
        """
        with self._lock:
            return self._build_progress_state()

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _build_progress_state(self) -> Dict:
        """Build the progress state dictionary (must hold lock)."""
        # Calculate overall progress
        if self._total_count == 0:
            progress = 0.0
        else:
            epoch_fraction = 0.0
            if self._current_total_epochs > 0:
                epoch_fraction = self._current_epoch / self._current_total_epochs
            progress = (self._completed_count + epoch_fraction) / self._total_count

        # Calculate ETA
        remaining_count = self._total_count - self._completed_count
        eta_seconds = self._eta_calculator.estimate_remaining_seconds(remaining_count)
        eta_display = self._eta_calculator.format_eta(eta_seconds)

        # Build completed results list
        completed_results = [
            {
                "symbol": r.symbol,
                "status": r.status,
                "duration_seconds": r.duration_seconds,
            }
            for r in self._results
        ]

        return {
            "active": True,
            "current_symbol": self._current_symbol,
            "current_symbol_start_time": self._current_symbol_start_time,
            "current_epoch": self._current_epoch,
            "total_epochs": self._current_total_epochs,
            "completed_count": self._completed_count,
            "total_count": self._total_count,
            "overall_progress": progress,
            "eta_display": eta_display,
            "eta_seconds": eta_seconds,
            "average_duration": self._eta_calculator.average_duration,
            "heartbeat_timestamp": self._heartbeat_timestamp,
            "completed_results": completed_results,
            "mode": self._mode,
        }

    def _sync_session_state(self) -> None:
        """Sync progress state to the session_state dict (must hold lock).

        In the Streamlit app, the UI reads directly from the tracker via
        get_progress_state() — no writing to session_state from background
        threads. For tests using plain dicts, we still write.
        """
        if self._session_state_dict is None:
            return
        # Only write to plain dicts (tests), skip Streamlit's SessionState
        # which triggers ScriptRunContext warnings from background threads
        if type(self._session_state_dict).__name__ == "dict":
            try:
                state = self._build_progress_state()
                self._session_state_dict["training_progress"] = state
            except Exception:
                pass

    def _save_checkpoint(self) -> None:
        """Save session checkpoint to disk (must hold lock).

        Creates a SessionCheckpoint from current state and persists it
        via SessionCheckpointManager.
        """
        try:
            completed_names = [r.symbol for r in self._results]
            completed_set = set(completed_names)
            pending = [s for s in self._symbols if s not in completed_set]

            # Build duration map for completed symbols
            symbol_durations = {
                r.symbol: r.duration_seconds
                for r in self._results
                if r.status == "completed"
            }

            checkpoint = SessionCheckpoint(
                completed_symbols=completed_names,
                pending_symbols=pending,
                mode=self._mode,
                session_start_time=datetime.datetime.now().isoformat(),
                symbol_durations=symbol_durations,
                total_epochs_per_symbol=self._total_epochs_per_symbol,
            )
            self._checkpoint_manager.save(checkpoint)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
