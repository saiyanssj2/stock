"""
Background training manager for concurrent training and inference.

Provides:
- BackgroundTrainingManager: orchestrates background training with hot-swap
- GPUMemoryMonitor: monitors GPU memory and pauses/resumes training
- SymbolQueue: thread-safe queue for incremental training additions from UI
- TrainingStatus: dataclass for UI status display

Key behaviors:
- Training runs in a background thread, inference continues uninterrupted
- GPU memory monitoring: pause training if >90% utilization, resume at <80%
- Model hot-swap within 3 seconds without dropping in-flight requests
- Symbol queue for incremental training additions from UI
- Inference completes within 5 seconds regardless of training activity
- Per-symbol progress tracking with checkpoint/resume capability

Requirements: 7.1, 7.2, 7.3, 7.5, 7.6, 6.11, 12.4, 5.4, 5.5, 5.6, 5.7, 5.8, 6.3, 6.4, 6.5, 6.6
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

from engine.session_checkpoint import SessionCheckpointManager
from engine.symbol_progress_tracker import SymbolProgressTracker

logger = logging.getLogger(__name__)

# Re-export support classes so existing imports from this module still work
from engine.background_training_support import GPUMemoryMonitor, SymbolQueue  # noqa: E402


# ==============================================================================
# Training State Enum
# ==============================================================================


class TrainingState(Enum):
    """Current state of background training."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"  # Paused due to GPU memory pressure
    COMPLETING = "completing"  # Finishing current epoch before hot-swap
    STOPPED = "stopped"


# ==============================================================================
# Training Status (for UI display)
# ==============================================================================


@dataclass
class TrainingStatus:
    """Status information for UI display.

    Updated at least every 10 seconds during training (Requirement 7.5).
    """

    state: TrainingState = TrainingState.IDLE
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    current_val_loss: float = 0.0
    eta_seconds: float = 0.0
    symbols_training: List[str] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    paused_reason: Optional[str] = None
    model_version: int = 0
    last_training_completed: Optional[datetime] = None


# ==============================================================================
# Background Training Manager
# ==============================================================================


class BackgroundTrainingManager:
    """Manages background training with hot-swap and GPU memory coordination.

    Orchestrates:
    - Background training thread that doesn't block inference
    - GPU memory monitoring to pause/resume training
    - Model hot-swap within 3 seconds after training completes
    - Symbol queue for incremental training additions
    - Status reporting for UI (updated every 10 seconds)

    The manager ensures inference always completes within 5 seconds
    regardless of training activity (Requirement 7.1).

    Thread Safety:
        - Training runs in a dedicated daemon thread
        - All state access is protected by locks
        - Model hot-swap uses ModelManager's existing lock
        - Inference priority is enforced via GPU memory monitoring

    Requirements: 7.1, 7.2, 7.3, 7.5, 7.6, 6.11, 12.4
    """

    def __init__(
        self,
        model_manager,
        training_pipeline=None,
        gpu_monitor: Optional[GPUMemoryMonitor] = None,
        hot_swap_timeout: float = 3.0,
    ):
        """Initialize BackgroundTrainingManager.

        Args:
            model_manager: ModelManager instance for hot-swap operations.
            training_pipeline: TrainingPipeline instance (can be set later).
            gpu_monitor: GPUMemoryMonitor instance. Created with defaults if None.
            hot_swap_timeout: Maximum seconds for model hot-swap (default 3.0).
        """
        self._model_manager = model_manager
        self._training_pipeline = training_pipeline
        self._gpu_monitor = gpu_monitor or GPUMemoryMonitor(
            high_threshold=0.75, low_threshold=0.60
        )
        self._hot_swap_timeout = hot_swap_timeout

        # Symbol queue for UI additions
        self._symbol_queue = SymbolQueue()

        # Training state
        self._status = TrainingStatus()
        self._status_lock = threading.Lock()

        # Training thread control
        self._training_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unpaused

        # Inference priority: this event is cleared during hot-swap to let
        # in-flight inference requests complete first
        self._inference_gate = threading.Event()
        self._inference_gate.set()  # Inference always allowed by default

        # Callback for UI notification
        self._on_status_change: Optional[Callable[[TrainingStatus], None]] = None
        self._on_training_paused: Optional[Callable[[str], None]] = None

        # Progress tracking state
        self._resume_checkpoint = None
        self._session_state_dict: Optional[Dict] = None
        self._progress_tracker: Optional[SymbolProgressTracker] = None
        self._current_tracker: Optional[SymbolProgressTracker] = None
        self._current_training_symbol: Optional[str] = None

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def status(self) -> TrainingStatus:
        """Current training status (thread-safe copy)."""
        with self._status_lock:
            return TrainingStatus(
                state=self._status.state,
                current_epoch=self._status.current_epoch,
                total_epochs=self._status.total_epochs,
                current_loss=self._status.current_loss,
                current_val_loss=self._status.current_val_loss,
                eta_seconds=self._status.eta_seconds,
                symbols_training=list(self._status.symbols_training),
                last_updated=self._status.last_updated,
                paused_reason=self._status.paused_reason,
                model_version=self._status.model_version,
                last_training_completed=self._status.last_training_completed,
            )

    @property
    def is_training(self) -> bool:
        """Whether training is currently active (running or paused)."""
        with self._status_lock:
            return self._status.state in (TrainingState.RUNNING, TrainingState.PAUSED)

    @property
    def is_paused(self) -> bool:
        """Whether training is paused due to GPU constraints."""
        with self._status_lock:
            return self._status.state == TrainingState.PAUSED

    @property
    def symbol_queue(self) -> SymbolQueue:
        """Access to the symbol queue for UI additions."""
        return self._symbol_queue

    @property
    def gpu_monitor(self) -> GPUMemoryMonitor:
        """Access to the GPU memory monitor."""
        return self._gpu_monitor

    @property
    def progress_tracker(self) -> Optional[SymbolProgressTracker]:
        """Access to the current progress tracker (None when not training)."""
        return self._progress_tracker

    # --------------------------------------------------------------------------
    # Training Pipeline Configuration
    # --------------------------------------------------------------------------

    def set_training_pipeline(self, pipeline) -> None:
        """Set or update the training pipeline.

        Args:
            pipeline: TrainingPipeline instance.
        """
        self._training_pipeline = pipeline

    def set_status_callback(self, callback: Callable[[TrainingStatus], None]) -> None:
        """Set callback for status changes (for UI updates).

        Args:
            callback: Function called with updated TrainingStatus.
        """
        self._on_status_change = callback

    def set_paused_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for training pause notifications.

        Args:
            callback: Function called with pause reason string.
        """
        self._on_training_paused = callback

    # --------------------------------------------------------------------------
    # Start/Stop Training
    # --------------------------------------------------------------------------

    def start_training(
        self,
        symbol_data: Dict[str, "pd.DataFrame"],
        mode: str = "incremental",
        data_dir: Optional[str] = None,
        resume_from_checkpoint: bool = False,
        session_state_dict: Optional[Dict] = None,
    ) -> bool:
        """Start background training.

        Launches a background thread that runs training without blocking
        inference. Training can be paused/resumed based on GPU memory.

        Args:
            symbol_data: Dict mapping symbol names to DataFrames.
            mode: "full" for full retraining, "incremental" for fine-tuning.
            data_dir: Base directory for additional data loading.
            resume_from_checkpoint: If True, resume from existing checkpoint.
            session_state_dict: Reference to st.session_state for progress updates.

        Returns:
            True if training started successfully, False if already running.

        Requirements: 5.4, 5.5, 5.7, 5.8
        """
        if self.is_training:
            logger.warning("Training already in progress, cannot start new session")
            return False

        if self._training_pipeline is None:
            logger.error("Training pipeline not configured")
            return False

        # Include any queued symbols
        queued_symbols = self._symbol_queue.get_pending_symbols()
        if queued_symbols:
            import pandas as pd

            for symbol in queued_symbols:
                if symbol not in symbol_data and data_dir:
                    csv_path = Path(data_dir) / f"{symbol}.csv"
                    if csv_path.exists():
                        try:
                            symbol_data[symbol] = pd.read_csv(csv_path)
                            logger.info(f"Loaded queued symbol '{symbol}' from {csv_path}")
                        except Exception as e:
                            logger.warning(f"Failed to load queued symbol '{symbol}': {e}")

        # Check for existing checkpoint (Requirement 5.4)
        checkpoint_manager = SessionCheckpointManager()
        self._resume_checkpoint = None
        if resume_from_checkpoint and checkpoint_manager.exists():
            checkpoint = checkpoint_manager.load()
            if checkpoint is not None:
                self._resume_checkpoint = checkpoint
                logger.info(
                    f"Resuming from checkpoint: "
                    f"{len(checkpoint.completed_symbols)} completed, "
                    f"{len(checkpoint.pending_symbols)} pending"
                )
            else:
                # Checkpoint was corrupt, load() already deleted it
                logger.warning("Checkpoint was corrupt, starting fresh session")
        elif not resume_from_checkpoint and checkpoint_manager.exists():
            # User chose fresh start - delete old checkpoint (Requirement 6.4)
            checkpoint_manager.delete()
            logger.info("User chose fresh start, deleted existing checkpoint")

        # Store session state dict reference for progress tracking
        self._session_state_dict = session_state_dict

        # Reset stop event
        self._stop_event.clear()
        self._pause_event.set()

        # Update status
        with self._status_lock:
            self._status.state = TrainingState.RUNNING
            self._status.symbols_training = list(symbol_data.keys())
            self._status.current_epoch = 0
            self._status.last_updated = datetime.now()
            self._status.paused_reason = None

        # Launch background training thread
        self._training_thread = threading.Thread(
            target=self._training_loop,
            args=(symbol_data, mode, data_dir),
            daemon=True,
            name="BackgroundTraining",
        )
        self._training_thread.start()

        logger.info(f"Background training started: mode={mode}, symbols={len(symbol_data)}")
        return True

    def stop_training(self) -> None:
        """Stop background training gracefully.

        Signals the training thread to stop after the current epoch completes.
        Does not interrupt in-progress computation.
        """
        self._stop_event.set()
        self._pause_event.set()  # Unpause if paused so thread can exit

        if self._training_thread and self._training_thread.is_alive():
            self._training_thread.join(timeout=10.0)

        with self._status_lock:
            self._status.state = TrainingState.STOPPED
            self._status.last_updated = datetime.now()

        logger.info("Background training stopped")

    # --------------------------------------------------------------------------
    # Training Loop (runs in background thread)
    # --------------------------------------------------------------------------

    def _training_loop(
        self,
        symbol_data: Dict[str, "pd.DataFrame"],
        mode: str,
        data_dir: Optional[str],
    ) -> None:
        """Main training loop running in background thread.

        This method:
        1. Determines the symbol list (from checkpoint or fresh)
        2. Creates a SymbolProgressTracker for per-symbol tracking
        3. Iterates over symbols, training each one
        4. Tracks progress, handles failures, and saves checkpoints
        5. Deletes checkpoint file on successful completion

        Args:
            symbol_data: Symbol DataFrames for training.
            mode: "full" or "incremental".
            data_dir: Base directory for data loading.

        Requirements: 5.4, 5.5, 5.6, 6.3, 6.5, 6.6
        """
        try:
            # Suppress noisy Streamlit "missing ScriptRunContext" warnings
            # in the background thread — they are harmless
            import logging as _logging
            _st_logger = _logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
            _st_logger.setLevel(_logging.ERROR)

            self._run_tracked_training(symbol_data, mode, data_dir)
        except Exception as e:
            logger.error(f"Background training failed: {e}", exc_info=True)
            with self._status_lock:
                self._status.state = TrainingState.STOPPED
                self._status.paused_reason = f"Error: {str(e)}"
                self._status.last_updated = datetime.now()
        finally:
            if not self._stop_event.is_set():
                with self._status_lock:
                    self._status.state = TrainingState.IDLE
                    self._status.last_updated = datetime.now()

    def _run_tracked_training(
        self,
        symbol_data: Dict[str, "pd.DataFrame"],
        mode: str,
        data_dir: Optional[str],
    ) -> None:
        """Run training with per-symbol progress tracking.

        Handles checkpoint resume logic, missing CSV detection, and
        per-symbol iteration with tracker callbacks.

        Delegates to engine.background_training_impl.run_tracked_training.

        Args:
            symbol_data: Symbol DataFrames for training.
            mode: "full" or "incremental".
            data_dir: Base directory for data loading.

        Requirements: 5.4, 5.5, 5.6, 6.5, 6.6
        """
        from engine.background_training_impl import run_tracked_training

        run_tracked_training(self, symbol_data, mode, data_dir)

    def _run_monitored_training(
        self,
        pipeline,
        symbol_data: Dict[str, "pd.DataFrame"],
        mode: str,
        data_dir: Optional[str],
    ) -> Optional["TrainingResult"]:
        """Run training with GPU memory monitoring between epochs.

        Delegates to engine.background_training_impl.run_monitored_training.

        Args:
            pipeline: TrainingPipeline instance.
            symbol_data: Training data.
            mode: "full" or "incremental".
            data_dir: Data directory.

        Returns:
            TrainingResult if completed, None if stopped/failed.
        """
        from engine.background_training_impl import run_monitored_training

        return run_monitored_training(
            manager=self,
            pipeline=pipeline,
            symbol_data=symbol_data,
            mode=mode,
            data_dir=data_dir,
        )

    def notify_epoch_complete(self, epoch: int, total_epochs: int) -> None:
        """Notify the progress tracker that an epoch has completed.

        Called from the training implementation after each epoch to update
        per-symbol progress tracking and heartbeat.

        Args:
            epoch: The epoch number that just completed (1-based).
            total_epochs: Total epochs for this symbol.
        """
        tracker = getattr(self, "_current_tracker", None)
        symbol = getattr(self, "_current_training_symbol", None)
        if tracker is not None and symbol is not None:
            tracker.on_epoch_complete(symbol, epoch, total_epochs)
            tracker.update_heartbeat()

    # --------------------------------------------------------------------------
    # GPU Memory Pause/Resume
    # --------------------------------------------------------------------------

    def _handle_gpu_pause(self) -> None:
        """Handle GPU memory pressure by pausing training.

        Sets training state to PAUSED, notifies UI, and waits for
        memory to drop below the low threshold.

        Requirement 7.6: Prioritize inference, pause training.
        """
        reason = (
            f"GPU memory utilization exceeds "
            f"{self._gpu_monitor.high_threshold * 100:.0f}%"
        )
        logger.warning(f"Pausing training: {reason}")

        with self._status_lock:
            self._status.state = TrainingState.PAUSED
            self._status.paused_reason = reason
            self._status.last_updated = datetime.now()

        # Clear pause event to block training thread
        self._pause_event.clear()

        # Notify UI callback
        if self._on_training_paused:
            try:
                self._on_training_paused(reason)
            except Exception:
                pass

        if self._on_status_change:
            try:
                self._on_status_change(self.status)
            except Exception:
                pass

        # Wait for memory to drop (in separate monitor thread or loop)
        self._start_memory_watch()

    def _start_memory_watch(self) -> None:
        """Start watching GPU memory for resume condition.

        Spawns a lightweight thread that checks memory utilization
        and resumes training when it drops below the low threshold.
        """

        def _watch():
            while not self._stop_event.is_set():
                if self._gpu_monitor.can_resume_training():
                    self._resume_from_pause()
                    return
                time.sleep(1.0)

        watch_thread = threading.Thread(
            target=_watch, daemon=True, name="GPUMemoryWatch"
        )
        watch_thread.start()

    def _resume_from_pause(self) -> None:
        """Resume training after GPU memory pressure subsides."""
        logger.info("GPU memory pressure resolved, resuming training")

        with self._status_lock:
            self._status.state = TrainingState.RUNNING
            self._status.paused_reason = None
            self._status.last_updated = datetime.now()

        # Set pause event to unblock training thread
        self._pause_event.set()

        if self._on_status_change:
            try:
                self._on_status_change(self.status)
            except Exception:
                pass

    # --------------------------------------------------------------------------
    # Model Hot-Swap
    # --------------------------------------------------------------------------

    def _perform_hot_swap(self, model_path: str) -> None:
        """Hot-swap the model within the timeout constraint.

        Atomically replaces the active model without dropping in-flight
        inference requests. Uses ModelManager's thread-safe hot_swap method.

        Requirement 7.3: Hot-swap within 3 seconds without dropping requests.

        Args:
            model_path: Path to the newly trained model file.
        """
        logger.info(f"Starting model hot-swap from {model_path}")
        swap_start = time.time()

        with self._status_lock:
            self._status.state = TrainingState.COMPLETING

        try:
            self._model_manager.hot_swap(model_path)

            swap_time = time.time() - swap_start

            with self._status_lock:
                self._status.model_version = self._model_manager.version
                self._status.last_training_completed = datetime.now()
                self._status.state = TrainingState.IDLE
                self._status.last_updated = datetime.now()

            if swap_time > self._hot_swap_timeout:
                logger.warning(
                    f"Hot-swap took {swap_time:.2f}s "
                    f"(exceeds {self._hot_swap_timeout}s target)"
                )
            else:
                logger.info(
                    f"Hot-swap complete in {swap_time:.2f}s "
                    f"(model version {self._model_manager.version})"
                )

        except Exception as e:
            logger.error(f"Hot-swap failed: {e}")
            with self._status_lock:
                self._status.state = TrainingState.IDLE
                self._status.paused_reason = f"Hot-swap failed: {str(e)}"
                self._status.last_updated = datetime.now()

    # --------------------------------------------------------------------------
    # Queue Management
    # --------------------------------------------------------------------------

    def queue_symbol(self, symbol: str) -> None:
        """Queue a symbol for the next incremental training session.

        Called from UI when user adds new symbols.

        Requirement 6.11: Queue without restarting the application.

        Args:
            symbol: Stock ticker symbol to add.
        """
        self._symbol_queue.add_symbol(symbol)

    def queue_symbols(self, symbols: List[str]) -> None:
        """Queue multiple symbols for incremental training.

        Args:
            symbols: List of stock ticker symbols.
        """
        self._symbol_queue.add_symbols(symbols)

    def get_queued_symbols(self) -> List[str]:
        """Get currently queued symbols (for display purposes).

        Note: This doesn't drain the queue - just peeks.

        Returns:
            List of queued symbol names.
        """
        with self._symbol_queue._lock:
            return list(self._symbol_queue._pending_symbols)
