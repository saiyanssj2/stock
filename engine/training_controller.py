"""
Training Controller - Manages graceful stop and training lifecycle.

Implements the graceful stop mechanism that allows training to complete the
current epoch cleanly before stopping, saving a valid checkpoint. Also provides
training status information for UI display.

Requirements: 14.2, 14.3, 14.5, 14.6, 14.7
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from engine.config import EngineError


# ==============================================================================
# TrainingStatus Dataclass
# ==============================================================================


@dataclass
class GracefulTrainingStatus:
    """Training status for graceful stop UI display.

    Provides information about the current training state, including
    whether a graceful stop is pending, the current epoch progress,
    and estimated time remaining.

    Attributes:
        is_active: Whether a training session is currently active.
        current_epoch: The epoch currently being processed (1-indexed).
        total_epochs: Total number of epochs planned for this session.
        stop_pending: Whether a graceful stop has been requested.
        status_message: Human-readable status message for UI display.
        estimated_epoch_remaining: Estimated seconds remaining for the current epoch.
    """

    is_active: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    stop_pending: bool = False
    status_message: str = ""
    estimated_epoch_remaining: float = 0.0


# ==============================================================================
# TrainingController
# ==============================================================================


class TrainingController:
    """Manages graceful stop and coordinates training lifecycle.

    The TrainingController provides a mechanism to stop training gracefully
    after the current epoch completes. It tracks training state and provides
    status information for UI display.

    The key design principle: when a stop is requested, the current epoch
    is allowed to finish completely (forward pass, backward pass, parameter
    update for all remaining batches), a checkpoint is saved, and then
    the training loop exits cleanly. No new epoch begins after stop is requested.

    Usage:
        controller = TrainingController()
        controller.start_training(total_epochs=100)

        for epoch in range(1, 101):
            if not controller.should_continue_training():
                break
            # ... train epoch ...
            controller._on_epoch_complete(epoch, metrics)

    Requirements:
        - 14.2: Complete current epoch before stopping
        - 14.3: Save checkpoint before terminating
        - 14.5: Do not begin next epoch after stop requested
        - 14.6: Resume seamlessly from graceful-stop checkpoint
        - 14.7: Display notification when no training is active
    """

    def __init__(
        self,
        checkpoint_save_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ):
        """Initialize TrainingController.

        Args:
            checkpoint_save_fn: Optional callback to save checkpoint.
                Called with (epoch, metrics) when epoch completes.
                If not provided, checkpoint saving is a no-op.
        """
        self._stop_requested: bool = False
        self._is_training: bool = False
        self._current_epoch: int = 0
        self._total_epochs: int = 0
        self._epoch_start_time: float = 0.0
        self._last_epoch_duration: float = 0.0
        self._checkpoint_save_fn = checkpoint_save_fn

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def is_training(self) -> bool:
        """Whether a training session is currently active."""
        return self._is_training

    @property
    def stop_requested(self) -> bool:
        """Whether a graceful stop has been requested."""
        return self._stop_requested

    @property
    def current_epoch(self) -> int:
        """The epoch currently being processed (1-indexed)."""
        return self._current_epoch

    @property
    def total_epochs(self) -> int:
        """Total number of epochs planned for this training session."""
        return self._total_epochs

    # ==========================================================================
    # Public Methods
    # ==========================================================================

    def request_stop(self) -> None:
        """Request graceful stop of training.

        Training completes the current epoch, saves a checkpoint, then stops.
        No new epoch begins after this is called.

        Raises:
            EngineError: If no training session is currently active.
                Error code: NO_ACTIVE_TRAINING

        Requirements:
            - 14.2: Complete current epoch before stopping
            - 14.5: Do not begin next epoch after stop
            - 14.7: Raise error if no training active
        """
        if not self._is_training:
            raise EngineError(
                "No training session is currently running",
                error_code="NO_ACTIVE_TRAINING",
            )
        self._stop_requested = True

    def should_continue_training(self) -> bool:
        """Check if training should continue to the next epoch.

        Called by the training loop BEFORE beginning a new epoch.
        Returns False when a stop has been requested, preventing
        the next epoch from starting.

        Returns:
            True if training should continue, False if it should stop.

        Requirements:
            - 14.5: Returns False when stop requested (no new epoch begins)
        """
        return not self._stop_requested

    def start_training(self, total_epochs: int) -> None:
        """Begin a training session with graceful stop support.

        Resets the stop flag and training state, preparing the controller
        for a new training session.

        Args:
            total_epochs: Total number of epochs planned for this session.
        """
        self._stop_requested = False
        self._is_training = True
        self._current_epoch = 0
        self._total_epochs = total_epochs
        self._epoch_start_time = 0.0
        self._last_epoch_duration = 0.0

    def end_training(self) -> None:
        """Mark training session as complete.

        Called when training finishes naturally or after graceful stop
        checkpoint has been saved.
        """
        self._is_training = False
        self._stop_requested = False

    def begin_epoch(self, epoch: int) -> None:
        """Mark the start of a new epoch.

        Called at the beginning of each epoch to track timing.

        Args:
            epoch: The epoch number (1-indexed).
        """
        self._current_epoch = epoch
        self._epoch_start_time = time.time()

    def get_status(self) -> GracefulTrainingStatus:
        """Get current training status for UI display.

        Returns a GracefulTrainingStatus with all relevant information
        including whether a stop is pending with appropriate message.

        Returns:
            GracefulTrainingStatus with current state information.

        Requirements:
            - 14.4: Show "Stopping after current epoch..." when stop pending
            - 14.7: Show appropriate message when no training is running
        """
        if not self._is_training:
            return GracefulTrainingStatus(
                is_active=False,
                current_epoch=0,
                total_epochs=0,
                stop_pending=False,
                status_message="No training is currently running",
                estimated_epoch_remaining=0.0,
            )

        # Estimate remaining time for current epoch
        estimated_remaining = self._estimate_epoch_remaining()

        if self._stop_requested:
            status_message = (
                f"Stopping after current epoch {self._current_epoch}..."
                f" (ETA: {estimated_remaining:.1f}s)"
            )
        else:
            status_message = (
                f"Training epoch {self._current_epoch}/{self._total_epochs}"
            )

        return GracefulTrainingStatus(
            is_active=True,
            current_epoch=self._current_epoch,
            total_epochs=self._total_epochs,
            stop_pending=self._stop_requested,
            status_message=status_message,
            estimated_epoch_remaining=estimated_remaining,
        )

    def _on_epoch_complete(self, epoch: int, metrics: Dict[str, Any]) -> bool:
        """Hook called after each epoch completes.

        Saves a checkpoint via the checkpoint_save_fn callback and checks
        if stop was requested. If stop was requested, training should
        terminate after this call.

        Args:
            epoch: The completed epoch number (1-indexed).
            metrics: Training metrics for the completed epoch (loss, val_loss, etc.).

        Returns:
            True if training should continue to the next epoch.
            False if stop was requested (training should terminate).

        Requirements:
            - 14.3: Save checkpoint before terminating
            - 14.5: Do not begin next epoch after stop requested
        """
        # Track epoch timing for ETA estimation
        if self._epoch_start_time > 0:
            self._last_epoch_duration = time.time() - self._epoch_start_time

        self._current_epoch = epoch

        # Save checkpoint (required by Req 14.3)
        if self._checkpoint_save_fn is not None:
            self._checkpoint_save_fn(epoch, metrics)

        # Check if stop was requested - if so, terminate loop
        if self._stop_requested:
            self._is_training = False
            return False

        return True

    # ==========================================================================
    # Private Helpers
    # ==========================================================================

    def _estimate_epoch_remaining(self) -> float:
        """Estimate seconds remaining for the current epoch.

        Uses the duration of the last completed epoch as an estimate.
        If no epoch has completed yet, returns 0.0 (unknown).

        Returns:
            Estimated seconds remaining, or 0.0 if unknown.
        """
        if self._last_epoch_duration <= 0:
            # No previous epoch to estimate from
            if self._epoch_start_time > 0:
                # We're in the first epoch, can't estimate completion
                return 0.0
            return 0.0

        # Estimate based on elapsed time within current epoch
        if self._epoch_start_time > 0:
            elapsed = time.time() - self._epoch_start_time
            remaining = max(0.0, self._last_epoch_duration - elapsed)
            return remaining

        return self._last_epoch_duration
