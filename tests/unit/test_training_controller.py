"""
Unit tests for TrainingController with graceful stop logic.

Tests:
- TrainingController initialization
- request_stop() raises EngineError when no active training
- request_stop() sets stop flag when training is active
- should_continue_training() returns True/False based on stop state
- start_training() resets state properly
- _on_epoch_complete() saves checkpoint and respects stop flag
- get_status() returns correct GracefulTrainingStatus
- Full training lifecycle with graceful stop

Requirements: 14.2, 14.3, 14.5, 14.6, 14.7
"""

import time
from unittest.mock import MagicMock

import pytest

from engine.config import EngineError
from engine.training_controller import GracefulTrainingStatus, TrainingController


# ==============================================================================
# Initialization Tests
# ==============================================================================


class TestTrainingControllerInit:
    """Tests for TrainingController initialization."""

    def test_default_state_not_training(self):
        """Controller starts in non-training state."""
        controller = TrainingController()
        assert controller.is_training is False

    def test_default_stop_not_requested(self):
        """Stop is not requested by default."""
        controller = TrainingController()
        assert controller.stop_requested is False

    def test_default_epoch_zero(self):
        """Current epoch starts at 0."""
        controller = TrainingController()
        assert controller.current_epoch == 0

    def test_default_total_epochs_zero(self):
        """Total epochs starts at 0."""
        controller = TrainingController()
        assert controller.total_epochs == 0

    def test_accepts_checkpoint_save_fn(self):
        """Controller accepts a checkpoint save function."""
        save_fn = MagicMock()
        controller = TrainingController(checkpoint_save_fn=save_fn)
        assert controller._checkpoint_save_fn is save_fn


# ==============================================================================
# request_stop() Tests
# ==============================================================================


class TestRequestStop:
    """Tests for request_stop() method."""

    def test_raises_error_when_not_training(self):
        """Raises EngineError with NO_ACTIVE_TRAINING when no training active."""
        controller = TrainingController()
        with pytest.raises(EngineError) as exc_info:
            controller.request_stop()
        assert exc_info.value.error_code == "NO_ACTIVE_TRAINING"
        assert "No training session is currently running" in str(exc_info.value)

    def test_sets_stop_flag_when_training(self):
        """Sets _stop_requested to True when training is active."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)
        controller.request_stop()
        assert controller.stop_requested is True

    def test_multiple_stops_are_idempotent(self):
        """Calling request_stop multiple times doesn't error if training active."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)
        controller.request_stop()
        # Second call should not raise
        controller.request_stop()
        assert controller.stop_requested is True


# ==============================================================================
# should_continue_training() Tests
# ==============================================================================


class TestShouldContinueTraining:
    """Tests for should_continue_training() method."""

    def test_returns_true_when_no_stop_requested(self):
        """Returns True when stop has not been requested."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)
        assert controller.should_continue_training() is True

    def test_returns_false_when_stop_requested(self):
        """Returns False when stop has been requested (prevents next epoch)."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)
        controller.request_stop()
        assert controller.should_continue_training() is False

    def test_returns_true_before_training_starts(self):
        """Returns True when controller is freshly initialized (no stop)."""
        controller = TrainingController()
        assert controller.should_continue_training() is True


# ==============================================================================
# start_training() Tests
# ==============================================================================


class TestStartTraining:
    """Tests for start_training() method."""

    def test_sets_is_training_true(self):
        """start_training sets is_training to True."""
        controller = TrainingController()
        controller.start_training(total_epochs=50)
        assert controller.is_training is True

    def test_resets_stop_flag(self):
        """start_training resets any previous stop request."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)
        controller.request_stop()
        # Start a new training session
        controller.start_training(total_epochs=20)
        assert controller.stop_requested is False

    def test_sets_total_epochs(self):
        """start_training sets total_epochs correctly."""
        controller = TrainingController()
        controller.start_training(total_epochs=75)
        assert controller.total_epochs == 75

    def test_resets_current_epoch(self):
        """start_training resets current_epoch to 0."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)
        controller.begin_epoch(5)
        # Start new training - should reset
        controller.start_training(total_epochs=20)
        assert controller.current_epoch == 0


# ==============================================================================
# end_training() Tests
# ==============================================================================


class TestEndTraining:
    """Tests for end_training() method."""

    def test_sets_is_training_false(self):
        """end_training marks training as inactive."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)
        controller.end_training()
        assert controller.is_training is False

    def test_resets_stop_flag(self):
        """end_training resets stop flag."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)
        controller.request_stop()
        controller.end_training()
        assert controller.stop_requested is False


# ==============================================================================
# begin_epoch() Tests
# ==============================================================================


class TestBeginEpoch:
    """Tests for begin_epoch() method."""

    def test_updates_current_epoch(self):
        """begin_epoch sets the current epoch number."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)
        controller.begin_epoch(3)
        assert controller.current_epoch == 3

    def test_tracks_epoch_start_time(self):
        """begin_epoch records the start time for ETA estimation."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)
        controller.begin_epoch(1)
        assert controller._epoch_start_time > 0


# ==============================================================================
# _on_epoch_complete() Tests
# ==============================================================================


class TestOnEpochComplete:
    """Tests for _on_epoch_complete() method."""

    def test_calls_checkpoint_save_fn(self):
        """_on_epoch_complete calls the checkpoint save function."""
        save_fn = MagicMock()
        controller = TrainingController(checkpoint_save_fn=save_fn)
        controller.start_training(total_epochs=10)
        metrics = {"loss": 0.5, "val_loss": 0.6}
        controller._on_epoch_complete(1, metrics)
        save_fn.assert_called_once_with(1, metrics)

    def test_returns_true_when_no_stop(self):
        """Returns True to continue training when no stop requested."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)
        result = controller._on_epoch_complete(1, {"loss": 0.5})
        assert result is True

    def test_returns_false_when_stop_requested(self):
        """Returns False to terminate training when stop was requested."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)
        controller.request_stop()
        result = controller._on_epoch_complete(1, {"loss": 0.5})
        assert result is False

    def test_sets_is_training_false_on_stop(self):
        """Sets is_training to False when stop requested and epoch completes."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)
        controller.request_stop()
        controller._on_epoch_complete(1, {"loss": 0.5})
        assert controller.is_training is False

    def test_saves_checkpoint_even_on_stop(self):
        """Checkpoint is saved even when stopping (Req 14.3)."""
        save_fn = MagicMock()
        controller = TrainingController(checkpoint_save_fn=save_fn)
        controller.start_training(total_epochs=10)
        controller.request_stop()
        controller._on_epoch_complete(5, {"loss": 0.3})
        save_fn.assert_called_once_with(5, {"loss": 0.3})

    def test_no_save_fn_does_not_error(self):
        """If no checkpoint_save_fn provided, no error on epoch complete."""
        controller = TrainingController(checkpoint_save_fn=None)
        controller.start_training(total_epochs=10)
        # Should not raise
        result = controller._on_epoch_complete(1, {"loss": 0.5})
        assert result is True

    def test_updates_current_epoch(self):
        """_on_epoch_complete updates current epoch tracking."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)
        controller._on_epoch_complete(3, {"loss": 0.4})
        assert controller.current_epoch == 3


# ==============================================================================
# get_status() Tests
# ==============================================================================


class TestGetStatus:
    """Tests for get_status() method."""

    def test_returns_inactive_when_not_training(self):
        """Returns inactive status when no training is running."""
        controller = TrainingController()
        status = controller.get_status()
        assert status.is_active is False
        assert status.current_epoch == 0
        assert status.total_epochs == 0
        assert status.stop_pending is False
        assert "No training is currently running" in status.status_message

    def test_returns_active_status_during_training(self):
        """Returns active status with epoch info during training."""
        controller = TrainingController()
        controller.start_training(total_epochs=50)
        controller.begin_epoch(7)
        status = controller.get_status()
        assert status.is_active is True
        assert status.current_epoch == 7
        assert status.total_epochs == 50
        assert status.stop_pending is False
        assert "Training epoch 7/50" in status.status_message

    def test_returns_stop_pending_status(self):
        """Returns stop pending status with epoch and ETA info."""
        controller = TrainingController()
        controller.start_training(total_epochs=50)
        controller.begin_epoch(7)
        controller.request_stop()
        status = controller.get_status()
        assert status.is_active is True
        assert status.stop_pending is True
        assert "Stopping after current epoch 7" in status.status_message

    def test_returns_graceful_training_status_type(self):
        """get_status returns a GracefulTrainingStatus instance."""
        controller = TrainingController()
        status = controller.get_status()
        assert isinstance(status, GracefulTrainingStatus)

    def test_estimated_remaining_zero_for_first_epoch(self):
        """Estimated remaining is 0 for the first epoch (no history)."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)
        controller.begin_epoch(1)
        status = controller.get_status()
        assert status.estimated_epoch_remaining == 0.0


# ==============================================================================
# Full Lifecycle Tests
# ==============================================================================


class TestTrainingLifecycle:
    """Integration-style tests for the full training lifecycle."""

    def test_normal_training_completion(self):
        """Simulates normal training from start to finish."""
        checkpoints_saved = []

        def save_fn(epoch, metrics):
            checkpoints_saved.append((epoch, metrics))

        controller = TrainingController(checkpoint_save_fn=save_fn)
        controller.start_training(total_epochs=3)

        for epoch in range(1, 4):
            if not controller.should_continue_training():
                break
            controller.begin_epoch(epoch)
            # Simulate epoch work
            controller._on_epoch_complete(epoch, {"loss": 1.0 / epoch})

        controller.end_training()

        assert len(checkpoints_saved) == 3
        assert controller.is_training is False

    def test_graceful_stop_mid_training(self):
        """Simulates graceful stop requested during training."""
        checkpoints_saved = []

        def save_fn(epoch, metrics):
            checkpoints_saved.append((epoch, metrics))

        controller = TrainingController(checkpoint_save_fn=save_fn)
        controller.start_training(total_epochs=10)

        epochs_completed = 0
        for epoch in range(1, 11):
            if not controller.should_continue_training():
                break
            controller.begin_epoch(epoch)
            # Request stop after epoch 3 starts (will complete epoch 3)
            if epoch == 3:
                controller.request_stop()
            # Complete the current epoch
            should_continue = controller._on_epoch_complete(
                epoch, {"loss": 1.0 / epoch}
            )
            epochs_completed = epoch
            if not should_continue:
                break

        # Exactly 3 epochs completed (stop after epoch 3)
        assert epochs_completed == 3
        # Checkpoint saved for all 3 epochs including the stop epoch
        assert len(checkpoints_saved) == 3
        # Training is now inactive
        assert controller.is_training is False

    def test_stop_prevents_next_epoch(self):
        """Verifies that stop request prevents the next epoch from starting."""
        controller = TrainingController()
        controller.start_training(total_epochs=10)

        # Complete epoch 1
        controller.begin_epoch(1)
        controller._on_epoch_complete(1, {"loss": 0.5})

        # Request stop
        controller.request_stop()

        # should_continue_training should prevent epoch 2
        assert controller.should_continue_training() is False

    def test_resume_after_graceful_stop(self):
        """Simulates resuming training after a graceful stop (Req 14.6)."""
        checkpoints_saved = []

        def save_fn(epoch, metrics):
            checkpoints_saved.append((epoch, metrics))

        controller = TrainingController(checkpoint_save_fn=save_fn)

        # First training session - stop at epoch 3
        controller.start_training(total_epochs=10)
        for epoch in range(1, 11):
            if not controller.should_continue_training():
                break
            controller.begin_epoch(epoch)
            if epoch == 3:
                controller.request_stop()
            if not controller._on_epoch_complete(epoch, {"loss": 1.0 / epoch}):
                break

        assert controller.is_training is False
        last_checkpoint = checkpoints_saved[-1]
        assert last_checkpoint[0] == 3  # Stopped at epoch 3

        # Resume from epoch 4 (seamless continuation)
        controller.start_training(total_epochs=10)
        for epoch in range(4, 11):
            if not controller.should_continue_training():
                break
            controller.begin_epoch(epoch)
            controller._on_epoch_complete(epoch, {"loss": 1.0 / epoch})

        controller.end_training()

        # Epochs 4-10 were saved (7 more)
        assert len(checkpoints_saved) == 10  # 3 + 7
        assert checkpoints_saved[-1][0] == 10
