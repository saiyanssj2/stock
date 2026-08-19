"""
Unit tests for TrainingController integration with TrainingPipeline.

Tests:
- train_full() respects graceful stop (no new epoch after stop)
- train_full() saves checkpoint before exiting on graceful stop
- train_incremental() respects graceful stop
- Resume from graceful-stop checkpoint produces seamless continuation (N+1)
- TrainingPipeline works normally without a controller (backward compat)

Requirements: 14.2, 14.3, 14.5, 14.6
"""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from engine.config import ModelConfig, TrainingConfig
from engine.training_controller import TrainingController
from engine.training_pipeline import TrainingPipeline


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create a temporary directory for model checkpoints."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return str(model_dir)


@pytest.fixture
def training_config(tmp_model_dir):
    """Create a TrainingConfig pointing to temp directory with fast settings."""
    return TrainingConfig(
        learning_rate=1e-3,
        batch_size=16,
        max_epochs_full=10,
        max_epochs_incremental=5,
        min_sessions_per_symbol=50,  # Lower for testing
        checkpoint_dir=tmp_model_dir,
        log_file=str(Path(tmp_model_dir) / "training_log.jsonl"),
        early_stopping_patience=100,  # High to avoid early stopping interfering
    )


@pytest.fixture
def model_config():
    """Create a small ModelConfig for fast testing."""
    return ModelConfig(
        num_features=61,
        lookback=20,  # Small lookback for faster tests
        tcn_channels=[32, 32, 16],
        kernel_size=3,
        dilations=[1, 2, 4],
        attention_heads=2,
        attention_dim=16,
        dropout=0.0,
    )


@pytest.fixture
def sample_symbol_data():
    """Create sample symbol data for training (enough rows for min_sessions=50)."""
    np.random.seed(42)
    n = 100  # More than min_sessions_per_symbol=50
    base_price = 50.0
    returns = np.random.normal(0, 0.02, n)
    returns[0] = 0.0
    closes = base_price * np.cumprod(1.0 + returns)
    dates = pd.bdate_range(end="2024-06-30", periods=n)

    df = pd.DataFrame({
        "time": dates,
        "open": closes * 0.99,
        "high": closes * 1.01,
        "low": closes * 0.98,
        "close": closes,
        "volume": np.random.uniform(100000, 5000000, n),
    })

    return {"TEST_STOCK": df, "VNINDEX": df.copy()}


# ==============================================================================
# Tests: TrainingPipeline with TrainingController (train_full)
# ==============================================================================


class TestTrainFullGracefulStop:
    """Tests for train_full() integration with TrainingController."""

    def test_pipeline_without_controller_works(
        self, training_config, model_config, sample_symbol_data
    ):
        """TrainingPipeline works normally without a controller (backward compat)."""
        pipeline = TrainingPipeline(
            config=training_config,
            model_config=model_config,
            training_controller=None,
        )

        result = pipeline.train_full(sample_symbol_data, resume=False)

        # Should complete normally (either all epochs or early stopping)
        assert result.epochs_completed > 0
        assert result.final_train_loss > 0

    def test_graceful_stop_completes_current_epoch(
        self, training_config, model_config, sample_symbol_data
    ):
        """Graceful stop allows current epoch to finish before stopping (Req 14.2)."""
        controller = TrainingController()
        training_config.max_epochs_full = 10

        pipeline = TrainingPipeline(
            config=training_config,
            model_config=model_config,
            training_controller=controller,
        )

        # Request stop after a short delay (during epoch 1 or 2)
        def request_stop_after_delay():
            # Wait for training to start, then request stop
            while not controller.is_training:
                time.sleep(0.01)
            # Let at least one epoch complete before requesting stop
            while controller.current_epoch < 1:
                time.sleep(0.01)
            controller.request_stop()

        stop_thread = threading.Thread(target=request_stop_after_delay, daemon=True)
        stop_thread.start()

        result = pipeline.train_full(sample_symbol_data, resume=False)
        stop_thread.join(timeout=5)

        # Should have completed at least 1 epoch (stop doesn't interrupt mid-epoch)
        assert result.epochs_completed >= 1
        # Should NOT have completed all 10 epochs
        assert result.epochs_completed < 10

    def test_no_new_epoch_after_stop_requested(
        self, training_config, model_config, sample_symbol_data
    ):
        """No new epoch begins after stop is requested (Req 14.5)."""
        controller = TrainingController()
        training_config.max_epochs_full = 10

        pipeline = TrainingPipeline(
            config=training_config,
            model_config=model_config,
            training_controller=controller,
        )

        # We'll request stop immediately after start_training is called by pipeline
        # by patching should_continue_training to return False after 2 calls
        call_count = [0]
        original_should_continue = controller.should_continue_training

        def mock_should_continue():
            call_count[0] += 1
            if call_count[0] <= 2:
                return True
            return False

        controller.should_continue_training = mock_should_continue

        result = pipeline.train_full(sample_symbol_data, resume=False)

        # Should have completed exactly 2 epochs (called should_continue 3 times,
        # third call returns False preventing epoch 3)
        assert result.epochs_completed == 2

    def test_checkpoint_saved_before_stop_exit(
        self, training_config, model_config, sample_symbol_data
    ):
        """Checkpoint is saved before training loop exits on graceful stop (Req 14.3)."""
        controller = TrainingController()
        training_config.max_epochs_full = 10

        pipeline = TrainingPipeline(
            config=training_config,
            model_config=model_config,
            training_controller=controller,
        )

        # Allow 1 epoch, then stop
        call_count = [0]

        def mock_should_continue():
            call_count[0] += 1
            return call_count[0] <= 1

        controller.should_continue_training = mock_should_continue

        result = pipeline.train_full(sample_symbol_data, resume=False)

        # Verify checkpoint was saved
        checkpoint_dir = Path(training_config.checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        assert len(checkpoints) >= 1

        # Load the checkpoint and verify it's valid
        cp = torch.load(str(checkpoints[0]), map_location="cpu", weights_only=False)
        assert "model_state_dict" in cp
        assert "optimizer_state_dict" in cp
        assert "epoch" in cp

    def test_controller_end_training_called(
        self, training_config, model_config, sample_symbol_data
    ):
        """Controller's end_training() is called when training finishes."""
        controller = TrainingController()
        training_config.max_epochs_full = 2

        pipeline = TrainingPipeline(
            config=training_config,
            model_config=model_config,
            training_controller=controller,
        )

        pipeline.train_full(sample_symbol_data, resume=False)

        # After train_full completes, controller should be inactive
        assert controller.is_training is False


# ==============================================================================
# Tests: TrainingPipeline with TrainingController (train_incremental)
# ==============================================================================


class TestTrainIncrementalGracefulStop:
    """Tests for train_incremental() integration with TrainingController."""

    def _run_initial_training(self, training_config, model_config, sample_symbol_data):
        """Helper: run a quick full training to create initial checkpoint."""
        initial_config = TrainingConfig(
            learning_rate=training_config.learning_rate,
            batch_size=training_config.batch_size,
            max_epochs_full=2,  # Just 2 epochs for the initial model
            min_sessions_per_symbol=training_config.min_sessions_per_symbol,
            checkpoint_dir=training_config.checkpoint_dir,
            log_file=training_config.log_file,
            early_stopping_patience=100,
        )
        pipeline = TrainingPipeline(
            config=initial_config,
            model_config=model_config,
        )
        return pipeline.train_full(sample_symbol_data, resume=False)

    def test_incremental_graceful_stop(
        self, training_config, model_config, sample_symbol_data
    ):
        """train_incremental() respects graceful stop (Req 14.2, 14.5)."""
        # First, create initial checkpoint
        self._run_initial_training(training_config, model_config, sample_symbol_data)

        # Now test incremental with graceful stop
        controller = TrainingController()
        training_config.max_epochs_incremental = 5

        pipeline = TrainingPipeline(
            config=training_config,
            model_config=model_config,
            training_controller=controller,
        )

        # Allow 2 epochs, then stop
        call_count = [0]

        def mock_should_continue():
            call_count[0] += 1
            return call_count[0] <= 2

        controller.should_continue_training = mock_should_continue

        result = pipeline.train_incremental(sample_symbol_data)

        # Should have completed exactly 2 epochs
        assert result.epochs_completed == 2
        assert result.epochs_completed < 5  # Not all epochs

    def test_incremental_without_controller(
        self, training_config, model_config, sample_symbol_data
    ):
        """train_incremental() works normally without controller."""
        self._run_initial_training(training_config, model_config, sample_symbol_data)

        training_config.max_epochs_incremental = 3
        pipeline = TrainingPipeline(
            config=training_config,
            model_config=model_config,
            training_controller=None,
        )

        result = pipeline.train_incremental(sample_symbol_data)
        assert result.epochs_completed == 3

    def test_incremental_checkpoint_saved_on_stop(
        self, training_config, model_config, sample_symbol_data
    ):
        """Checkpoint saved before exit on graceful stop in incremental (Req 14.3)."""
        self._run_initial_training(training_config, model_config, sample_symbol_data)

        controller = TrainingController()
        training_config.max_epochs_incremental = 5

        pipeline = TrainingPipeline(
            config=training_config,
            model_config=model_config,
            training_controller=controller,
        )

        # Allow 1 epoch, then stop
        call_count = [0]

        def mock_should_continue():
            call_count[0] += 1
            return call_count[0] <= 1

        controller.should_continue_training = mock_should_continue

        result = pipeline.train_incremental(sample_symbol_data)

        # Verify incremental checkpoint was saved
        checkpoint_dir = Path(training_config.checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_incremental_epoch_*.pt"))
        assert len(checkpoints) >= 1


# ==============================================================================
# Tests: Resume from Graceful Stop Checkpoint
# ==============================================================================


class TestResumeFromGracefulStop:
    """Tests for seamless resume after graceful stop (Req 14.6)."""

    def test_resume_continues_from_next_epoch(
        self, training_config, model_config, sample_symbol_data
    ):
        """Resume from graceful-stop checkpoint continues from epoch N+1 (Req 14.6)."""
        # First: train 3 epochs then graceful stop
        controller = TrainingController()
        training_config.max_epochs_full = 10

        pipeline = TrainingPipeline(
            config=training_config,
            model_config=model_config,
            training_controller=controller,
        )

        # Allow 3 epochs
        call_count = [0]

        def mock_should_continue():
            call_count[0] += 1
            return call_count[0] <= 3

        controller.should_continue_training = mock_should_continue

        result1 = pipeline.train_full(sample_symbol_data, resume=False)
        assert result1.epochs_completed == 3

        # Verify checkpoint for epoch 2 (0-indexed) exists
        checkpoint_dir = Path(training_config.checkpoint_dir)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        assert len(checkpoints) >= 1
        last_cp = torch.load(str(checkpoints[-1]), map_location="cpu", weights_only=False)
        stopped_epoch = last_cp["epoch"]

        # Second: resume training without controller (should continue from epoch N+1)
        pipeline2 = TrainingPipeline(
            config=training_config,
            model_config=model_config,
            training_controller=None,
        )
        training_config.max_epochs_full = stopped_epoch + 4  # Run a few more

        result2 = pipeline2.train_full(sample_symbol_data, resume=True)

        # Should have resumed and completed more epochs
        assert result2.epochs_completed > result1.epochs_completed

    def test_resume_produces_valid_model_output(
        self, training_config, model_config, sample_symbol_data
    ):
        """Model loaded from graceful-stop checkpoint produces valid output."""
        from engine.evaluation_model import StockEvalNet

        controller = TrainingController()
        training_config.max_epochs_full = 3

        pipeline = TrainingPipeline(
            config=training_config,
            model_config=model_config,
            training_controller=controller,
        )

        # Train 2 epochs then stop
        call_count = [0]

        def mock_should_continue():
            call_count[0] += 1
            return call_count[0] <= 2

        controller.should_continue_training = mock_should_continue

        result = pipeline.train_full(sample_symbol_data, resume=False)

        # Load model from checkpoint and verify it produces valid output
        checkpoint_dir = Path(training_config.checkpoint_dir)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        assert len(checkpoints) >= 1

        cp = torch.load(str(checkpoints[-1]), map_location="cpu", weights_only=False)
        model = StockEvalNet(model_config)
        model.load_state_dict(cp["model_state_dict"])
        model.eval()

        # Create dummy input and verify output is bounded [-1, 1]
        dummy_input = torch.randn(1, model_config.lookback, model_config.num_features)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (1, 1)
        assert -1.0 <= output.item() <= 1.0
