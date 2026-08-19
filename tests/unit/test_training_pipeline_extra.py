"""
Unit tests for engine.training_pipeline module (part 2).

Tests full training, incremental training, and feature matrix construction.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.config import DataError, ModelConfig, ModelError, TrainingConfig
from engine.training_pipeline import (
    TrainingPipeline,
    TrainingResult,
)


# ==============================================================================
# Tests for train_full()
# ==============================================================================


class TestTrainFull:
    """Test full training from scratch."""

    @pytest.fixture
    def training_data(self):
        """Create synthetic training data with enough sessions."""
        np.random.seed(42)
        n = 300
        base_price = 50.0
        returns = np.random.normal(0, 0.02, n)
        returns[0] = 0.0
        closes = base_price * np.cumprod(1.0 + returns)

        df = pd.DataFrame({
            "open": closes * 0.99,
            "high": closes * 1.01,
            "low": closes * 0.98,
            "close": closes,
            "volume": np.random.uniform(100000, 5000000, n),
        })
        return df

    @pytest.fixture
    def pipeline_with_temp(self, tmp_path):
        """Create a pipeline with temp directories for checkpoints/logs."""
        config = TrainingConfig(
            max_epochs_full=3,
            early_stopping_patience=5,
            batch_size=32,
            checkpoint_dir=str(tmp_path / "models"),
            log_file=str(tmp_path / "models" / "training_log.jsonl"),
        )
        model_config = ModelConfig(num_features=61, lookback=60)
        return TrainingPipeline(config=config, model_config=model_config)

    def test_train_full_produces_model(self, pipeline_with_temp, training_data):
        """Full training should produce a saved model file."""
        symbol_data = {
            "VNM": training_data,
            "VNINDEX": training_data.copy(),
        }

        result = pipeline_with_temp.train_full(symbol_data, resume=False)

        assert result.epochs_completed > 0
        assert result.model_path is not None
        assert Path(result.model_path).exists()
        assert result.total_time_seconds > 0

    def test_train_full_produces_valid_result(self, pipeline_with_temp, training_data):
        """TrainingResult should contain valid metrics."""
        symbol_data = {
            "VNM": training_data,
            "VNINDEX": training_data.copy(),
        }

        result = pipeline_with_temp.train_full(symbol_data, resume=False)

        assert result.final_train_loss > 0
        assert result.final_val_loss > 0
        assert result.final_mae >= 0
        assert result.best_val_loss > 0
        assert result.best_epoch >= 0
        assert "VNM" in result.symbols_trained
        assert "VNINDEX" in result.symbols_trained

    def test_train_full_creates_checkpoints(self, pipeline_with_temp, training_data, tmp_path):
        """Should create checkpoint files after each epoch."""
        symbol_data = {"VNM": training_data, "VNINDEX": training_data.copy()}

        pipeline_with_temp.train_full(symbol_data, resume=False)

        checkpoint_dir = tmp_path / "models"
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        assert len(checkpoints) > 0

    def test_train_full_creates_log(self, pipeline_with_temp, training_data, tmp_path):
        """Should create a training log file."""
        symbol_data = {"VNM": training_data, "VNINDEX": training_data.copy()}

        pipeline_with_temp.train_full(symbol_data, resume=False)

        log_file = tmp_path / "models" / "training_log.jsonl"
        assert log_file.exists()

        with open(log_file, "r") as f:
            lines = f.readlines()
        assert len(lines) > 0

    def test_train_full_saves_norm_params(self, pipeline_with_temp, training_data, tmp_path):
        """Should save normalization parameters."""
        symbol_data = {"VNM": training_data, "VNINDEX": training_data.copy()}

        pipeline_with_temp.train_full(symbol_data, resume=False)

        norm_path = tmp_path / "models" / "norm_params.json"
        assert norm_path.exists()

    def test_train_full_no_valid_data_raises_error(self, pipeline_with_temp):
        """Should raise DataError when no symbols have enough data."""
        small_df = pd.DataFrame({"close": np.ones(50)})
        symbol_data = {"VNM": small_df}

        with pytest.raises(DataError):
            pipeline_with_temp.train_full(symbol_data, resume=False)


# ==============================================================================
# Tests for train_incremental()
# ==============================================================================


class TestTrainIncremental:
    """Test incremental fine-tuning with new data."""

    @pytest.fixture
    def training_data(self):
        """Create synthetic training data."""
        np.random.seed(42)
        n = 300
        base_price = 50.0
        returns = np.random.normal(0, 0.02, n)
        returns[0] = 0.0
        closes = base_price * np.cumprod(1.0 + returns)

        df = pd.DataFrame({
            "open": closes * 0.99,
            "high": closes * 1.01,
            "low": closes * 0.98,
            "close": closes,
            "volume": np.random.uniform(100000, 5000000, n),
        })
        return df

    @pytest.fixture
    def pretrained_pipeline(self, tmp_path, training_data):
        """Create a pipeline that has already been fully trained."""
        config = TrainingConfig(
            max_epochs_full=2,
            max_epochs_incremental=3,
            early_stopping_patience=5,
            batch_size=32,
            checkpoint_dir=str(tmp_path / "models"),
            log_file=str(tmp_path / "models" / "training_log.jsonl"),
        )
        model_config = ModelConfig(num_features=61, lookback=60)
        pipeline = TrainingPipeline(config=config, model_config=model_config)

        # Run full training first
        symbol_data = {"VNM": training_data, "VNINDEX": training_data.copy()}
        pipeline.train_full(symbol_data, resume=False)

        return pipeline

    def test_incremental_training_completes(self, pretrained_pipeline, training_data):
        """Incremental training should complete successfully."""
        new_data = {"VNM": training_data.tail(100).copy()}

        result = pretrained_pipeline.train_incremental(new_data)

        assert result.epochs_completed > 0
        assert result.model_path is not None
        assert Path(result.model_path).exists()

    def test_incremental_respects_max_epochs(self, pretrained_pipeline, training_data):
        """Should not exceed max_epochs_incremental."""
        new_data = {"VNM": training_data.tail(100).copy()}

        result = pretrained_pipeline.train_incremental(new_data, max_epochs=3)

        assert result.epochs_completed <= 3

    def test_incremental_without_checkpoint_raises_error(self, tmp_path):
        """Should raise ModelError if no checkpoint exists."""
        config = TrainingConfig(
            max_epochs_incremental=3,
            checkpoint_dir=str(tmp_path / "empty_models"),
            log_file=str(tmp_path / "empty_models" / "log.jsonl"),
        )
        model_config = ModelConfig(num_features=61, lookback=60)
        pipeline = TrainingPipeline(config=config, model_config=model_config)

        df = pd.DataFrame({
            "open": np.ones(100),
            "high": np.ones(100),
            "low": np.ones(100),
            "close": np.ones(100),
            "volume": np.ones(100),
        })

        with pytest.raises((ModelError, DataError)):
            pipeline.train_incremental({"VNM": df})

    def test_incremental_insufficient_data_raises_error(self, pretrained_pipeline):
        """Should raise DataError if new data is too small."""
        tiny_df = pd.DataFrame({
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1000.0],
        })

        with pytest.raises(DataError):
            pretrained_pipeline.train_incremental({"VNM": tiny_df})


# ==============================================================================
# Tests for _build_feature_matrix()
# ==============================================================================


class TestBuildFeatureMatrix:
    """Test feature matrix construction for training."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline with standard config."""
        model_config = ModelConfig(num_features=61, lookback=60)
        return TrainingPipeline(model_config=model_config)

    @pytest.fixture
    def sample_df(self):
        """DataFrame with 100 rows of OHLCV data."""
        np.random.seed(42)
        n = 100
        closes = 50.0 * np.cumprod(1.0 + np.random.normal(0, 0.02, n))
        return pd.DataFrame({
            "open": closes * 0.99,
            "high": closes * 1.01,
            "low": closes * 0.98,
            "close": closes,
            "volume": np.random.uniform(100000, 5000000, n),
        })

    def test_produces_correct_shapes(self, pipeline, sample_df):
        """Feature matrix should have correct dimensions."""
        features, labels = pipeline._build_feature_matrix(sample_df, lookback=60)

        expected_windows = len(sample_df) - 60 + 1  # 41
        assert features.shape == (expected_windows, 60, 61)
        assert labels.shape == (expected_windows,)

    def test_no_nan_in_features(self, pipeline, sample_df):
        """Output features should not contain NaN or Inf."""
        features, _ = pipeline._build_feature_matrix(sample_df, lookback=60)

        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_insufficient_data_returns_empty(self, pipeline):
        """Should return empty arrays if data has fewer rows than lookback."""
        small_df = pd.DataFrame({
            "open": [1.0] * 10,
            "high": [1.0] * 10,
            "low": [1.0] * 10,
            "close": [1.0] * 10,
            "volume": [1000.0] * 10,
        })

        features, labels = pipeline._build_feature_matrix(small_df, lookback=60)

        assert len(features) == 0
        assert len(labels) == 0
