"""
Unit tests for engine.training_pipeline module (part 1).

Tests label generation, chronological splitting, data validation,
data preparation, CheckpointManager, and EpochLogger.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from engine.config import DataError, ModelConfig, ModelError
from engine.training_pipeline import (
    CheckpointManager,
    EpochLogger,
    TrainingPipeline,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def pipeline():
    """Create a TrainingPipeline with default config."""
    return TrainingPipeline()


@pytest.fixture
def sample_price_df():
    """Create a simple DataFrame with known close prices."""
    closes = [100.0, 105.0, 110.0, 108.0, 112.0, 115.0, 120.0, 118.0, 125.0, 130.0]
    return pd.DataFrame({"close": closes})


@pytest.fixture
def large_price_df():
    """Create a DataFrame with 300 sessions for validation tests."""
    np.random.seed(42)
    n = 300
    base_price = 50.0
    returns = np.random.normal(0, 0.02, n)
    returns[0] = 0.0
    closes = base_price * np.cumprod(1.0 + returns)
    dates = pd.bdate_range(end="2024-06-30", periods=n)
    return pd.DataFrame({
        "time": dates,
        "open": closes * 0.99,
        "high": closes * 1.01,
        "low": closes * 0.98,
        "close": closes,
        "volume": np.random.uniform(100000, 5000000, n),
    })


# ==============================================================================
# Tests for _generate_labels()
# ==============================================================================


class TestGenerateLabels:
    """Test label generation from future returns with tanh scaling."""

    def test_basic_label_generation(self, pipeline, sample_price_df):
        """Labels should be computed from future 5-session returns."""
        labels = pipeline._generate_labels(sample_price_df)

        # Should have same length as input
        assert len(labels) == len(sample_price_df)

        # Last 5 entries should be NaN (no future data)
        assert all(np.isnan(labels[-5:]))

        # First 5 entries should have valid labels
        assert all(~np.isnan(labels[:5]))

    def test_labels_bounded(self, pipeline, sample_price_df):
        """All valid labels should be in [-1.0, +1.0]."""
        labels = pipeline._generate_labels(sample_price_df)
        valid_labels = labels[~np.isnan(labels)]

        assert all(valid_labels >= -1.0)
        assert all(valid_labels <= 1.0)

    def test_positive_return_gives_positive_label(self, pipeline):
        """A positive future return should produce a positive label."""
        # Price goes from 100 to 150 in 5 sessions (50% return)
        df = pd.DataFrame({"close": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]})
        labels = pipeline._generate_labels(df, horizon=5)

        # First label: (150-100)/100 = 0.50, tanh(0.50 * 10) ≈ 1.0
        assert labels[0] > 0.0

    def test_negative_return_gives_negative_label(self, pipeline):
        """A negative future return should produce a negative label."""
        # Price goes from 100 to 50 in 5 sessions (-50% return)
        df = pd.DataFrame({"close": [100.0, 90.0, 80.0, 70.0, 60.0, 50.0]})
        labels = pipeline._generate_labels(df, horizon=5)

        # First label: (50-100)/100 = -0.50, tanh(-0.50 * 10) ≈ -1.0
        assert labels[0] < 0.0

    def test_zero_return_gives_zero_label(self, pipeline):
        """No price change should produce a label of 0.0."""
        df = pd.DataFrame({"close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]})
        labels = pipeline._generate_labels(df, horizon=5)

        assert labels[0] == pytest.approx(0.0, abs=1e-10)

    def test_custom_horizon(self, pipeline, sample_price_df):
        """Custom horizon should control look-ahead distance."""
        labels_h3 = pipeline._generate_labels(sample_price_df, horizon=3)
        labels_h5 = pipeline._generate_labels(sample_price_df, horizon=5)

        # horizon=3: last 3 are NaN
        assert all(np.isnan(labels_h3[-3:]))
        assert not np.isnan(labels_h3[-4])

        # horizon=5: last 5 are NaN
        assert all(np.isnan(labels_h5[-5:]))
        assert not np.isnan(labels_h5[-6])

    def test_custom_sensitivity(self, pipeline):
        """Higher sensitivity should produce more extreme labels."""
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]})

        labels_low = pipeline._generate_labels(df, horizon=5, sensitivity=1.0)
        labels_high = pipeline._generate_labels(df, horizon=5, sensitivity=100.0)

        # Same return, higher sensitivity → more extreme label
        assert abs(labels_high[0]) > abs(labels_low[0])

    def test_monotonically_increasing_with_return(self, pipeline):
        """Labels should increase monotonically as returns increase."""
        # Create scenarios with increasing returns
        returns = [-0.10, -0.05, 0.0, 0.05, 0.10]
        labels = []

        for r in returns:
            future_price = 100.0 * (1.0 + r)
            df = pd.DataFrame({"close": [100.0, 100.0, 100.0, 100.0, 100.0, future_price]})
            lab = pipeline._generate_labels(df, horizon=5)
            labels.append(lab[0])

        # Should be monotonically increasing
        for i in range(len(labels) - 1):
            assert labels[i] < labels[i + 1]

    def test_missing_close_column_raises_error(self, pipeline):
        """Should raise DataError if 'close' column is missing."""
        df = pd.DataFrame({"open": [100.0, 105.0]})
        with pytest.raises(DataError, match="missing required 'close' column"):
            pipeline._generate_labels(df)

    def test_empty_dataframe_raises_error(self, pipeline):
        """Should raise DataError for empty DataFrame."""
        df = pd.DataFrame({"close": []})
        with pytest.raises(DataError, match="Empty DataFrame"):
            pipeline._generate_labels(df)

    def test_tanh_scaling_with_default_sensitivity(self, pipeline):
        """Verify tanh scaling: 10% return → tanh(0.1 * 10) = tanh(1.0)."""
        df = pd.DataFrame({"close": [100.0, 100.0, 100.0, 100.0, 100.0, 110.0]})
        labels = pipeline._generate_labels(df, horizon=5)

        expected = np.tanh(0.1 * 10.0)  # tanh(1.0) ≈ 0.7616
        assert labels[0] == pytest.approx(expected, rel=1e-9)


# ==============================================================================
# Tests for _chronological_split()
# ==============================================================================


class TestChronologicalSplit:
    """Test chronological data splitting."""

    def test_basic_split_ratios(self, pipeline):
        """70/15/15 split should be approximately correct."""
        n = 100
        data = np.arange(n).reshape(-1, 1)
        labels = np.random.uniform(-1, 1, n)

        (train_d, train_l), (val_d, val_l), (test_d, test_l) = \
            pipeline._chronological_split(data, labels)

        total = len(train_d) + len(val_d) + len(test_d)
        assert total == n

        # Approximate ratios (±2 due to rounding)
        assert abs(len(train_d) - 70) <= 2
        assert abs(len(val_d) - 15) <= 2
        assert abs(len(test_d) - 15) <= 2

    def test_chronological_ordering(self, pipeline):
        """Train indices < val indices < test indices (no future leakage)."""
        n = 100
        data = np.arange(n).reshape(-1, 1)
        labels = np.random.uniform(-1, 1, n)

        (train_d, _), (val_d, _), (test_d, _) = \
            pipeline._chronological_split(data, labels)

        # All train values should be less than all val values
        assert train_d.max() < val_d.min()
        # All val values should be less than all test values
        assert val_d.max() < test_d.min()

    def test_nan_labels_excluded(self, pipeline):
        """Rows with NaN labels should not appear in any split."""
        n = 100
        data = np.arange(n).reshape(-1, 1)
        labels = np.random.uniform(-1, 1, n)
        # Set last 10 to NaN (as would happen with label horizon)
        labels[-10:] = np.nan

        (train_d, train_l), (val_d, val_l), (test_d, test_l) = \
            pipeline._chronological_split(data, labels)

        # No NaN in any split
        assert not np.any(np.isnan(train_l))
        assert not np.any(np.isnan(val_l))
        assert not np.any(np.isnan(test_l))

        # Total should be 90 (100 - 10 NaN)
        total = len(train_l) + len(val_l) + len(test_l)
        assert total == 90

    def test_mismatched_lengths_raises_error(self, pipeline):
        """Should raise DataError if data and labels have different lengths."""
        data = np.ones((100, 5))
        labels = np.ones(50)

        with pytest.raises(DataError, match="length mismatch"):
            pipeline._chronological_split(data, labels)

    def test_empty_data_raises_error(self, pipeline):
        """Should raise DataError for empty arrays."""
        data = np.array([]).reshape(0, 5)
        labels = np.array([])

        with pytest.raises(DataError, match="Empty data"):
            pipeline._chronological_split(data, labels)

    def test_all_nan_labels_raises_error(self, pipeline):
        """Should raise DataError if all labels are NaN."""
        data = np.ones((10, 5))
        labels = np.full(10, np.nan)

        with pytest.raises(DataError, match="No valid"):
            pipeline._chronological_split(data, labels)

    def test_custom_split_ratios(self, pipeline):
        """Custom ratios should be respected."""
        n = 100
        data = np.arange(n).reshape(-1, 1)
        labels = np.random.uniform(-1, 1, n)

        (train_d, _), (val_d, _), (test_d, _) = \
            pipeline._chronological_split(data, labels, train_ratio=0.80, val_ratio=0.10)

        assert abs(len(train_d) - 80) <= 2
        assert abs(len(val_d) - 10) <= 2
        assert abs(len(test_d) - 10) <= 2


# ==============================================================================
# Tests for validate_symbol_data()
# ==============================================================================


class TestValidateSymbolData:
    """Test data validation for training symbols."""

    def test_sufficient_data_passes(self, pipeline, large_price_df):
        """Symbol with >= 250 sessions should pass validation."""
        assert pipeline.validate_symbol_data(large_price_df, "VNM") is True

    def test_insufficient_data_fails(self, pipeline):
        """Symbol with < 250 sessions should fail validation."""
        small_df = pd.DataFrame({"close": np.ones(100)})
        assert pipeline.validate_symbol_data(small_df, "VNM") is False

    def test_none_data_fails(self, pipeline):
        """None DataFrame should fail validation."""
        assert pipeline.validate_symbol_data(None, "VNM") is False

    def test_custom_min_sessions(self, pipeline):
        """Custom minimum sessions threshold should be respected."""
        df = pd.DataFrame({"close": np.ones(50)})
        assert pipeline.validate_symbol_data(df, "VNM", min_sessions=50) is True
        assert pipeline.validate_symbol_data(df, "VNM", min_sessions=51) is False

    def test_exactly_minimum_passes(self, pipeline):
        """Exactly 250 sessions should pass."""
        df = pd.DataFrame({"close": np.ones(250)})
        assert pipeline.validate_symbol_data(df, "VNM") is True


# ==============================================================================
# Tests for prepare_training_data()
# ==============================================================================


class TestPrepareTrainingData:
    """Test training data preparation with VNINDEX inclusion."""

    def test_valid_symbols_included(self, pipeline, large_price_df):
        """Symbols with sufficient data should be included."""
        symbol_data = {
            "VNM": large_price_df,
            "FPT": large_price_df.copy(),
            "VNINDEX": large_price_df.copy(),
        }

        valid_symbols, valid_data = pipeline.prepare_training_data(symbol_data)

        assert "VNM" in valid_symbols
        assert "FPT" in valid_symbols
        assert "VNINDEX" in valid_symbols
        assert len(valid_symbols) == 3

    def test_insufficient_symbols_skipped(self, pipeline, large_price_df):
        """Symbols with insufficient data should be skipped with warning."""
        small_df = pd.DataFrame({"close": np.ones(100)})
        symbol_data = {
            "VNM": large_price_df,
            "SMALL": small_df,
            "VNINDEX": large_price_df.copy(),
        }

        valid_symbols, valid_data = pipeline.prepare_training_data(symbol_data)

        assert "VNM" in valid_symbols
        assert "VNINDEX" in valid_symbols
        assert "SMALL" not in valid_symbols

    def test_no_valid_symbols_raises_error(self, pipeline):
        """Should raise DataError if no symbols have sufficient data."""
        small_df = pd.DataFrame({"close": np.ones(100)})
        symbol_data = {"VNM": small_df, "FPT": small_df}

        with pytest.raises(DataError, match="No symbols with sufficient data"):
            pipeline.prepare_training_data(symbol_data)

    def test_vnindex_always_checked(self, pipeline, large_price_df):
        """VNINDEX should be validated like other symbols."""
        symbol_data = {
            "VNM": large_price_df,
            "VNINDEX": large_price_df.copy(),
        }

        valid_symbols, _ = pipeline.prepare_training_data(symbol_data)
        assert "VNINDEX" in valid_symbols


# ==============================================================================
# Tests for CheckpointManager
# ==============================================================================


class TestCheckpointManager:
    """Test CheckpointManager save, load, and cleanup."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temp directory for checkpoints."""
        return str(tmp_path / "checkpoints")

    @pytest.fixture
    def manager(self, temp_dir):
        """Create a CheckpointManager with temp directory."""
        return CheckpointManager(checkpoint_dir=temp_dir)

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        from engine.evaluation_model import StockEvalNet

        config = ModelConfig(num_features=61, lookback=60)
        return StockEvalNet(config)

    def test_save_creates_checkpoint_file(self, manager, simple_model, temp_dir):
        """Saving should create a checkpoint file."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

        path = manager.save(
            model=simple_model,
            optimizer=optimizer,
            epoch=0,
            train_loss=0.5,
            val_loss=0.6,
            best_val_loss=0.6,
        )

        assert Path(path).exists()

    def test_save_and_load_restores_state(self, manager, simple_model, temp_dir):
        """Loading a checkpoint should restore model and optimizer state."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

        # Do one forward/backward pass so optimizer has state
        dummy_input = torch.randn(2, 60, 61)
        output = simple_model(dummy_input)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Save weights before checkpoint
        original_weights = {k: v.clone() for k, v in simple_model.state_dict().items()}

        path = manager.save(
            model=simple_model,
            optimizer=optimizer,
            epoch=5,
            train_loss=0.3,
            val_loss=0.4,
            best_val_loss=0.35,
        )

        # Create a new model and optimizer
        from engine.evaluation_model import StockEvalNet

        new_model = StockEvalNet(ModelConfig(num_features=61, lookback=60))
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)

        # Load checkpoint
        info = manager.load(new_model, new_optimizer, path)

        assert info["epoch"] == 5
        assert info["train_loss"] == pytest.approx(0.3)
        assert info["val_loss"] == pytest.approx(0.4)
        assert info["best_val_loss"] == pytest.approx(0.35)

        # Weights should match
        for key in original_weights:
            assert torch.allclose(
                new_model.state_dict()[key],
                original_weights[key],
                atol=1e-7,
            )

    def test_find_latest_checkpoint(self, manager, simple_model, temp_dir):
        """Should find the most recent checkpoint file."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

        # Save multiple checkpoints
        for epoch in range(3):
            manager.save(
                model=simple_model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=0.5 - epoch * 0.1,
                val_loss=0.6 - epoch * 0.1,
                best_val_loss=0.5,
            )

        latest = manager.get_latest_checkpoint_path()
        assert latest is not None
        assert "epoch_0002" in latest

    def test_load_nonexistent_checkpoint_raises_error(self, manager, simple_model):
        """Should raise ModelError when checkpoint doesn't exist."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

        with pytest.raises(ModelError, match="not found"):
            manager.load(simple_model, optimizer, "/nonexistent/path.pt")

    def test_cleanup_old_checkpoints(self, manager, simple_model, temp_dir):
        """Should keep only the last N checkpoints."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

        # Save 5 checkpoints
        for epoch in range(5):
            manager.save(
                model=simple_model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=0.5,
                val_loss=0.6,
                best_val_loss=0.5,
            )

        manager.cleanup_old_checkpoints(keep_last=2)

        remaining = sorted(Path(temp_dir).glob("checkpoint_epoch_*.pt"))
        assert len(remaining) == 2
        # Should keep epochs 3 and 4
        assert "0003" in remaining[0].name
        assert "0004" in remaining[1].name

    def test_save_with_norm_params(self, manager, simple_model, temp_dir):
        """Should save and restore normalization parameters."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        norm_params = {
            "min_vals": [0.0, 1.0, 2.0],
            "max_vals": [10.0, 11.0, 12.0],
        }

        path = manager.save(
            model=simple_model,
            optimizer=optimizer,
            epoch=0,
            train_loss=0.5,
            val_loss=0.6,
            best_val_loss=0.6,
            norm_params=norm_params,
        )

        from engine.evaluation_model import StockEvalNet

        new_model = StockEvalNet(ModelConfig(num_features=61, lookback=60))
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        info = manager.load(new_model, new_optimizer, path)

        assert info["norm_params"] == norm_params


# ==============================================================================
# Tests for EpochLogger
# ==============================================================================


class TestEpochLogger:
    """Test epoch logging to JSONL file."""

    @pytest.fixture
    def logger(self, tmp_path):
        """Create an EpochLogger with temp file."""
        log_file = str(tmp_path / "training_log.jsonl")
        return EpochLogger(log_file=log_file)

    def test_log_epoch_creates_file(self, logger):
        """Logging should create the log file."""
        logger.log_epoch(epoch=0, train_loss=0.5, val_loss=0.6, mae=0.3)
        assert Path(logger.log_file).exists()

    def test_log_epoch_writes_valid_json(self, logger):
        """Each line should be valid JSON."""
        logger.log_epoch(epoch=0, train_loss=0.5, val_loss=0.6, mae=0.3)
        logger.log_epoch(epoch=1, train_loss=0.4, val_loss=0.5, mae=0.25)

        with open(logger.log_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2
        for line in lines:
            entry = json.loads(line.strip())
            assert "epoch" in entry
            assert "train_loss" in entry
            assert "val_loss" in entry
            assert "mae" in entry
            assert "timestamp" in entry

    def test_log_epoch_preserves_values(self, logger):
        """Logged values should match input."""
        logger.log_epoch(
            epoch=5,
            train_loss=0.123,
            val_loss=0.456,
            mae=0.789,
            learning_rate=0.001,
            epoch_time_seconds=12.5,
            mode="full",
        )

        with open(logger.log_file, "r") as f:
            entry = json.loads(f.readline())

        assert entry["epoch"] == 5
        assert entry["train_loss"] == pytest.approx(0.123)
        assert entry["val_loss"] == pytest.approx(0.456)
        assert entry["mae"] == pytest.approx(0.789)
        assert entry["learning_rate"] == pytest.approx(0.001)
        assert entry["epoch_time_seconds"] == pytest.approx(12.5)
        assert entry["mode"] == "full"

    def test_log_appends_not_overwrites(self, logger):
        """Multiple log calls should append, not overwrite."""
        for i in range(5):
            logger.log_epoch(epoch=i, train_loss=0.5 - i * 0.1, val_loss=0.6, mae=0.3)

        with open(logger.log_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 5
