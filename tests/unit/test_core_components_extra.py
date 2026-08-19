"""
Unit tests for core components: error handling and configuration validation.

Covers:
- Data validation error handling (Requirements 1.4, 1.5)
- Error class hierarchy and attributes
- Configuration dataclass edge cases (ModelConfig, EngineConfig, SearchConfig, TrainingConfig)

Validates: Requirements 1.4, 1.5
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from engine.config import (
    ConfigError,
    DataError,
    EngineConfig,
    EngineError,
    ModelConfig,
    ModelError,
    ResourceError,
    SearchConfig,
    TrainingConfig,
)
from engine.evaluation_model import StockEvalNet


# ==============================================================================
# 3. Error Handling Tests
# ==============================================================================


class TestDataValidationErrors:
    """Test error handling for invalid inputs in DecisionEngine.validate_csv."""

    def setup_method(self):
        """Create a DecisionEngine instance with a temp directory."""
        from engine.decision_engine import DecisionEngine

        self.temp_dir = tempfile.mkdtemp()
        config = EngineConfig(base_dir=self.temp_dir)
        self.engine = DecisionEngine(base_dir=self.temp_dir, config=config)

    def test_missing_csv_raises_data_error(self):
        """Missing CSV file raises DataError with FILE_NOT_FOUND code."""
        fake_path = Path(self.temp_dir) / "nonexistent.csv"
        with pytest.raises(DataError) as exc_info:
            self.engine.validate_csv(fake_path)
        assert exc_info.value.error_code == "FILE_NOT_FOUND"
        assert "nonexistent.csv" in str(exc_info.value.message)

    def test_missing_csv_error_includes_path_details(self):
        """DataError for missing file includes path in details."""
        fake_path = Path(self.temp_dir) / "missing_stock.csv"
        with pytest.raises(DataError) as exc_info:
            self.engine.validate_csv(fake_path)
        assert "path" in exc_info.value.details
        assert "missing_stock.csv" in exc_info.value.details["path"]

    def test_malformed_csv_missing_columns(self):
        """CSV with missing required columns raises DataError with MISSING_COLUMNS."""
        csv_path = Path(self.temp_dir) / "bad_columns.csv"
        # Write CSV with only 'time' and 'close' — missing open, high, low, volume
        pd.DataFrame({"time": ["2024-01-01"], "close": [100.0]}).to_csv(
            csv_path, index=False
        )

        with pytest.raises(DataError) as exc_info:
            self.engine.validate_csv(csv_path)
        assert exc_info.value.error_code == "MISSING_COLUMNS"
        assert "missing_columns" in exc_info.value.details
        missing = exc_info.value.details["missing_columns"]
        assert "open" in missing
        assert "high" in missing
        assert "low" in missing
        assert "volume" in missing

    def test_insufficient_rows_raises_data_error(self):
        """CSV with fewer rows than min_data_rows raises DataError."""
        csv_path = Path(self.temp_dir) / "too_short.csv"
        # min_data_rows default is 5, write only 3 rows
        pd.DataFrame(
            {
                "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000, 1100, 1200],
            }
        ).to_csv(csv_path, index=False)

        with pytest.raises(DataError) as exc_info:
            self.engine.validate_csv(csv_path)
        assert exc_info.value.error_code == "INSUFFICIENT_DATA"
        assert exc_info.value.details["rows"] == 3
        assert exc_info.value.details["min_required"] == 5

    def test_valid_csv_passes_validation(self):
        """A CSV with all required columns and enough rows passes validation."""
        csv_path = Path(self.temp_dir) / "valid.csv"
        n = 10
        pd.DataFrame(
            {
                "time": pd.bdate_range("2024-01-01", periods=n),
                "open": np.random.uniform(90, 110, n),
                "high": np.random.uniform(100, 120, n),
                "low": np.random.uniform(80, 100, n),
                "close": np.random.uniform(90, 110, n),
                "volume": np.random.uniform(100000, 500000, n),
            }
        ).to_csv(csv_path, index=False)

        # Should not raise
        self.engine.validate_csv(csv_path)

    def test_validate_multiple_csvs_collects_all_errors(self):
        """validate_multiple_csvs returns errors for all invalid files."""
        valid_path = Path(self.temp_dir) / "valid.csv"
        missing_path = Path(self.temp_dir) / "missing.csv"
        bad_cols_path = Path(self.temp_dir) / "bad_cols.csv"

        # Create valid CSV
        pd.DataFrame(
            {
                "time": pd.bdate_range("2024-01-01", periods=10),
                "open": range(10),
                "high": range(10),
                "low": range(10),
                "close": range(10),
                "volume": range(10),
            }
        ).to_csv(valid_path, index=False)

        # Create malformed CSV
        pd.DataFrame({"time": ["2024-01-01"], "close": [100]}).to_csv(
            bad_cols_path, index=False
        )

        errors = self.engine.validate_multiple_csvs(
            [valid_path, missing_path, bad_cols_path]
        )
        # Valid file should not be in errors
        assert str(valid_path) not in errors
        # Missing and malformed files should both have errors
        assert str(missing_path) in errors
        assert str(bad_cols_path) in errors
        assert errors[str(missing_path)].error_code == "FILE_NOT_FOUND"
        assert errors[str(bad_cols_path)].error_code == "MISSING_COLUMNS"


class TestErrorHierarchy:
    """Test the error class hierarchy and attributes."""

    def test_data_error_is_engine_error(self):
        """DataError inherits from EngineError."""
        err = DataError("test")
        assert isinstance(err, EngineError)

    def test_model_error_is_engine_error(self):
        """ModelError inherits from EngineError."""
        err = ModelError("test")
        assert isinstance(err, EngineError)

    def test_resource_error_is_engine_error(self):
        """ResourceError inherits from EngineError."""
        err = ResourceError("test")
        assert isinstance(err, EngineError)

    def test_config_error_is_engine_error(self):
        """ConfigError inherits from EngineError."""
        err = ConfigError("test")
        assert isinstance(err, EngineError)

    def test_engine_error_attributes(self):
        """EngineError stores message, error_code, and details."""
        err = EngineError("msg", "CODE_123", {"key": "val"})
        assert err.message == "msg"
        assert err.error_code == "CODE_123"
        assert err.details == {"key": "val"}

    def test_engine_error_str_format(self):
        """EngineError __str__ includes error_code and message."""
        err = EngineError("something failed", "MY_CODE")
        assert "[MY_CODE]" in str(err)
        assert "something failed" in str(err)

    def test_data_error_default_code(self):
        """DataError uses DATA_ERROR as default error_code."""
        err = DataError("test message")
        assert err.error_code == "DATA_ERROR"

    def test_model_error_default_code(self):
        """ModelError uses MODEL_ERROR as default error_code."""
        err = ModelError("test message")
        assert err.error_code == "MODEL_ERROR"


# ==============================================================================
# 4. Configuration Validation Edge Cases
# ==============================================================================


class TestModelConfigEdgeCases:
    """Test ModelConfig dataclass edge cases."""

    def test_default_values(self):
        """Default ModelConfig has expected values."""
        config = ModelConfig()
        assert config.num_features == 61
        assert config.lookback == 60
        assert config.tcn_channels == [128, 128, 64]
        assert config.kernel_size == 3
        assert config.dilations == [1, 2, 4]
        assert config.attention_heads == 4
        assert config.attention_dim == 64
        assert config.dropout == 0.1
        assert config.max_batch_size == 50
        assert config.output_range == (-1.0, 1.0)

    def test_custom_values_accepted(self):
        """ModelConfig accepts custom values."""
        config = ModelConfig(
            num_features=32,
            lookback=30,
            tcn_channels=[64, 64],
            dilations=[1, 2],
            attention_heads=2,
            attention_dim=32,
            dropout=0.2,
            max_batch_size=10,
        )
        assert config.num_features == 32
        assert config.lookback == 30
        assert config.tcn_channels == [64, 64]
        assert config.max_batch_size == 10

    def test_model_config_zero_dropout(self):
        """ModelConfig with zero dropout is valid."""
        config = ModelConfig(dropout=0.0)
        assert config.dropout == 0.0

    def test_model_config_single_channel(self):
        """ModelConfig with single TCN channel works."""
        config = ModelConfig(tcn_channels=[64], dilations=[1])
        model = StockEvalNet(config)
        x = torch.randn(1, config.lookback, config.num_features)
        model.eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1)


class TestEngineConfigEdgeCases:
    """Test EngineConfig dataclass edge cases."""

    def test_default_values(self):
        """Default EngineConfig has expected values."""
        config = EngineConfig()
        assert config.lookback == 60
        assert config.lookback_min == 20
        assert config.lookback_max == 200
        assert config.default_capital == 100_000_000.0
        assert config.max_position_pct == 0.20
        assert config.lot_size == 100
        assert config.daily_price_limit == 0.07
        assert config.settlement_days == 2.5
        assert config.confidence_hold_threshold == 0.3
        assert config.min_data_rows == 5

    def test_required_columns_default(self):
        """Default required columns match expected list."""
        config = EngineConfig()
        expected = ["time", "open", "high", "low", "close", "volume"]
        assert config.csv_required_columns == expected

    def test_custom_lookback_in_range(self):
        """Custom lookback within [20, 200] is accepted."""
        config = EngineConfig(lookback=100)
        assert config.lookback == 100

    def test_custom_capital(self):
        """Custom initial capital is accepted."""
        config = EngineConfig(default_capital=50_000_000.0)
        assert config.default_capital == 50_000_000.0

    def test_custom_lot_size(self):
        """Custom lot size is accepted."""
        config = EngineConfig(lot_size=500)
        assert config.lot_size == 500


class TestSearchConfigEdgeCases:
    """Test SearchConfig dataclass edge cases."""

    def test_default_values(self):
        """Default SearchConfig has expected values."""
        config = SearchConfig()
        assert config.default_depth == 3
        assert config.max_depth == 5
        assert config.timeout_seconds == 5.0
        assert config.default_scenarios == 5
        assert config.min_scenarios == 3
        assert config.max_scenarios == 7
        assert config.min_history_days == 30
        assert config.top_scenarios_report == 3
        assert config.top_indicators_report == 5

    def test_adaptive_settings(self):
        """Adaptive branching settings are correct."""
        config = SearchConfig()
        assert config.adaptive_depth_threshold == 3
        assert config.adaptive_scenario_count == 3

    def test_custom_depth(self):
        """Custom depth values are accepted."""
        config = SearchConfig(default_depth=2, max_depth=4)
        assert config.default_depth == 2
        assert config.max_depth == 4

    def test_custom_timeout(self):
        """Custom timeout is accepted."""
        config = SearchConfig(timeout_seconds=10.0)
        assert config.timeout_seconds == 10.0


class TestTrainingConfigEdgeCases:
    """Test TrainingConfig dataclass edge cases."""

    def test_default_values(self):
        """Default TrainingConfig has expected values."""
        config = TrainingConfig()
        assert config.learning_rate == 1e-3
        assert config.batch_size == 64
        assert config.max_epochs_full == 100
        assert config.max_epochs_incremental == 10
        assert config.train_split == 0.70
        assert config.val_split == 0.15
        assert config.test_split == 0.15
        assert config.label_horizon == 5
        assert config.label_sensitivity == 10.0
        assert config.min_sessions_per_symbol == 250

    def test_splits_sum_to_one(self):
        """Train/val/test splits sum to 1.0."""
        config = TrainingConfig()
        total = config.train_split + config.val_split + config.test_split
        assert abs(total - 1.0) < 1e-10

    def test_custom_learning_rate(self):
        """Custom learning rate is accepted."""
        config = TrainingConfig(learning_rate=5e-4)
        assert config.learning_rate == 5e-4

    def test_custom_batch_size(self):
        """Custom batch size is accepted."""
        config = TrainingConfig(batch_size=32)
        assert config.batch_size == 32

    def test_custom_epochs(self):
        """Custom epoch counts are accepted."""
        config = TrainingConfig(max_epochs_full=50, max_epochs_incremental=5)
        assert config.max_epochs_full == 50
        assert config.max_epochs_incremental == 5
