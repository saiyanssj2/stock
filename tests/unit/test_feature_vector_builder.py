"""
Unit tests for FeatureVectorBuilder.

Tests cover:
- build() with valid data: normalization to [0, 1]
- build() with NaN data: forward-fill then zero-fill
- denormalize() round-trip
- save_params() / load from JSON
- from_training_data() classmethod
- Degenerate features (min == max) produce 0.5
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.market_state import (
    FeatureVectorBuilder,
    MarketState,
    INDICATOR_COLUMNS,
    OHLCV_COLUMNS,
    NUM_FEATURES,
    ALL_FEATURE_COLUMNS,
    NUM_INDICATORS,
)
from engine.config import DataError


class TestFeatureVectorBuilderBuild:
    """Tests for FeatureVectorBuilder.build() method."""

    def _make_state(self, lookback=60, with_nan=False):
        """Helper to create a valid MarketState for testing."""
        np.random.seed(42)
        ohlcv = np.random.uniform(10, 100, (lookback, 5))
        # Make OHLCV consistent: high >= open,close; low <= open,close
        ohlcv[:, 1] = np.maximum(ohlcv[:, 1], np.maximum(ohlcv[:, 0], ohlcv[:, 3]))
        ohlcv[:, 2] = np.minimum(ohlcv[:, 2], np.minimum(ohlcv[:, 0], ohlcv[:, 3]))

        num_indicators = len(INDICATOR_COLUMNS)
        indicators = np.random.uniform(-100, 500, (lookback, num_indicators))

        if with_nan:
            # Inject NaN in various places
            indicators[0, :5] = np.nan  # First row, first 5 indicators
            indicators[10, 10:15] = np.nan  # Middle row
            indicators[:, -1] = np.nan  # Entire last column NaN

        return MarketState(
            symbol="VNM",
            timestamp=pd.Timestamp("2024-01-15"),
            ohlcv=ohlcv,
            indicators=indicators,
            lookback=lookback,
        )

    def _make_builder(self):
        """Helper to create a FeatureVectorBuilder with known min/max."""
        np.random.seed(42)
        min_vals = np.random.uniform(-100, 0, NUM_FEATURES)
        max_vals = min_vals + np.random.uniform(10, 200, NUM_FEATURES)
        return FeatureVectorBuilder(min_vals=min_vals, max_vals=max_vals)

    def test_build_output_shape(self):
        """build() returns array of shape (lookback, num_features)."""
        state = self._make_state(lookback=60)
        builder = self._make_builder()
        result = builder.build(state)
        assert result.shape == (60, NUM_FEATURES)

    def test_build_values_in_zero_one(self):
        """build() produces values in [0, 1] for non-degenerate data."""
        state = self._make_state(lookback=60)
        builder = self._make_builder()
        result = builder.build(state)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_build_no_nan_in_output(self):
        """build() produces no NaN or Inf values even with NaN input."""
        state = self._make_state(lookback=60, with_nan=True)
        builder = self._make_builder()
        result = builder.build(state)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_build_forward_fill_nan(self):
        """build() forward-fills NaN along time axis before normalization."""
        lookback = 20
        ohlcv = np.ones((lookback, 5)) * 50.0
        num_indicators = len(INDICATOR_COLUMNS)
        indicators = np.ones((lookback, num_indicators)) * 25.0

        # Create a column where first few values are valid, then NaN
        indicators[5:10, 0] = np.nan  # Rows 5-9 of column 0 are NaN

        state = MarketState(
            symbol="FPT",
            timestamp=pd.Timestamp("2024-01-15"),
            ohlcv=ohlcv,
            indicators=indicators,
            lookback=lookback,
        )

        # min/max that make the non-NaN value normalize to something specific
        min_vals = np.zeros(NUM_FEATURES)
        max_vals = np.ones(NUM_FEATURES) * 100.0
        builder = FeatureVectorBuilder(min_vals=min_vals, max_vals=max_vals)

        result = builder.build(state)

        # After forward-fill, rows 5-9 of indicator 0 should be filled with 25.0
        # Indicator 0 is at column index 5 (after OHLCV)
        # Normalized: (25 - 0) / (100 - 0) = 0.25
        indicator_col_idx = 5  # First indicator after 5 OHLCV cols
        for row in range(5, 10):
            assert abs(result[row, indicator_col_idx] - 0.25) < 1e-10

    def test_build_degenerate_feature_gives_half(self):
        """When min == max for a feature, build() outputs 0.5."""
        state = self._make_state(lookback=30)

        # Make min == max for feature index 0
        min_vals = np.zeros(NUM_FEATURES)
        max_vals = np.ones(NUM_FEATURES) * 100.0
        min_vals[0] = 50.0
        max_vals[0] = 50.0  # Degenerate

        builder = FeatureVectorBuilder(min_vals=min_vals, max_vals=max_vals)
        result = builder.build(state)

        # All values in column 0 should be 0.5
        assert np.all(result[:, 0] == 0.5)


class TestFeatureVectorBuilderDenormalize:
    """Tests for FeatureVectorBuilder.denormalize() method."""

    def test_denormalize_round_trip(self):
        """normalize then denormalize recovers original values within tolerance."""
        np.random.seed(123)
        min_vals = np.random.uniform(-50, 0, NUM_FEATURES)
        max_vals = min_vals + np.random.uniform(10, 200, NUM_FEATURES)
        builder = FeatureVectorBuilder(min_vals=min_vals, max_vals=max_vals)

        # Generate raw values within the min/max range
        raw = np.random.uniform(0, 1, (60, NUM_FEATURES))
        for i in range(NUM_FEATURES):
            raw[:, i] = raw[:, i] * (max_vals[i] - min_vals[i]) + min_vals[i]

        # Normalize
        range_vals = max_vals - min_vals
        normalized = (raw - min_vals) / range_vals

        # Denormalize
        recovered = builder.denormalize(normalized)

        # Check tolerance
        assert np.allclose(raw, recovered, rtol=1e-4)

    def test_denormalize_zeros(self):
        """denormalize(zeros) returns min_vals."""
        min_vals = np.array([1.0, 2.0, 3.0] + [0.0] * (NUM_FEATURES - 3))
        max_vals = np.array([10.0, 20.0, 30.0] + [1.0] * (NUM_FEATURES - 3))
        builder = FeatureVectorBuilder(min_vals=min_vals, max_vals=max_vals)

        zeros = np.zeros((1, NUM_FEATURES))
        result = builder.denormalize(zeros)
        assert np.allclose(result, min_vals)

    def test_denormalize_ones(self):
        """denormalize(ones) returns max_vals."""
        min_vals = np.array([1.0, 2.0, 3.0] + [0.0] * (NUM_FEATURES - 3))
        max_vals = np.array([10.0, 20.0, 30.0] + [1.0] * (NUM_FEATURES - 3))
        builder = FeatureVectorBuilder(min_vals=min_vals, max_vals=max_vals)

        ones = np.ones((1, NUM_FEATURES))
        result = builder.denormalize(ones)
        assert np.allclose(result, max_vals)


class TestFeatureVectorBuilderSaveLoad:
    """Tests for save_params() and loading from JSON."""

    def test_save_and_load_params(self, tmp_path):
        """save_params() creates a JSON file that can be loaded back."""
        np.random.seed(42)
        min_vals = np.random.uniform(-100, 0, NUM_FEATURES)
        max_vals = min_vals + np.random.uniform(10, 200, NUM_FEATURES)

        builder = FeatureVectorBuilder(min_vals=min_vals, max_vals=max_vals)

        # Save
        path = str(tmp_path / "norm_params.json")
        builder.save_params(path)

        # Load
        loaded_builder = FeatureVectorBuilder(norm_params_path=path)

        assert np.allclose(builder.min_vals, loaded_builder.min_vals, atol=1e-15)
        assert np.allclose(builder.max_vals, loaded_builder.max_vals, atol=1e-15)

    def test_save_params_creates_directory(self, tmp_path):
        """save_params() creates parent directories if needed."""
        builder = FeatureVectorBuilder(
            min_vals=np.zeros(NUM_FEATURES),
            max_vals=np.ones(NUM_FEATURES),
        )

        path = str(tmp_path / "subdir" / "nested" / "params.json")
        builder.save_params(path)

        assert Path(path).exists()

    def test_load_missing_file_raises_error(self):
        """Loading from a non-existent file raises DataError."""
        with pytest.raises(DataError, match="not found"):
            FeatureVectorBuilder(norm_params_path="/nonexistent/path.json")

    def test_save_load_preserves_feature_columns(self, tmp_path):
        """Saved JSON includes feature column names."""
        builder = FeatureVectorBuilder(
            min_vals=np.zeros(NUM_FEATURES),
            max_vals=np.ones(NUM_FEATURES),
        )

        path = str(tmp_path / "params.json")
        builder.save_params(path)

        with open(path, "r") as f:
            data = json.load(f)

        assert data["num_features"] == NUM_FEATURES
        assert data["feature_columns"] == ALL_FEATURE_COLUMNS


class TestFeatureVectorBuilderFromTrainingData:
    """Tests for from_training_data() classmethod."""

    def _make_training_df(self, num_rows=100, seed=42):
        """Create a DataFrame with OHLCV and indicator columns."""
        np.random.seed(seed)
        df = pd.DataFrame()
        df["time"] = pd.bdate_range(end="2024-06-30", periods=num_rows)
        df["open"] = np.random.uniform(20, 100, num_rows)
        df["high"] = df["open"] * 1.02
        df["low"] = df["open"] * 0.98
        df["close"] = df["open"] * np.random.uniform(0.99, 1.01, num_rows)
        df["volume"] = np.random.uniform(100000, 5000000, num_rows)

        # Add indicator columns with realistic ranges
        for col in INDICATOR_COLUMNS:
            df[col] = np.random.uniform(-50, 200, num_rows)

        return df

    def test_from_training_data_basic(self):
        """from_training_data computes valid min/max from dataframes."""
        dfs = [self._make_training_df(seed=i) for i in range(3)]
        builder = FeatureVectorBuilder.from_training_data(dfs)

        assert builder.min_vals.shape == (NUM_FEATURES,)
        assert builder.max_vals.shape == (NUM_FEATURES,)
        # min should be less than or equal to max
        assert np.all(builder.min_vals <= builder.max_vals)

    def test_from_training_data_empty_raises_error(self):
        """from_training_data with empty list raises DataError."""
        with pytest.raises(DataError, match="No DataFrames"):
            FeatureVectorBuilder.from_training_data([])

    def test_from_training_data_global_min_max(self):
        """from_training_data computes global min/max across all DataFrames."""
        # Create two DataFrames with known min/max
        df1 = self._make_training_df(num_rows=50, seed=1)
        df2 = self._make_training_df(num_rows=50, seed=2)

        builder = FeatureVectorBuilder.from_training_data([df1, df2])

        # The global min should be <= individual DataFrames' mins
        # Check close column (index 3 in features)
        close_idx = 3  # close is the 4th OHLCV column
        all_close = np.concatenate([df1["close"].values, df2["close"].values])
        assert builder.min_vals[close_idx] <= all_close.min() + 1e-10
        assert builder.max_vals[close_idx] >= all_close.max() - 1e-10


class TestIndicatorColumns:
    """Tests for INDICATOR_COLUMNS constant."""

    def test_indicator_columns_count(self):
        """INDICATOR_COLUMNS has the expected number of indicators."""
        # 22 trend + 11 momentum + 14 volatility + 9 volume = 56
        assert len(INDICATOR_COLUMNS) == 56

    def test_num_features(self):
        """NUM_FEATURES = OHLCV + indicators."""
        assert NUM_FEATURES == len(OHLCV_COLUMNS) + len(INDICATOR_COLUMNS)
        assert NUM_FEATURES == 61

    def test_all_feature_columns_order(self):
        """ALL_FEATURE_COLUMNS starts with OHLCV then indicators."""
        assert ALL_FEATURE_COLUMNS[:5] == OHLCV_COLUMNS
        assert ALL_FEATURE_COLUMNS[5:] == INDICATOR_COLUMNS

    def test_no_duplicate_columns(self):
        """No duplicate column names in ALL_FEATURE_COLUMNS."""
        assert len(ALL_FEATURE_COLUMNS) == len(set(ALL_FEATURE_COLUMNS))
