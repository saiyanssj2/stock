"""
Property-based tests for FeatureVectorBuilder.

Tests the following correctness properties from the design document:
- Property 1: Feature vector normalization bounds (all values in [0.0, 1.0])
- Property 3: Normalize/denormalize round-trip (within 0.01% relative tolerance)
- Property 4: Normalization parameters serialization round-trip (save/load JSON identical within 1e-15)
- Property 6: NaN handling produces clean feature vectors (no NaN or Inf in output)

**Validates: Requirements 2.4, 2.7, 8.2, 8.3, 8.4, 8.6**
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.market_state import (
    FeatureVectorBuilder,
    MarketState,
    NUM_FEATURES,
    NUM_INDICATORS,
    OHLCV_COLUMNS,
    INDICATOR_COLUMNS,
)


# ---------------------------------------------------------------------------
# Custom strategies adapted for the actual engine's NUM_FEATURES (61)
# ---------------------------------------------------------------------------


@st.composite
def normalization_params_strategy(draw, num_features=NUM_FEATURES):
    """
    Generate normalization parameters (per-feature min and max values)
    compatible with FeatureVectorBuilder.

    Ensures min_vals[i] < max_vals[i] for all i (non-degenerate).
    """
    min_vals = draw(
        st.lists(
            st.floats(min_value=-1000.0, max_value=500.0,
                      allow_nan=False, allow_infinity=False),
            min_size=num_features,
            max_size=num_features,
        )
    )
    min_vals = np.array(min_vals)

    # Ensure max > min by adding a positive offset
    offsets = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=500.0,
                      allow_nan=False, allow_infinity=False),
            min_size=num_features,
            max_size=num_features,
        )
    )
    offsets = np.array(offsets)
    max_vals = min_vals + offsets

    return {
        "min_vals": min_vals,
        "max_vals": max_vals,
    }


@st.composite
def market_state_within_range_strategy(draw, params):
    """
    Generate a MarketState whose raw feature values lie within the
    normalization min/max range, ensuring build() produces values in [0, 1].

    Parameters
    ----------
    params : dict
        Normalization params with 'min_vals' and 'max_vals' arrays of shape (NUM_FEATURES,).
    """
    min_vals = params["min_vals"]
    max_vals = params["max_vals"]

    lookback = draw(st.integers(min_value=20, max_value=60))

    # Generate OHLCV data within the first 5 features' min/max range
    ohlcv = np.zeros((lookback, 5))
    for col in range(5):
        col_min = min_vals[col]
        col_max = max_vals[col]
        vals = draw(
            st.lists(
                st.floats(min_value=float(col_min), max_value=float(col_max),
                          allow_nan=False, allow_infinity=False),
                min_size=lookback,
                max_size=lookback,
            )
        )
        ohlcv[:, col] = vals

    # Generate indicator data within features 5..NUM_FEATURES range
    indicators = np.zeros((lookback, NUM_INDICATORS))
    for col in range(NUM_INDICATORS):
        feat_idx = col + 5  # offset by OHLCV columns
        col_min = min_vals[feat_idx]
        col_max = max_vals[feat_idx]
        vals = draw(
            st.lists(
                st.floats(min_value=float(col_min), max_value=float(col_max),
                          allow_nan=False, allow_infinity=False),
                min_size=lookback,
                max_size=lookback,
            )
        )
        indicators[:, col] = vals

    return MarketState(
        symbol="VNM",
        timestamp=pd.Timestamp("2024-01-15"),
        ohlcv=ohlcv,
        indicators=indicators,
        lookback=lookback,
    )


# ---------------------------------------------------------------------------
# Property 1: Feature vector normalization bounds
# ---------------------------------------------------------------------------


class TestProperty1NormalizationBounds:
    """
    Property 1: Feature vector normalization bounds.

    For any valid MarketState with non-degenerate data (at least one feature
    has distinct min and max values), the resulting Feature_Vector after
    min-max normalization SHALL have all values in [0.0, 1.0].

    **Validates: Requirements 2.4, 8.3**
    """

    @given(params=normalization_params_strategy())
    @settings(max_examples=15)
    def test_build_output_in_zero_one_range(self, params):
        """
        All normalized feature values are in [0.0, 1.0] when input data
        is within the normalization min/max range.

        **Validates: Requirements 2.4, 8.3**
        """
        min_vals = params["min_vals"]
        max_vals = params["max_vals"]

        builder = FeatureVectorBuilder(min_vals=min_vals, max_vals=max_vals)

        lookback = 30

        # Generate OHLCV within range
        ohlcv = np.zeros((lookback, 5))
        for col in range(5):
            ohlcv[:, col] = np.random.uniform(min_vals[col], max_vals[col], lookback)

        # Generate indicators within range
        indicators = np.zeros((lookback, NUM_INDICATORS))
        for col in range(NUM_INDICATORS):
            feat_idx = col + 5
            indicators[:, col] = np.random.uniform(
                min_vals[feat_idx], max_vals[feat_idx], lookback
            )

        state = MarketState(
            symbol="FPT",
            timestamp=pd.Timestamp("2024-01-15"),
            ohlcv=ohlcv,
            indicators=indicators,
            lookback=lookback,
        )

        result = builder.build(state)

        # Assert all values in [0.0, 1.0]
        assert result.shape == (lookback, NUM_FEATURES)
        assert np.all(result >= 0.0), (
            f"Found values below 0.0: min={result.min()}"
        )
        assert np.all(result <= 1.0), (
            f"Found values above 1.0: max={result.max()}"
        )

    @given(params=normalization_params_strategy())
    @settings(max_examples=15)
    def test_build_clips_out_of_range_to_zero_one(self, params):
        """
        Values outside training range are clipped to [0.0, 1.0].

        **Validates: Requirements 2.4, 8.3**
        """
        min_vals = params["min_vals"]
        max_vals = params["max_vals"]

        builder = FeatureVectorBuilder(min_vals=min_vals, max_vals=max_vals)

        lookback = 20

        # Generate data deliberately outside the min/max range
        ohlcv = np.zeros((lookback, 5))
        for col in range(5):
            # Go beyond max
            ohlcv[:, col] = max_vals[col] + 100.0

        indicators = np.zeros((lookback, NUM_INDICATORS))
        for col in range(NUM_INDICATORS):
            feat_idx = col + 5
            # Go below min
            indicators[:, col] = min_vals[feat_idx] - 100.0

        state = MarketState(
            symbol="HPG",
            timestamp=pd.Timestamp("2024-01-15"),
            ohlcv=ohlcv,
            indicators=indicators,
            lookback=lookback,
        )

        result = builder.build(state)

        # Even out-of-range data must be clipped to [0, 1]
        assert np.all(result >= 0.0), f"Found values below 0.0: min={result.min()}"
        assert np.all(result <= 1.0), f"Found values above 1.0: max={result.max()}"


# ---------------------------------------------------------------------------
# Property 3: Normalize/denormalize round-trip
# ---------------------------------------------------------------------------


class TestProperty3NormalizeDenormalizeRoundTrip:
    """
    Property 3: Normalize/denormalize round-trip.

    For any valid MarketState and trained normalization parameters, applying
    normalization (to [0,1]) followed by denormalization SHALL produce
    indicator values within 0.01% relative tolerance of the originals.

    **Validates: Requirements 8.6**
    """

    @given(params=normalization_params_strategy())
    @settings(max_examples=15)
    def test_normalize_denormalize_round_trip(self, params):
        """
        Normalizing then denormalizing recovers original values within
        0.01% relative tolerance for non-degenerate features.

        **Validates: Requirements 8.6**
        """
        min_vals = params["min_vals"]
        max_vals = params["max_vals"]

        builder = FeatureVectorBuilder(min_vals=min_vals, max_vals=max_vals)

        lookback = 30

        # Generate values strictly within range to avoid clipping
        # Use a margin to stay away from boundaries
        ohlcv = np.zeros((lookback, 5))
        for col in range(5):
            margin = (max_vals[col] - min_vals[col]) * 0.01
            ohlcv[:, col] = np.random.uniform(
                min_vals[col] + margin, max_vals[col] - margin, lookback
            )

        indicators = np.zeros((lookback, NUM_INDICATORS))
        for col in range(NUM_INDICATORS):
            feat_idx = col + 5
            margin = (max_vals[feat_idx] - min_vals[feat_idx]) * 0.01
            indicators[:, col] = np.random.uniform(
                min_vals[feat_idx] + margin, max_vals[feat_idx] - margin, lookback
            )

        state = MarketState(
            symbol="VCB",
            timestamp=pd.Timestamp("2024-01-15"),
            ohlcv=ohlcv,
            indicators=indicators,
            lookback=lookback,
        )

        # Original raw features (OHLCV + indicators concatenated)
        original = np.concatenate([ohlcv, indicators], axis=1)

        # Normalize via build()
        normalized = builder.build(state)

        # Denormalize
        recovered = builder.denormalize(normalized)

        # Check 0.01% relative tolerance for non-degenerate features
        range_vals = max_vals - min_vals
        non_degenerate = range_vals > 0

        for feat_idx in range(NUM_FEATURES):
            if not non_degenerate[feat_idx]:
                continue
            for t in range(lookback):
                orig_val = original[t, feat_idx]
                rec_val = recovered[t, feat_idx]
                if abs(orig_val) < 1e-10:
                    # For near-zero values, use absolute tolerance
                    assert abs(rec_val - orig_val) < 1e-6, (
                        f"Absolute error too large at [{t}, {feat_idx}]: "
                        f"original={orig_val}, recovered={rec_val}"
                    )
                else:
                    rel_error = abs(rec_val - orig_val) / abs(orig_val)
                    assert rel_error < 1e-4, (  # 0.01% = 1e-4
                        f"Relative error {rel_error:.2e} exceeds 0.01% at "
                        f"[{t}, {feat_idx}]: original={orig_val}, recovered={rec_val}"
                    )


# ---------------------------------------------------------------------------
# Property 4: Normalization parameters serialization round-trip
# ---------------------------------------------------------------------------


class TestProperty4ParamsSerializationRoundTrip:
    """
    Property 4: Normalization parameters serialization round-trip.

    For any set of normalization parameters (per-feature min and max values),
    saving to JSON and loading back SHALL produce numerically identical
    parameters (within floating-point representation limits of 1e-15).

    **Validates: Requirements 8.4**
    """

    @given(params=normalization_params_strategy())
    @settings(max_examples=15)
    def test_save_load_params_identical(self, params):
        """
        save_params() then loading produces min_vals and max_vals
        identical within 1e-15 tolerance.

        **Validates: Requirements 8.4**
        """
        min_vals = params["min_vals"]
        max_vals = params["max_vals"]

        original_builder = FeatureVectorBuilder(min_vals=min_vals, max_vals=max_vals)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "norm_params.json")

            # Save
            original_builder.save_params(path)

            # Load into new builder
            loaded_builder = FeatureVectorBuilder(norm_params_path=path)

            # Assert min_vals match within 1e-15
            assert np.allclose(
                original_builder.min_vals,
                loaded_builder.min_vals,
                atol=1e-15,
                rtol=0,
            ), (
                f"min_vals mismatch: max diff = "
                f"{np.max(np.abs(original_builder.min_vals - loaded_builder.min_vals))}"
            )

            # Assert max_vals match within 1e-15
            assert np.allclose(
                original_builder.max_vals,
                loaded_builder.max_vals,
                atol=1e-15,
                rtol=0,
            ), (
                f"max_vals mismatch: max diff = "
                f"{np.max(np.abs(original_builder.max_vals - loaded_builder.max_vals))}"
            )


# ---------------------------------------------------------------------------
# Property 6: NaN handling produces clean feature vectors
# ---------------------------------------------------------------------------


class TestProperty6NaNHandling:
    """
    Property 6: NaN handling produces clean feature vectors.

    For any MarketState where some Technical_Indicator values are NaN, the
    FeatureVectorBuilder SHALL produce a tensor containing no NaN or Inf values,
    applying forward-fill along the time axis then zero-fill for remaining gaps.

    **Validates: Requirements 2.7, 8.2**
    """

    @given(
        params=normalization_params_strategy(),
        nan_fraction=st.floats(min_value=0.01, max_value=0.8),
    )
    @settings(max_examples=15)
    def test_nan_input_produces_clean_output(self, params, nan_fraction):
        """
        Even when a fraction of indicator values are NaN, build() output
        contains no NaN or Inf.

        **Validates: Requirements 2.7, 8.2**
        """
        min_vals = params["min_vals"]
        max_vals = params["max_vals"]

        builder = FeatureVectorBuilder(min_vals=min_vals, max_vals=max_vals)

        lookback = 30

        # Generate valid OHLCV (no NaN in price data)
        ohlcv = np.zeros((lookback, 5))
        for col in range(5):
            ohlcv[:, col] = np.random.uniform(min_vals[col], max_vals[col], lookback)

        # Generate indicators with random NaN values injected
        indicators = np.zeros((lookback, NUM_INDICATORS))
        for col in range(NUM_INDICATORS):
            feat_idx = col + 5
            indicators[:, col] = np.random.uniform(
                min_vals[feat_idx], max_vals[feat_idx], lookback
            )

        # Inject NaN at random positions
        nan_mask = np.random.random((lookback, NUM_INDICATORS)) < nan_fraction
        indicators[nan_mask] = np.nan

        state = MarketState(
            symbol="MWG",
            timestamp=pd.Timestamp("2024-01-15"),
            ohlcv=ohlcv,
            indicators=indicators,
            lookback=lookback,
        )

        result = builder.build(state)

        # Assert no NaN or Inf in output
        assert not np.any(np.isnan(result)), (
            f"Found NaN values in output with nan_fraction={nan_fraction:.2f}"
        )
        assert not np.any(np.isinf(result)), (
            f"Found Inf values in output with nan_fraction={nan_fraction:.2f}"
        )
        # Output should still be in [0, 1]
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    @given(params=normalization_params_strategy())
    @settings(max_examples=15)
    def test_all_nan_column_produces_clean_output(self, params):
        """
        When an entire indicator column is NaN (e.g., PSAR_up/PSAR_down),
        build() still produces no NaN or Inf values (zero-fill applies).

        **Validates: Requirements 2.7, 8.2**
        """
        min_vals = params["min_vals"]
        max_vals = params["max_vals"]

        builder = FeatureVectorBuilder(min_vals=min_vals, max_vals=max_vals)

        lookback = 25

        # Valid OHLCV
        ohlcv = np.zeros((lookback, 5))
        for col in range(5):
            ohlcv[:, col] = np.random.uniform(min_vals[col], max_vals[col], lookback)

        # Indicators: make some entire columns NaN
        indicators = np.zeros((lookback, NUM_INDICATORS))
        for col in range(NUM_INDICATORS):
            feat_idx = col + 5
            indicators[:, col] = np.random.uniform(
                min_vals[feat_idx], max_vals[feat_idx], lookback
            )

        # Set entire columns to NaN (mimics PSAR_up/PSAR_down behavior)
        num_nan_cols = min(5, NUM_INDICATORS)
        nan_col_indices = np.random.choice(
            NUM_INDICATORS, size=num_nan_cols, replace=False
        )
        for col_idx in nan_col_indices:
            indicators[:, col_idx] = np.nan

        state = MarketState(
            symbol="TCB",
            timestamp=pd.Timestamp("2024-01-15"),
            ohlcv=ohlcv,
            indicators=indicators,
            lookback=lookback,
        )

        result = builder.build(state)

        # Assert clean output
        assert not np.any(np.isnan(result)), "Found NaN in output with all-NaN columns"
        assert not np.any(np.isinf(result)), "Found Inf in output with all-NaN columns"
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    @given(params=normalization_params_strategy())
    @settings(max_examples=15)
    def test_nan_at_start_handled_by_forward_fill_then_zero(self, params):
        """
        NaN at the beginning of a column (before any valid value) gets
        zero-filled since forward-fill has no preceding value.

        **Validates: Requirements 2.7, 8.2**
        """
        min_vals = params["min_vals"]
        max_vals = params["max_vals"]

        builder = FeatureVectorBuilder(min_vals=min_vals, max_vals=max_vals)

        lookback = 20

        # Valid OHLCV
        ohlcv = np.zeros((lookback, 5))
        for col in range(5):
            ohlcv[:, col] = np.random.uniform(min_vals[col], max_vals[col], lookback)

        # Indicators: first several rows are NaN, rest are valid
        indicators = np.zeros((lookback, NUM_INDICATORS))
        for col in range(NUM_INDICATORS):
            feat_idx = col + 5
            indicators[:, col] = np.random.uniform(
                min_vals[feat_idx], max_vals[feat_idx], lookback
            )

        # Make first 10 rows NaN for a few columns
        indicators[:10, 0] = np.nan
        indicators[:10, 1] = np.nan
        indicators[:10, 2] = np.nan

        state = MarketState(
            symbol="ACB",
            timestamp=pd.Timestamp("2024-01-15"),
            ohlcv=ohlcv,
            indicators=indicators,
            lookback=lookback,
        )

        result = builder.build(state)

        # Assert clean output
        assert not np.any(np.isnan(result)), "Found NaN from start-of-column NaN values"
        assert not np.any(np.isinf(result)), "Found Inf from start-of-column NaN values"
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
