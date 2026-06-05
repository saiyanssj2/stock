"""
Unit tests for engine/market_state.py - MarketState dataclass.

Tests construction, validation, serialization/deserialization.
"""

import numpy as np
import pandas as pd
import pytest

from engine.config import ConfigError, DataError
from engine.market_state import (
    INDICATOR_COLUMNS,
    NUM_INDICATORS,
    NUM_OHLCV,
    MarketState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv_df(num_rows: int = 250) -> pd.DataFrame:
    """Create a valid OHLCV DataFrame for testing."""
    np.random.seed(42)
    base_price = 50.0
    returns = np.random.normal(0, 0.02, num_rows)
    returns[0] = 0.0
    closes = base_price * np.cumprod(1.0 + returns)
    highs = closes * 1.01
    lows = closes * 0.99
    opens = (highs + lows) / 2
    volumes = np.random.uniform(100000, 5000000, num_rows)
    dates = pd.bdate_range(end="2024-06-30", periods=num_rows)

    return pd.DataFrame({
        "time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


# ---------------------------------------------------------------------------
# Test MarketState construction from DataFrame
# ---------------------------------------------------------------------------


class TestMarketStateFromDataframe:
    """Tests for MarketState.from_dataframe() classmethod."""

    def test_basic_construction(self):
        """MarketState is constructed with correct shapes and values."""
        df = _make_ohlcv_df(250)
        state = MarketState.from_dataframe(df, symbol="VNM", lookback=60)

        assert state.symbol == "VNM"
        assert state.lookback == 60
        assert state.ohlcv.shape == (60, NUM_OHLCV)
        assert state.indicators.shape == (60, NUM_INDICATORS)
        assert isinstance(state.timestamp, pd.Timestamp)

    def test_default_lookback(self):
        """Default lookback is 60."""
        df = _make_ohlcv_df(250)
        state = MarketState.from_dataframe(df, symbol="FPT")

        assert state.lookback == 60

    def test_custom_lookback(self):
        """Custom lookback within valid range succeeds."""
        df = _make_ohlcv_df(250)
        state = MarketState.from_dataframe(df, symbol="HPG", lookback=30)

        assert state.lookback == 30
        assert state.ohlcv.shape == (30, NUM_OHLCV)
        assert state.indicators.shape == (30, NUM_INDICATORS)

    def test_lookback_at_minimum(self):
        """Lookback at minimum boundary (20) succeeds."""
        df = _make_ohlcv_df(250)
        state = MarketState.from_dataframe(df, symbol="VCB", lookback=20)

        assert state.lookback == 20
        assert state.ohlcv.shape == (20, NUM_OHLCV)

    def test_lookback_at_maximum(self):
        """Lookback at maximum boundary (200) succeeds."""
        df = _make_ohlcv_df(250)
        state = MarketState.from_dataframe(df, symbol="VCB", lookback=200)

        assert state.lookback == 200
        assert state.ohlcv.shape == (200, NUM_OHLCV)

    def test_lookback_below_minimum_raises(self):
        """Lookback below 20 raises ConfigError."""
        df = _make_ohlcv_df(250)
        with pytest.raises(ConfigError, match="Lookback must be between"):
            MarketState.from_dataframe(df, symbol="VNM", lookback=19)

    def test_lookback_above_maximum_raises(self):
        """Lookback above 200 raises ConfigError."""
        df = _make_ohlcv_df(250)
        with pytest.raises(ConfigError, match="Lookback must be between"):
            MarketState.from_dataframe(df, symbol="VNM", lookback=201)

    def test_insufficient_data_raises(self):
        """DataFrame with fewer rows than lookback raises DataError."""
        df = _make_ohlcv_df(30)
        with pytest.raises(DataError, match="Insufficient data rows"):
            MarketState.from_dataframe(df, symbol="VNM", lookback=60)

    def test_missing_columns_raises(self):
        """DataFrame missing required columns raises DataError."""
        df = pd.DataFrame({
            "time": pd.bdate_range(end="2024-06-30", periods=100),
            "open": np.ones(100),
            "close": np.ones(100),
            # Missing: high, low, volume
        })
        with pytest.raises(DataError, match="missing required columns"):
            MarketState.from_dataframe(df, symbol="VNM", lookback=20)

    def test_ohlcv_contains_no_nan(self):
        """OHLCV values from a valid DataFrame should not contain NaN."""
        df = _make_ohlcv_df(250)
        state = MarketState.from_dataframe(df, symbol="VNM", lookback=30)

        assert not np.isnan(state.ohlcv).any()

    def test_timestamp_from_time_column(self):
        """Timestamp is extracted from 'time' column if present."""
        df = _make_ohlcv_df(250)
        state = MarketState.from_dataframe(df, symbol="VNM", lookback=60)

        # Should be a valid timestamp matching the last row's time
        assert isinstance(state.timestamp, pd.Timestamp)

    def test_indicators_have_correct_shape(self):
        """Indicators array has shape (lookback, NUM_INDICATORS)."""
        df = _make_ohlcv_df(250)
        state = MarketState.from_dataframe(df, symbol="VNM", lookback=60)

        assert state.indicators.shape == (60, NUM_INDICATORS)
        assert state.indicators.dtype == np.float64


# ---------------------------------------------------------------------------
# Test MarketState serialization round-trip
# ---------------------------------------------------------------------------


class TestMarketStateSerialization:
    """Tests for serialize() and deserialize() methods."""

    def _make_state(self, lookback: int = 60) -> MarketState:
        """Create a MarketState for testing serialization."""
        np.random.seed(123)
        ohlcv = np.random.uniform(10, 200, (lookback, NUM_OHLCV))
        indicators = np.random.uniform(-100, 100, (lookback, NUM_INDICATORS))
        return MarketState(
            symbol="VNM",
            timestamp=pd.Timestamp("2024-06-30"),
            ohlcv=ohlcv,
            indicators=indicators,
            lookback=lookback,
        )

    def test_serialize_returns_bytes(self):
        """serialize() returns bytes."""
        state = self._make_state()
        data = state.serialize()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_deserialize_returns_market_state(self):
        """deserialize() returns a MarketState."""
        state = self._make_state()
        data = state.serialize()
        restored = MarketState.deserialize(data)
        assert isinstance(restored, MarketState)

    def test_round_trip_preserves_symbol(self):
        """Round-trip preserves symbol."""
        state = self._make_state()
        restored = MarketState.deserialize(state.serialize())
        assert restored.symbol == state.symbol

    def test_round_trip_preserves_timestamp(self):
        """Round-trip preserves timestamp."""
        state = self._make_state()
        restored = MarketState.deserialize(state.serialize())
        assert restored.timestamp == state.timestamp

    def test_round_trip_preserves_lookback(self):
        """Round-trip preserves lookback."""
        state = self._make_state()
        restored = MarketState.deserialize(state.serialize())
        assert restored.lookback == state.lookback

    def test_round_trip_preserves_ohlcv(self):
        """Round-trip preserves ohlcv within 1e-9 tolerance."""
        state = self._make_state()
        restored = MarketState.deserialize(state.serialize())
        np.testing.assert_allclose(restored.ohlcv, state.ohlcv, atol=1e-9)

    def test_round_trip_preserves_indicators(self):
        """Round-trip preserves indicators within 1e-9 tolerance."""
        state = self._make_state()
        restored = MarketState.deserialize(state.serialize())
        np.testing.assert_allclose(restored.indicators, state.indicators, atol=1e-9)

    def test_round_trip_preserves_nan_indicators(self):
        """Round-trip preserves NaN values in indicators."""
        state = self._make_state()
        # Introduce some NaN values
        state.indicators[0, 0] = np.nan
        state.indicators[5, 10] = np.nan
        state.indicators[-1, -1] = np.nan

        restored = MarketState.deserialize(state.serialize())

        # NaN positions should remain NaN
        assert np.isnan(restored.indicators[0, 0])
        assert np.isnan(restored.indicators[5, 10])
        assert np.isnan(restored.indicators[-1, -1])

        # Non-NaN positions should be preserved
        non_nan_mask = ~np.isnan(state.indicators)
        np.testing.assert_allclose(
            restored.indicators[non_nan_mask],
            state.indicators[non_nan_mask],
            atol=1e-9,
        )

    def test_deserialize_invalid_bytes_raises(self):
        """Deserializing invalid bytes raises DataError."""
        with pytest.raises(DataError, match="Failed to deserialize"):
            MarketState.deserialize(b"not valid json")

    def test_deserialize_missing_fields_raises(self):
        """Deserializing JSON with missing fields raises DataError."""
        incomplete = json.dumps({"symbol": "VNM"}).encode("utf-8")
        with pytest.raises(DataError, match="Malformed MarketState data"):
            MarketState.deserialize(incomplete)


# Need to import json for the test above
import json  # noqa: E402
