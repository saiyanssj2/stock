"""
Smoke test to verify Hypothesis strategies and test infrastructure work correctly.
"""

import numpy as np
from hypothesis import given, settings

from tests.conftest import (
    market_state_strategy,
    feature_vector_strategy,
    ohlcv_dataframe_strategy,
    normalization_params_strategy,
    trade_sequence_strategy,
    NUM_FEATURES,
    NUM_OHLCV_COLS,
    NUM_INDICATOR_COLS,
    MIN_LOOKBACK,
    MAX_LOOKBACK,
    DEFAULT_LOOKBACK,
)


@given(state=market_state_strategy())
@settings(max_examples=10)
def test_market_state_strategy_generates_valid_data(state):
    """Market state strategy produces well-formed data."""
    assert state["symbol"] is not None
    assert state["timestamp"] is not None
    assert state["ohlcv"].shape == (state["lookback"], NUM_OHLCV_COLS)
    assert state["indicators"].shape == (state["lookback"], NUM_INDICATOR_COLS)
    assert MIN_LOOKBACK <= state["lookback"] <= MAX_LOOKBACK

    # OHLCV invariants: high >= low
    ohlcv = state["ohlcv"]
    assert np.all(ohlcv[:, 1] >= ohlcv[:, 2]), "High must be >= Low"


@given(fv=feature_vector_strategy())
@settings(max_examples=10)
def test_feature_vector_strategy_generates_normalized_data(fv):
    """Feature vector strategy produces data in [0, 1] by default."""
    assert fv.shape == (DEFAULT_LOOKBACK, NUM_FEATURES)
    assert np.all(fv >= 0.0)
    assert np.all(fv <= 1.0)
    assert not np.any(np.isnan(fv))
    assert not np.any(np.isinf(fv))


@given(df=ohlcv_dataframe_strategy())
@settings(max_examples=10)
def test_ohlcv_dataframe_strategy_generates_valid_dataframe(df):
    """OHLCV DataFrame strategy produces correct schema."""
    required_cols = ["time", "open", "high", "low", "close", "volume"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) >= 30
    assert len(df) <= 500


@given(params=normalization_params_strategy())
@settings(max_examples=10)
def test_normalization_params_strategy_generates_valid_params(params):
    """Normalization params strategy produces min < max for all features."""
    min_vals = params["min_vals"]
    max_vals = params["max_vals"]
    assert min_vals.shape == (NUM_FEATURES,)
    assert max_vals.shape == (NUM_FEATURES,)
    assert np.all(max_vals > min_vals), "max_vals must be > min_vals"


@given(trades=trade_sequence_strategy())
@settings(max_examples=10)
def test_trade_sequence_strategy_generates_valid_trades(trades):
    """Trade sequence strategy produces market-rule compliant trades."""
    assert len(trades) >= 1
    for trade in trades:
        # Shares are multiples of 100
        assert trade["shares"] % 100 == 0
        assert trade["shares"] > 0
        # Exit date after entry date
        assert trade["exit_date"] > trade["entry_date"]
        # Prices are positive
        assert trade["entry_price"] > 0
        assert trade["exit_price"] > 0
