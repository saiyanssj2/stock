"""
Shared Hypothesis strategies and pytest fixtures for the Stock Decision Engine test suite.

Provides reusable data generators for property-based testing across all test modules.
Strategies generate data compatible with engine/config.py dataclasses:
- ModelConfig (input_channels=63, hidden_channels=[128,128,64], lookback=60)
- Action enum (BUY, HOLD, SELL)
- TrainingConfig, EngineConfig, SearchConfig
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import settings, HealthCheck
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Hypothesis global settings
# ---------------------------------------------------------------------------

settings.register_profile(
    "default",
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci",
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "quick",
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("default")


# ---------------------------------------------------------------------------
# Constants matching engine/config.py
# ---------------------------------------------------------------------------

NUM_FEATURES = 63  # 5 OHLCV + 58 indicators
DEFAULT_LOOKBACK = 60
MIN_LOOKBACK = 20
MAX_LOOKBACK = 200
NUM_OHLCV_COLS = 5  # open, high, low, close, volume
NUM_INDICATOR_COLS = 58


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------


@st.composite
def market_state_strategy(
    draw,
    lookback=None,
    num_indicators=NUM_INDICATOR_COLS,
    allow_nan=False,
):
    """
    Generate a valid MarketState-compatible dictionary with OHLCV data
    and indicator values.

    Parameters
    ----------
    lookback : int or None
        Number of historical sessions. If None, draws from [MIN_LOOKBACK, MAX_LOOKBACK].
    num_indicators : int
        Number of indicator columns (default: 58 matching engine config).
    allow_nan : bool
        If True, some indicator values may be NaN (tests NaN handling).

    Returns
    -------
    dict with keys: symbol, timestamp, ohlcv, indicators, lookback
    """
    if lookback is None:
        lookback = draw(st.integers(min_value=MIN_LOOKBACK, max_value=MAX_LOOKBACK))

    symbol = draw(st.sampled_from([
        "VNM", "VCB", "FPT", "HPG", "MWG", "VHM", "VIC", "MSN",
        "TCB", "ACB", "BID", "CTG", "GAS", "PLX", "SAB", "VRE",
    ]))

    # Use a seed from Hypothesis to generate numpy arrays efficiently
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)

    # Generate realistic OHLCV data via random walk
    base_price = draw(st.floats(min_value=10.0, max_value=200.0))
    returns = rng.uniform(-0.069, 0.069, lookback)  # ±7% VN market limit
    returns[0] = 0.0
    closes = base_price * np.cumprod(1.0 + returns)

    # Derive OHLCV from close prices
    ohlcv = np.zeros((lookback, NUM_OHLCV_COLS))
    spreads = np.abs(closes * 0.02)  # ~2% spread
    ohlcv[:, 3] = closes  # close
    ohlcv[:, 0] = closes + rng.uniform(-1, 1, lookback) * spreads  # open
    ohlcv[:, 1] = np.maximum(ohlcv[:, 0], closes) + rng.uniform(0, 1, lookback) * spreads  # high
    ohlcv[:, 2] = np.minimum(ohlcv[:, 0], closes) - rng.uniform(0, 1, lookback) * spreads  # low
    ohlcv[:, 4] = rng.uniform(10000.0, 50000000.0, lookback)  # volume

    # Ensure high >= max(open, close) and low <= min(open, close)
    ohlcv[:, 1] = np.maximum(ohlcv[:, 1], np.maximum(ohlcv[:, 0], ohlcv[:, 3]))
    ohlcv[:, 2] = np.minimum(ohlcv[:, 2], np.minimum(ohlcv[:, 0], ohlcv[:, 3]))

    # Generate indicator values
    indicators = rng.uniform(-1000.0, 1000.0, (lookback, num_indicators))

    if allow_nan:
        nan_mask = rng.random((lookback, num_indicators)) < 0.1
        indicators[nan_mask] = np.nan

    # Generate timestamp
    end_date = pd.Timestamp("2024-01-15")
    timestamps = pd.bdate_range(end=end_date, periods=lookback)

    return {
        "symbol": symbol,
        "timestamp": timestamps[-1],
        "ohlcv": ohlcv,
        "indicators": indicators,
        "lookback": lookback,
    }


@st.composite
def feature_vector_strategy(
    draw,
    lookback=DEFAULT_LOOKBACK,
    num_features=NUM_FEATURES,
    normalized=True,
):
    """
    Generate a feature vector tensor compatible with the EvaluationModel input.

    Parameters
    ----------
    lookback : int
        Number of time steps (default 60).
    num_features : int
        Number of features per time step (default 63).
    normalized : bool
        If True, values are in [0, 1] (post-normalization).
        If False, values can be any realistic range.

    Returns
    -------
    np.ndarray of shape (lookback, num_features)
    """
    # Use a seed from Hypothesis to generate numpy arrays efficiently
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)

    if normalized:
        data = rng.uniform(0.0, 1.0, (lookback, num_features))
    else:
        data = rng.uniform(-1000.0, 1000.0, (lookback, num_features))

    return data


@st.composite
def ohlcv_dataframe_strategy(
    draw,
    min_rows=30,
    max_rows=500,
    include_indicators=False,
):
    """
    Generate a pandas DataFrame with OHLCV data matching the CSV format
    used by the engine (columns: time, open, high, low, close, volume).

    Parameters
    ----------
    min_rows : int
        Minimum number of rows (trading sessions).
    max_rows : int
        Maximum number of rows.
    include_indicators : bool
        If True, adds indicator-like columns (random values for testing).

    Returns
    -------
    pd.DataFrame with columns matching engine expectations.
    """
    num_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    # Generate realistic price series via random walk
    base_price = draw(st.floats(min_value=10.0, max_value=200.0))
    returns = np.array(draw(
        st.lists(
            st.floats(min_value=-0.069, max_value=0.069),
            min_size=num_rows,
            max_size=num_rows,
        )
    ))
    returns[0] = 0.0  # First return is 0
    closes = base_price * np.cumprod(1.0 + returns)

    # Build OHLCV
    spread_pct = 0.015
    highs = closes * (1.0 + np.abs(np.random.uniform(0, spread_pct, num_rows)))
    lows = closes * (1.0 - np.abs(np.random.uniform(0, spread_pct, num_rows)))
    opens = lows + (highs - lows) * np.random.uniform(0.2, 0.8, num_rows)
    volumes = np.random.uniform(100000, 50000000, num_rows)

    # Generate business day timestamps
    end_date = pd.Timestamp("2024-06-30")
    dates = pd.bdate_range(end=end_date, periods=num_rows)

    df = pd.DataFrame({
        "time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    if include_indicators:
        # Add mock indicator columns matching engine's expected indicators
        indicator_names = _get_indicator_column_names()
        for col_name in indicator_names:
            df[col_name] = np.random.uniform(-100, 100, num_rows)

    return df


@st.composite
def normalization_params_strategy(
    draw,
    num_features=NUM_FEATURES,
):
    """
    Generate normalization parameters (per-feature min and max values)
    compatible with FeatureVectorBuilder.

    Parameters
    ----------
    num_features : int
        Number of features (default 63).

    Returns
    -------
    dict with keys 'min_vals' and 'max_vals', each np.ndarray of shape (num_features,).
    The constraint min_vals[i] < max_vals[i] is enforced for non-degenerate normalization.
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
def trade_sequence_strategy(
    draw,
    min_trades=1,
    max_trades=20,
):
    """
    Generate a sequence of trades compatible with BacktestEngine validation.

    Each trade has: entry_date, exit_date, entry_price, exit_price, shares, pnl, pnl_pct.
    Trades respect Vietnamese market rules:
    - Shares are multiples of 100
    - Price changes within ±7% daily limits
    - T+2.5 settlement (exit >= entry + 3 business days)

    Parameters
    ----------
    min_trades : int
        Minimum number of trades.
    max_trades : int
        Maximum number of trades.

    Returns
    -------
    list of dicts, each representing a Trade.
    """
    num_trades = draw(st.integers(min_value=min_trades, max_value=max_trades))

    trades = []
    current_date = pd.Timestamp("2023-01-05")

    for _ in range(num_trades):
        # Entry date advances by some business days
        days_gap = draw(st.integers(min_value=1, max_value=10))
        entry_date = current_date + pd.tseries.offsets.BDay(days_gap)

        # Exit must be at least 3 business days after entry (T+2.5 rule)
        exit_offset = draw(st.integers(min_value=3, max_value=30))
        exit_date = entry_date + pd.tseries.offsets.BDay(exit_offset)

        # Entry price
        entry_price = draw(st.floats(min_value=10.0, max_value=200.0))

        # Exit price within ±7% of entry (single day limit, simplified)
        max_change = entry_price * 0.07
        price_change = draw(st.floats(min_value=-max_change, max_value=max_change))
        exit_price = entry_price + price_change

        # Shares must be multiple of 100
        lot_count = draw(st.integers(min_value=1, max_value=100))
        shares = lot_count * 100

        # Compute PnL
        pnl = (exit_price - entry_price) * shares
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0

        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": shares,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })

        # Next trade starts after this one exits
        current_date = exit_date

    return trades


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_indicator_column_names():
    """
    Return the list of indicator column names matching the fixed ordering
    from analysis.py (as specified in the design document).
    """
    return [
        # Trend (22)
        "EMA_9", "EMA_20", "EMA_50", "EMA_200",
        "SMA_20", "SMA_50",
        "MACD", "MACD_signal", "MACD_hist",
        "ADX", "ADX_pos", "ADX_neg",
        "Aroon_up", "Aroon_down",
        "CCI_20",
        "PSAR", "PSAR_up", "PSAR_down",
        "Ichimoku_conv", "Ichimoku_base", "Ichimoku_a", "Ichimoku_b",
        # Momentum (11 - as listed in design: RSI_14, RSI_7, STOCH_k, STOCH_d,
        #   StochRSI_k, StochRSI_d, Williams_R, ROC_10, TSI, UO, AO)
        "RSI_14", "RSI_7",
        "STOCH_k", "STOCH_d",
        "StochRSI_k", "StochRSI_d",
        "Williams_R", "ROC_10", "TSI", "UO", "AO",
        # Volatility (14)
        "BB_upper", "BB_middle", "BB_lower", "BB_pband", "BB_wband",
        "ATR_14", "ATR_7",
        "KC_upper", "KC_middle", "KC_lower",
        "DC_upper", "DC_middle", "DC_lower",
        "Ulcer_14",
        # Volume (9)
        "OBV", "MFI", "CMF", "VWAP", "ADI", "FI", "VPT", "EOM", "NVI",
    ]


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ohlcv_df():
    """Provide a small but valid OHLCV DataFrame for unit tests."""
    np.random.seed(42)
    num_rows = 100
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


@pytest.fixture
def sample_normalization_params():
    """Provide sample normalization parameters for unit tests."""
    np.random.seed(42)
    min_vals = np.random.uniform(-100, 0, NUM_FEATURES)
    max_vals = min_vals + np.random.uniform(1, 200, NUM_FEATURES)
    return {
        "min_vals": min_vals,
        "max_vals": max_vals,
    }


@pytest.fixture
def sample_feature_vector():
    """Provide a normalized feature vector for unit tests."""
    np.random.seed(42)
    return np.random.uniform(0, 1, (DEFAULT_LOOKBACK, NUM_FEATURES))
