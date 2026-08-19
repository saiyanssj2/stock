"""
MarketState dataclass and construction logic.

Provides the core representation of market conditions at a given point in time,
including OHLCV price data and computed technical indicators over a lookback window.
Supports JSON-based serialization/deserialization for caching.
"""

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from engine.config import ConfigError, DataError, EngineConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed ordering of indicator columns from analysis.add_indicators()
INDICATOR_COLUMNS = [
    # Trend (22)
    "EMA_9", "EMA_20", "EMA_50", "EMA_200",
    "SMA_20", "SMA_50",
    "MACD", "MACD_signal", "MACD_hist",
    "ADX", "ADX_pos", "ADX_neg",
    "Aroon_up", "Aroon_down",
    "CCI_20",
    "PSAR", "PSAR_up", "PSAR_down",
    "Ichimoku_conv", "Ichimoku_base", "Ichimoku_a", "Ichimoku_b",
    # Momentum (11)
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
    # Wyckoff / Volume-Price Analysis (5)
    "WK_effort_result", "WK_vol_climax", "WK_spread_pos", "WK_spring", "WK_obv_slope",
    # Value Investing Proxies (6)
    "VAL_mean_reversion", "VAL_52w_position", "VAL_drawdown",
    "VAL_recovery_ratio", "VAL_vol_contraction", "VAL_smart_accumulation",
    # Market Context — VNINDEX + VN30 (6)
    "MKT_vni_ret5", "MKT_vni_vol_ratio", "MKT_vs_vni",
    "MKT_vn30_ret5", "MKT_vn30_vol_ratio", "MKT_vs_vn30",
]

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
NUM_OHLCV = 5
NUM_INDICATORS = len(INDICATOR_COLUMNS)  # 73 (56 cũ + 5 Wyckoff + 6 Value + 6 Market Context)


# ---------------------------------------------------------------------------
# MarketState Dataclass
# ---------------------------------------------------------------------------


@dataclass
class MarketState:
    """
    Represents the market state at a given timestamp for a specific symbol.

    Attributes
    ----------
    symbol : str
        Stock ticker symbol (e.g., "VNM", "FPT").
    timestamp : pd.Timestamp
        The timestamp of the most recent data point in the lookback window.
    ohlcv : np.ndarray
        OHLCV price/volume data, shape (lookback, 5).
        Column order: open, high, low, close, volume.
    indicators : np.ndarray
        Technical indicator values, shape (lookback, num_indicators).
        Column order matches INDICATOR_COLUMNS.
    lookback : int
        Number of historical trading sessions included (range [20, 200]).
    """

    symbol: str
    timestamp: pd.Timestamp
    ohlcv: np.ndarray  # shape: (lookback, 5)
    indicators: np.ndarray  # shape: (lookback, num_indicators)
    lookback: int

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        symbol: str,
        lookback: int = 60,
        config: Optional[EngineConfig] = None,
    ) -> "MarketState":
        """
        Construct a MarketState from a pandas DataFrame.

        The DataFrame should contain OHLCV columns. Technical indicators will
        be computed using analysis.add_indicators(). The last `lookback` rows
        of the resulting data are used.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least columns: open, high, low, close, volume.
            May also contain a 'time' column for timestamps.
        symbol : str
            Stock ticker symbol.
        lookback : int
            Number of historical sessions to include (default 60, range [20, 200]).
        config : EngineConfig, optional
            Engine configuration. Uses defaults if not provided.

        Returns
        -------
        MarketState
            Constructed market state.

        Raises
        ------
        ConfigError
            If lookback is outside [20, 200].
        DataError
            If the DataFrame has insufficient rows or missing required columns.
        """
        if config is None:
            config = EngineConfig()

        # Validate lookback range
        if lookback < config.lookback_min or lookback > config.lookback_max:
            raise ConfigError(
                f"Lookback must be between {config.lookback_min} and "
                f"{config.lookback_max}, got {lookback}",
                error_code="INVALID_LOOKBACK",
                details={
                    "lookback": lookback,
                    "min": config.lookback_min,
                    "max": config.lookback_max,
                },
            )

        # Validate required columns
        missing_cols = [col for col in OHLCV_COLUMNS if col not in df.columns]
        if missing_cols:
            raise DataError(
                f"DataFrame missing required columns: {missing_cols}",
                error_code="MISSING_COLUMNS",
                details={"missing_columns": missing_cols},
            )

        # Compute indicators using analysis.add_indicators()
        from analysis import add_indicators

        df_with_indicators = add_indicators(df)

        # Validate sufficient rows after indicator computation
        if len(df_with_indicators) < lookback:
            raise DataError(
                f"Insufficient data rows: need at least {lookback}, "
                f"got {len(df_with_indicators)}",
                error_code="INSUFFICIENT_DATA",
                details={"required": lookback, "available": len(df_with_indicators)},
            )

        # Extract the last `lookback` rows
        tail = df_with_indicators.iloc[-lookback:]

        # Extract OHLCV data
        ohlcv = tail[OHLCV_COLUMNS].values.astype(np.float64)

        # Extract indicator values (in fixed column order)
        # Some indicators may not exist if computation requires more data
        indicator_data = np.full((lookback, NUM_INDICATORS), np.nan, dtype=np.float64)
        for i, col in enumerate(INDICATOR_COLUMNS):
            if col in tail.columns:
                indicator_data[:, i] = tail[col].values.astype(np.float64)

        # Determine timestamp
        if "time" in tail.columns:
            timestamp = pd.Timestamp(tail["time"].iloc[-1])
        elif isinstance(tail.index, pd.DatetimeIndex):
            timestamp = pd.Timestamp(tail.index[-1])
        else:
            timestamp = pd.Timestamp.now()

        return cls(
            symbol=symbol,
            timestamp=timestamp,
            ohlcv=ohlcv,
            indicators=indicator_data,
            lookback=lookback,
        )

    def serialize(self) -> bytes:
        """
        Serialize the MarketState to bytes (JSON-based) for caching.

        Numpy arrays are converted to nested lists for JSON compatibility.
        NaN values are preserved as null in JSON.

        Returns
        -------
        bytes
            JSON-encoded representation of the MarketState.
        """
        data = {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "ohlcv": _ndarray_to_json_safe(self.ohlcv),
            "indicators": _ndarray_to_json_safe(self.indicators),
            "lookback": self.lookback,
        }
        return json.dumps(data, ensure_ascii=False).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "MarketState":
        """
        Deserialize a MarketState from bytes.

        Parameters
        ----------
        data : bytes
            JSON-encoded MarketState (produced by serialize()).

        Returns
        -------
        MarketState
            Reconstructed market state.

        Raises
        ------
        DataError
            If the data cannot be parsed or is malformed.
        """
        try:
            payload = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise DataError(
                f"Failed to deserialize MarketState: {e}",
                error_code="DESERIALIZE_ERROR",
                details={"error": str(e)},
            )

        try:
            ohlcv = _json_safe_to_ndarray(payload["ohlcv"])
            indicators = _json_safe_to_ndarray(payload["indicators"])

            return cls(
                symbol=payload["symbol"],
                timestamp=pd.Timestamp(payload["timestamp"]),
                ohlcv=ohlcv,
                indicators=indicators,
                lookback=payload["lookback"],
            )
        except (KeyError, TypeError, ValueError) as e:
            raise DataError(
                f"Malformed MarketState data: {e}",
                error_code="DESERIALIZE_ERROR",
                details={"error": str(e)},
            )


# ---------------------------------------------------------------------------
# JSON serialization helpers for numpy arrays with NaN support
# ---------------------------------------------------------------------------


def _ndarray_to_json_safe(arr: np.ndarray) -> list:
    """
    Convert a numpy array to a JSON-safe nested list.

    NaN values are converted to None (JSON null) to preserve them
    through the round-trip.
    """
    result = []
    for row in arr:
        json_row = []
        for val in row:
            if np.isnan(val):
                json_row.append(None)
            else:
                json_row.append(float(val))
        result.append(json_row)
    return result


def _json_safe_to_ndarray(data: list) -> np.ndarray:
    """
    Convert a JSON-safe nested list back to a numpy array.

    None values are converted back to NaN.
    """
    result = []
    for row in data:
        numpy_row = []
        for val in row:
            if val is None:
                numpy_row.append(np.nan)
            else:
                numpy_row.append(float(val))
        result.append(numpy_row)
    return np.array(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# Feature Vector Constants
# ---------------------------------------------------------------------------

ALL_FEATURE_COLUMNS = OHLCV_COLUMNS + INDICATOR_COLUMNS
NUM_FEATURES = len(ALL_FEATURE_COLUMNS)  # 5 OHLCV + 73 indicators = 78


# ---------------------------------------------------------------------------
# FeatureVectorBuilder
# ---------------------------------------------------------------------------


class FeatureVectorBuilder:
    """
    Converts MarketState into model-ready feature arrays with min-max normalization.

    The builder maintains per-feature min/max values computed from training data.
    It handles NaN values via forward-fill then zero-fill, and normalizes all
    features to [0, 1].

    For degenerate features where min == max, the normalized output is 0.5.

    Attributes
    ----------
    min_vals : np.ndarray
        Per-feature minimum values, shape (num_features,).
    max_vals : np.ndarray
        Per-feature maximum values, shape (num_features,).
    """

    # Class-level constants
    INDICATOR_COLUMNS = INDICATOR_COLUMNS
    OHLCV_COLUMNS = OHLCV_COLUMNS
    ALL_FEATURE_COLUMNS = ALL_FEATURE_COLUMNS
    NUM_FEATURES = NUM_FEATURES

    def __init__(
        self,
        min_vals: Optional[np.ndarray] = None,
        max_vals: Optional[np.ndarray] = None,
        norm_params_path: Optional[str] = None,
    ):
        """
        Initialize FeatureVectorBuilder.

        Parameters
        ----------
        min_vals : np.ndarray or None
            Per-feature minimums of shape (num_features,). If None and
            norm_params_path is given, loads from the JSON file.
        max_vals : np.ndarray or None
            Per-feature maximums of shape (num_features,).
        norm_params_path : str or None
            Path to a JSON file containing normalization parameters.
            If provided and min_vals/max_vals are None, loads from file.

        Raises
        ------
        DataError
            If norm_params_path is given but the file is missing or invalid.
        """
        if min_vals is not None and max_vals is not None:
            self.min_vals = np.asarray(min_vals, dtype=np.float64)
            self.max_vals = np.asarray(max_vals, dtype=np.float64)
        elif norm_params_path is not None:
            self._load_params(norm_params_path)
        else:
            # Default: identity normalization (no-op)
            self.min_vals = np.zeros(NUM_FEATURES, dtype=np.float64)
            self.max_vals = np.ones(NUM_FEATURES, dtype=np.float64)

    def build(self, state: "MarketState") -> np.ndarray:
        """
        Build normalized feature array from a MarketState.

        Steps:
        1. Concatenate OHLCV and indicators into a single feature matrix.
        2. Forward-fill NaN along the time axis (axis=0) for each feature.
        3. Zero-fill any remaining NaN values.
        4. Apply min-max normalization to [0, 1].

        For degenerate features (min == max), output is 0.5.

        Parameters
        ----------
        state : MarketState
            The market state to convert.

        Returns
        -------
        np.ndarray
            Normalized feature array of shape (lookback, num_features)
            with all values in [0, 1].
        """
        # Step 1: Concatenate OHLCV + indicators
        if state.indicators.shape[1] > 0:
            raw = np.concatenate([state.ohlcv, state.indicators], axis=1)
        else:
            raw = state.ohlcv.copy()

        # Ensure correct number of features (pad or trim)
        current_cols = raw.shape[1]
        if current_cols < NUM_FEATURES:
            padding = np.full(
                (raw.shape[0], NUM_FEATURES - current_cols), np.nan, dtype=np.float64
            )
            raw = np.concatenate([raw, padding], axis=1)
        elif current_cols > NUM_FEATURES:
            raw = raw[:, :NUM_FEATURES]

        raw = raw.astype(np.float64)

        # Step 2: Forward-fill NaN along time axis (axis=0)
        raw = self._forward_fill(raw)

        # Step 3: Zero-fill remaining NaN / Inf
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 4: Min-max normalize to [0, 1]
        normalized = self._normalize(raw)

        return normalized

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """
        Inverse normalization: convert [0, 1] values back to original scale.

        Parameters
        ----------
        normalized : np.ndarray
            Normalized array of shape (..., num_features) with values in [0, 1].

        Returns
        -------
        np.ndarray
            Denormalized array in original value scale.
        """
        range_vals = self.max_vals - self.min_vals
        return normalized * range_vals + self.min_vals

    def save_params(self, path: str) -> None:
        """
        Save normalization parameters (min_vals, max_vals) to a JSON file.

        Parameters
        ----------
        path : str
            File path for the JSON output. Parent directories are created
            if they don't exist.
        """
        from pathlib import Path

        params = {
            "min_vals": self.min_vals.tolist(),
            "max_vals": self.max_vals.tolist(),
            "num_features": NUM_FEATURES,
            "feature_columns": ALL_FEATURE_COLUMNS,
        }
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

    @classmethod
    def load_params(cls, path: str) -> "FeatureVectorBuilder":
        """
        Load normalization parameters from a JSON file and return a builder.

        Parameters
        ----------
        path : str
            Path to the JSON normalization parameters file.

        Returns
        -------
        FeatureVectorBuilder
            A builder initialized with the loaded parameters.

        Raises
        ------
        DataError
            If the file is missing or contains invalid data.
        """
        return cls(norm_params_path=path)

    def _load_params(self, path: str) -> None:
        """
        Load normalization parameters from a JSON file into this builder.

        Parameters
        ----------
        path : str
            Path to the JSON normalization parameters file.

        Raises
        ------
        DataError
            If the file is missing or contains invalid data.
        """
        from pathlib import Path

        filepath = Path(path)
        if not filepath.exists():
            raise DataError(
                f"Normalization parameter file not found: {path}",
                error_code="NORM_PARAMS_MISSING",
                details={"path": path},
            )
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                params = json.load(f)
            self.min_vals = np.array(params["min_vals"], dtype=np.float64)
            self.max_vals = np.array(params["max_vals"], dtype=np.float64)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise DataError(
                f"Failed to load normalization parameters from {path}: {e}",
                error_code="NORM_PARAMS_INVALID",
                details={"path": path, "error": str(e)},
            )

    @classmethod
    def from_training_data(cls, dataframes: list) -> "FeatureVectorBuilder":
        """
        Compute normalization parameters (min/max) from training DataFrames.

        Each DataFrame should already have indicators computed via
        `analysis.add_indicators()`. This method computes per-feature
        global min and max across all provided DataFrames.

        Parameters
        ----------
        dataframes : list of pd.DataFrame
            Training DataFrames, each with OHLCV and indicator columns.

        Returns
        -------
        FeatureVectorBuilder
            A builder with min/max computed from the training data.

        Raises
        ------
        DataError
            If no valid DataFrames are provided.
        """
        if not dataframes:
            raise DataError(
                "No DataFrames provided to compute normalization parameters",
                error_code="NO_TRAINING_DATA",
            )

        global_min = np.full(NUM_FEATURES, np.inf, dtype=np.float64)
        global_max = np.full(NUM_FEATURES, -np.inf, dtype=np.float64)

        valid_count = 0
        for df in dataframes:
            features = cls._extract_features(df)
            if features is None:
                continue

            valid_count += 1

            # Forward-fill before computing stats
            features = _forward_fill_array(features)

            # Compute per-feature min/max ignoring NaN
            with np.errstate(invalid="ignore"):
                df_min = np.nanmin(features, axis=0)
                df_max = np.nanmax(features, axis=0)

            global_min = np.minimum(global_min, df_min)
            global_max = np.maximum(global_max, df_max)

        if valid_count == 0:
            raise DataError(
                "No valid DataFrames with required OHLCV columns found",
                error_code="NO_VALID_TRAINING_DATA",
            )

        # Handle features with no valid data (still inf/-inf)
        no_data_mask = np.isinf(global_min) | np.isinf(global_max)
        global_min[no_data_mask] = 0.0
        global_max[no_data_mask] = 1.0

        return cls(min_vals=global_min, max_vals=global_max)

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        """
        Apply min-max normalization to [0, 1].

        For degenerate features where min == max, output 0.5.

        Parameters
        ----------
        raw : np.ndarray
            Raw feature values, shape (lookback, num_features).

        Returns
        -------
        np.ndarray
            Normalized values clipped to [0, 1].
        """
        range_vals = self.max_vals - self.min_vals

        # Identify degenerate features (range == 0)
        degenerate_mask = range_vals == 0.0

        # Avoid division by zero: use 1.0 for degenerate ranges
        safe_range = np.where(degenerate_mask, 1.0, range_vals)

        normalized = (raw - self.min_vals) / safe_range

        # Set degenerate features to 0.5
        normalized[..., degenerate_mask] = 0.5

        # Clip to [0, 1] for values outside training range
        normalized = np.clip(normalized, 0.0, 1.0)

        return normalized

    @staticmethod
    def _forward_fill(arr: np.ndarray) -> np.ndarray:
        """
        Forward-fill NaN values along the time axis (axis=0).

        For each feature column, propagates the last valid value forward
        to fill subsequent NaN entries.
        """
        return _forward_fill_array(arr)

    @classmethod
    def _extract_features(cls, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract feature matrix from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV and optionally indicator columns.

        Returns
        -------
        np.ndarray or None
            Feature matrix of shape (rows, NUM_FEATURES), or None if
            the DataFrame is missing required OHLCV columns.
        """
        # Check for required OHLCV columns
        missing_ohlcv = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing_ohlcv:
            return None

        # Extract OHLCV
        ohlcv = df[OHLCV_COLUMNS].values.astype(np.float64)

        # Extract available indicators in fixed order
        indicator_data = np.full(
            (len(df), NUM_INDICATORS), np.nan, dtype=np.float64
        )
        for i, col in enumerate(INDICATOR_COLUMNS):
            if col in df.columns:
                indicator_data[:, i] = df[col].values.astype(np.float64)

        # Concatenate: OHLCV (5) + indicators (56) = 61 features
        features = np.concatenate([ohlcv, indicator_data], axis=1)

        return features


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def _forward_fill_array(arr: np.ndarray) -> np.ndarray:
    """
    Forward-fill NaN values along axis 0 for each column.

    Parameters
    ----------
    arr : np.ndarray
        2D array of shape (time_steps, features).

    Returns
    -------
    np.ndarray
        Array with NaN forward-filled along axis 0.
    """
    result = arr.copy()
    for col in range(result.shape[1]):
        col_data = result[:, col]
        mask = np.isnan(col_data)
        if not mask.any():
            continue
        # Use pandas Series.ffill() for efficient forward-fill
        result[:, col] = pd.Series(col_data).ffill().values
    return result
