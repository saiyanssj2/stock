"""
Training pipeline for the Stock Decision Engine evaluation model.

Orchestrates model training including:
- Label generation from future returns (tanh-scaled)
- Chronological train/val/test splitting (70/15/15)
- Data validation (minimum 250 sessions per symbol)
- VNINDEX inclusion for market context
- Full training from scratch (train_full)
- Incremental fine-tuning with new data (train_incremental)
- Epoch logging (loss, MAE, validation loss) to local file
- CheckpointManager: save/resume training state

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.11, 6.12, 6.13, 13.1
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from engine.config import DataError, ModelConfig, ModelError, TrainingConfig

logger = logging.getLogger(__name__)


# ==============================================================================
# Training Result
# ==============================================================================


@dataclass
class TrainingResult:
    """Result of a training session."""

    epochs_completed: int = 0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    final_mae: float = 0.0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_time_seconds: float = 0.0
    model_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    symbols_trained: List[str] = field(default_factory=list)


# ==============================================================================
# CheckpointManager
# ==============================================================================


class CheckpointManager:
    """Manages training checkpoints: save after each epoch, resume from last.

    Saves complete training state including:
    - Model state_dict (architecture weights)
    - Optimizer state_dict
    - Current epoch number
    - Best validation loss
    - Normalization parameters (min/max values)
    - Training configuration

    Requirements: 6.8, 13.1
    """

    def __init__(self, checkpoint_dir: str = "engine/models"):
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory to store checkpoint files.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        train_loss: float,
        val_loss: float,
        best_val_loss: float,
        model_config: Optional[ModelConfig] = None,
        norm_params: Optional[Dict] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Save a training checkpoint after an epoch.

        Args:
            model: The model being trained.
            optimizer: The optimizer with current state.
            epoch: Current epoch number (0-indexed).
            train_loss: Training loss for this epoch.
            val_loss: Validation loss for this epoch.
            best_val_loss: Best validation loss seen so far.
            model_config: Model architecture configuration.
            norm_params: Normalization parameters (min_vals, max_vals).
            filename: Optional custom filename. Defaults to 'checkpoint_epoch_{epoch}.pt'.

        Returns:
            Path to the saved checkpoint file.
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "saved_at": datetime.now().isoformat(),
        }

        if model_config is not None:
            checkpoint["model_config"] = {
                "num_features": model_config.num_features,
                "lookback": model_config.lookback,
                "tcn_channels": model_config.tcn_channels,
                "kernel_size": model_config.kernel_size,
                "dilations": model_config.dilations,
                "attention_heads": model_config.attention_heads,
                "attention_dim": model_config.attention_dim,
                "dropout": model_config.dropout,
            }

        if norm_params is not None:
            checkpoint["norm_params"] = norm_params

        torch.save(checkpoint, str(checkpoint_path))
        logger.info(f"Checkpoint saved: epoch={epoch}, val_loss={val_loss:.6f}, path={checkpoint_path}")

        return str(checkpoint_path)

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Optional[str] = None,
    ) -> Dict:
        """Load a training checkpoint and restore model/optimizer state.

        Args:
            model: The model to restore weights into.
            optimizer: The optimizer to restore state into.
            checkpoint_path: Path to specific checkpoint file.
                If None, loads the latest checkpoint in checkpoint_dir.

        Returns:
            Dict with keys: epoch, train_loss, val_loss, best_val_loss, norm_params.

        Raises:
            ModelError: If checkpoint file is not found or corrupted.
        """
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path is None:
            raise ModelError(
                "No checkpoint found to resume from",
                error_code="CHECKPOINT_NOT_FOUND",
                details={"checkpoint_dir": str(self.checkpoint_dir)},
            )

        path = Path(checkpoint_path)
        if not path.exists():
            raise ModelError(
                f"Checkpoint file not found: {checkpoint_path}",
                error_code="CHECKPOINT_NOT_FOUND",
                details={"path": checkpoint_path},
            )

        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as e:
            raise ModelError(
                f"Failed to load checkpoint: {e}",
                error_code="CHECKPOINT_CORRUPTED",
                details={"path": checkpoint_path, "error": str(e)},
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return {
            "epoch": checkpoint["epoch"],
            "train_loss": checkpoint.get("train_loss", 0.0),
            "val_loss": checkpoint.get("val_loss", 0.0),
            "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
            "norm_params": checkpoint.get("norm_params"),
        }

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint file in checkpoint_dir.

        Returns:
            Path to the latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None
        return str(checkpoints[-1])

    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Public accessor for the latest checkpoint path.

        Returns:
            Path to the latest checkpoint, or None if no checkpoints exist.
        """
        return self._find_latest_checkpoint()

    def cleanup_old_checkpoints(self, keep_last: int = 3) -> None:
        """Remove old checkpoint files, keeping only the most recent ones.

        Args:
            keep_last: Number of recent checkpoints to keep.
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) <= keep_last:
            return

        for cp in checkpoints[:-keep_last]:
            cp.unlink()
            logger.debug(f"Removed old checkpoint: {cp}")


# ==============================================================================
# Epoch Logger
# ==============================================================================


class EpochLogger:
    """Logs training metrics (loss, MAE, validation loss) to a JSONL file.

    Each line in the log file is a JSON object with epoch metrics.

    Requirements: 6.7
    """

    def __init__(self, log_file: str = "engine/models/training_log.jsonl"):
        """Initialize EpochLogger.

        Args:
            log_file: Path to the JSONL log file.
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        mae: float,
        learning_rate: float = 0.0,
        epoch_time_seconds: float = 0.0,
        mode: str = "full",
    ) -> None:
        """Log metrics for a single epoch.

        Args:
            epoch: Epoch number (0-indexed).
            train_loss: Training loss for this epoch.
            val_loss: Validation loss for this epoch.
            mae: Mean absolute error on validation set.
            learning_rate: Current learning rate.
            epoch_time_seconds: Time taken for this epoch.
            mode: Training mode ('full' or 'incremental').
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "mae": float(mae),
            "learning_rate": float(learning_rate),
            "epoch_time_seconds": float(epoch_time_seconds),
            "mode": mode,
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, mae={mae:.6f}"
        )


class TrainingPipeline:
    """Orchestrates model training from historical data.

    Handles label generation, data splitting, and validation of training data.
    Ensures chronological integrity (no future data leakage) and includes
    VNINDEX market context in all training runs.
    """

    def __init__(self, config: Optional[TrainingConfig] = None, model_config: Optional[ModelConfig] = None):
        """Initialize TrainingPipeline.

        Args:
            config: Training configuration. Uses defaults if None.
            model_config: Model architecture configuration. Uses defaults if None.
        """
        self.config = config or TrainingConfig()
        self.model_config = model_config or ModelConfig()
        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)
        self.epoch_logger = EpochLogger(self.config.log_file)

    # --------------------------------------------------------------------------
    # Label Generation
    # --------------------------------------------------------------------------

    def _generate_labels(
        self,
        df: pd.DataFrame,
        horizon: Optional[int] = None,
        sensitivity: Optional[float] = None,
    ) -> np.ndarray:
        """Compute position score targets from future returns.

        For each row i, computes the percentage return from close[i] to
        close[i + horizon], then maps to [-1, 1] using tanh scaling:

            label[i] = tanh(pct_return * sensitivity)

        Rows where there is insufficient future data (last `horizon` rows)
        receive NaN labels.

        Args:
            df: DataFrame with at least a 'close' column.
            horizon: Number of sessions to look ahead (default from config: 5).
            sensitivity: Tanh scaling factor (default from config: 10.0).

        Returns:
            np.ndarray of shape (len(df),) with labels in [-1.0, +1.0].
            Last `horizon` entries are NaN (insufficient future data).

        Raises:
            DataError: If DataFrame missing 'close' column or has no rows.
        """
        if horizon is None:
            horizon = self.config.label_horizon
        if sensitivity is None:
            sensitivity = self.config.label_sensitivity

        if "close" not in df.columns:
            raise DataError(
                "DataFrame missing required 'close' column for label generation",
                error_code="MISSING_COLUMNS",
                details={"missing_columns": ["close"]},
            )

        if len(df) == 0:
            raise DataError(
                "Empty DataFrame provided for label generation",
                error_code="INSUFFICIENT_DATA",
                details={"rows": 0},
            )

        close_prices = df["close"].values.astype(np.float64)
        n = len(close_prices)
        labels = np.full(n, np.nan, dtype=np.float64)

        # Compute future returns and apply tanh scaling
        for i in range(n - horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + horizon]

            if current_price <= 0 or np.isnan(current_price) or np.isnan(future_price):
                continue

            pct_return = (future_price - current_price) / current_price
            labels[i] = np.tanh(pct_return * sensitivity)

        return labels

    # --------------------------------------------------------------------------
    # Chronological Split
    # --------------------------------------------------------------------------

    def _chronological_split(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """Split data chronologically into train/validation/test sets.

        Ensures no future data leaks into earlier sets. The split is
        purely positional (first 70% train, next 15% val, last 15% test).

        Only rows with valid (non-NaN) labels are included in the output sets.

        Args:
            data: Feature array of shape (n_samples, ...).
            labels: Label array of shape (n_samples,).
            train_ratio: Fraction for training (default from config: 0.70).
            val_ratio: Fraction for validation (default from config: 0.15).

        Returns:
            Tuple of ((train_data, train_labels), (val_data, val_labels),
            (test_data, test_labels)).

        Raises:
            DataError: If data and labels have mismatched lengths or are empty.
        """
        if train_ratio is None:
            train_ratio = self.config.train_split
        if val_ratio is None:
            val_ratio = self.config.val_split

        if len(data) != len(labels):
            raise DataError(
                f"Data and labels length mismatch: {len(data)} vs {len(labels)}",
                error_code="DATA_LABEL_MISMATCH",
                details={"data_len": len(data), "labels_len": len(labels)},
            )

        if len(data) == 0:
            raise DataError(
                "Empty data provided for splitting",
                error_code="INSUFFICIENT_DATA",
                details={"rows": 0},
            )

        # Filter out rows with NaN labels
        valid_mask = ~np.isnan(labels)
        valid_data = data[valid_mask]
        valid_labels = labels[valid_mask]

        if len(valid_data) == 0:
            raise DataError(
                "No valid (non-NaN) labels found in data",
                error_code="INSUFFICIENT_DATA",
                details={"total_rows": len(data), "valid_rows": 0},
            )

        n = len(valid_data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        # Ensure at least 1 sample in each split
        train_end = max(1, train_end)
        val_end = max(train_end + 1, val_end)
        val_end = min(val_end, n - 1)  # Leave at least 1 for test

        train_data = valid_data[:train_end]
        train_labels = valid_labels[:train_end]

        val_data = valid_data[train_end:val_end]
        val_labels = valid_labels[train_end:val_end]

        test_data = valid_data[val_end:]
        test_labels = valid_labels[val_end:]

        return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)

    # --------------------------------------------------------------------------
    # Data Validation & Loading
    # --------------------------------------------------------------------------

    def validate_symbol_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        min_sessions: Optional[int] = None,
    ) -> bool:
        """Validate that a symbol has sufficient data for training.

        Args:
            df: DataFrame with OHLCV data for the symbol.
            symbol: Stock ticker symbol (for logging).
            min_sessions: Minimum required sessions (default from config: 250).

        Returns:
            True if symbol has sufficient data, False otherwise.
            Logs a warning if data is insufficient.
        """
        if min_sessions is None:
            min_sessions = self.config.min_sessions_per_symbol

        if df is None or len(df) < min_sessions:
            actual = 0 if df is None else len(df)
            logger.warning(
                f"Skipping symbol '{symbol}': insufficient data "
                f"({actual} sessions, minimum {min_sessions} required)"
            )
            return False

        return True

    def prepare_training_data(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        data_dir: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
        """Prepare and validate training data for multiple symbols.

        Validates each symbol meets minimum session requirements,
        ensures VNINDEX is always included for market context.

        Args:
            symbol_data: Dict mapping symbol names to their DataFrames.
            data_dir: Optional base directory for loading VNINDEX if not
                already in symbol_data.

        Returns:
            Tuple of (valid_symbols list, valid_data dict).
            Symbols that fail validation are excluded with a warning.

        Raises:
            DataError: If no valid symbols remain after validation, or if
                VNINDEX data is unavailable.
        """
        valid_symbols: List[str] = []
        valid_data: Dict[str, pd.DataFrame] = {}

        # Ensure VNINDEX is present for market context
        if "VNINDEX" not in symbol_data:
            if data_dir is not None:
                vnindex_path = Path(data_dir) / "VNINDEX.csv"
                if vnindex_path.exists():
                    try:
                        vnindex_df = pd.read_csv(vnindex_path)
                        symbol_data["VNINDEX"] = vnindex_df
                    except Exception as e:
                        logger.warning(f"Failed to load VNINDEX from {vnindex_path}: {e}")
                else:
                    logger.warning(
                        f"VNINDEX data file not found at {vnindex_path}. "
                        "Market context will be limited."
                    )

        # Validate each symbol
        for symbol, df in symbol_data.items():
            if self.validate_symbol_data(df, symbol):
                valid_symbols.append(symbol)
                valid_data[symbol] = df

        # VNINDEX must always be included if available
        if "VNINDEX" in symbol_data and "VNINDEX" not in valid_symbols:
            # VNINDEX failed validation - this is a critical warning
            logger.warning(
                "VNINDEX has insufficient data for training. "
                "Market context will not be available."
            )

        if not valid_symbols:
            raise DataError(
                "No symbols with sufficient data for training",
                error_code="NO_VALID_TRAINING_DATA",
                details={
                    "symbols_checked": list(symbol_data.keys()),
                    "min_sessions_required": self.config.min_sessions_per_symbol,
                },
            )

        # Log summary
        skipped = set(symbol_data.keys()) - set(valid_symbols)
        if skipped:
            logger.info(
                f"Training data prepared: {len(valid_symbols)} valid symbols, "
                f"{len(skipped)} skipped ({', '.join(sorted(skipped))})"
            )
        else:
            logger.info(f"Training data prepared: {len(valid_symbols)} valid symbols")

        has_vnindex = "VNINDEX" in valid_symbols
        if has_vnindex:
            logger.info("VNINDEX included for market context")
        else:
            logger.warning("VNINDEX not available - training without market context")

        return valid_symbols, valid_data

    # --------------------------------------------------------------------------
    # Feature Extraction for Training
    # --------------------------------------------------------------------------

    def _build_feature_matrix(
        self,
        df: pd.DataFrame,
        lookback: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build windowed feature matrix and corresponding labels from a DataFrame.

        Creates sliding windows of size `lookback` from the OHLCV + indicator data,
        paired with the label for each window's last timestep.

        Args:
            df: DataFrame with OHLCV columns and computed indicators.
            lookback: Window size (default from model_config).

        Returns:
            Tuple of (features, labels) where:
                features: shape (num_windows, lookback, num_features)
                labels: shape (num_windows,)
        """
        from engine.market_state import (
            INDICATOR_COLUMNS,
            NUM_FEATURES,
            NUM_INDICATORS,
            OHLCV_COLUMNS,
        )

        if lookback is None:
            lookback = self.model_config.lookback

        # Extract OHLCV
        ohlcv_cols = [c for c in OHLCV_COLUMNS if c in df.columns]
        ohlcv = df[ohlcv_cols].values.astype(np.float64) if ohlcv_cols else np.zeros((len(df), 5))

        # Extract indicators in fixed order
        indicator_data = np.full((len(df), NUM_INDICATORS), np.nan, dtype=np.float64)
        for i, col in enumerate(INDICATOR_COLUMNS):
            if col in df.columns:
                indicator_data[:, i] = df[col].values.astype(np.float64)

        # Concatenate: OHLCV + indicators
        raw_features = np.concatenate([ohlcv, indicator_data], axis=1)

        # Pad/trim to NUM_FEATURES
        if raw_features.shape[1] < NUM_FEATURES:
            padding = np.full(
                (raw_features.shape[0], NUM_FEATURES - raw_features.shape[1]),
                np.nan, dtype=np.float64,
            )
            raw_features = np.concatenate([raw_features, padding], axis=1)
        elif raw_features.shape[1] > NUM_FEATURES:
            raw_features = raw_features[:, :NUM_FEATURES]

        # Forward-fill NaN along time axis
        for col in range(raw_features.shape[1]):
            col_data = raw_features[:, col]
            mask = np.isnan(col_data)
            if mask.any():
                raw_features[:, col] = pd.Series(col_data).ffill().values

        # Zero-fill remaining NaN
        raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Generate labels
        labels = self._generate_labels(df)

        # Create sliding windows
        n = len(raw_features)
        num_windows = n - lookback + 1

        if num_windows <= 0:
            return np.array([]).reshape(0, lookback, NUM_FEATURES), np.array([])

        features = np.zeros((num_windows, lookback, NUM_FEATURES), dtype=np.float64)
        window_labels = np.zeros(num_windows, dtype=np.float64)

        for i in range(num_windows):
            features[i] = raw_features[i : i + lookback]
            # Label corresponds to the last timestep of the window
            window_labels[i] = labels[i + lookback - 1]

        return features, window_labels

    # --------------------------------------------------------------------------
    # Full Training
    # --------------------------------------------------------------------------

    def train_full(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        data_dir: Optional[str] = None,
        resume: bool = True,
    ) -> TrainingResult:
        """Train the evaluation model from scratch on VN30 + VNINDEX.

        Full training pipeline:
        1. Validate and prepare data for all symbols
        2. Compute normalization parameters from training data
        3. Build feature matrices with sliding windows
        4. Split chronologically (70/15/15)
        5. Train for max_epochs_full epochs with early stopping
        6. Save checkpoints after each epoch
        7. Log metrics to local file

        Target: Complete within 4 hours on i5-10400F + RTX 2060.

        Args:
            symbol_data: Dict mapping symbol names to DataFrames with OHLCV + indicators.
            data_dir: Optional base directory for loading additional data (VNINDEX).
            resume: If True, attempt to resume from last checkpoint.

        Returns:
            TrainingResult with training metrics and model path.

        Raises:
            DataError: If no valid training data is available.

        Requirements: 6.4, 6.5, 6.6, 6.7, 6.8, 13.1
        """
        from engine.evaluation_model import ModelManager, StockEvalNet
        from engine.market_state import FeatureVectorBuilder

        start_time = time.time()

        # Step 1: Validate and prepare data
        valid_symbols, valid_data = self.prepare_training_data(symbol_data, data_dir)

        # Step 2: Compute normalization parameters from all training data
        logger.info("Computing normalization parameters from training data...")
        feature_builder = FeatureVectorBuilder.from_training_data(list(valid_data.values()))

        # Handle any NaN in normalization parameters (columns with all-NaN data)
        nan_mask = np.isnan(feature_builder.min_vals) | np.isnan(feature_builder.max_vals)
        feature_builder.min_vals[nan_mask] = 0.0
        feature_builder.max_vals[nan_mask] = 1.0

        # Save normalization params
        norm_params_path = str(Path(self.config.checkpoint_dir) / "norm_params.json")
        feature_builder.save_params(norm_params_path)
        norm_params = {
            "min_vals": feature_builder.min_vals.tolist(),
            "max_vals": feature_builder.max_vals.tolist(),
        }

        # Step 3: Build feature matrices for all symbols
        logger.info(f"Building feature matrices for {len(valid_symbols)} symbols...")
        all_features = []
        all_labels = []

        for symbol in valid_symbols:
            df = valid_data[symbol]
            features, labels = self._build_feature_matrix(df)

            if len(features) == 0:
                logger.warning(f"No valid windows for symbol '{symbol}', skipping")
                continue

            # Normalize features using computed params
            for i in range(len(features)):
                features[i] = feature_builder._normalize(features[i])

            all_features.append(features)
            all_labels.append(labels)

        if not all_features:
            raise DataError(
                "No valid training samples could be generated",
                error_code="NO_VALID_TRAINING_DATA",
            )

        # Concatenate all data (maintains chronological order within each symbol)
        combined_features = np.concatenate(all_features, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)

        logger.info(
            f"Total training samples: {len(combined_features)}, "
            f"feature shape: {combined_features.shape}"
        )

        # Step 4: Chronological split
        (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = \
            self._chronological_split(combined_features, combined_labels)

        logger.info(
            f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
        )

        # Step 5: Set up model, optimizer, and loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on device: {device}")

        model = StockEvalNet(self.model_config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.MSELoss()

        # Resume from checkpoint if available
        start_epoch = 0
        best_val_loss = float("inf")

        if resume:
            latest_cp = self.checkpoint_manager.get_latest_checkpoint_path()
            if latest_cp is not None:
                try:
                    cp_info = self.checkpoint_manager.load(model, optimizer, latest_cp)
                    start_epoch = cp_info["epoch"] + 1
                    best_val_loss = cp_info["best_val_loss"]
                    model.to(device)
                    logger.info(f"Resumed from checkpoint: epoch={start_epoch}, best_val_loss={best_val_loss:.6f}")
                except Exception as e:
                    logger.warning(f"Failed to resume from checkpoint: {e}. Starting fresh.")
                    start_epoch = 0
                    best_val_loss = float("inf")

        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.from_numpy(train_data.astype(np.float32)),
            torch.from_numpy(train_labels.astype(np.float32)),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(val_data.astype(np.float32)),
            torch.from_numpy(val_labels.astype(np.float32)),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        # Step 6: Training loop
        patience_counter = 0
        result = TrainingResult(symbols_trained=valid_symbols)

        for epoch in range(start_epoch, self.config.max_epochs_full):
            epoch_start = time.time()

            # Training phase
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device).unsqueeze(1)

                optimizer.zero_grad()
                predictions = model(batch_features)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * len(batch_features)
                train_count += len(batch_features)

            train_loss = train_loss_sum / max(train_count, 1)

            # Validation phase
            model.eval()
            val_loss_sum = 0.0
            val_mae_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device).unsqueeze(1)

                    predictions = model(batch_features)
                    loss = criterion(predictions, batch_labels)
                    mae = torch.mean(torch.abs(predictions - batch_labels))

                    val_loss_sum += loss.item() * len(batch_features)
                    val_mae_sum += mae.item() * len(batch_features)
                    val_count += len(batch_features)

            val_loss = val_loss_sum / max(val_count, 1)
            val_mae = val_mae_sum / max(val_count, 1)

            epoch_time = time.time() - epoch_start

            # Log epoch metrics
            current_lr = optimizer.param_groups[0]["lr"]
            self.epoch_logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                mae=val_mae,
                learning_rate=current_lr,
                epoch_time_seconds=epoch_time,
                mode="full",
            )

            # Save checkpoint after each epoch
            self.checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                model_config=self.model_config,
                norm_params=norm_params,
            )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                result.best_val_loss = best_val_loss
                result.best_epoch = epoch
                patience_counter = 0

                # Save best model
                best_model_path = str(Path(self.config.checkpoint_dir) / "stock_eval_net.pt")
                self._save_complete_model(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    model_config=self.model_config,
                    norm_params=norm_params,
                    path=best_model_path,
                )
                result.model_path = best_model_path
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(patience={self.config.early_stopping_patience})"
                    )
                    break

            # Update result
            result.epochs_completed = epoch + 1
            result.final_train_loss = train_loss
            result.final_val_loss = val_loss
            result.final_mae = val_mae

        # Cleanup old checkpoints (keep last 3)
        self.checkpoint_manager.cleanup_old_checkpoints(keep_last=3)

        result.total_time_seconds = time.time() - start_time
        logger.info(
            f"Full training complete: {result.epochs_completed} epochs, "
            f"best_val_loss={result.best_val_loss:.6f} at epoch {result.best_epoch}, "
            f"time={result.total_time_seconds:.1f}s"
        )

        return result

    # --------------------------------------------------------------------------
    # Incremental Training
    # --------------------------------------------------------------------------

    def train_incremental(
        self,
        new_data: Dict[str, pd.DataFrame],
        checkpoint_path: Optional[str] = None,
        max_epochs: Optional[int] = None,
    ) -> TrainingResult:
        """Incremental fine-tuning with new data.

        Fine-tunes an existing model with new daily data for a limited number
        of epochs. Used for daily updates without full retraining.

        Target: Complete within 10 minutes on i5-10400F + RTX 2060.

        Args:
            new_data: Dict mapping symbol names to DataFrames containing new data.
            checkpoint_path: Path to model checkpoint to fine-tune from.
                If None, uses the latest checkpoint.
            max_epochs: Maximum epochs for fine-tuning (default from config: 10).

        Returns:
            TrainingResult with training metrics.

        Raises:
            DataError: If no valid new data is available.
            ModelError: If no checkpoint is available to resume from.

        Requirements: 6.9, 6.10, 6.11
        """
        from engine.evaluation_model import StockEvalNet
        from engine.market_state import FeatureVectorBuilder

        start_time = time.time()

        if max_epochs is None:
            max_epochs = self.config.max_epochs_incremental

        # Load normalization parameters
        norm_params_path = str(Path(self.config.checkpoint_dir) / "norm_params.json")
        try:
            feature_builder = FeatureVectorBuilder.load_params(norm_params_path)
        except Exception as e:
            raise ModelError(
                f"Cannot load normalization parameters for incremental training: {e}",
                error_code="NORM_PARAMS_MISSING",
                details={"path": norm_params_path, "error": str(e)},
            )

        norm_params = {
            "min_vals": feature_builder.min_vals.tolist(),
            "max_vals": feature_builder.max_vals.tolist(),
        }

        # Build feature matrices from new data
        logger.info(f"Building feature matrices from {len(new_data)} symbols of new data...")
        all_features = []
        all_labels = []
        valid_symbols = []

        for symbol, df in new_data.items():
            if df is None or len(df) < self.model_config.lookback:
                logger.warning(
                    f"Skipping symbol '{symbol}' for incremental training: "
                    f"insufficient data ({0 if df is None else len(df)} rows, "
                    f"need at least {self.model_config.lookback})"
                )
                continue

            features, labels = self._build_feature_matrix(df)
            if len(features) == 0:
                continue

            # Normalize using existing params
            for i in range(len(features)):
                features[i] = feature_builder._normalize(features[i])

            all_features.append(features)
            all_labels.append(labels)
            valid_symbols.append(symbol)

        if not all_features:
            raise DataError(
                "No valid training samples from new data",
                error_code="NO_VALID_TRAINING_DATA",
                details={"symbols_provided": list(new_data.keys())},
            )

        combined_features = np.concatenate(all_features, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)

        # Filter NaN labels
        valid_mask = ~np.isnan(combined_labels)
        combined_features = combined_features[valid_mask]
        combined_labels = combined_labels[valid_mask]

        if len(combined_features) == 0:
            raise DataError(
                "No valid samples after filtering NaN labels",
                error_code="NO_VALID_TRAINING_DATA",
            )

        logger.info(f"Incremental training samples: {len(combined_features)}")

        # Split: use 85% train, 15% val for incremental (no separate test)
        n = len(combined_features)
        split_idx = max(1, int(n * 0.85))
        train_data = combined_features[:split_idx]
        train_labels = combined_labels[:split_idx]
        val_data = combined_features[split_idx:]
        val_labels = combined_labels[split_idx:]

        # Ensure val has at least 1 sample
        if len(val_data) == 0:
            val_data = train_data[-1:]
            val_labels = train_labels[-1:]

        # Set up model and optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StockEvalNet(self.model_config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate * 0.1,  # Lower LR for fine-tuning
            weight_decay=self.config.weight_decay,
        )

        # Load checkpoint
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_manager.get_latest_checkpoint_path()

        if checkpoint_path is None:
            # Try loading the saved best model
            best_model_path = str(Path(self.config.checkpoint_dir) / "stock_eval_net.pt")
            if Path(best_model_path).exists():
                checkpoint_path = best_model_path
            else:
                raise ModelError(
                    "No checkpoint available for incremental training. Run full training first.",
                    error_code="CHECKPOINT_NOT_FOUND",
                )

        # Load model state from checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.to(device)
            logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
        except Exception as e:
            raise ModelError(
                f"Failed to load model for incremental training: {e}",
                error_code="CHECKPOINT_CORRUPTED",
                details={"path": checkpoint_path, "error": str(e)},
            )

        # Create DataLoaders
        criterion = nn.MSELoss()

        train_dataset = TensorDataset(
            torch.from_numpy(train_data.astype(np.float32)),
            torch.from_numpy(train_labels.astype(np.float32)),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(val_data.astype(np.float32)),
            torch.from_numpy(val_labels.astype(np.float32)),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        # Training loop (max_epochs, no early stopping for incremental)
        result = TrainingResult(symbols_trained=valid_symbols)
        best_val_loss = float("inf")

        for epoch in range(max_epochs):
            epoch_start = time.time()

            # Training phase
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device).unsqueeze(1)

                optimizer.zero_grad()
                predictions = model(batch_features)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * len(batch_features)
                train_count += len(batch_features)

            train_loss = train_loss_sum / max(train_count, 1)

            # Validation phase
            model.eval()
            val_loss_sum = 0.0
            val_mae_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device).unsqueeze(1)

                    predictions = model(batch_features)
                    loss = criterion(predictions, batch_labels)
                    mae = torch.mean(torch.abs(predictions - batch_labels))

                    val_loss_sum += loss.item() * len(batch_features)
                    val_mae_sum += mae.item() * len(batch_features)
                    val_count += len(batch_features)

            val_loss = val_loss_sum / max(val_count, 1)
            val_mae = val_mae_sum / max(val_count, 1)

            epoch_time = time.time() - epoch_start

            # Log epoch metrics
            current_lr = optimizer.param_groups[0]["lr"]
            self.epoch_logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                mae=val_mae,
                learning_rate=current_lr,
                epoch_time_seconds=epoch_time,
                mode="incremental",
            )

            # Save checkpoint
            self.checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                model_config=self.model_config,
                norm_params=norm_params,
                filename=f"checkpoint_incremental_epoch_{epoch:04d}.pt",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                result.best_val_loss = best_val_loss
                result.best_epoch = epoch

            # Update result
            result.epochs_completed = epoch + 1
            result.final_train_loss = train_loss
            result.final_val_loss = val_loss
            result.final_mae = val_mae

        # Save final model
        model_path = str(Path(self.config.checkpoint_dir) / "stock_eval_net.pt")
        self._save_complete_model(
            model=model,
            optimizer=optimizer,
            epoch=result.epochs_completed - 1,
            model_config=self.model_config,
            norm_params=norm_params,
            path=model_path,
        )
        result.model_path = model_path
        result.checkpoint_path = checkpoint_path

        result.total_time_seconds = time.time() - start_time
        logger.info(
            f"Incremental training complete: {result.epochs_completed} epochs, "
            f"best_val_loss={result.best_val_loss:.6f}, "
            f"time={result.total_time_seconds:.1f}s"
        )

        return result

    # --------------------------------------------------------------------------
    # Model Saving
    # --------------------------------------------------------------------------

    def _save_complete_model(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        model_config: ModelConfig,
        norm_params: Dict,
        path: str,
    ) -> None:
        """Save complete model state in PyTorch format.

        Saves architecture config, weights, optimizer state, epoch, and
        normalization parameters in a single file.

        Requirements: 6.6, 13.1

        Args:
            model: Trained model.
            optimizer: Optimizer with current state.
            epoch: Current epoch number.
            model_config: Model architecture configuration.
            norm_params: Normalization parameters dict.
            path: Destination file path.
        """
        import hashlib

        state_dict = model.state_dict()

        # Compute checksum for integrity validation
        hasher = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            if isinstance(state_dict[key], torch.Tensor):
                hasher.update(key.encode("utf-8"))
                hasher.update(state_dict[key].cpu().numpy().tobytes())
        checksum = hasher.hexdigest()

        checkpoint = {
            "state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "model_config": {
                "num_features": model_config.num_features,
                "lookback": model_config.lookback,
                "tcn_channels": model_config.tcn_channels,
                "kernel_size": model_config.kernel_size,
                "dilations": model_config.dilations,
                "attention_heads": model_config.attention_heads,
                "attention_dim": model_config.attention_dim,
                "dropout": model_config.dropout,
            },
            "norm_params": norm_params,
            "checksum": checksum,
            "saved_at": datetime.now().isoformat(),
        }

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(filepath))
        logger.info(f"Complete model saved: path={path}, checksum={checksum[:12]}...")
