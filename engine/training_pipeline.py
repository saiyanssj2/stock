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
- CheckpointManager: save/resume training state with portable checkpoints
- Hardware-aware batch size adjustment on resume

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.11, 6.12, 6.13, 13.1, 15.3, 15.5, 15.8
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from engine.config import DataError, ModelConfig, ModelError, TrainingConfig
from engine.hardware_profile import HardwareProfile

if TYPE_CHECKING:
    from engine.training_controller import TrainingController

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

    Uses HardwareProfile for portable cross-device checkpoints (all tensors
    saved on CPU with hardware metadata for traceability).

    Requirements: 6.8, 13.1, 15.3, 15.5, 15.8
    """

    def __init__(self, checkpoint_dir: str = "engine/models"):
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory to store checkpoint files.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._hardware_profile = HardwareProfile()

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
        """Save a training checkpoint after an epoch using portable format.

        Returns:
            Path to the saved checkpoint file.

        Requirements: 15.3, 15.5
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        }

        if norm_params is not None:
            metrics["norm_params"] = norm_params

        HardwareProfile.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            hardware=self._hardware_profile,
            path=str(checkpoint_path),
            model_config=model_config,
        )

        logger.info(
            f"Portable checkpoint saved: epoch={epoch}, val_loss={val_loss:.6f}, "
            f"device={self._hardware_profile.device_type}, path={checkpoint_path}"
        )

        return str(checkpoint_path)

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Optional[str] = None,
        target_device: Optional[torch.device] = None,
    ) -> Dict:
        """Load a training checkpoint using portable load and restore state.

        Returns:
            Dict with keys: epoch, train_loss, val_loss, best_val_loss,
            norm_params, hardware_metadata.

        Raises:
            ModelError: If checkpoint file is not found or corrupted.

        Requirements: 15.3
        """
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path is None:
            raise ModelError(
                "No checkpoint found to resume from",
                error_code="CHECKPOINT_NOT_FOUND",
                details={"checkpoint_dir": str(self.checkpoint_dir)},
            )

        if target_device is None:
            target_device = self._hardware_profile.get_device()

        # Use portable load_checkpoint from HardwareProfile
        checkpoint = HardwareProfile.load_checkpoint(
            path=checkpoint_path,
            target_device=target_device,
        )

        # Restore model and optimizer state
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Extract metrics from the portable checkpoint format
        metrics = checkpoint.get("metrics", {})
        hardware_metadata = checkpoint.get("hardware_metadata")

        if hardware_metadata:
            logger.info(
                f"Loaded checkpoint from device={hardware_metadata.get('device_type')}, "
                f"gpu={hardware_metadata.get('gpu_model')}, "
                f"pytorch={hardware_metadata.get('pytorch_version')}"
            )

        return {
            "epoch": checkpoint.get("epoch", 0),
            "train_loss": metrics.get("train_loss", 0.0),
            "val_loss": metrics.get("val_loss", 0.0),
            "best_val_loss": metrics.get("best_val_loss", float("inf")),
            "norm_params": metrics.get("norm_params"),
            "hardware_metadata": hardware_metadata,
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
    Also supports session-level logging with HardwareProfile metadata.

    Requirements: 6.7, 15.8
    """

    def __init__(self, log_file: str = "engine/models/training_log.jsonl"):
        """Initialize EpochLogger.

        Args:
            log_file: Path to the JSONL log file.
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._session_log_file = self.log_file.parent / "training_sessions.jsonl"

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

    def log_training_session(
        self,
        mode: str,
        hardware_profile: "HardwareProfile",
        symbols: List[str],
        batch_size: int,
        total_epochs: int = 0,
        total_time_seconds: float = 0.0,
        final_train_loss: float = 0.0,
        final_val_loss: float = 0.0,
        resumed_from_checkpoint: bool = False,
    ) -> None:
        """Log a training session with HardwareProfile metadata.

        Records device type, GPU model, PyTorch version, and other
        hardware details for each training session, enabling cross-machine
        training history traceability.

        Args:
            mode: Training mode ('full' or 'incremental').
            hardware_profile: HardwareProfile of the current machine.
            symbols: List of symbols trained in this session.
            batch_size: Batch size used (may be auto-adjusted based on hardware).
            total_epochs: Number of epochs completed.
            total_time_seconds: Total wall-clock time for the session.
            final_train_loss: Final training loss.
            final_val_loss: Final validation loss.
            resumed_from_checkpoint: Whether session was resumed from a checkpoint.

        Requirements: 15.8
        """
        session_entry = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "hardware": hardware_profile.to_metadata(),
            "symbols": symbols,
            "batch_size": batch_size,
            "total_epochs": total_epochs,
            "total_time_seconds": float(total_time_seconds),
            "final_train_loss": float(final_train_loss),
            "final_val_loss": float(final_val_loss),
            "resumed_from_checkpoint": resumed_from_checkpoint,
        }

        with open(self._session_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(session_entry, ensure_ascii=False) + "\n")

        hw_meta = hardware_profile.to_metadata()
        logger.info(
            f"Training session logged: mode={mode}, "
            f"device={hw_meta.get('device_type')}, "
            f"gpu={hw_meta.get('gpu_model')}, "
            f"pytorch={hw_meta.get('pytorch_version')}, "
            f"batch_size={batch_size}, epochs={total_epochs}"
        )


class TrainingPipeline:
    """Orchestrates model training from historical data.

    Handles label generation, data splitting, and validation of training data.
    Ensures chronological integrity (no future data leakage) and includes
    VNINDEX market context in all training runs.

    Supports graceful stop via an optional TrainingController. When a controller
    is provided, the training loop checks should_continue_training() before each
    new epoch and saves a checkpoint before exiting on graceful stop.

    Requirements: 6.1-6.13, 13.1, 14.2, 14.3, 14.5, 14.6
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        model_config: Optional[ModelConfig] = None,
        training_controller: Optional["TrainingController"] = None,
    ):
        """Initialize TrainingPipeline.

        Args:
            config: Training configuration. Uses defaults if None.
            model_config: Model architecture configuration. Uses defaults if None.
            training_controller: Optional TrainingController for graceful stop support.
                When provided, training loops check should_continue_training() before
                each epoch and respect stop requests.
        """
        self.config = config or TrainingConfig()
        self.model_config = model_config or ModelConfig()
        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)
        self.epoch_logger = EpochLogger(self.config.log_file)
        self._training_controller = training_controller
        self._hardware_profile = HardwareProfile()

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
    # Chronological Split (thin delegate)
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
        from engine.training_pipeline_impl import chronological_split_impl

        return chronological_split_impl(self, data, labels, train_ratio, val_ratio)

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
        from engine.training_pipeline_impl import prepare_training_data_impl

        return prepare_training_data_impl(self, symbol_data, data_dir)

    # --------------------------------------------------------------------------
    # Feature Extraction for Training (thin delegate)
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
        from engine.training_pipeline_impl import build_feature_matrix_impl

        return build_feature_matrix_impl(self, df, lookback)

    # --------------------------------------------------------------------------
    # Full Training (thin delegate)
    # --------------------------------------------------------------------------

    def train_full(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        data_dir: Optional[str] = None,
        resume: bool = True,
    ) -> "TrainingResult":
        """Train the evaluation model from scratch on VN30 + VNINDEX.

        Full training pipeline:
        1. Validate and prepare data (VNINDEX for market context)
        2. Compute normalization parameters from training data
        3. Build feature matrices with sliding windows
        4. Split chronologically (70/15/15)
        5. Train for max_epochs_full epochs with early stopping
        6. Save checkpoints after each epoch
        7. Log metrics to local file

        Target: Complete within 8 hours on i5-10400F + RTX 2060.

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
        from engine.training_pipeline_impl import train_full_impl

        return train_full_impl(self, symbol_data, data_dir, resume)

    # --------------------------------------------------------------------------
    # Incremental Training (thin delegate)
    # --------------------------------------------------------------------------

    def train_incremental(
        self,
        new_data: Dict[str, pd.DataFrame],
        checkpoint_path: Optional[str] = None,
        max_epochs: Optional[int] = None,
    ) -> "TrainingResult":
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
        from engine.training_pipeline_incr import train_incremental_impl

        return train_incremental_impl(self, new_data, checkpoint_path, max_epochs)

    # --------------------------------------------------------------------------
    # Model Saving (thin delegate)
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
        from engine.training_pipeline_incr import save_complete_model_impl

        save_complete_model_impl(self, model, optimizer, epoch, model_config, norm_params, path)
