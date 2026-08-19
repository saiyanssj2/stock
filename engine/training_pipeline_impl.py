"""
Implementation module for TrainingPipeline heavy methods (full training).

Contains chronological_split_impl, prepare_training_data_impl,
build_feature_matrix_impl, and train_full_impl as module-level functions
that accept the pipeline instance as the first argument.

This module is imported lazily by training_pipeline.py to keep the main
module under 700 lines while preserving all public APIs.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from engine.config import DataError, ModelError
from engine.hardware_profile import HardwareProfile

logger = logging.getLogger(__name__)


def chronological_split_impl(
    pipeline,
    data: np.ndarray,
    labels: np.ndarray,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Implementation of TrainingPipeline._chronological_split."""
    if train_ratio is None:
        train_ratio = pipeline.config.train_split
    if val_ratio is None:
        val_ratio = pipeline.config.val_split

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


def prepare_training_data_impl(
    pipeline,
    symbol_data: Dict[str, "pd.DataFrame"],
    data_dir: Optional[str] = None,
) -> Tuple[List[str], Dict[str, "pd.DataFrame"]]:
    """Implementation of TrainingPipeline.prepare_training_data."""
    import pandas as pd

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
        if pipeline.validate_symbol_data(df, symbol):
            valid_symbols.append(symbol)
            valid_data[symbol] = df

    # VNINDEX must always be included if available
    if "VNINDEX" in symbol_data and "VNINDEX" not in valid_symbols:
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
                "min_sessions_required": pipeline.config.min_sessions_per_symbol,
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


def build_feature_matrix_impl(
    pipeline,
    df: "pd.DataFrame",
    lookback: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Implementation of TrainingPipeline._build_feature_matrix."""
    import pandas as pd
    from engine.market_state import (
        INDICATOR_COLUMNS,
        NUM_FEATURES,
        NUM_INDICATORS,
        OHLCV_COLUMNS,
    )

    if lookback is None:
        lookback = pipeline.model_config.lookback

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
    labels = pipeline._generate_labels(df)

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


def train_full_impl(
    pipeline,
    symbol_data: Dict[str, "pd.DataFrame"],
    data_dir: Optional[str] = None,
    resume: bool = True,
):
    """Implementation of TrainingPipeline.train_full. Requirements: 6.4-6.8, 13.1."""
    from engine.evaluation_model import ModelManager, StockEvalNet
    from engine.market_state import FeatureVectorBuilder
    from engine.training_pipeline import TrainingResult
    from engine.training_pipeline_incr import save_complete_model_impl

    import pandas as pd

    start_time = time.time()

    # Step 1: Validate and prepare data
    valid_symbols, valid_data = pipeline.prepare_training_data(symbol_data, data_dir)

    # Step 2: Compute normalization parameters from all training data
    logger.info("Computing normalization parameters from training data...")
    feature_builder = FeatureVectorBuilder.from_training_data(list(valid_data.values()))

    # Handle any NaN in normalization parameters (columns with all-NaN data)
    nan_mask = np.isnan(feature_builder.min_vals) | np.isnan(feature_builder.max_vals)
    feature_builder.min_vals[nan_mask] = 0.0
    feature_builder.max_vals[nan_mask] = 1.0

    # Save normalization params
    norm_params_path = str(Path(pipeline.config.checkpoint_dir) / "norm_params.json")
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
        features, labels = pipeline._build_feature_matrix(df)

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
        pipeline._chronological_split(combined_features, combined_labels)

    logger.info(
        f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
    )

    # Step 5: Set up model, optimizer, and loss
    device = pipeline._hardware_profile.get_device()
    logger.info(f"Training on device: {device} ({pipeline._hardware_profile})")

    model = StockEvalNet(pipeline.model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=pipeline.config.learning_rate,
        weight_decay=pipeline.config.weight_decay,
    )
    criterion = nn.MSELoss()

    # LR Scheduler: giảm LR khi val_loss không cải thiện
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # Auto-adjust batch size based on detected hardware (Req 15.5)
    batch_size = pipeline._hardware_profile.get_optimal_batch_size(mode="training")
    logger.info(f"Auto-adjusted batch size for training: {batch_size}")

    # Resume from checkpoint if available
    start_epoch = 0
    best_val_loss = float("inf")
    resumed_from_checkpoint = False

    if resume:
        latest_cp = pipeline.checkpoint_manager.get_latest_checkpoint_path()
        if latest_cp is not None:
            try:
                cp_info = pipeline.checkpoint_manager.load(
                    model, optimizer, latest_cp, target_device=device
                )
                start_epoch = cp_info["epoch"] + 1
                best_val_loss = cp_info["best_val_loss"]
                model.to(device)
                resumed_from_checkpoint = True
                hw_meta = cp_info.get("hardware_metadata")
                if hw_meta:
                    logger.info(
                        f"Resumed from checkpoint saved on "
                        f"device={hw_meta.get('device_type')}, "
                        f"gpu={hw_meta.get('gpu_model')}"
                    )
                logger.info(
                    f"Resumed from checkpoint: epoch={start_epoch}, "
                    f"best_val_loss={best_val_loss:.6f}, batch_size={batch_size}"
                )
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Step 6: Training loop
    patience_counter = 0
    result = TrainingResult(symbols_trained=valid_symbols)
    graceful_stopped = False

    total_epochs = pipeline.config.max_epochs_full - start_epoch
    if pipeline._training_controller is not None:
        pipeline._training_controller.start_training(total_epochs=total_epochs)

    for epoch in range(start_epoch, pipeline.config.max_epochs_full):
        # Check graceful stop BEFORE beginning a new epoch (Req 14.5)
        if pipeline._training_controller is not None:
            if not pipeline._training_controller.should_continue_training():
                logger.info(
                    f"Graceful stop: not starting epoch {epoch} "
                    "(stop requested after previous epoch)"
                )
                graceful_stopped = True
                break

        if pipeline._training_controller is not None:
            pipeline._training_controller.begin_epoch(epoch + 1)

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
            # Gradient clipping để tránh exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        pipeline.epoch_logger.log_epoch(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            mae=val_mae, learning_rate=current_lr,
            epoch_time_seconds=epoch_time, mode="full",
        )

        # Step LR scheduler dựa trên val_loss
        scheduler.step(val_loss)

        # Save checkpoint after each epoch
        pipeline.checkpoint_manager.save(
            model=model, optimizer=optimizer, epoch=epoch,
            train_loss=train_loss, val_loss=val_loss,
            best_val_loss=best_val_loss, model_config=pipeline.model_config,
            norm_params=norm_params,
        )

        # Notify TrainingController that epoch is complete
        if pipeline._training_controller is not None:
            epoch_metrics = {
                "train_loss": train_loss, "val_loss": val_loss,
                "mae": val_mae, "epoch_time": epoch_time,
            }
            should_continue = pipeline._training_controller._on_epoch_complete(
                epoch=epoch + 1, metrics=epoch_metrics,
            )
            if not should_continue:
                logger.info(
                    f"Graceful stop: epoch {epoch} completed, checkpoint saved, "
                    "training terminated cleanly."
                )
                graceful_stopped = True
                result.epochs_completed = epoch + 1
                result.final_train_loss = train_loss
                result.final_val_loss = val_loss
                result.final_mae = val_mae
                break

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            result.best_val_loss = best_val_loss
            result.best_epoch = epoch
            patience_counter = 0

            best_model_path = str(Path(pipeline.config.checkpoint_dir) / "stock_eval_net.pt")
            save_complete_model_impl(
                pipeline=pipeline, model=model, optimizer=optimizer,
                epoch=epoch, model_config=pipeline.model_config,
                norm_params=norm_params, path=best_model_path,
            )
            result.model_path = best_model_path
        else:
            patience_counter += 1
            if patience_counter >= pipeline.config.early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(patience={pipeline.config.early_stopping_patience})"
                )
                break

        result.epochs_completed = epoch + 1
        result.final_train_loss = train_loss
        result.final_val_loss = val_loss
        result.final_mae = val_mae

    # Cleanup old checkpoints (keep last 3)
    pipeline.checkpoint_manager.cleanup_old_checkpoints(keep_last=3)

    if pipeline._training_controller is not None:
        pipeline._training_controller.end_training()

    result.total_time_seconds = time.time() - start_time

    stop_reason = " (graceful stop)" if graceful_stopped else ""

    # Log training session with HardwareProfile metadata (Req 15.8)
    pipeline.epoch_logger.log_training_session(
        mode="full", hardware_profile=pipeline._hardware_profile,
        symbols=valid_symbols, batch_size=batch_size,
        total_epochs=result.epochs_completed,
        total_time_seconds=result.total_time_seconds,
        final_train_loss=result.final_train_loss,
        final_val_loss=result.final_val_loss,
        resumed_from_checkpoint=resumed_from_checkpoint,
    )

    logger.info(
        f"Full training complete{stop_reason}: {result.epochs_completed} epochs, "
        f"best_val_loss={result.best_val_loss:.6f} at epoch {result.best_epoch}, "
        f"time={result.total_time_seconds:.1f}s"
    )

    return result
