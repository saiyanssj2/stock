"""
Background training implementation details.

Contains the heavy training loop logic extracted from BackgroundTrainingManager
to keep the main module under 700 lines.

This module is an internal implementation detail and should not be imported
directly by external code. All public APIs remain in engine.background_training.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from engine.background_training import BackgroundTrainingManager

logger = logging.getLogger(__name__)


def run_monitored_training(
    manager: "BackgroundTrainingManager",
    pipeline,
    symbol_data: Dict[str, "pd.DataFrame"],
    mode: str,
    data_dir: Optional[str],
) -> Optional["TrainingResult"]:
    """Run training with GPU memory monitoring between epochs.

    Integrates the existing TrainingPipeline with background monitoring:
    - Before each epoch: check GPU memory, pause if needed
    - After each epoch: update status, check stop event
    - After training: hot-swap model

    This wraps the training pipeline's execution with monitoring hooks.

    Args:
        manager: BackgroundTrainingManager instance (provides state/events).
        pipeline: TrainingPipeline instance.
        symbol_data: Training data.
        mode: "full" or "incremental".
        data_dir: Data directory.

    Returns:
        TrainingResult if completed, None if stopped/failed.
    """
    from engine.evaluation_model import StockEvalNet
    from engine.market_state import FeatureVectorBuilder
    from engine.training_pipeline import TrainingResult

    start_time = time.time()

    # Prepare training data using pipeline methods
    try:
        valid_symbols, valid_data = pipeline.prepare_training_data(
            symbol_data, data_dir
        )
    except Exception as e:
        logger.error(f"Failed to prepare training data: {e}")
        return None

    # Compute normalization parameters
    feature_builder = FeatureVectorBuilder.from_training_data(
        list(valid_data.values())
    )

    # Handle NaN in normalization parameters
    nan_mask = np.isnan(feature_builder.min_vals) | np.isnan(feature_builder.max_vals)
    feature_builder.min_vals[nan_mask] = 0.0
    feature_builder.max_vals[nan_mask] = 1.0

    norm_params_path = str(
        Path(pipeline.config.checkpoint_dir) / "norm_params.json"
    )
    feature_builder.save_params(norm_params_path)
    norm_params = {
        "min_vals": feature_builder.min_vals.tolist(),
        "max_vals": feature_builder.max_vals.tolist(),
    }

    # Build feature matrices
    all_features = []
    all_labels = []

    for symbol in valid_symbols:
        df = valid_data[symbol]
        features, labels = pipeline._build_feature_matrix(df)
        if len(features) == 0:
            continue
        for i in range(len(features)):
            features[i] = feature_builder._normalize(features[i])
        all_features.append(features)
        all_labels.append(labels)

    if not all_features:
        logger.error("No valid training samples generated")
        return None

    combined_features = np.concatenate(all_features, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    # Split data
    train_data, train_labels, val_data, val_labels = _split_data(
        pipeline, combined_features, combined_labels, mode
    )

    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockEvalNet(pipeline.model_config).to(device)
    lr = pipeline.config.learning_rate
    if mode == "incremental":
        lr *= 0.1  # Lower LR for fine-tuning

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=pipeline.config.weight_decay
    )
    criterion = nn.MSELoss()

    # LR Scheduler: giảm LR khi val_loss không cải thiện
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # Resume from checkpoint if available
    start_epoch, best_val_loss = _resume_from_checkpoint(
        pipeline, model, optimizer, device, mode
    )

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
        train_dataset, batch_size=pipeline.config.batch_size, shuffle=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=pipeline.config.batch_size, shuffle=False
    )

    # Determine max epochs
    max_epochs = (
        pipeline.config.max_epochs_incremental
        if mode == "incremental"
        else pipeline.config.max_epochs_full
    )

    # Update status with total epochs
    with manager._status_lock:
        manager._status.total_epochs = max_epochs

    # Training loop with monitoring
    result = TrainingResult(symbols_trained=valid_symbols)
    epoch_times: List[float] = []
    train_loss = 0.0
    val_loss = 0.0
    epoch = start_epoch  # Initialize in case loop doesn't execute

    for epoch in range(start_epoch, max_epochs):
        # Check stop event
        if manager._stop_event.is_set():
            logger.info("Training stopped by user request")
            break

        # GPU memory check before epoch
        if manager._gpu_monitor.should_pause_training():
            manager._handle_gpu_pause()

        # Wait if paused
        manager._pause_event.wait()

        # Check stop again after potential pause
        if manager._stop_event.is_set():
            break

        epoch_start = time.time()

        # Training phase
        train_loss, stopped = _run_training_epoch(
            manager, model, train_loader, optimizer, criterion, device
        )
        if stopped:
            break

        # Validation phase
        val_loss, val_mae = _run_validation_epoch(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Log metrics
        pipeline.epoch_logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            mae=val_mae,
            learning_rate=optimizer.param_groups[0]["lr"],
            epoch_time_seconds=epoch_time,
            mode=mode,
        )

        # Step LR scheduler dựa trên val_loss
        scheduler.step(val_loss)

        # Save checkpoint
        pipeline.checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            model_config=pipeline.model_config,
            norm_params=norm_params,
        )

        # Track best model và early stopping
        # Kiểm tra improvement TRƯỚC khi update best_val_loss
        is_improvement = val_loss < best_val_loss

        if is_improvement:
            best_val_loss = val_loss
            result.best_val_loss = best_val_loss
            result.best_epoch = epoch

            # Save best model
            best_model_path = str(
                Path(pipeline.config.checkpoint_dir) / "stock_eval_net.pt"
            )
            pipeline._save_complete_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                model_config=pipeline.model_config,
                norm_params=norm_params,
                path=best_model_path,
            )
            result.model_path = best_model_path

        # Update status (Requirement 7.5: updated every 10s)
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
        remaining_epochs = max_epochs - epoch - 1
        eta = remaining_epochs * avg_epoch_time

        with manager._status_lock:
            manager._status.current_epoch = epoch + 1
            manager._status.total_epochs = max_epochs
            manager._status.current_loss = train_loss
            manager._status.current_val_loss = val_loss
            manager._status.eta_seconds = eta
            manager._status.last_updated = datetime.now()

        # Notify progress tracker of epoch completion
        manager.notify_epoch_complete(epoch + 1, max_epochs)

        # Notify callback
        if manager._on_status_change:
            try:
                manager._on_status_change(manager.status)
            except Exception:
                pass

        # Early stopping (full training only)
        if mode == "full":
            patience_counter = getattr(manager, "_patience_counter", 0)
            if is_improvement:
                manager._patience_counter = 0
            else:
                manager._patience_counter = patience_counter + 1
                if manager._patience_counter >= pipeline.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

    # Update result
    result.epochs_completed = epoch + 1 if not manager._stop_event.is_set() else epoch
    result.final_train_loss = train_loss
    result.final_val_loss = val_loss
    result.total_time_seconds = time.time() - start_time

    # Cleanup old checkpoints
    pipeline.checkpoint_manager.cleanup_old_checkpoints(keep_last=3)

    logger.info(
        f"Background training complete: mode={mode}, "
        f"epochs={result.epochs_completed}, "
        f"best_val_loss={result.best_val_loss:.6f}"
    )

    return result


def _split_data(pipeline, combined_features, combined_labels, mode):
    """Split combined features/labels into train and validation sets.

    Args:
        pipeline: TrainingPipeline instance.
        combined_features: All feature data.
        combined_labels: All label data.
        mode: "full" or "incremental".

    Returns:
        Tuple of (train_data, train_labels, val_data, val_labels).
    """
    if mode == "full":
        (train_data, train_labels), (val_data, val_labels), _ = (
            pipeline._chronological_split(combined_features, combined_labels)
        )
    else:
        # Incremental: 85/15 split
        valid_mask = ~np.isnan(combined_labels)
        combined_features = combined_features[valid_mask]
        combined_labels = combined_labels[valid_mask]
        n = len(combined_features)
        split_idx = max(1, int(n * 0.85))
        train_data = combined_features[:split_idx]
        train_labels = combined_labels[:split_idx]
        val_data = combined_features[split_idx:]
        val_labels = combined_labels[split_idx:]
        if len(val_data) == 0:
            val_data = train_data[-1:]
            val_labels = train_labels[-1:]

    return train_data, train_labels, val_data, val_labels


def _resume_from_checkpoint(pipeline, model, optimizer, device, mode):
    """Resume model from checkpoint if available.

    Args:
        pipeline: TrainingPipeline instance.
        model: The model to load weights into.
        optimizer: The optimizer to restore state for.
        device: Target device.
        mode: "full" or "incremental".

    Returns:
        Tuple of (start_epoch, best_val_loss).
    """
    start_epoch = 0
    best_val_loss = float("inf")

    if mode == "incremental":
        latest_cp = pipeline.checkpoint_manager.get_latest_checkpoint_path()
        if latest_cp:
            try:
                checkpoint = torch.load(
                    latest_cp, map_location="cpu", weights_only=False
                )
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                model.to(device)
                logger.info(f"Resumed from checkpoint: {latest_cp}")
            except Exception as e:
                logger.warning(f"Failed to resume: {e}")
    elif mode == "full":
        latest_cp = pipeline.checkpoint_manager.get_latest_checkpoint_path()
        if latest_cp:
            try:
                cp_info = pipeline.checkpoint_manager.load(
                    model, optimizer, latest_cp
                )
                start_epoch = cp_info["epoch"] + 1
                best_val_loss = cp_info["best_val_loss"]
                model.to(device)
            except Exception as e:
                logger.warning(f"Failed to resume: {e}")

    return start_epoch, best_val_loss


def _run_training_epoch(manager, model, train_loader, optimizer, criterion, device):
    """Run a single training epoch with GPU monitoring.

    Args:
        manager: BackgroundTrainingManager instance.
        model: The neural network model.
        train_loader: DataLoader for training data.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Target device.

    Returns:
        Tuple of (train_loss, stopped) where stopped indicates if
        training was interrupted.
    """
    model.train()
    train_loss_sum = 0.0
    train_count = 0

    for batch_features, batch_labels in train_loader:
        if manager._stop_event.is_set():
            return train_loss_sum / max(train_count, 1), True

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

        # Periodic GPU memory check within epoch
        if manager._gpu_monitor.should_pause_training():
            manager._handle_gpu_pause()
            manager._pause_event.wait()
            if manager._stop_event.is_set():
                return train_loss_sum / max(train_count, 1), True

    return train_loss_sum / max(train_count, 1), False


def _run_validation_epoch(model, val_loader, criterion, device):
    """Run a single validation epoch.

    Args:
        model: The neural network model.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Target device.

    Returns:
        Tuple of (val_loss, val_mae).
    """
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
    return val_loss, val_mae


def run_tracked_training(
    manager: "BackgroundTrainingManager",
    symbol_data: Dict[str, "pd.DataFrame"],
    mode: str,
    data_dir: Optional[str],
) -> None:
    """Run training with per-symbol progress tracking.

    Handles checkpoint resume logic, missing CSV detection, and
    per-symbol iteration with tracker callbacks.

    Args:
        manager: BackgroundTrainingManager instance.
        symbol_data: Symbol DataFrames for training.
        mode: "full" or "incremental".
        data_dir: Base directory for data loading.

    Requirements: 5.4, 5.5, 5.6, 6.5, 6.6
    """
    from engine.resource_limiter import apply_resource_limits
    from engine.session_checkpoint import SessionCheckpointManager
    from engine.symbol_progress_tracker import SymbolProgressTracker

    # Apply resource limits (CPU threads, GPU memory) before training
    apply_resource_limits()

    checkpoint_manager = SessionCheckpointManager()

    # Determine training symbol list
    all_symbols = list(symbol_data.keys())
    training_symbols = all_symbols

    # Handle resume from checkpoint (Requirement 5.5)
    if manager._resume_checkpoint is not None:
        pending_symbols = manager._resume_checkpoint.pending_symbols

        # Filter out symbols with missing CSV files (Requirement 6.5)
        valid_pending = []
        for symbol in pending_symbols:
            if symbol in symbol_data:
                valid_pending.append(symbol)
            elif data_dir:
                csv_path = Path(data_dir) / f"{symbol}.csv"
                if csv_path.exists():
                    try:
                        symbol_data[symbol] = pd.read_csv(csv_path)
                        valid_pending.append(symbol)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load CSV for symbol '{symbol}': {e}. Skipping."
                        )
                else:
                    logger.warning(
                        f"CSV file missing for pending symbol '{symbol}': "
                        f"{csv_path}. Skipping."
                    )
            else:
                logger.warning(
                    f"No data available for pending symbol '{symbol}'. Skipping."
                )

        # If ALL pending symbols are missing, start fresh (Requirement 6.6)
        if len(valid_pending) == 0 and len(pending_symbols) > 0:
            logger.warning(
                "All pending symbols from checkpoint have missing CSV files. "
                "Deleting checkpoint and starting fresh."
            )
            checkpoint_manager.delete()
            training_symbols = all_symbols
            manager._resume_checkpoint = None
        else:
            training_symbols = valid_pending

    # Determine epochs per symbol
    total_epochs_per_symbol = (
        manager._training_pipeline.config.max_epochs_full
        if mode == "full"
        else manager._training_pipeline.config.max_epochs_incremental
    )

    # Create SymbolProgressTracker
    session_state_dict = getattr(manager, "_session_state_dict", None)
    tracker = SymbolProgressTracker(
        symbols=training_symbols,
        mode=mode,
        total_epochs_per_symbol=total_epochs_per_symbol,
        session_state_dict=session_state_dict,
    )

    # Store tracker reference for external access
    manager._progress_tracker = tracker

    # Per-symbol training loop
    for symbol in training_symbols:
        if manager._stop_event.is_set():
            logger.info("Training stopped by user request")
            break

        # Get symbol data
        if symbol not in symbol_data:
            if data_dir:
                csv_path = Path(data_dir) / f"{symbol}.csv"
                if csv_path.exists():
                    try:
                        symbol_data[symbol] = pd.read_csv(csv_path)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load CSV for '{symbol}': {e}. Skipping."
                        )
                        tracker.on_symbol_failed(symbol, f"CSV load error: {e}")
                        continue
                else:
                    logger.warning(
                        f"CSV file not found for '{symbol}': {csv_path}. Skipping."
                    )
                    tracker.on_symbol_failed(symbol, "CSV file not found")
                    continue
            else:
                logger.warning(f"No data for symbol '{symbol}'. Skipping.")
                tracker.on_symbol_failed(symbol, "No data available")
                continue

        # Signal training start for this symbol
        tracker.on_symbol_start(symbol)
        symbol_start_time = time.time()

        try:
            # Train this symbol
            single_symbol_data = {symbol: symbol_data[symbol]}
            _run_single_symbol_training(
                manager, single_symbol_data, mode, data_dir, tracker, symbol
            )

            # Record successful completion
            duration = time.time() - symbol_start_time
            tracker.on_symbol_complete(symbol, duration)

        except Exception as e:
            duration = time.time() - symbol_start_time
            logger.error(f"Training failed for symbol '{symbol}': {e}")
            tracker.on_symbol_failed(symbol, str(e))

    # Session complete - delete checkpoint (Requirement 5.6)
    if not manager._stop_event.is_set():
        checkpoint_manager.delete()
        logger.info("Training session complete, checkpoint deleted")


def _run_single_symbol_training(
    manager: "BackgroundTrainingManager",
    symbol_data: Dict[str, "pd.DataFrame"],
    mode: str,
    data_dir: Optional[str],
    tracker,
    symbol: str,
) -> None:
    """Train a single symbol with epoch-level progress tracking.

    Wraps the existing monitored training with tracker callbacks
    for epoch completion and heartbeat updates.

    Args:
        manager: BackgroundTrainingManager instance.
        symbol_data: Single-symbol data dict.
        mode: "full" or "incremental".
        data_dir: Data directory.
        tracker: SymbolProgressTracker instance.
        symbol: Symbol being trained.
    """
    # Store tracker and symbol on manager for epoch callbacks
    manager._current_tracker = tracker
    manager._current_training_symbol = symbol

    try:
        # Clear per-epoch checkpoints before each symbol so training starts
        # from epoch 0. The per-epoch checkpoint is per-training-session, not
        # per-symbol — reusing it would skip all epochs for subsequent symbols.
        try:
            manager._training_pipeline.checkpoint_manager.cleanup_old_checkpoints(
                keep_last=0
            )
        except Exception:
            pass

        result = run_monitored_training(
            manager=manager,
            pipeline=manager._training_pipeline,
            symbol_data=symbol_data,
            mode=mode,
            data_dir=data_dir,
        )

        # Hot-swap model after successful training
        if result and result.model_path:
            manager._perform_hot_swap(result.model_path)

    finally:
        manager._current_tracker = None
        manager._current_training_symbol = None
