"""
Implementation module for TrainingPipeline incremental training and model saving.

Contains train_incremental_impl and save_complete_model_impl as module-level
functions that accept the pipeline instance as the first argument.

This module is imported lazily by training_pipeline.py.
"""

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from engine.config import DataError, ModelError
from engine.hardware_profile import HardwareProfile

logger = logging.getLogger(__name__)


def train_incremental_impl(
    pipeline,
    new_data: Dict[str, "pd.DataFrame"],
    checkpoint_path: Optional[str] = None,
    max_epochs: Optional[int] = None,
):
    """Implementation of TrainingPipeline.train_incremental. Requirements: 6.9-6.11."""
    from engine.evaluation_model import StockEvalNet
    from engine.market_state import FeatureVectorBuilder
    from engine.training_pipeline import TrainingResult

    import pandas as pd

    start_time = time.time()

    if max_epochs is None:
        max_epochs = pipeline.config.max_epochs_incremental

    # Load normalization parameters
    norm_params_path = str(Path(pipeline.config.checkpoint_dir) / "norm_params.json")
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
        if df is None or len(df) < pipeline.model_config.lookback:
            logger.warning(
                f"Skipping symbol '{symbol}' for incremental training: "
                f"insufficient data ({0 if df is None else len(df)} rows, "
                f"need at least {pipeline.model_config.lookback})"
            )
            continue

        features, labels = pipeline._build_feature_matrix(df)
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
    device = pipeline._hardware_profile.get_device()
    logger.info(f"Incremental training on device: {device} ({pipeline._hardware_profile})")

    model = StockEvalNet(pipeline.model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=pipeline.config.learning_rate * 0.1,  # Lower LR for fine-tuning
        weight_decay=pipeline.config.weight_decay,
    )

    # LR Scheduler: giảm LR khi val_loss không cải thiện (incremental dùng patience thấp hơn)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
    )

    # Auto-adjust batch size based on detected hardware (Req 15.5)
    batch_size = pipeline._hardware_profile.get_optimal_batch_size(mode="training")
    logger.info(f"Auto-adjusted batch size for incremental training: {batch_size}")

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = pipeline.checkpoint_manager.get_latest_checkpoint_path()

    if checkpoint_path is None:
        # Try loading the saved best model
        best_model_path = str(Path(pipeline.config.checkpoint_dir) / "stock_eval_net.pt")
        if Path(best_model_path).exists():
            checkpoint_path = best_model_path
        else:
            raise ModelError(
                "No checkpoint available for incremental training. Run full training first.",
                error_code="CHECKPOINT_NOT_FOUND",
            )

    # Load model state from checkpoint using portable load
    try:
        checkpoint = HardwareProfile.load_checkpoint(
            path=checkpoint_path,
            target_device=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        # Log source hardware info if available
        hw_meta = checkpoint.get("hardware_metadata")
        if hw_meta:
            logger.info(
                f"Loaded checkpoint from device={hw_meta.get('device_type')}, "
                f"gpu={hw_meta.get('gpu_model')}"
            )
        logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
    except ModelError:
        raise
    except Exception as e:
        # Fallback: try legacy checkpoint format (no portable format)
        try:
            legacy_cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in legacy_cp:
                model.load_state_dict(legacy_cp["model_state_dict"])
            elif "state_dict" in legacy_cp:
                model.load_state_dict(legacy_cp["state_dict"])
            else:
                model.load_state_dict(legacy_cp)
            model.to(device)
            logger.info(f"Loaded model from legacy checkpoint: {checkpoint_path}")
        except Exception as e2:
            raise ModelError(
                f"Failed to load model for incremental training: {e2}",
                error_code="CHECKPOINT_CORRUPTED",
                details={"path": checkpoint_path, "error": str(e2)},
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
        train_dataset, batch_size=batch_size, shuffle=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Training loop (max_epochs, no early stopping for incremental)
    result = TrainingResult(symbols_trained=valid_symbols)
    best_val_loss = float("inf")
    graceful_stopped = False

    # Notify TrainingController that training is starting (if connected)
    if pipeline._training_controller is not None:
        pipeline._training_controller.start_training(total_epochs=max_epochs)

    for epoch in range(max_epochs):
        # Check graceful stop BEFORE beginning a new epoch (Req 14.5)
        if pipeline._training_controller is not None:
            if not pipeline._training_controller.should_continue_training():
                logger.info(
                    f"Graceful stop: not starting incremental epoch {epoch} "
                    "(stop requested after previous epoch)"
                )
                graceful_stopped = True
                break

        # Notify controller of epoch start
        if pipeline._training_controller is not None:
            pipeline._training_controller.begin_epoch(epoch + 1)  # 1-indexed for UI

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
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            mae=val_mae,
            learning_rate=current_lr,
            epoch_time_seconds=epoch_time,
            mode="incremental",
        )

        # Step LR scheduler dựa trên val_loss
        scheduler.step(val_loss)

        # Save checkpoint (Req 14.3: always save before potential stop)
        pipeline.checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            model_config=pipeline.model_config,
            norm_params=norm_params,
            filename=f"checkpoint_incremental_epoch_{epoch:04d}.pt",
        )

        # Notify TrainingController that epoch is complete
        if pipeline._training_controller is not None:
            epoch_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mae": val_mae,
                "epoch_time": epoch_time,
            }
            should_continue = pipeline._training_controller._on_epoch_complete(
                epoch=epoch + 1,  # 1-indexed
                metrics=epoch_metrics,
            )
            if not should_continue:
                # Graceful stop: checkpoint already saved above (Req 14.3)
                logger.info(
                    f"Graceful stop: incremental epoch {epoch} completed, "
                    "checkpoint saved, training terminated cleanly."
                )
                graceful_stopped = True
                # Update result before breaking
                result.epochs_completed = epoch + 1
                result.final_train_loss = train_loss
                result.final_val_loss = val_loss
                result.final_mae = val_mae
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    result.best_val_loss = best_val_loss
                    result.best_epoch = epoch
                break

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
    model_path = str(Path(pipeline.config.checkpoint_dir) / "stock_eval_net.pt")
    save_complete_model_impl(
        pipeline=pipeline,
        model=model,
        optimizer=optimizer,
        epoch=result.epochs_completed - 1,
        model_config=pipeline.model_config,
        norm_params=norm_params,
        path=model_path,
    )
    result.model_path = model_path
    result.checkpoint_path = checkpoint_path

    # Notify TrainingController that training has ended
    if pipeline._training_controller is not None:
        pipeline._training_controller.end_training()

    result.total_time_seconds = time.time() - start_time

    stop_reason = ""
    if graceful_stopped:
        stop_reason = " (graceful stop)"

    # Log training session with HardwareProfile metadata (Req 15.8)
    pipeline.epoch_logger.log_training_session(
        mode="incremental",
        hardware_profile=pipeline._hardware_profile,
        symbols=valid_symbols,
        batch_size=batch_size,
        total_epochs=result.epochs_completed,
        total_time_seconds=result.total_time_seconds,
        final_train_loss=result.final_train_loss,
        final_val_loss=result.final_val_loss,
        resumed_from_checkpoint=True,  # Incremental always resumes
    )

    logger.info(
        f"Incremental training complete{stop_reason}: {result.epochs_completed} epochs, "
        f"best_val_loss={result.best_val_loss:.6f}, "
        f"time={result.total_time_seconds:.1f}s"
    )

    return result


def save_complete_model_impl(
    pipeline,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config,
    norm_params: Dict,
    path: str,
) -> None:
    """Implementation of TrainingPipeline._save_complete_model. Requirements: 6.6, 13.1."""
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
