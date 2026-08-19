"""
Shared utility cho việc retrain model từ hard examples.

Bridge functions chuyển đổi hard_examples (List[dict]) thành training tensors
và orchestrate retrain flow. Được sử dụng bởi cả:
- engine/auto_learner.py (_retrain_model)
- engine/auto_learner_wiring.py (_retrain_with_hard_examples)

References: Requirements 2.1, 2.2, 2.3, 2.4
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from engine.config import ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


def convert_hard_examples_to_tensors(
    hard_examples: List[dict],
    data_dir: str,
    model_config: ModelConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chuyển đổi hard_examples thành training tensors.

    Load OHLCV data cho symbols trong hard_examples, extract features
    quanh buy_date, tạo sliding windows và labels phù hợp cho training.

    Args:
        hard_examples: Danh sách dict với keys: symbol, buy_date, sell_date,
            buy_price, sell_price, pnl, pnl_pct, holding_days, outcome.
        data_dir: Đường dẫn thư mục chứa file CSV data (VD: "data").
        model_config: Cấu hình model (num_features, lookback).

    Returns:
        Tuple (features_tensor, labels_tensor):
            - features_tensor: shape (N, lookback, num_features), dtype float32
            - labels_tensor: shape (N, 1), dtype float32

    Raises:
        ValueError: Nếu không có hard example nào tạo được training data hợp lệ.
    """
    from engine.market_state import (
        INDICATOR_COLUMNS,
        NUM_FEATURES,
        NUM_INDICATORS,
        OHLCV_COLUMNS,
    )

    lookback = model_config.lookback
    all_features = []
    all_labels = []

    for example in hard_examples:
        symbol = example.get("symbol", "")
        buy_date = example.get("buy_date", "")
        pnl_pct = example.get("pnl_pct", 0.0)

        # Load CSV data cho symbol
        csv_path = os.path.join(data_dir, f"{symbol}.csv")
        if not os.path.exists(csv_path):
            logger.warning(
                f"[HardExampleTrainer] Bỏ qua hard example — "
                f"không tìm thấy file data: {csv_path}"
            )
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning(
                f"[HardExampleTrainer] Lỗi đọc file {csv_path}: {e}"
            )
            continue

        # Kiểm tra đủ cột OHLCV
        missing_cols = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing_cols:
            logger.warning(
                f"[HardExampleTrainer] File {csv_path} thiếu cột: {missing_cols}"
            )
            continue

        # Tìm index gần buy_date nhất
        buy_idx = _find_date_index(df, buy_date)
        if buy_idx is None:
            # Nếu không tìm được date, dùng window cuối cùng có đủ lookback
            if len(df) >= lookback:
                buy_idx = len(df) - 1
            else:
                logger.warning(
                    f"[HardExampleTrainer] Không đủ data cho {symbol} "
                    f"(cần {lookback} rows, có {len(df)})"
                )
                continue

        # Extract window: lấy lookback rows kết thúc tại buy_idx
        start_idx = max(0, buy_idx - lookback + 1)
        end_idx = buy_idx + 1

        # Cần đủ lookback rows
        if (end_idx - start_idx) < lookback:
            # Nếu không đủ rows phía trước buy_idx, dùng từ đầu nếu có đủ
            if len(df) >= lookback:
                start_idx = 0
                end_idx = lookback
            else:
                logger.warning(
                    f"[HardExampleTrainer] Không đủ lookback data cho {symbol}"
                )
                continue

        window_df = df.iloc[start_idx:end_idx]

        # Extract features: OHLCV + indicators
        ohlcv = window_df[OHLCV_COLUMNS].values.astype(np.float64)
        indicator_data = np.full(
            (len(window_df), NUM_INDICATORS), np.nan, dtype=np.float64
        )
        for i, col in enumerate(INDICATOR_COLUMNS):
            if col in window_df.columns:
                indicator_data[:, i] = window_df[col].values.astype(np.float64)

        raw_features = np.concatenate([ohlcv, indicator_data], axis=1)

        # Pad/trim features đến NUM_FEATURES
        if raw_features.shape[1] < NUM_FEATURES:
            padding = np.full(
                (raw_features.shape[0], NUM_FEATURES - raw_features.shape[1]),
                0.0, dtype=np.float64,
            )
            raw_features = np.concatenate([raw_features, padding], axis=1)
        elif raw_features.shape[1] > NUM_FEATURES:
            raw_features = raw_features[:, :NUM_FEATURES]

        # Forward-fill NaN rồi zero-fill
        for col in range(raw_features.shape[1]):
            col_data = raw_features[:, col]
            mask = np.isnan(col_data)
            if mask.any():
                raw_features[:, col] = pd.Series(col_data).ffill().values
        raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Per-window min-max normalize (nhất quán với backtest inference)
        col_min = raw_features.min(axis=0)
        col_max = raw_features.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0
        raw_features = (raw_features - col_min) / col_range

        # Đảm bảo shape đúng (lookback, num_features)
        if raw_features.shape[0] != lookback:
            # Pad hoặc trim để đúng lookback
            if raw_features.shape[0] > lookback:
                raw_features = raw_features[-lookback:]
            else:
                pad_rows = lookback - raw_features.shape[0]
                padding = np.zeros((pad_rows, NUM_FEATURES), dtype=np.float64)
                raw_features = np.concatenate([padding, raw_features], axis=0)

        all_features.append(raw_features)

        # Label: dùng pnl_pct từ hard example, map qua tanh để nằm trong [-1, 1]
        # Đây là "ground truth" — outcome thực tế của trade
        label = float(np.tanh(pnl_pct * 10.0))
        all_labels.append(label)

    if not all_features:
        raise ValueError(
            "Không tạo được training data từ hard examples — "
            "tất cả examples bị skip do thiếu data"
        )

    # Stack thành tensors
    features_array = np.stack(all_features, axis=0).astype(np.float32)
    labels_array = np.array(all_labels, dtype=np.float32).reshape(-1, 1)

    features_tensor = torch.from_numpy(features_array)
    labels_tensor = torch.from_numpy(labels_array)

    logger.info(
        f"[HardExampleTrainer] Tạo training tensors: "
        f"features={features_tensor.shape}, labels={labels_tensor.shape}"
    )

    return features_tensor, labels_tensor


def retrain_from_hard_examples(
    hard_examples: List[dict],
    checkpoint_dir: str,
    data_dir: str,
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
) -> bool:
    """
    Orchestrate full retrain flow từ hard examples.

    Load model → create training data → train (forward/loss/backward/step)
    → save checkpoint. Xử lý graceful khi data file hoặc checkpoint không tồn tại.

    Args:
        hard_examples: Danh sách hard examples cần học.
        checkpoint_dir: Thư mục chứa model checkpoints (VD: "engine/models").
        data_dir: Thư mục chứa OHLCV CSV files (VD: "data").
        model_config: Cấu hình model architecture. None → dùng default.
        training_config: Cấu hình training. None → dùng default.

    Returns:
        True nếu training thành công, False nếu có lỗi hoặc skip.
    """
    from engine.evaluation_model import StockEvalNet
    from engine.hardware_profile import HardwareProfile

    # Validation: skip nếu hard_examples rỗng
    if not hard_examples:
        logger.info("[HardExampleTrainer] Hard examples rỗng — skip training")
        return False

    if model_config is None:
        model_config = ModelConfig()
    if training_config is None:
        training_config = TrainingConfig()

    logger.info(
        f"[HardExampleTrainer] Bắt đầu retrain với {len(hard_examples)} hard examples"
    )

    # Bước 1: Tạo training data từ hard examples
    try:
        features_tensor, labels_tensor = convert_hard_examples_to_tensors(
            hard_examples, data_dir, model_config
        )
    except ValueError as e:
        logger.warning(f"[HardExampleTrainer] Không tạo được training data: {e}")
        return False
    except Exception as e:
        logger.error(f"[HardExampleTrainer] Lỗi khi tạo training data: {e}")
        return False

    # Bước 2: Load hoặc tạo model
    device = torch.device("cpu")  # Dùng CPU cho safety, tránh OOM
    model = StockEvalNet(model_config).to(device)

    checkpoint_path = _find_checkpoint(checkpoint_dir)
    if checkpoint_path is not None:
        try:
            checkpoint = HardwareProfile.load_checkpoint(
                path=checkpoint_path,
                target_device=device,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                f"[HardExampleTrainer] Loaded checkpoint: {checkpoint_path}"
            )
        except Exception as e:
            logger.warning(
                f"[HardExampleTrainer] Không load được checkpoint ({e}) — "
                f"dùng fresh model"
            )
    else:
        logger.info(
            "[HardExampleTrainer] Không tìm thấy checkpoint — dùng fresh model"
        )

    # Bước 3: Setup optimizer và loss function
    # Dùng learning rate thấp hơn (lr * 0.1) cho fine-tuning
    lr = training_config.learning_rate * 0.1
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=training_config.weight_decay,
    )
    criterion = nn.MSELoss()

    # Bước 4: Training loop
    model.train()
    num_epochs = min(training_config.max_epochs_incremental, 5)  # Cap ở 5 epochs cho fine-tuning
    batch_size = min(training_config.batch_size, len(features_tensor))

    features_tensor = features_tensor.to(device)
    labels_tensor = labels_tensor.to(device)

    total_loss = 0.0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Mini-batch training
        for start in range(0, len(features_tensor), batch_size):
            end = min(start + batch_size, len(features_tensor))
            batch_features = features_tensor[start:end]
            batch_labels = labels_tensor[start:end]

            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        total_loss += avg_epoch_loss

    avg_loss = total_loss / max(num_epochs, 1)
    logger.info(
        f"[HardExampleTrainer] Training hoàn thành: "
        f"{num_epochs} epochs, avg_loss={avg_loss:.6f}"
    )

    # Bước 5: Save checkpoint
    try:
        _save_model_checkpoint(model, optimizer, checkpoint_dir, model_config)
        logger.info(
            f"[HardExampleTrainer] Đã lưu checkpoint vào {checkpoint_dir}"
        )
    except Exception as e:
        logger.error(f"[HardExampleTrainer] Lỗi khi lưu checkpoint: {e}")
        return False

    return True


def _find_date_index(df: pd.DataFrame, date_str: str) -> Optional[int]:
    """
    Tìm index trong DataFrame tương ứng với date_str.

    Args:
        df: DataFrame chứa cột 'time'.
        date_str: Date string cần tìm (VD: "2024-01-15").

    Returns:
        Index trong DataFrame hoặc None nếu không tìm thấy.
    """
    if not date_str or "time" not in df.columns:
        return None

    try:
        # Thử tìm exact match hoặc closest date
        dates = pd.to_datetime(df["time"], errors="coerce")
        target = pd.to_datetime(date_str, errors="coerce")

        if pd.isna(target):
            return None

        # Tìm exact match
        exact_match = dates == target
        if exact_match.any():
            return int(exact_match.idxmax())

        # Tìm ngày gần nhất trước target
        before_target = dates[dates <= target]
        if not before_target.empty:
            return int(before_target.index[-1])

        # Fallback: trả về index cuối cùng
        return len(df) - 1

    except Exception:
        return None


def _find_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Tìm checkpoint file mới nhất trong thư mục.

    Ưu tiên: stock_eval_net.pt (finetuned) → checkpoint_epoch_XXXX.pt (initial training)

    Args:
        checkpoint_dir: Thư mục chứa checkpoints.

    Returns:
        Path đến checkpoint file hoặc None nếu không tìm thấy.
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    # Ưu tiên stock_eval_net.pt — file chính được retrain liên tục
    default_model = checkpoint_path / "stock_eval_net.pt"
    if default_model.exists():
        return str(default_model)

    # Fallback: tìm epoch checkpoint mới nhất (từ initial training)
    epoch_checkpoints = sorted(checkpoint_path.glob("checkpoint_epoch_*.pt"))
    if epoch_checkpoints:
        return str(epoch_checkpoints[-1])

    return None


def _save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    model_config: ModelConfig,
) -> None:
    """
    Lưu model checkpoint sau khi train.

    Dùng format tương thích với HardwareProfile.load_checkpoint.

    Args:
        model: Model đã train xong.
        optimizer: Optimizer state.
        checkpoint_dir: Thư mục lưu checkpoint.
        model_config: Cấu hình model architecture.
    """
    from engine.hardware_profile import HardwareProfile

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Lưu đè lên stock_eval_net.pt — đây là file mà inference pipeline load
    save_path = str(checkpoint_path / "stock_eval_net.pt")

    hardware = HardwareProfile()
    HardwareProfile.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=0,
        metrics={"train_loss": 0.0, "val_loss": 0.0, "best_val_loss": 0.0},
        hardware=hardware,
        path=save_path,
        model_config=model_config,
    )
