# -*- coding: utf-8 -*-
"""
Stage 1: Supervised Learning Pre-Training.

Train TCN+Attention model để nhận diện market patterns.
Label = tanh(future_return * scale) — dự đoán hướng + magnitude.
Chronological split với embargo gap.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from engine.wf_trainer.config import WFConfig

logger = logging.getLogger(__name__)


@dataclass
class SLTrainResult:
    """Kết quả Stage 1 training."""

    epochs_completed: int = 0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_val_loss: float = float("inf")
    duration_seconds: float = 0.0
    symbols_used: List[str] = None

    def __post_init__(self):
        if self.symbols_used is None:
            self.symbols_used = []


def build_sl_dataset(
    symbols: List[str],
    config: WFConfig,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Xây dựng dataset cho supervised learning.

    Load CSV → extract features → create sliding windows → generate labels.
    Chronological split: train / val (với embargo gap).

    Returns:
        (train_X, train_y, val_X, val_y) hoặc (None, None, None, None) nếu thiếu data.
    """
    from engine.market_state import INDICATOR_COLUMNS, OHLCV_COLUMNS, NUM_INDICATORS

    all_windows = []
    all_labels = []

    for symbol in symbols:
        csv_path = Path(config.data_dir) / f"{symbol}.csv"
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if len(df) < config.min_sessions:
            continue

        # Extract features
        ohlcv = df[OHLCV_COLUMNS].values.astype(np.float64)
        indicators = np.zeros((len(df), NUM_INDICATORS), dtype=np.float64)
        for i, col in enumerate(INDICATOR_COLUMNS):
            if col in df.columns:
                indicators[:, i] = df[col].values.astype(np.float64)

        features = np.concatenate([ohlcv, indicators], axis=1)

        # Pad/trim đến num_features
        if features.shape[1] < config.num_features:
            pad = np.zeros((len(df), config.num_features - features.shape[1]))
            features = np.concatenate([features, pad], axis=1)
        elif features.shape[1] > config.num_features:
            features = features[:, :config.num_features]

        # NaN handling
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Tạo sliding windows + labels
        close_prices = df["close"].values.astype(np.float64)
        lookback = config.lookback
        horizon = config.label_horizon

        for i in range(lookback, len(df) - horizon):
            window = features[i - lookback:i]  # (lookback, num_features)

            # Per-window min-max normalize
            col_min = window.min(axis=0)
            col_max = window.max(axis=0)
            col_range = col_max - col_min
            col_range[col_range == 0] = 1.0
            window_norm = (window - col_min) / col_range

            # Label: tanh(future_return * scale)
            future_price = close_prices[i + horizon]
            current_price = close_prices[i]
            if current_price > 0:
                future_return = (future_price - current_price) / current_price
                label = float(np.tanh(future_return * config.label_scale))
            else:
                label = 0.0

            all_windows.append(window_norm)
            all_labels.append(label)

    if not all_windows:
        return None, None, None, None

    # Stack
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32).reshape(-1, 1)

    # Chronological split: 70% train | embargo | 15% val | embargo | 15% test
    # SL chỉ dùng train + val. Test dành cho RL + Backtest.
    n = len(X)
    train_end = int(n * config.train_ratio)
    val_start = train_end + config.embargo_days
    val_end = val_start + int(n * config.val_ratio)

    # Đảm bảo không vượt quá data
    if val_end > n:
        val_end = n
    if val_start >= val_end:
        val_start = train_end
        val_end = min(train_end + 100, n)

    train_X = torch.from_numpy(X[:train_end])
    train_y = torch.from_numpy(y[:train_end])
    val_X = torch.from_numpy(X[val_start:val_end])
    val_y = torch.from_numpy(y[val_start:val_end])

    logger.info(
        f"[SL] Dataset: train={len(train_X)}, val={len(val_X)}, "
        f"symbols={len(symbols)}, features={config.num_features}"
    )

    return train_X, train_y, val_X, val_y


def train_supervised(
    model: nn.Module,
    config: WFConfig,
    symbols: List[str],
) -> SLTrainResult:
    """
    Chạy supervised pre-training.

    Train model trên labeled data (future return prediction).
    Dùng MSE loss, Adam optimizer, early stopping nếu val_loss không giảm.

    Args:
        model: TCN+Attention model (StockEvalNet)
        config: WFConfig
        symbols: Danh sách symbols để train

    Returns:
        SLTrainResult với metrics
    """
    start_time = time.time()

    # Build dataset
    train_X, train_y, val_X, val_y = build_sl_dataset(symbols, config)
    if train_X is None or len(train_X) == 0:
        logger.warning("[SL] Không đủ data để train")
        return SLTrainResult()

    # DataLoader
    train_ds = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_ds, batch_size=config.sl_batch_size, shuffle=True)

    # Setup
    device = torch.device("cpu")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.sl_lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(config.sl_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X.to(device))
            val_loss = criterion(val_pred, val_y.to(device)).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[SL] Early stopping at epoch {epoch+1}")
                break

    duration = time.time() - start_time
    logger.info(
        f"[SL] Done: {epoch+1} epochs, train_loss={avg_train_loss:.4f}, "
        f"val_loss={val_loss:.4f}, best_val={best_val_loss:.4f}, {duration:.1f}s"
    )

    return SLTrainResult(
        epochs_completed=epoch + 1,
        final_train_loss=avg_train_loss,
        final_val_loss=val_loss,
        best_val_loss=best_val_loss,
        duration_seconds=duration,
        symbols_used=symbols,
    )
