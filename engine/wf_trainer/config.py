# -*- coding: utf-8 -*-
"""
Cấu hình cho Walk-Forward Hybrid SL+RL Training.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class WFConfig:
    """Cấu hình Walk-Forward Training Cycle."""

    # --- Data ---
    data_dir: str = "data"
    lookback: int = 60  # Số phiên lịch sử làm input
    num_features: int = 78  # OHLCV(5) + indicators(73: 56 cũ + 5 Wyckoff + 6 Value + 6 Market Context)
    train_ratio: float = 0.70  # 70% train (2014-~2021)
    val_ratio: float = 0.15  # 15% validation (SL early stopping)
    # test_ratio = 1 - train - val = 15% (RL + Backtest OOS)
    embargo_days: int = 5  # Khoảng cách giữa mỗi phần (tránh leakage)
    min_sessions: int = 250  # Tối thiểu 250 phiên để train

    # --- Supervised Learning (Stage 1) ---
    sl_epochs: int = 20  # Số epochs cho SL pretrain
    sl_lr: float = 1e-3  # Learning rate
    sl_batch_size: int = 64
    label_horizon: int = 5  # Nhìn trước 5 ngày để tạo label
    label_scale: float = 10.0  # tanh(return * scale) cho label

    # --- Reinforcement Learning (Stage 2) ---
    rl_episodes: int = 300  # Số episodes RL training (tăng để policy converge)
    rl_lr: float = 3e-4  # Learning rate cho RL (thấp hơn SL)
    rl_gamma: float = 0.99  # Discount factor
    rl_clip_eps: float = 0.2  # PPO clip epsilon
    rl_transaction_cost: float = 0.0015  # 0.15% phí giao dịch VN
    rl_initial_capital: float = 1_000_000_000.0  # 1 tỉ VND

    # --- Model ---
    tcn_channels: List[int] = field(default_factory=lambda: [128, 128, 64])
    kernel_size: int = 3
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4])
    attention_heads: int = 4
    attention_dim: int = 64
    dropout: float = 0.1

    # --- Cycle ---
    checkpoint_dir: str = "engine/models"
    report_dir: str = "data/engine/reports"
    history_dir: str = "data/engine/history"
    max_symbols_per_cycle: int = 70  # Tất cả symbols mỗi cycle

    # --- Evaluation ---
    # Nếu model mới xấu hơn model cũ → rollback
    rollback_if_worse: bool = True
    min_sharpe_improvement: float = -0.1  # Chấp nhận Sharpe giảm tối đa 0.1
