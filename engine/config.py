"""
Engine configuration constants, data types, and error hierarchy.

Defines all configuration dataclasses (ModelConfig, TrainingConfig, EngineConfig,
SearchConfig), the Action enum, result dataclasses (DecisionReport, ScenarioResult,
IndicatorContribution, Trade, BacktestResult, ComparisonResult), and the error
hierarchy (EngineError, DataError, ModelError, ResourceError, ConfigError).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd


# ==============================================================================
# Action Enum
# ==============================================================================


class Action(Enum):
    """Trading action that can be recommended by the decision engine."""

    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


# ==============================================================================
# Configuration Dataclasses
# ==============================================================================


@dataclass
class ModelConfig:
    """Configuration for the TCN+Attention evaluation model architecture."""

    num_features: int = 78  # OHLCV (5) + indicators (73: 56 cũ + 5 Wyckoff + 6 Value + 6 Market Context)
    lookback: int = 60  # Default lookback window (trading sessions)
    tcn_channels: List[int] = field(default_factory=lambda: [128, 128, 64])
    kernel_size: int = 3
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4])
    attention_heads: int = 4
    attention_dim: int = 64
    dropout: float = 0.1
    max_batch_size: int = 50
    output_range: tuple = (-1.0, 1.0)  # Position score bounds


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""

    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs_full: int = 100
    max_epochs_incremental: int = 10
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    label_horizon: int = 5  # Sessions to look ahead for label generation
    label_sensitivity: float = 10.0  # tanh scaling factor
    min_sessions_per_symbol: int = 250
    checkpoint_dir: str = "engine/models"
    log_file: str = "engine/models/training_log.jsonl"
    max_vram_training_gb: float = 5.5
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10


@dataclass
class ResourceConfig:
    """Resource limits to prevent system overload during training.

    Defaults are conservative (50% CPU, GPU pause at 75%) to keep
    the system responsive and avoid crashes from 100% utilization.
    """

    # CPU: max threads for PyTorch (0 = auto, uses ~50% cores)
    cpu_threads: int = 0
    # CPU: max inter-op parallelism threads
    cpu_interop_threads: int = 2
    # GPU: pause training if utilization exceeds this (fraction 0-1)
    gpu_pause_threshold: float = 0.75
    # GPU: resume training when utilization drops below this
    gpu_resume_threshold: float = 0.60
    # GPU: max VRAM fraction to use (0-1)
    gpu_max_memory_fraction: float = 0.70
    # Training: max workers for DataLoader (0 = main thread only)
    dataloader_workers: int = 0
    # Training: pin memory for faster GPU transfer
    pin_memory: bool = False
    # Training: reduce batch size from default to save memory
    batch_size_reduction_factor: float = 1.0


@dataclass
class EngineConfig:
    """Top-level configuration for the Decision Engine."""

    base_dir: str = "."
    model_path: str = "engine/models/stock_eval_net.pt"
    norm_params_path: str = "engine/models/norm_params.json"
    lookback: int = 60  # Default lookback window [20, 200]
    lookback_min: int = 20
    lookback_max: int = 200
    default_capital: float = 100_000_000.0  # 100M VND
    max_position_pct: float = 0.20  # 20% max single position
    lot_size: int = 100  # Minimum share lot size
    daily_price_limit: float = 0.07  # ±7%
    settlement_days: float = 2.5  # T+2.5
    max_inference_time_ms: float = 50.0  # Per-sample inference target
    max_search_time_s: float = 5.0  # Search timeout
    max_analysis_time_s: float = 10.0  # Full analysis timeout
    confidence_hold_threshold: float = 0.3  # Below this → force HOLD
    max_vram_inference_gb: float = 4.0
    # Price in CSV is in units of 1000 VND. Multiply by this to get actual VND.
    # Set to 1.0 if data is already in VND (used in tests).
    # Production code (DecisionEngine) sets this to 1000.0.
    price_scale: float = 1.0
    csv_required_columns: List[str] = field(
        default_factory=lambda: ["time", "open", "high", "low", "close", "volume"]
    )
    min_data_rows: int = 5  # Minimum rows for valid CSV


@dataclass
class SearchConfig:
    """Configuration for the Minimax/Alpha-Beta search module."""

    default_depth: int = 3
    max_depth: int = 5
    timeout_seconds: float = 5.0
    default_scenarios: int = 5  # Scenarios per node
    min_scenarios: int = 3
    max_scenarios: int = 7
    min_history_days: int = 30  # Minimum days for scenario generation
    top_scenarios_report: int = 3  # Top scenarios in DecisionReport
    top_indicators_report: int = 5  # Top indicators in DecisionReport
    # Adaptive branching: reduce scenarios when depth > 3
    adaptive_depth_threshold: int = 3
    adaptive_scenario_count: int = 3


# ==============================================================================
# Result Dataclasses
# ==============================================================================


@dataclass
class ScenarioResult:
    """A single scenario explored by the search module."""

    action_sequence: List[Action]
    leaf_score: float
    description: str


@dataclass
class IndicatorContribution:
    """An indicator's contribution to the position score."""

    name: str
    value: float
    contribution: float  # Absolute contribution to score


@dataclass
class DecisionReport:
    """Complete decision report output from the engine."""

    symbol: str
    recommended_action: Action
    confidence: float  # [0.0, 1.0]
    position_score: float  # [-1.0, 1.0]
    top_scenarios: List[ScenarioResult]  # Top 3
    top_indicators: List[IndicatorContribution]  # Top 5, sorted by |contribution|
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Trade:
    """A single executed trade in a backtest."""

    entry_date: Optional[pd.Timestamp] = None
    exit_date: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    """Result of a backtest run for a single strategy."""

    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    equity_curve: Optional[pd.Series] = None
    trades: List[Trade] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Result of comparing multiple strategies."""

    results: Dict[str, BacktestResult] = field(default_factory=dict)
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    initial_capital: float = 100_000_000.0
    excluded_strategies: List[str] = field(default_factory=list)
    exclusion_reasons: Dict[str, str] = field(default_factory=dict)


# ==============================================================================
# Error Hierarchy
# ==============================================================================


class EngineError(Exception):
    """Base exception for all engine errors."""

    def __init__(self, message: str, error_code: str = "ENGINE_ERROR", details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class DataError(EngineError):
    """Data loading and validation errors (missing files, malformed data, insufficient rows)."""

    def __init__(self, message: str, error_code: str = "DATA_ERROR", details: dict = None):
        super().__init__(message, error_code, details)


class ModelError(EngineError):
    """Model loading, inference, and validation errors (checksum, incompatible architecture)."""

    def __init__(self, message: str, error_code: str = "MODEL_ERROR", details: dict = None):
        super().__init__(message, error_code, details)


class ResourceError(EngineError):
    """Hardware resource errors (GPU OOM, CPU fallback failure)."""

    def __init__(self, message: str, error_code: str = "RESOURCE_ERROR", details: dict = None):
        super().__init__(message, error_code, details)


class ConfigError(EngineError):
    """Configuration validation errors (invalid parameters, missing config files)."""

    def __init__(self, message: str, error_code: str = "CONFIG_ERROR", details: dict = None):
        super().__init__(message, error_code, details)
