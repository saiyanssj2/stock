"""
Cấu hình và data models cho module đánh giá mô hình sau huấn luyện.

Định nghĩa EvaluationConfig dataclass và tất cả result dataclasses dùng trong
quy trình đánh giá: OverfittingSeverity enum, DistributionStats, OverfittingResult,
ConfusionMatrix, ScoreDistributionResult, PredictionQualityResult,
SymbolEvaluationResult, ModelMetadata, EvaluationResult.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from engine.config import BacktestResult, ComparisonResult


# ==============================================================================
# Cấu hình đánh giá
# ==============================================================================


@dataclass
class EvaluationConfig:
    """Cấu hình cho module đánh giá sau huấn luyện."""

    # Signal thresholds: chuyển Position_Score → Action
    buy_threshold: float = 0.3       # Score > threshold → BUY
    sell_threshold: float = -0.3     # Score < threshold → SELL

    # Baseline comparison
    random_seed: int = 42            # Seed cho Random strategy
    sma_short: int = 20              # SMA ngắn hạn
    sma_long: int = 50               # SMA dài hạn

    # Overfitting thresholds
    overfitting_mild: float = 1.5    # ratio >= 1.5 → mild
    overfitting_moderate: float = 2.0  # ratio >= 2.0 → moderate
    overfitting_severe: float = 3.0  # ratio >= 3.0 → severe

    # Score distribution flags
    conservative_std_threshold: float = 0.1  # std < 0.1 → too conservative
    extreme_std_threshold: float = 0.8       # std > 0.8 → too extreme

    # Report output
    report_dir: str = "engine/reports"
    json_filename: str = "evaluation_report.json"
    html_filename: str = "evaluation_report.html"


# ==============================================================================
# Enums
# ==============================================================================


class OverfittingSeverity(Enum):
    """Mức độ overfitting."""

    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


# ==============================================================================
# Result Dataclasses
# ==============================================================================


@dataclass
class DistributionStats:
    """Thống kê phân phối một tập predictions."""

    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0


@dataclass
class OverfittingResult:
    """Kết quả phân tích overfitting."""

    loss_ratio: float = 0.0              # test_loss / train_loss
    severity: OverfittingSeverity = OverfittingSeverity.NONE
    ks_statistic: float = 0.0            # Kolmogorov-Smirnov statistic
    ks_p_value: float = 0.0              # KS test p-value
    train_distribution: DistributionStats = field(default_factory=DistributionStats)
    test_distribution: DistributionStats = field(default_factory=DistributionStats)
    warning_message: Optional[str] = None


@dataclass
class ConfusionMatrix:
    """Ma trận nhầm lẫn hướng dự đoán."""

    true_positive: int = 0   # Predicted up, actual up
    true_negative: int = 0   # Predicted down, actual down
    false_positive: int = 0  # Predicted up, actual down
    false_negative: int = 0  # Predicted down, actual up

    @property
    def total(self) -> int:
        """Tổng số predictions."""
        return (
            self.true_positive
            + self.true_negative
            + self.false_positive
            + self.false_negative
        )

    @property
    def accuracy(self) -> float:
        """Tỷ lệ dự đoán đúng (percentage [0, 100])."""
        if self.total == 0:
            return 0.0
        return (self.true_positive + self.true_negative) / self.total * 100.0


@dataclass
class ScoreDistributionResult:
    """Kết quả phân tích phân phối Position_Score."""

    mean: float = 0.0
    std: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    too_conservative: bool = False  # std < 0.1
    too_extreme: bool = False       # std > 0.8
    flag_message: Optional[str] = None


@dataclass
class PredictionQualityResult:
    """Kết quả phân tích chất lượng dự đoán."""

    directional_accuracy: float = 0.0      # Percentage [0, 100]
    pearson_correlation: float = 0.0       # [-1, 1]
    calibration_quartiles: Dict[str, float] = field(default_factory=dict)
    score_distribution: ScoreDistributionResult = field(
        default_factory=ScoreDistributionResult
    )
    confusion_matrix: ConfusionMatrix = field(default_factory=ConfusionMatrix)


@dataclass
class SymbolEvaluationResult:
    """Kết quả đánh giá cho một symbol."""

    symbol: str = ""
    backtest_result: Optional[BacktestResult] = None
    prediction_quality: Optional[PredictionQualityResult] = None


@dataclass
class ModelMetadata:
    """Thông tin metadata của model."""

    architecture: str = "StockEvalNet (TCN + Attention)"
    parameter_count: int = 0
    training_timestamp: Optional[str] = None
    model_path: Optional[str] = None
    symbols_trained: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Kết quả tổng hợp toàn bộ đánh giá."""

    # Metadata
    model_metadata: ModelMetadata = field(default_factory=ModelMetadata)
    evaluation_timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    # Per-symbol results
    symbol_results: Dict[str, SymbolEvaluationResult] = field(default_factory=dict)

    # Aggregated results
    aggregated_backtest: Optional[BacktestResult] = None
    comparison_result: Optional[ComparisonResult] = None
    overfitting_result: Optional[OverfittingResult] = None
    aggregated_prediction_quality: Optional[PredictionQualityResult] = None
