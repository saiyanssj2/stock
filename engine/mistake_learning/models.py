"""
Data models cho Mistake-Driven Learning.

Định nghĩa tất cả dataclass và enum dùng trong feedback loop:
TradeRecord, MistakeReport, MissedOpportunity, HardExampleSet,
CorrectedLabels, AntiPattern, FeedbackCycleResult, TradeClassification.

Mỗi dataclass có validation trong __post_init__, raise ValueError
khi field không hợp lệ.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ==============================================================================
# Enum
# ==============================================================================


class TradeClassification(Enum):
    """Phân loại trade sau khi phân tích."""

    GOOD = "GOOD"
    BAD = "BAD"
    MISSED = "MISSED"


# ==============================================================================
# Hằng số validation
# ==============================================================================

_VALID_CLASSIFICATIONS = {"GOOD", "BAD", "MISSED", "UNKNOWN"}
_FEATURE_VECTOR_LENGTH = 61


# ==============================================================================
# TradeRecord
# ==============================================================================


@dataclass
class TradeRecord:
    """Bản ghi trade mở rộng với features và context cho mistake learning.

    Validation:
    - signal_strength phải trong [-1.0, 1.0]
    - classification phải là GOOD, BAD, MISSED, hoặc UNKNOWN
    - pnl_pct phải là float hữu hạn (không NaN, không Inf)
    - feature_vector khi không None phải có đúng 61 phần tử
    """

    # Thông tin trade cơ bản
    entry_date: Optional[pd.Timestamp] = None
    exit_date: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0

    # Thông tin mở rộng cho mistake learning
    cycle_number: int = 0
    signal_strength: float = 0.0  # Output gốc của model [-1, 1]
    action_taken: str = "HOLD"  # BUY/HOLD/SELL
    classification: str = "UNKNOWN"  # GOOD/BAD/MISSED/UNKNOWN

    # Feature snapshot tại thời điểm entry (61 features)
    feature_vector: Optional[List[float]] = None

    # Market context tại thời điểm entry
    market_volatility: float = 0.0
    market_trend: float = 0.0  # Dương = uptrend
    volume_ratio: float = 0.0  # Volume so với trung bình 20 ngày

    def __post_init__(self) -> None:
        """Validate tất cả fields sau khi khởi tạo."""
        self._validate_signal_strength()
        self._validate_classification()
        self._validate_pnl_pct()
        self._validate_feature_vector()

    def _validate_signal_strength(self) -> None:
        """signal_strength phải trong [-1.0, 1.0]."""
        if not (-1.0 <= self.signal_strength <= 1.0):
            raise ValueError(
                f"signal_strength={self.signal_strength} nằm ngoài "
                f"valid range [-1.0, 1.0]"
            )

    def _validate_classification(self) -> None:
        """classification phải là GOOD, BAD, MISSED, hoặc UNKNOWN."""
        if self.classification not in _VALID_CLASSIFICATIONS:
            raise ValueError(
                f"classification='{self.classification}' không hợp lệ. "
                f"Phải là một trong: {_VALID_CLASSIFICATIONS}"
            )

    def _validate_pnl_pct(self) -> None:
        """pnl_pct phải là float hữu hạn (không NaN, không Inf)."""
        if not math.isfinite(self.pnl_pct):
            raise ValueError(
                f"pnl_pct={self.pnl_pct} không hợp lệ. "
                f"Phải là float hữu hạn (không NaN, không Inf)"
            )

    def _validate_feature_vector(self) -> None:
        """feature_vector khi không None phải có đúng 61 phần tử."""
        if self.feature_vector is not None:
            if len(self.feature_vector) != _FEATURE_VECTOR_LENGTH:
                raise ValueError(
                    f"feature_vector có {len(self.feature_vector)} phần tử, "
                    f"yêu cầu đúng {_FEATURE_VECTOR_LENGTH} phần tử"
                )


# ==============================================================================
# MissedOpportunity
# ==============================================================================


@dataclass
class MissedOpportunity:
    """Cơ hội giao dịch bị bỏ lỡ, phát hiện qua hindsight.

    Khi model giữ HOLD nhưng lẽ ra nên BUY/SELL dựa trên
    kết quả thực tế trong lookforward window.
    """

    date: Optional[pd.Timestamp] = None
    actual_action: str = "HOLD"  # Action model đã thực hiện
    optimal_action: str = "BUY"  # Action lẽ ra nên thực hiện
    potential_gain_pct: float = 0.0  # Phần trăm lợi nhuận bỏ lỡ
    feature_vector: Optional[List[float]] = None


# ==============================================================================
# MistakeReport
# ==============================================================================


@dataclass
class MistakeReport:
    """Báo cáo tổng hợp phân tích mistake cho một cycle.

    Validation:
    - Nếu total_trades == 0: bad_trade_ratio phải là 0.0
    - Nếu total_trades > 0: bad_trade_ratio == len(bad_trades) / total_trades
    """

    cycle_number: int = 0
    total_trades: int = 0
    good_trades: List[TradeRecord] = field(default_factory=list)
    bad_trades: List[TradeRecord] = field(default_factory=list)
    missed_opportunities: List[MissedOpportunity] = field(default_factory=list)

    # Thống kê
    bad_trade_ratio: float = 0.0  # bad / total
    avg_bad_loss: float = 0.0  # Trung bình loss của bad trades
    most_common_mistake_pattern: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate bad_trade_ratio dựa trên total_trades."""
        self._validate_bad_trade_ratio()

    def _validate_bad_trade_ratio(self) -> None:
        """Kiểm tra tính nhất quán của bad_trade_ratio."""
        if self.total_trades == 0:
            if self.bad_trade_ratio != 0.0:
                raise ValueError(
                    f"bad_trade_ratio={self.bad_trade_ratio} phải là 0.0 "
                    f"khi total_trades=0"
                )
        else:
            expected = len(self.bad_trades) / self.total_trades
            if not math.isclose(self.bad_trade_ratio, expected, rel_tol=1e-9):
                raise ValueError(
                    f"bad_trade_ratio={self.bad_trade_ratio} không khớp với "
                    f"len(bad_trades)/total_trades={expected}"
                )


# ==============================================================================
# HardExampleSet
# ==============================================================================


@dataclass
class HardExampleSet:
    """Tập hard examples được mine từ bad trades.

    Validation:
    - features, labels, weights phải có cùng first dimension
      (features.shape[0] == labels.shape[0] == weights.shape[0])
    """

    features: np.ndarray = field(default_factory=lambda: np.array([]))
    labels: np.ndarray = field(default_factory=lambda: np.array([]))
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    source_trades: List[TradeRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate kích thước arrays."""
        self._validate_array_shapes()

    def _validate_array_shapes(self) -> None:
        """features, labels, weights phải có cùng first dimension."""
        f_len = self.features.shape[0] if self.features.ndim > 0 else 0
        l_len = self.labels.shape[0] if self.labels.ndim > 0 else 0
        w_len = self.weights.shape[0] if self.weights.ndim > 0 else 0

        # Trường hợp tất cả rỗng thì OK
        if f_len == 0 and l_len == 0 and w_len == 0:
            return

        if not (f_len == l_len == w_len):
            raise ValueError(
                f"Kích thước arrays không khớp: "
                f"features.shape[0]={f_len}, "
                f"labels.shape[0]={l_len}, "
                f"weights.shape[0]={w_len}. "
                f"Tất cả phải bằng nhau."
            )

    @property
    def count(self) -> int:
        """Số lượng hard examples."""
        if self.labels.ndim == 0 or self.labels.size == 0:
            return 0
        return len(self.labels)


# ==============================================================================
# CorrectedLabels
# ==============================================================================


@dataclass
class CorrectedLabels:
    """Kết quả của quá trình label correction.

    Chứa labels gốc, labels đã sửa, mask chỉ ra vị trí được sửa,
    và tổng số corrections.
    """

    original_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    corrected_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    correction_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    correction_count: int = 0

    @property
    def correction_ratio(self) -> float:
        """Tỷ lệ labels được sửa so với tổng labels."""
        if len(self.original_labels) == 0:
            return 0.0
        return self.correction_count / len(self.original_labels)


# ==============================================================================
# AntiPattern
# ==============================================================================


@dataclass
class AntiPattern:
    """Một pattern lịch sử dẫn đến thua lỗ.

    Được tích lũy qua nhiều cycles, merge khi cosine similarity > 0.8.
    """

    pattern_id: str = ""  # UUID
    indicator_conditions: Dict[str, float] = field(default_factory=dict)
    centroid: List[float] = field(default_factory=list)  # Mean feature vector
    outcome_avg_pnl: float = 0.0  # PnL trung bình khi pattern kích hoạt
    frequency: int = 0  # Số lần quan sát được
    first_seen_cycle: int = 0
    last_seen_cycle: int = 0
    confidence: float = 0.0  # frequency / total_cycles_observed


# ==============================================================================
# FeedbackCycleResult
# ==============================================================================


@dataclass
class FeedbackCycleResult:
    """Kết quả của một feedback cycle hoàn chỉnh.

    Nếu skipped=True, reason giải thích lý do bỏ qua (ví dụ:
    không đủ trades, config disabled, exception xảy ra).
    """

    skipped: bool = False
    reason: Optional[str] = None
    mistake_report: Optional[MistakeReport] = None
    hard_examples_count: int = 0
    corrections_count: int = 0
    retrain_result: Optional[Any] = None
