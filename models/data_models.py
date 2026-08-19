"""
Data models - Định nghĩa các model liên quan đến data pipeline và auto-learner config.

Bao gồm:
- UpdateResult: Kết quả data update
- AutoLearnerConfig: Cấu hình Auto-Learner
- CycleResult: Kết quả một auto-learning cycle
- RetryPolicy: Chính sách retry cho operations có thể fail

References: Req 8.2, 3.4, 3.5
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Type


@dataclass
class UpdateResult:
    """
    Kết quả data update từ DataPipeline.

    Cho biết bao nhiêu symbol update thành công, bao nhiêu thất bại,
    và thời gian update cho từng symbol.

    Attributes:
        total_symbols: Tổng số symbol cần update
        success_count: Số symbol update thành công
        failed_symbols: Danh sách symbol update thất bại
        errors: Map symbol → error message cho các symbol thất bại
        duration_seconds: Tổng thời gian update (giây)
        last_update_timestamps: Map symbol → timestamp lần update cuối
    """

    total_symbols: int
    success_count: int
    failed_symbols: List[str]
    errors: Dict[str, str]
    duration_seconds: float
    last_update_timestamps: Dict[str, datetime] = field(default_factory=dict)


@dataclass
class AutoLearnerConfig:
    """
    Cấu hình cho Auto-Learner module.

    Định nghĩa các tham số cho vòng lặp tự động:
    train → backtest → phân tích sai lầm → retrain.

    Attributes:
        cycle_interval_hours: Khoảng cách giữa các cycle (giờ)
        min_improvement_cycles: Số cycle liên tiếp cần cải thiện để ghi nhận trend
        phase_c_sharpe_threshold: Số benchmark strategies cần beat để chuyển Phase B
        phase_b_stable_cycles: Số cycles loss ổn định để chuyển Phase A
        data_update_time: Giờ auto-update data sau market close (HH:MM)
        api_rate_limit: Giới hạn requests/phút khi fetch data
        confidence_threshold: Ngưỡng confidence tối thiểu để recommend
    """

    cycle_interval_hours: float = 24.0
    min_improvement_cycles: int = 3
    phase_c_sharpe_threshold: int = 2
    phase_b_stable_cycles: int = 5
    data_update_time: str = "15:30"
    api_rate_limit: int = 20
    confidence_threshold: float = 0.5


@dataclass
class CycleResult:
    """
    Kết quả một auto-learning cycle hoàn chỉnh.

    Lưu lại performance metrics sau mỗi cycle để track improvement trend.
    Dùng để quyết định phase transition và log history.

    Attributes:
        cycle_number: Số thứ tự cycle
        phase: Phase training tại thời điểm cycle
        sharpe_ratio: Sharpe ratio đạt được
        win_rate: Tỷ lệ thắng (%)
        total_return: Tổng lợi nhuận (%)
        strategies_beaten: Số strategies đã beat (0-4)
        validation_loss: Loss trên validation set
        is_improving: True nếu tốt hơn cycle trước
        timestamp: Thời điểm hoàn thành cycle
        duration_seconds: Thời gian chạy cycle (giây)
        notes: Ghi chú bổ sung (VD: "phase transition triggered")
    """

    cycle_number: int
    phase: str
    sharpe_ratio: float
    win_rate: float
    total_return: float
    strategies_beaten: int
    validation_loss: float
    is_improving: bool
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    notes: str = ""


@dataclass
class RetryPolicy:
    """
    Chính sách retry cho các operation có thể fail.

    Dùng exponential backoff: delay = base_delay * (backoff_factor ^ attempt).
    Chỉ retry cho các loại error trong retryable_errors.

    Attributes:
        max_retries: Số lần retry tối đa
        base_delay_seconds: Delay cơ sở ban đầu (giây)
        max_delay_seconds: Delay tối đa (giây)
        backoff_factor: Hệ số nhân cho exponential backoff
        retryable_errors: Danh sách error types được phép retry
    """

    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_factor: float = 2.0
    retryable_errors: List[Type[Exception]] = field(default_factory=list)
