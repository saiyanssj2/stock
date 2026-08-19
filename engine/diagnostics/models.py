"""
Shared data models cho diagnostics module.

Chứa tất cả dataclass và enum dùng chung giữa các component:
- Training Trigger Monitor
- Timing Profiler
- Data Leakage Detector
- Stub Detector
- Verification Runner
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# === Enums ===


class TriggerSource(Enum):
    """Nguồn trigger training."""

    USER_MANUAL = "user_manual"
    AUTO_LEARNER_CYCLE = "auto_learner_cycle"
    BACKGROUND_SCHEDULED = "background_scheduled"
    DATA_UPDATE_HOOK = "data_update_hook"


class ViolationType(Enum):
    """Loại vi phạm data leakage."""

    SIGNAL_FUTURE_LEAK = "SIGNAL_FUTURE_LEAK"
    SPLIT_OVERLAP = "SPLIT_OVERLAP"
    LABEL_LOOK_AHEAD = "LABEL_LOOK_AHEAD"
    POSSIBLE_LEAKAGE = "POSSIBLE_LEAKAGE"


class CheckStatus(Enum):
    """Trạng thái kết quả check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    ERROR = "error"


# === Training Trigger Monitor models ===


@dataclass
class TriggerEvent:
    """Một event trigger training.

    Attributes:
        session_id: ID duy nhất của training session.
        timestamp: Thời điểm trigger, format ISO-8601.
        trigger_source: Nguồn trigger (user/auto/scheduled/hook).
        caller_component: Tên component đã trigger training.
        metadata: Thông tin bổ sung (cycle_number, elapsed_since_last_cycle, etc.).
    """

    session_id: str
    timestamp: str
    trigger_source: TriggerSource
    caller_component: str
    metadata: Dict[str, object] = field(default_factory=dict)


# === Timing Profiler models ===


@dataclass
class PhaseTimingResult:
    """Kết quả timing cho một phase.

    Attributes:
        phase_name: Tên phase (data_loading, feature_extraction, etc.).
        duration_ms: Thời gian thực thi tính bằng milliseconds.
        started_at: Thời điểm bắt đầu, format ISO-8601.
        ended_at: Thời điểm kết thúc, format ISO-8601.
    """

    phase_name: str
    duration_ms: float
    started_at: str
    ended_at: str


@dataclass
class EpochValidation:
    """Kết quả validation cho một epoch.

    Attributes:
        epoch_number: Số thứ tự epoch.
        has_forward_pass: Có forward pass hay không.
        has_backward_pass: Có backward pass (loss.backward()) hay không.
        has_optimizer_step: Có optimizer.step() hay không.
        is_valid: Epoch hợp lệ khi có đủ cả 3 operations.
        duration_ms: Thời gian epoch tính bằng milliseconds.
    """

    epoch_number: int
    has_forward_pass: bool
    has_backward_pass: bool
    has_optimizer_step: bool
    is_valid: bool
    duration_ms: float


@dataclass
class TimingProfileReport:
    """Báo cáo timing profile đầy đủ cho training session.

    Attributes:
        session_id: ID duy nhất của session.
        timestamp: Thời điểm tạo report, format ISO-8601.
        symbol_count: Số lượng symbols được train.
        per_phase_duration_ms: Thời gian từng phase (key=tên phase, value=ms).
        total_duration_ms: Tổng thời gian training.
        epoch_count: Số epochs thực sự chạy.
        is_stub: True nếu phát hiện stub function.
        is_real_computation: True nếu có computation thật (epoch_count > 0).
        warnings: Danh sách mã cảnh báo.
    """

    session_id: str
    timestamp: str
    symbol_count: int
    per_phase_duration_ms: Dict[str, float]
    total_duration_ms: float
    epoch_count: int
    is_stub: bool
    is_real_computation: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class BacktestTimingReport:
    """Báo cáo timing cho backtest per-symbol.

    Attributes:
        symbol: Mã chứng khoán.
        sessions_count: Số phiên giao dịch.
        total_time_ms: Tổng thời gian backtest cho symbol.
        strategy_signal_generation_ms: Thời gian generate signal.
        trade_execution_ms: Thời gian execute trades.
        metrics_computation_ms: Thời gian tính metrics.
        processed_days: Số ngày đã xử lý.
        total_days: Tổng số ngày trong dataset.
        warnings: Danh sách cảnh báo cho symbol này.
    """

    symbol: str
    sessions_count: int
    total_time_ms: float
    strategy_signal_generation_ms: float
    trade_execution_ms: float
    metrics_computation_ms: float
    processed_days: int
    total_days: int
    warnings: List[str] = field(default_factory=list)


# === Data Leakage Detector models ===


@dataclass
class LeakageViolation:
    """Một vi phạm data leakage.

    Attributes:
        violation_type: Loại vi phạm (SIGNAL_FUTURE_LEAK, SPLIT_OVERLAP, etc.).
        symbol: Mã chứng khoán liên quan.
        details: Chi tiết vi phạm (index, rows_leaked, split_pairs, etc.).
    """

    violation_type: ViolationType
    symbol: str
    details: Dict[str, object] = field(default_factory=dict)


@dataclass
class LeakageReport:
    """Báo cáo kiểm tra data leakage.

    Attributes:
        timestamp: Thời điểm tạo report, format ISO-8601.
        symbol: Mã chứng khoán được kiểm tra.
        violations: Danh sách các vi phạm phát hiện được.
        violation_counts: Số lượng vi phạm theo từng loại.
        overall_status: "pass" nếu 0 violations, "fail" nếu >= 1.
    """

    timestamp: str
    symbol: str
    violations: List[LeakageViolation]
    violation_counts: Dict[str, int]
    overall_status: str


# === Stub Detector models ===


@dataclass
class StubFinding:
    """Một function bị phát hiện là stub.

    Attributes:
        function_name: Tên function.
        file_path: Đường dẫn file chứa function.
        reason: Lý do bị đánh dấu stub (stub_keyword, no_computation, empty_body, target_not_found).
        severity: Mức độ nghiêm trọng ("critical" hoặc "warning").
        line_number: Số dòng trong file (optional).
        details: Mô tả chi tiết (optional).
    """

    function_name: str
    file_path: str
    reason: str
    severity: str
    line_number: Optional[int] = None
    details: Optional[str] = None


@dataclass
class StubReport:
    """Báo cáo stub detection.

    Attributes:
        timestamp: Thời điểm tạo report, format ISO-8601.
        stubs: Danh sách stub findings.
        summary: Tổng hợp (total_stubs, critical_count, warning_count).
    """

    timestamp: str
    stubs: List[StubFinding]
    summary: Dict[str, int]


# === Verification Runner models ===


@dataclass
class CheckResult:
    """Kết quả một check trong verification.

    Attributes:
        check_name: Tên check (trigger_audit, timing_profile, etc.).
        status: Trạng thái (pass/warn/fail/error).
        findings_count: Số findings phát hiện.
        duration_ms: Thời gian chạy check tính bằng ms.
        error_message: Thông báo lỗi nếu status là error.
    """

    check_name: str
    status: CheckStatus
    findings_count: int
    duration_ms: float
    error_message: Optional[str] = None


@dataclass
class VerificationReport:
    """Báo cáo verification tổng hợp.

    Attributes:
        timestamp: Thời điểm tạo report, format ISO-8601.
        checks: Danh sách kết quả từng check.
        overall_status: Trạng thái tổng hợp (pass/warn/fail).
    """

    timestamp: str
    checks: List[CheckResult]
    overall_status: str
