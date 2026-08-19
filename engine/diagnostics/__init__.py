"""
Engine Diagnostics Module — Xác minh tính đúng đắn của quy trình train/backtest.

Module này hoạt động như một lớp instrumentation độc lập,
không thay đổi logic hiện tại của engine mà chỉ quan sát, đo lường và báo cáo.

Components:
- Training Trigger Monitor: Phát hiện auto-training không mong muốn
- Timing Profiler: Đo lường thời gian thực tế train/backtest
- Data Leakage Detector: Kiểm tra data leakage
- Stub Detector: Phát hiện stub code
- Verification Runner: Orchestrate và tạo unified report
"""

from engine.diagnostics.models import (
    # Enums
    TriggerSource,
    ViolationType,
    CheckStatus,
    # Training Trigger Monitor
    TriggerEvent,
    # Timing Profiler
    PhaseTimingResult,
    EpochValidation,
    TimingProfileReport,
    BacktestTimingReport,
    # Data Leakage Detector
    LeakageViolation,
    LeakageReport,
    # Stub Detector
    StubFinding,
    StubReport,
    # Verification Runner
    CheckResult,
    VerificationReport,
)

__all__ = [
    # Enums
    "TriggerSource",
    "ViolationType",
    "CheckStatus",
    # Training Trigger Monitor
    "TriggerEvent",
    # Timing Profiler
    "PhaseTimingResult",
    "EpochValidation",
    "TimingProfileReport",
    "BacktestTimingReport",
    # Data Leakage Detector
    "LeakageViolation",
    "LeakageReport",
    # Stub Detector
    "StubFinding",
    "StubReport",
    # Verification Runner
    "CheckResult",
    "VerificationReport",
]
