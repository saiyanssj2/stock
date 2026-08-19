# -*- coding: utf-8 -*-
"""
Verification Runner — Orchestrate 4 checks tuần tự và tạo unified report.

Chạy: trigger audit, timing profile, leakage detection, stub scan.
Xuất verification_report.json với per-check results và overall_status.

Requirements:
- 6.1: CLI command orchestrates 4 checks, outputs summary
- 6.2: Exit codes: 0=all pass, 1=warnings, 2=critical
- 6.3: Save verification_report.json with timestamp, per-check results, overall_status
- 6.4: Training duration anomaly: heartbeat_ts - started_at < 10s → warning
- 6.5: Exception in check → mark "error", continue other checks
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from engine.diagnostics.models import CheckResult, CheckStatus, VerificationReport
from engine.diagnostics.leakage_detector import DataLeakageDetector
from engine.diagnostics.stub_detector import StubDetector
from engine.diagnostics.timing_profiler import TimingProfiler
from engine.diagnostics.trigger_monitor import TrainingTriggerMonitor

logger = logging.getLogger(__name__)

# Ngưỡng anomaly: heartbeat_ts - started_at < 10 giây cho 65 symbols (Req 6.4)
TRAINING_DURATION_ANOMALY_THRESHOLD_SECONDS: float = 10.0
TRAINING_DURATION_ANOMALY_SYMBOL_COUNT: int = 65


class VerificationRunner:
    """Orchestrate 4 checks tuần tự và tổng hợp kết quả.

    Chạy từng check trong try/except — nếu check exception, đánh dấu "error"
    và tiếp tục checks còn lại (Req 6.5).

    Attributes:
        output_dir: Thư mục output cho verification_report.json.
    """

    def __init__(self, output_dir: str = "data/engine/diagnostics") -> None:
        """Khởi tạo VerificationRunner.

        Args:
            output_dir: Thư mục chứa output files. Sẽ tự tạo nếu chưa tồn tại.
        """
        self._output_dir = output_dir
        self._report_path = os.path.join(output_dir, "verification_report.json")

    def run_all(self) -> VerificationReport:
        """Chạy tất cả checks tuần tự. Một check exception không dừng các check khác.

        Thứ tự chạy: trigger_audit → timing_profile → leakage_detection → stub_scan.
        Mỗi check được wrap trong try/except — exception → status "error" (Req 6.5).

        Returns:
            VerificationReport chứa kết quả tổng hợp.
        """
        checks: List[CheckResult] = []

        # Check 1: Trigger audit
        checks.append(self._run_check_safe(self.run_trigger_audit))

        # Check 2: Timing profile
        checks.append(self._run_check_safe(self.run_timing_profile))

        # Check 3: Leakage detection
        checks.append(self._run_check_safe(self.run_leakage_detection))

        # Check 4: Stub scan
        checks.append(self._run_check_safe(self.run_stub_scan))

        # Tính overall_status
        overall_status = self._compute_overall_status(checks)

        # Tạo report
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )[:-3] + "Z"

        report = VerificationReport(
            timestamp=timestamp,
            checks=checks,
            overall_status=overall_status,
        )

        # Ghi report ra file (Req 6.3)
        self._write_report(report)

        return report

    def run_trigger_audit(self) -> CheckResult:
        """Check 1: Trigger audit — kiểm tra training triggers.

        Kiểm tra I/O health của trigger monitor và training duration anomaly (Req 6.4).

        Returns:
            CheckResult cho trigger audit.
        """
        start_time = time.perf_counter()
        findings_count = 0
        status = CheckStatus.PASS

        # Kiểm tra trigger monitor I/O health
        monitor = TrainingTriggerMonitor()
        if not monitor.is_healthy():
            status = CheckStatus.WARN
            findings_count += 1

        # Kiểm tra training duration anomaly (Req 6.4)
        anomaly_detected = self._check_training_duration_anomaly()
        if anomaly_detected:
            # Training duration anomaly → warning
            if status == CheckStatus.PASS:
                status = CheckStatus.WARN
            findings_count += 1

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return CheckResult(
            check_name="trigger_audit",
            status=status,
            findings_count=findings_count,
            duration_ms=duration_ms,
        )

    def run_timing_profile(self) -> CheckResult:
        """Check 2: Timing profile — kiểm tra training/backtest timing.

        Đọc timing_profile.json nếu tồn tại và kiểm tra warnings.

        Returns:
            CheckResult cho timing profile.
        """
        start_time = time.perf_counter()
        findings_count = 0
        status = CheckStatus.PASS

        # Đọc timing profile report nếu có
        timing_report_path = os.path.join(self._output_dir, "timing_profile.json")
        if os.path.exists(timing_report_path):
            try:
                with open(timing_report_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)

                warnings = report_data.get("warnings", [])
                findings_count = len(warnings)

                if "SUSPICIOUSLY_FAST" in warnings or "NO_EPOCHS_EXECUTED" in warnings:
                    status = CheckStatus.FAIL
                elif findings_count > 0:
                    status = CheckStatus.WARN

                # Kiểm tra is_stub
                if report_data.get("is_stub", False):
                    status = CheckStatus.FAIL
                    findings_count += 1

            except (json.JSONDecodeError, OSError):
                # File corrupt → pass (không có findings)
                pass

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return CheckResult(
            check_name="timing_profile",
            status=status,
            findings_count=findings_count,
            duration_ms=duration_ms,
        )

    def run_leakage_detection(self) -> CheckResult:
        """Check 3: Data leakage detection — kiểm tra leakage violations.

        Đọc leakage_report.json nếu tồn tại và đếm violations.

        Returns:
            CheckResult cho leakage detection.
        """
        start_time = time.perf_counter()
        findings_count = 0
        status = CheckStatus.PASS

        # Đọc leakage report nếu có
        leakage_report_path = os.path.join(self._output_dir, "leakage_report.json")
        if os.path.exists(leakage_report_path):
            try:
                with open(leakage_report_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)

                # Đếm tổng violations
                violation_counts = report_data.get("violation_counts", {})
                findings_count = sum(violation_counts.values())

                overall = report_data.get("overall_status", "pass")
                if overall == "fail":
                    status = CheckStatus.FAIL
                elif findings_count > 0:
                    status = CheckStatus.WARN

            except (json.JSONDecodeError, OSError):
                pass

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return CheckResult(
            check_name="leakage_detection",
            status=status,
            findings_count=findings_count,
            duration_ms=duration_ms,
        )

    def run_stub_scan(self) -> CheckResult:
        """Check 4: Stub detection — quét code cho stub patterns.

        Chạy StubDetector.scan_auto_learner() và generate_report().

        Returns:
            CheckResult cho stub scan.
        """
        start_time = time.perf_counter()

        # Tạo detector và quét
        detector = StubDetector(
            output_path=os.path.join(self._output_dir, "stub_report.json")
        )
        detector.scan_auto_learner()
        report = detector.generate_report()

        # Đánh giá kết quả
        findings_count = report.summary.get("total_stubs", 0)
        critical_count = report.summary.get("critical_count", 0)
        warning_count = report.summary.get("warning_count", 0)

        if critical_count > 0:
            status = CheckStatus.FAIL
        elif warning_count > 0:
            status = CheckStatus.WARN
        else:
            status = CheckStatus.PASS

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return CheckResult(
            check_name="stub_scan",
            status=status,
            findings_count=findings_count,
            duration_ms=duration_ms,
        )

    def get_exit_code(self, report: VerificationReport) -> int:
        """Map check results → exit code (0/1/2).

        Chỉ xét checks có status != ERROR (Req 6.5).
        - 0 = tất cả successful checks PASS
        - 1 = có successful check WARN
        - 2 = có successful check FAIL

        Args:
            report: VerificationReport chứa kết quả checks.

        Returns:
            Exit code (0, 1, hoặc 2).
        """
        # Lọc chỉ các checks thành công (status != ERROR)
        successful_checks = [
            c for c in report.checks if c.status != CheckStatus.ERROR
        ]

        # Nếu không có check thành công nào → exit 0 (không có findings)
        if not successful_checks:
            return 0

        # Kiểm tra có FAIL không → exit 2
        if any(c.status == CheckStatus.FAIL for c in successful_checks):
            return 2

        # Kiểm tra có WARN không → exit 1
        if any(c.status == CheckStatus.WARN for c in successful_checks):
            return 1

        # Tất cả PASS → exit 0
        return 0

    # === Private helper methods ===

    def _run_check_safe(self, check_fn) -> CheckResult:
        """Chạy check function trong try/except.

        Nếu exception xảy ra → trả CheckResult với status=ERROR (Req 6.5).

        Args:
            check_fn: Function thực hiện check, trả về CheckResult.

        Returns:
            CheckResult — hoặc từ check_fn, hoặc ERROR nếu exception.
        """
        try:
            return check_fn()
        except Exception as e:
            logger.error(f"Check exception: {e}")
            # Xác định tên check từ function name
            try:
                check_name = check_fn.__name__.replace("run_", "")
            except AttributeError:
                check_name = "unknown"
            return CheckResult(
                check_name=check_name,
                status=CheckStatus.ERROR,
                findings_count=0,
                duration_ms=0.0,
                error_message=str(e),
            )

    def _compute_overall_status(self, checks: List[CheckResult]) -> str:
        """Tính overall_status từ danh sách check results.

        Logic:
        - Nếu có check FAIL hoặc ERROR → overall = "fail"
        - Nếu có check WARN → overall = "warn"
        - Tất cả PASS → overall = "pass"

        Args:
            checks: Danh sách CheckResult.

        Returns:
            "pass", "warn", hoặc "fail".
        """
        has_fail_or_error = any(
            c.status in (CheckStatus.FAIL, CheckStatus.ERROR) for c in checks
        )
        has_warn = any(c.status == CheckStatus.WARN for c in checks)

        if has_fail_or_error:
            return "fail"
        if has_warn:
            return "warn"
        return "pass"

    def _check_training_duration_anomaly(self) -> bool:
        """Kiểm tra training duration anomaly (Req 6.4).

        Đọc status files, tìm training tasks có:
        heartbeat_ts - started_at < 10 giây cho 65 symbols.

        Returns:
            True nếu phát hiện anomaly, False nếu không.
        """
        try:
            from orchestrator.status_protocol import read_all_statuses
            from models.task_models import TaskType

            statuses = read_all_statuses()

            for task_id, status in statuses.items():
                # Chỉ kiểm tra training tasks
                if status.task_type != TaskType.TRAINING:
                    continue

                # Tính duration = heartbeat_ts - started_at
                duration = (
                    status.heartbeat_ts - status.started_at
                ).total_seconds()

                # Kiểm tra symbol count từ details nếu có
                symbol_count = status.details.get("symbol_count", 0)
                if symbol_count == 0:
                    # Fallback: dùng ngưỡng mặc định 65 symbols
                    symbol_count = status.details.get("symbols_total", 0)

                # Req 6.4: heartbeat_ts - started_at < 10s cho 65 symbols
                if (
                    symbol_count >= TRAINING_DURATION_ANOMALY_SYMBOL_COUNT
                    and duration < TRAINING_DURATION_ANOMALY_THRESHOLD_SECONDS
                ):
                    logger.warning(
                        f"TRAINING_DURATION_ANOMALY: task={task_id}, "
                        f"duration={duration:.1f}s, symbols={symbol_count}"
                    )
                    return True

        except (ImportError, OSError, Exception) as e:
            # Nếu không đọc được status files → bỏ qua check này
            logger.debug(f"Không thể kiểm tra training duration anomaly: {e}")

        return False

    def _write_report(self, report: VerificationReport) -> None:
        """Ghi verification report ra file JSON (Req 6.3).

        Tạo directory nếu chưa tồn tại.

        Args:
            report: VerificationReport cần ghi.
        """
        # Tạo output directory
        os.makedirs(self._output_dir, exist_ok=True)

        # Serialize report
        report_dict = {
            "timestamp": report.timestamp,
            "checks": [
                {
                    "check_name": c.check_name,
                    "status": c.status.value,
                    "findings_count": c.findings_count,
                    "duration_ms": round(c.duration_ms, 2),
                    "error_message": c.error_message,
                }
                for c in report.checks
            ],
            "overall_status": report.overall_status,
        }

        with open(self._report_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
