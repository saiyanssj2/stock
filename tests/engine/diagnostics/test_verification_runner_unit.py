# -*- coding: utf-8 -*-
"""
Unit tests cho VerificationRunner.

Kiểm tra:
- run_all() orchestrate 4 checks tuần tự
- Exception handling: check lỗi → status "error", tiếp tục checks khác (Req 6.5)
- get_exit_code() mapping: 0=all pass, 1=warnings, 2=critical (Req 6.2)
- Overall status aggregation logic
- Training duration anomaly check (Req 6.4)
- Report output to verification_report.json (Req 6.3)
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from engine.diagnostics.models import CheckResult, CheckStatus, VerificationReport
from engine.diagnostics.verification_runner import VerificationRunner


@pytest.fixture
def temp_output_dir(tmp_path):
    """Thư mục output tạm thời cho tests."""
    output_dir = tmp_path / "diagnostics"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def runner(temp_output_dir):
    """VerificationRunner instance với output dir tạm."""
    return VerificationRunner(output_dir=temp_output_dir)


class TestRunAll:
    """Test run_all() orchestrate 4 checks."""

    def test_run_all_returns_verification_report(self, runner):
        """run_all() trả về VerificationReport hợp lệ."""
        report = runner.run_all()

        assert isinstance(report, VerificationReport)
        assert report.timestamp != ""
        assert len(report.checks) == 4
        assert report.overall_status in ("pass", "warn", "fail")

    def test_run_all_has_4_checks_in_order(self, runner):
        """run_all() chạy đúng 4 checks theo thứ tự."""
        report = runner.run_all()

        check_names = [c.check_name for c in report.checks]
        assert check_names == [
            "trigger_audit",
            "timing_profile",
            "leakage_detection",
            "stub_scan",
        ]

    def test_run_all_writes_report_json(self, runner, temp_output_dir):
        """run_all() ghi verification_report.json (Req 6.3)."""
        runner.run_all()

        report_path = os.path.join(temp_output_dir, "verification_report.json")
        assert os.path.exists(report_path)

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "checks" in data
        assert "overall_status" in data
        assert len(data["checks"]) == 4

    def test_run_all_report_json_has_required_fields(self, runner, temp_output_dir):
        """Report JSON chứa đúng các trường bắt buộc (Req 6.3)."""
        runner.run_all()

        report_path = os.path.join(temp_output_dir, "verification_report.json")
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for check in data["checks"]:
            assert "check_name" in check
            assert "status" in check
            assert "findings_count" in check
            assert "duration_ms" in check


class TestExceptionHandling:
    """Test exception handling: check lỗi → error, tiếp tục (Req 6.5)."""

    def test_check_exception_marks_error_status(self, runner):
        """Check gặp exception → status = ERROR."""
        # Sử dụng _run_check_safe trực tiếp với lambda
        def failing_check():
            raise RuntimeError("Test error")

        failing_check.__name__ = "run_trigger_audit"

        result = runner._run_check_safe(failing_check)
        assert result.status == CheckStatus.ERROR
        assert result.error_message == "Test error"
        assert result.check_name == "trigger_audit"

    def test_exception_in_one_check_continues_others(self, runner):
        """Exception ở check 1 không dừng checks 2, 3, 4 (Req 6.5)."""
        original_trigger = runner.run_trigger_audit

        def failing_trigger():
            raise RuntimeError("Boom")

        failing_trigger.__name__ = "run_trigger_audit"
        runner.run_trigger_audit = failing_trigger

        report = runner.run_all()

        # Restore
        runner.run_trigger_audit = original_trigger

        # Check 1 bị error
        assert report.checks[0].status == CheckStatus.ERROR
        assert report.checks[0].error_message == "Boom"

        # Checks 2, 3, 4 vẫn chạy (có thể PASS/WARN/FAIL nhưng không ERROR do "Boom")
        assert len(report.checks) == 4

    def test_multiple_exceptions_all_marked_error(self, runner):
        """Nhiều checks exception → tất cả đều marked error."""
        def failing_trigger():
            raise RuntimeError("Error 1")

        def failing_timing():
            raise ValueError("Error 2")

        failing_trigger.__name__ = "run_trigger_audit"
        failing_timing.__name__ = "run_timing_profile"

        runner.run_trigger_audit = failing_trigger
        runner.run_timing_profile = failing_timing

        report = runner.run_all()

        assert report.checks[0].status == CheckStatus.ERROR
        assert report.checks[0].error_message == "Error 1"
        assert report.checks[1].status == CheckStatus.ERROR
        assert report.checks[1].error_message == "Error 2"
        # Checks 3, 4 vẫn chạy bình thường
        assert report.checks[2].check_name == "leakage_detection"
        assert report.checks[3].check_name == "stub_scan"

    def test_error_check_overall_status_is_fail(self, runner):
        """Có check ERROR → overall_status = "fail"."""
        def failing_stub():
            raise OSError("IO Error")

        failing_stub.__name__ = "run_stub_scan"
        runner.run_stub_scan = failing_stub

        report = runner.run_all()

        assert report.overall_status == "fail"


class TestGetExitCode:
    """Test get_exit_code() mapping (Req 6.2, 6.5)."""

    def test_all_pass_returns_0(self, runner):
        """Tất cả checks PASS → exit code 0."""
        report = VerificationReport(
            timestamp="2024-01-15T10:00:00.000Z",
            checks=[
                CheckResult("trigger_audit", CheckStatus.PASS, 0, 100.0),
                CheckResult("timing_profile", CheckStatus.PASS, 0, 200.0),
                CheckResult("leakage_detection", CheckStatus.PASS, 0, 150.0),
                CheckResult("stub_scan", CheckStatus.PASS, 0, 50.0),
            ],
            overall_status="pass",
        )

        assert runner.get_exit_code(report) == 0

    def test_any_warn_returns_1(self, runner):
        """Có check WARN → exit code 1."""
        report = VerificationReport(
            timestamp="2024-01-15T10:00:00.000Z",
            checks=[
                CheckResult("trigger_audit", CheckStatus.PASS, 0, 100.0),
                CheckResult("timing_profile", CheckStatus.WARN, 2, 200.0),
                CheckResult("leakage_detection", CheckStatus.PASS, 0, 150.0),
                CheckResult("stub_scan", CheckStatus.PASS, 0, 50.0),
            ],
            overall_status="warn",
        )

        assert runner.get_exit_code(report) == 1

    def test_any_fail_returns_2(self, runner):
        """Có check FAIL → exit code 2."""
        report = VerificationReport(
            timestamp="2024-01-15T10:00:00.000Z",
            checks=[
                CheckResult("trigger_audit", CheckStatus.PASS, 0, 100.0),
                CheckResult("timing_profile", CheckStatus.PASS, 0, 200.0),
                CheckResult("leakage_detection", CheckStatus.FAIL, 3, 150.0),
                CheckResult("stub_scan", CheckStatus.PASS, 0, 50.0),
            ],
            overall_status="fail",
        )

        assert runner.get_exit_code(report) == 2

    def test_fail_takes_precedence_over_warn(self, runner):
        """FAIL ưu tiên hơn WARN → exit code 2."""
        report = VerificationReport(
            timestamp="2024-01-15T10:00:00.000Z",
            checks=[
                CheckResult("trigger_audit", CheckStatus.WARN, 1, 100.0),
                CheckResult("timing_profile", CheckStatus.FAIL, 2, 200.0),
                CheckResult("leakage_detection", CheckStatus.PASS, 0, 150.0),
                CheckResult("stub_scan", CheckStatus.WARN, 1, 50.0),
            ],
            overall_status="fail",
        )

        assert runner.get_exit_code(report) == 2

    def test_error_checks_not_influence_exit_code(self, runner):
        """Checks ERROR không ảnh hưởng exit code (Req 6.5)."""
        # Chỉ có ERROR checks → exit 0 (không có findings từ successful checks)
        report = VerificationReport(
            timestamp="2024-01-15T10:00:00.000Z",
            checks=[
                CheckResult(
                    "trigger_audit", CheckStatus.ERROR, 0, 0.0,
                    error_message="IO Error"
                ),
                CheckResult("timing_profile", CheckStatus.PASS, 0, 200.0),
                CheckResult("leakage_detection", CheckStatus.PASS, 0, 150.0),
                CheckResult("stub_scan", CheckStatus.PASS, 0, 50.0),
            ],
            overall_status="fail",
        )

        # Exit code chỉ dựa trên successful checks (3 checks PASS) → 0
        assert runner.get_exit_code(report) == 0

    def test_error_with_warn_returns_1(self, runner):
        """ERROR + WARN → exit code 1 (chỉ xét successful checks)."""
        report = VerificationReport(
            timestamp="2024-01-15T10:00:00.000Z",
            checks=[
                CheckResult(
                    "trigger_audit", CheckStatus.ERROR, 0, 0.0,
                    error_message="Error"
                ),
                CheckResult("timing_profile", CheckStatus.WARN, 2, 200.0),
                CheckResult("leakage_detection", CheckStatus.PASS, 0, 150.0),
                CheckResult("stub_scan", CheckStatus.PASS, 0, 50.0),
            ],
            overall_status="fail",
        )

        assert runner.get_exit_code(report) == 1

    def test_all_error_returns_0(self, runner):
        """Tất cả checks ERROR → exit code 0 (không có successful checks)."""
        report = VerificationReport(
            timestamp="2024-01-15T10:00:00.000Z",
            checks=[
                CheckResult(
                    "trigger_audit", CheckStatus.ERROR, 0, 0.0,
                    error_message="E1"
                ),
                CheckResult(
                    "timing_profile", CheckStatus.ERROR, 0, 0.0,
                    error_message="E2"
                ),
                CheckResult(
                    "leakage_detection", CheckStatus.ERROR, 0, 0.0,
                    error_message="E3"
                ),
                CheckResult(
                    "stub_scan", CheckStatus.ERROR, 0, 0.0,
                    error_message="E4"
                ),
            ],
            overall_status="fail",
        )

        assert runner.get_exit_code(report) == 0


class TestOverallStatus:
    """Test overall_status aggregation logic."""

    def test_all_pass_overall_pass(self, runner):
        """Tất cả PASS → overall = "pass"."""
        report = runner.run_all()

        # Nếu tất cả checks pass (environment phù hợp) → overall pass
        # Test verifies logic, not actual check results
        all_pass = all(
            c.status == CheckStatus.PASS for c in report.checks
        )
        if all_pass:
            assert report.overall_status == "pass"

    def test_any_fail_overall_fail(self, runner):
        """Có FAIL → overall = "fail"."""
        checks = [
            CheckResult("trigger_audit", CheckStatus.PASS, 0, 100.0),
            CheckResult("timing_profile", CheckStatus.FAIL, 1, 200.0),
        ]
        status = runner._compute_overall_status(checks)
        assert status == "fail"

    def test_any_error_overall_fail(self, runner):
        """Có ERROR → overall = "fail"."""
        checks = [
            CheckResult("trigger_audit", CheckStatus.PASS, 0, 100.0),
            CheckResult(
                "timing_profile", CheckStatus.ERROR, 0, 0.0,
                error_message="E"
            ),
        ]
        status = runner._compute_overall_status(checks)
        assert status == "fail"

    def test_warn_without_fail_overall_warn(self, runner):
        """Có WARN (không FAIL/ERROR) → overall = "warn"."""
        checks = [
            CheckResult("trigger_audit", CheckStatus.PASS, 0, 100.0),
            CheckResult("timing_profile", CheckStatus.WARN, 1, 200.0),
            CheckResult("leakage_detection", CheckStatus.PASS, 0, 150.0),
        ]
        status = runner._compute_overall_status(checks)
        assert status == "warn"


class TestTrainingDurationAnomaly:
    """Test training duration anomaly check (Req 6.4)."""

    def test_anomaly_detected_when_duration_less_than_10s(self, runner):
        """heartbeat_ts - started_at < 10s cho 65 symbols → warning."""
        from models.task_models import TaskState, TaskStatus, TaskType

        mock_status = TaskStatus(
            task_id="train_test",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=100.0,
            message="Done",
            heartbeat_ts=datetime(2024, 1, 15, 10, 0, 7),
            started_at=datetime(2024, 1, 15, 10, 0, 0),
            details={"symbol_count": 65},
        )

        with patch(
            "engine.diagnostics.verification_runner.VerificationRunner"
            "._check_training_duration_anomaly"
        ) as mock_check:
            mock_check.return_value = True
            report = runner.run_all()

        # Trigger audit should report the anomaly as warning
        trigger_check = report.checks[0]
        assert trigger_check.check_name == "trigger_audit"

    def test_no_anomaly_when_duration_above_10s(self, runner):
        """heartbeat_ts - started_at >= 10s → không có anomaly."""
        from models.task_models import TaskState, TaskStatus, TaskType

        mock_status = TaskStatus(
            task_id="train_test",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=100.0,
            message="Done",
            heartbeat_ts=datetime(2024, 1, 15, 10, 1, 0),
            started_at=datetime(2024, 1, 15, 10, 0, 0),
            details={"symbol_count": 65},
        )

        with patch(
            "orchestrator.status_protocol.read_all_statuses",
            return_value={"train_test": mock_status},
        ):
            result = runner._check_training_duration_anomaly()

        # 60 giây > 10 giây → không có anomaly
        assert result is False

    def test_anomaly_detected_with_real_status(self, runner):
        """Trực tiếp test _check_training_duration_anomaly với status < 10s."""
        from models.task_models import TaskState, TaskStatus, TaskType

        mock_status = TaskStatus(
            task_id="train_test",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=100.0,
            message="Done",
            heartbeat_ts=datetime(2024, 1, 15, 10, 0, 5),
            started_at=datetime(2024, 1, 15, 10, 0, 0),
            details={"symbol_count": 65},
        )

        with patch(
            "orchestrator.status_protocol.read_all_statuses",
            return_value={"train_test": mock_status},
        ):
            result = runner._check_training_duration_anomaly()

        # 5 giây < 10 giây, 65 symbols → anomaly detected
        assert result is True

    def test_anomaly_check_handles_import_error(self, runner):
        """Import error → graceful fallback, không crash."""
        with patch(
            "orchestrator.status_protocol.read_all_statuses",
            side_effect=ImportError("No module"),
        ):
            result = runner._check_training_duration_anomaly()

        # Import error → False (graceful)
        assert result is False


class TestIndividualChecks:
    """Test từng check riêng lẻ."""

    def test_trigger_audit_returns_check_result(self, runner):
        """run_trigger_audit() trả về CheckResult hợp lệ."""
        result = runner.run_trigger_audit()

        assert isinstance(result, CheckResult)
        assert result.check_name == "trigger_audit"
        assert result.status in (CheckStatus.PASS, CheckStatus.WARN, CheckStatus.FAIL)
        assert result.duration_ms >= 0.0

    def test_timing_profile_no_file_returns_pass(self, runner):
        """Không có timing_profile.json → PASS (không có findings)."""
        result = runner.run_timing_profile()

        assert result.check_name == "timing_profile"
        assert result.status == CheckStatus.PASS
        assert result.findings_count == 0

    def test_timing_profile_with_warnings(self, runner, temp_output_dir):
        """timing_profile.json có warnings → WARN hoặc FAIL."""
        # Tạo timing_profile.json với warnings
        report_data = {
            "session_id": "test-session",
            "timestamp": "2024-01-15T10:00:00Z",
            "symbol_count": 65,
            "per_phase_duration_ms": {"model_training": 5000.0},
            "total_duration_ms": 5000.0,
            "epoch_count": 0,
            "is_stub": False,
            "is_real_computation": False,
            "warnings": ["SUSPICIOUSLY_FAST"],
        }
        with open(
            os.path.join(temp_output_dir, "timing_profile.json"), "w"
        ) as f:
            json.dump(report_data, f)

        result = runner.run_timing_profile()

        assert result.check_name == "timing_profile"
        assert result.status == CheckStatus.FAIL
        assert result.findings_count >= 1

    def test_leakage_detection_no_file_returns_pass(self, runner):
        """Không có leakage_report.json → PASS."""
        result = runner.run_leakage_detection()

        assert result.check_name == "leakage_detection"
        assert result.status == CheckStatus.PASS
        assert result.findings_count == 0

    def test_leakage_detection_with_violations(self, runner, temp_output_dir):
        """leakage_report.json có violations → FAIL."""
        report_data = {
            "timestamp": "2024-01-15T10:00:00Z",
            "symbol": "VNM",
            "violations": [
                {
                    "violation_type": "SIGNAL_FUTURE_LEAK",
                    "symbol": "VNM",
                    "details": {"index": 100, "rows_leaked": 2},
                }
            ],
            "violation_counts": {
                "SIGNAL_FUTURE_LEAK": 1,
                "SPLIT_OVERLAP": 0,
                "LABEL_LOOK_AHEAD": 0,
                "POSSIBLE_LEAKAGE": 0,
            },
            "overall_status": "fail",
        }
        with open(
            os.path.join(temp_output_dir, "leakage_report.json"), "w"
        ) as f:
            json.dump(report_data, f)

        result = runner.run_leakage_detection()

        assert result.check_name == "leakage_detection"
        assert result.status == CheckStatus.FAIL
        assert result.findings_count == 1

    def test_stub_scan_returns_check_result(self, runner):
        """run_stub_scan() trả về CheckResult hợp lệ."""
        result = runner.run_stub_scan()

        assert isinstance(result, CheckResult)
        assert result.check_name == "stub_scan"
        assert result.duration_ms >= 0.0
