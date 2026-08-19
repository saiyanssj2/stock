"""
Unit tests cho engine/diagnostics/__main__.py — CLI entry point.

Kiểm tra:
- print_summary format đúng
- main() exit code với invalid args
- main() gọi verify pipeline
"""

import io
import sys
from unittest.mock import patch, MagicMock

import pytest

from engine.diagnostics.__main__ import print_summary, main
from engine.diagnostics.models import CheckResult, CheckStatus, VerificationReport


class TestPrintSummary:
    """Test print_summary xuất đúng format ra stdout."""

    def _make_report(self, checks, overall_status="pass"):
        """Helper tạo VerificationReport."""
        return VerificationReport(
            timestamp="2024-01-01T00:00:00.000Z",
            checks=checks,
            overall_status=overall_status,
        )

    def test_all_pass_output(self, capsys):
        """Test summary khi tất cả checks PASS."""
        checks = [
            CheckResult(
                check_name="trigger_audit",
                status=CheckStatus.PASS,
                findings_count=0,
                duration_ms=150.3,
            ),
            CheckResult(
                check_name="timing_profile",
                status=CheckStatus.PASS,
                findings_count=0,
                duration_ms=5200.0,
            ),
        ]
        report = self._make_report(checks, "pass")

        print_summary(report)

        captured = capsys.readouterr()
        assert "=== Verification Report ===" in captured.out
        assert "[PASS] trigger_audit" in captured.out
        assert "0 findings" in captured.out
        assert "150.3ms" in captured.out
        assert "Overall: pass" in captured.out

    def test_mixed_status_output(self, capsys):
        """Test summary với mixed statuses."""
        checks = [
            CheckResult(
                check_name="trigger_audit",
                status=CheckStatus.PASS,
                findings_count=0,
                duration_ms=100.0,
            ),
            CheckResult(
                check_name="timing_profile",
                status=CheckStatus.WARN,
                findings_count=2,
                duration_ms=5200.0,
            ),
            CheckResult(
                check_name="stub_scan",
                status=CheckStatus.FAIL,
                findings_count=1,
                duration_ms=80.2,
            ),
        ]
        report = self._make_report(checks, "fail")

        print_summary(report)

        captured = capsys.readouterr()
        assert "[PASS] trigger_audit" in captured.out
        assert "[WARN] timing_profile" in captured.out
        assert "[FAIL] stub_scan" in captured.out
        assert "2 findings" in captured.out
        assert "1 findings" in captured.out
        assert "Overall: fail" in captured.out

    def test_error_status_output(self, capsys):
        """Test summary khi có check ERROR."""
        checks = [
            CheckResult(
                check_name="leakage_detection",
                status=CheckStatus.ERROR,
                findings_count=0,
                duration_ms=0.0,
                error_message="ImportError: module not found",
            ),
        ]
        report = self._make_report(checks, "fail")

        print_summary(report)

        captured = capsys.readouterr()
        assert "[ERROR] leakage_detection" in captured.out


class TestMainFunction:
    """Test main() CLI logic."""

    def test_no_args_shows_usage(self):
        """Test main() hiển thị usage khi không có args."""
        with patch.object(sys, "argv", ["engine.diagnostics"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_invalid_command_shows_usage(self):
        """Test main() hiển thị usage khi command sai."""
        with patch.object(sys, "argv", ["engine.diagnostics", "check"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_verify_command_runs_pipeline(self):
        """Test main() với verify command chạy pipeline và exit đúng code."""
        mock_report = VerificationReport(
            timestamp="2024-01-01T00:00:00.000Z",
            checks=[
                CheckResult(
                    check_name="trigger_audit",
                    status=CheckStatus.PASS,
                    findings_count=0,
                    duration_ms=100.0,
                ),
            ],
            overall_status="pass",
        )

        with patch.object(sys, "argv", ["engine.diagnostics", "verify"]):
            with patch(
                "engine.diagnostics.__main__.VerificationRunner"
            ) as mock_runner_cls:
                mock_runner = MagicMock()
                mock_runner.run_all.return_value = mock_report
                mock_runner.get_exit_code.return_value = 0
                mock_runner_cls.return_value = mock_runner

                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 0
                mock_runner.run_all.assert_called_once()
                mock_runner.get_exit_code.assert_called_once_with(mock_report)

    def test_verify_exits_with_code_2_on_failure(self):
        """Test main() exit code 2 khi có critical issues."""
        mock_report = VerificationReport(
            timestamp="2024-01-01T00:00:00.000Z",
            checks=[
                CheckResult(
                    check_name="stub_scan",
                    status=CheckStatus.FAIL,
                    findings_count=1,
                    duration_ms=50.0,
                ),
            ],
            overall_status="fail",
        )

        with patch.object(sys, "argv", ["engine.diagnostics", "verify"]):
            with patch(
                "engine.diagnostics.__main__.VerificationRunner"
            ) as mock_runner_cls:
                mock_runner = MagicMock()
                mock_runner.run_all.return_value = mock_report
                mock_runner.get_exit_code.return_value = 2
                mock_runner_cls.return_value = mock_runner

                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 2
