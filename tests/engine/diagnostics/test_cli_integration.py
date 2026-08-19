# -*- coding: utf-8 -*-
"""
Integration tests cho CLI command `python -m engine.diagnostics verify`.

Chạy subprocess thực tế, kiểm tra:
- Exit code hợp lệ (0, 1, hoặc 2)
- stdout chứa header "=== Verification Report ==="
- stdout chứa tên 4 checks: trigger_audit, timing_profile, leakage_detection, stub_scan
- stdout chứa dòng "Overall:"
- stdout chứa dòng "Exit code:"
- Invalid command (e.g. `python -m engine.diagnostics foo`) exit code 1

Requirements: 6.1, 6.2
"""

import subprocess
import sys
import os

import pytest

# Đường dẫn project root — dùng làm cwd khi chạy subprocess
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


class TestCliIntegrationVerify:
    """Integration tests cho `python -m engine.diagnostics verify`."""

    def _run_verify(self, args: list[str] | None = None) -> subprocess.CompletedProcess:
        """Helper chạy CLI command và trả về CompletedProcess.

        Args:
            args: Arguments truyền cho module. Mặc định ["verify"].

        Returns:
            subprocess.CompletedProcess chứa returncode, stdout, stderr.
        """
        if args is None:
            args = ["verify"]

        cmd = [sys.executable, "-m", "engine.diagnostics"] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=120,  # Timeout 120s cho trường hợp checks chạy lâu
        )
        return result

    def test_exit_code_is_valid(self):
        """Exit code phải là 0, 1, hoặc 2 — tương ứng pass/warn/fail."""
        result = self._run_verify()

        assert result.returncode in (0, 1, 2), (
            f"Exit code không hợp lệ: {result.returncode}. "
            f"stdout: {result.stdout[:500]}, stderr: {result.stderr[:500]}"
        )

    def test_stdout_contains_report_header(self):
        """stdout phải chứa header "=== Verification Report ==="."""
        result = self._run_verify()

        assert "=== Verification Report ===" in result.stdout, (
            f"Thiếu header trong stdout. stdout: {result.stdout[:500]}"
        )

    def test_stdout_contains_trigger_audit_check(self):
        """stdout phải chứa tên check "trigger_audit"."""
        result = self._run_verify()

        assert "trigger_audit" in result.stdout, (
            f"Thiếu check 'trigger_audit' trong stdout. stdout: {result.stdout[:500]}"
        )

    def test_stdout_contains_timing_profile_check(self):
        """stdout phải chứa tên check "timing_profile"."""
        result = self._run_verify()

        assert "timing_profile" in result.stdout, (
            f"Thiếu check 'timing_profile' trong stdout. stdout: {result.stdout[:500]}"
        )

    def test_stdout_contains_leakage_detection_check(self):
        """stdout phải chứa tên check "leakage_detection"."""
        result = self._run_verify()

        assert "leakage_detection" in result.stdout, (
            f"Thiếu check 'leakage_detection' trong stdout. stdout: {result.stdout[:500]}"
        )

    def test_stdout_contains_stub_scan_check(self):
        """stdout phải chứa tên check "stub_scan"."""
        result = self._run_verify()

        assert "stub_scan" in result.stdout, (
            f"Thiếu check 'stub_scan' trong stdout. stdout: {result.stdout[:500]}"
        )

    def test_stdout_contains_overall_line(self):
        """stdout phải chứa dòng "Overall:" với status."""
        result = self._run_verify()

        assert "Overall:" in result.stdout, (
            f"Thiếu dòng 'Overall:' trong stdout. stdout: {result.stdout[:500]}"
        )

    def test_stdout_contains_exit_code_line(self):
        """stdout phải chứa dòng "Exit code:" hiển thị mã exit."""
        result = self._run_verify()

        assert "Exit code:" in result.stdout, (
            f"Thiếu dòng 'Exit code:' trong stdout. stdout: {result.stdout[:500]}"
        )

    def test_exit_code_line_matches_returncode(self):
        """Dòng "Exit code: X" trong stdout phải khớp với returncode thực tế."""
        result = self._run_verify()

        # Tìm dòng "Exit code: X"
        expected_line = f"Exit code: {result.returncode}"
        assert expected_line in result.stdout, (
            f"Exit code line không khớp returncode. "
            f"Expected '{expected_line}' trong stdout. stdout: {result.stdout[:500]}"
        )

    def test_invalid_command_exits_with_code_1(self):
        """Command không hợp lệ (ví dụ 'foo') phải exit code 1."""
        result = self._run_verify(args=["foo"])

        assert result.returncode == 1, (
            f"Invalid command 'foo' nên exit code 1, got {result.returncode}. "
            f"stdout: {result.stdout[:200]}"
        )

    def test_no_args_exits_with_code_1(self):
        """Không truyền args phải exit code 1."""
        result = self._run_verify(args=[])

        assert result.returncode == 1, (
            f"No args nên exit code 1, got {result.returncode}. "
            f"stdout: {result.stdout[:200]}"
        )

    def test_invalid_command_shows_usage(self):
        """Command không hợp lệ phải hiển thị thông báo usage."""
        result = self._run_verify(args=["foo"])

        assert "Usage:" in result.stdout or "usage:" in result.stdout.lower(), (
            f"Thiếu thông báo usage cho invalid command. stdout: {result.stdout[:200]}"
        )
