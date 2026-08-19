"""
CLI entry point: python -m engine.diagnostics verify

Chạy verification pipeline và xuất summary ra stdout.
Exit codes:
  0 = tất cả checks PASS
  1 = có warnings
  2 = có critical issues

Requirements: 6.1, 6.2
"""

import sys

from engine.diagnostics.models import CheckStatus
from engine.diagnostics.verification_runner import VerificationRunner


def print_summary(report) -> None:
    """In summary report ra stdout.

    Format:
        === Verification Report ===
        [PASS] trigger_audit       - 0 findings (150.3ms)
        [WARN] timing_profile     - 2 findings (5200.0ms)
        ...
        Overall: pass/warn/fail
        Exit code: 0/1/2

    Args:
        report: VerificationReport chứa kết quả checks.
    """
    # Status label mapping
    status_labels = {
        CheckStatus.PASS: "PASS",
        CheckStatus.WARN: "WARN",
        CheckStatus.FAIL: "FAIL",
        CheckStatus.ERROR: "ERROR",
    }

    print("=== Verification Report ===")

    for check in report.checks:
        label = status_labels.get(check.status, "UNKNOWN")
        # Pad tên check để căn cột
        name_padded = check.check_name.ljust(20)
        duration_str = f"{check.duration_ms:.1f}ms"
        print(f"[{label}] {name_padded}- {check.findings_count} findings ({duration_str})")

    print("")
    print(f"Overall: {report.overall_status}")


def main() -> None:
    """Entry point chính cho CLI command.

    Usage: python -m engine.diagnostics verify
    """
    if len(sys.argv) < 2 or sys.argv[1] != "verify":
        print("Usage: python -m engine.diagnostics verify")
        sys.exit(1)

    runner = VerificationRunner()
    report = runner.run_all()
    exit_code = runner.get_exit_code(report)

    print_summary(report)
    print(f"Exit code: {exit_code}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
