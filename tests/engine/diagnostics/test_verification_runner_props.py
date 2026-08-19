# -*- coding: utf-8 -*-
"""
Property-based tests cho VerificationRunner — Property 15: Exit code mapping correctness.

Feature: train-backtest-verification, Property 15: Exit code mapping correctness

**Validates: Requirements 6.2, 6.5**

Properties:
1. Exit code = 0 nếu tất cả successful checks (status != ERROR) đều PASS
2. Exit code = 1 nếu có ít nhất 1 successful check có status WARN (và không có FAIL)
3. Exit code = 2 nếu có ít nhất 1 successful check có status FAIL
4. Checks với status ERROR không ảnh hưởng exit code
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from typing import List

from engine.diagnostics.models import CheckResult, CheckStatus, VerificationReport
from engine.diagnostics.verification_runner import VerificationRunner


# ---------------------------------------------------------------------------
# Strategies — Sinh dữ liệu ngẫu nhiên hợp lệ
# ---------------------------------------------------------------------------

# Tên check hợp lệ
check_name_strategy = st.sampled_from([
    "trigger_audit",
    "timing_profile",
    "leakage_detection",
    "stub_scan",
    "custom_check",
])

# Duration ms hợp lệ (>= 0)
duration_ms_strategy = st.floats(min_value=0.0, max_value=10000.0, allow_nan=False)

# Findings count hợp lệ (>= 0)
findings_count_strategy = st.integers(min_value=0, max_value=100)

# Error message cho ERROR checks
error_message_strategy = st.text(min_size=1, max_size=50)


def check_result_with_status(status: CheckStatus) -> st.SearchStrategy:
    """Sinh CheckResult với status cố định."""
    if status == CheckStatus.ERROR:
        return st.builds(
            CheckResult,
            check_name=check_name_strategy,
            status=st.just(status),
            findings_count=st.just(0),
            duration_ms=st.just(0.0),
            error_message=error_message_strategy,
        )
    return st.builds(
        CheckResult,
        check_name=check_name_strategy,
        status=st.just(status),
        findings_count=findings_count_strategy,
        duration_ms=duration_ms_strategy,
        error_message=st.none(),
    )


# Strategy cho các status thành công (không phải ERROR)
successful_status_strategy = st.sampled_from([
    CheckStatus.PASS,
    CheckStatus.WARN,
    CheckStatus.FAIL,
])

# Strategy cho bất kỳ status nào
any_status_strategy = st.sampled_from([
    CheckStatus.PASS,
    CheckStatus.WARN,
    CheckStatus.FAIL,
    CheckStatus.ERROR,
])


@st.composite
def check_result_strategy(draw) -> CheckResult:
    """Sinh CheckResult ngẫu nhiên với bất kỳ status nào."""
    status = draw(any_status_strategy)
    name = draw(check_name_strategy)
    duration = draw(duration_ms_strategy)
    findings = draw(findings_count_strategy)

    if status == CheckStatus.ERROR:
        error_msg = draw(error_message_strategy)
        return CheckResult(
            check_name=name,
            status=status,
            findings_count=0,
            duration_ms=0.0,
            error_message=error_msg,
        )
    return CheckResult(
        check_name=name,
        status=status,
        findings_count=findings,
        duration_ms=duration,
        error_message=None,
    )


@st.composite
def all_pass_checks(draw) -> List[CheckResult]:
    """Sinh danh sách checks trong đó tất cả successful checks đều PASS."""
    # Ít nhất 1 check PASS
    num_pass = draw(st.integers(min_value=1, max_value=5))
    num_error = draw(st.integers(min_value=0, max_value=3))

    checks = []
    for _ in range(num_pass):
        checks.append(draw(check_result_with_status(CheckStatus.PASS)))
    for _ in range(num_error):
        checks.append(draw(check_result_with_status(CheckStatus.ERROR)))

    # Shuffle thứ tự
    draw(st.randoms()).shuffle(checks)
    return checks


@st.composite
def checks_with_warn_no_fail(draw) -> List[CheckResult]:
    """Sinh danh sách checks có ít nhất 1 WARN, không có FAIL."""
    # Ít nhất 1 WARN
    num_warn = draw(st.integers(min_value=1, max_value=3))
    num_pass = draw(st.integers(min_value=0, max_value=3))
    num_error = draw(st.integers(min_value=0, max_value=3))

    checks = []
    for _ in range(num_warn):
        checks.append(draw(check_result_with_status(CheckStatus.WARN)))
    for _ in range(num_pass):
        checks.append(draw(check_result_with_status(CheckStatus.PASS)))
    for _ in range(num_error):
        checks.append(draw(check_result_with_status(CheckStatus.ERROR)))

    # Shuffle thứ tự
    draw(st.randoms()).shuffle(checks)
    return checks


@st.composite
def checks_with_fail(draw) -> List[CheckResult]:
    """Sinh danh sách checks có ít nhất 1 FAIL (critical)."""
    # Ít nhất 1 FAIL
    num_fail = draw(st.integers(min_value=1, max_value=3))
    num_warn = draw(st.integers(min_value=0, max_value=2))
    num_pass = draw(st.integers(min_value=0, max_value=2))
    num_error = draw(st.integers(min_value=0, max_value=2))

    checks = []
    for _ in range(num_fail):
        checks.append(draw(check_result_with_status(CheckStatus.FAIL)))
    for _ in range(num_warn):
        checks.append(draw(check_result_with_status(CheckStatus.WARN)))
    for _ in range(num_pass):
        checks.append(draw(check_result_with_status(CheckStatus.PASS)))
    for _ in range(num_error):
        checks.append(draw(check_result_with_status(CheckStatus.ERROR)))

    # Shuffle thứ tự
    draw(st.randoms()).shuffle(checks)
    return checks


@st.composite
def all_error_checks(draw) -> List[CheckResult]:
    """Sinh danh sách checks toàn ERROR (không có successful check nào)."""
    num_error = draw(st.integers(min_value=1, max_value=5))

    checks = []
    for _ in range(num_error):
        checks.append(draw(check_result_with_status(CheckStatus.ERROR)))

    return checks


@st.composite
def random_checks_combination(draw) -> List[CheckResult]:
    """Sinh danh sách checks ngẫu nhiên (hỗn hợp mọi status)."""
    num_checks = draw(st.integers(min_value=1, max_value=8))
    checks = [draw(check_result_strategy()) for _ in range(num_checks)]
    return checks


def _make_report(checks: List[CheckResult]) -> VerificationReport:
    """Tạo VerificationReport từ danh sách checks."""
    return VerificationReport(
        timestamp="2024-01-15T10:00:00.000Z",
        checks=checks,
        overall_status="pass",  # overall_status không ảnh hưởng get_exit_code()
    )


def _compute_expected_exit_code(checks: List[CheckResult]) -> int:
    """Tính exit code mong đợi từ danh sách checks (reference implementation).

    Logic:
    - Lọc bỏ checks ERROR
    - Nếu không có successful check → 0
    - Nếu có FAIL trong successful checks → 2
    - Nếu có WARN trong successful checks → 1
    - Ngược lại (tất cả PASS) → 0
    """
    successful = [c for c in checks if c.status != CheckStatus.ERROR]

    if not successful:
        return 0

    if any(c.status == CheckStatus.FAIL for c in successful):
        return 2

    if any(c.status == CheckStatus.WARN for c in successful):
        return 1

    return 0


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestExitCodeMappingCorrectness:
    """
    Feature: train-backtest-verification, Property 15: Exit code mapping correctness

    For any combination of check results from the verification runner, the exit code
    SHALL be: 0 if all successful checks pass, 1 if any successful check has warnings,
    2 if any successful check has critical issues. Checks with status "error" SHALL NOT
    influence exit code.

    **Validates: Requirements 6.2, 6.5**
    """

    def setup_method(self):
        """Khởi tạo VerificationRunner cho mỗi test method."""
        self.runner = VerificationRunner(output_dir="/tmp/test_verification")

    @given(checks=all_pass_checks())
    @settings(max_examples=100)
    def test_all_pass_returns_exit_code_0(self, checks: List[CheckResult]):
        """
        Property: Nếu tất cả successful checks (status != ERROR) đều PASS → exit code = 0.
        ERROR checks có thể tồn tại nhưng không ảnh hưởng kết quả.

        **Validates: Requirements 6.2**
        """
        report = _make_report(checks)
        exit_code = self.runner.get_exit_code(report)

        assert exit_code == 0, (
            f"Expected exit code 0 (all successful checks PASS), got {exit_code}.\n"
            f"Checks: {[(c.check_name, c.status.value) for c in checks]}"
        )

    @given(checks=checks_with_warn_no_fail())
    @settings(max_examples=100)
    def test_warn_returns_exit_code_1(self, checks: List[CheckResult]):
        """
        Property: Nếu có ít nhất 1 successful check WARN và không có FAIL → exit code = 1.
        ERROR checks không ảnh hưởng.

        **Validates: Requirements 6.2**
        """
        report = _make_report(checks)
        exit_code = self.runner.get_exit_code(report)

        assert exit_code == 1, (
            f"Expected exit code 1 (has WARN, no FAIL), got {exit_code}.\n"
            f"Checks: {[(c.check_name, c.status.value) for c in checks]}"
        )

    @given(checks=checks_with_fail())
    @settings(max_examples=100)
    def test_fail_returns_exit_code_2(self, checks: List[CheckResult]):
        """
        Property: Nếu có ít nhất 1 successful check FAIL → exit code = 2.
        FAIL ưu tiên hơn WARN. ERROR checks không ảnh hưởng.

        **Validates: Requirements 6.2**
        """
        report = _make_report(checks)
        exit_code = self.runner.get_exit_code(report)

        assert exit_code == 2, (
            f"Expected exit code 2 (has FAIL), got {exit_code}.\n"
            f"Checks: {[(c.check_name, c.status.value) for c in checks]}"
        )

    @given(checks=all_error_checks())
    @settings(max_examples=100)
    def test_all_error_returns_exit_code_0(self, checks: List[CheckResult]):
        """
        Property: Nếu tất cả checks đều ERROR → exit code = 0
        (không có successful check nào để đánh giá).

        **Validates: Requirements 6.5**
        """
        report = _make_report(checks)
        exit_code = self.runner.get_exit_code(report)

        assert exit_code == 0, (
            f"Expected exit code 0 (all ERROR, no successful checks), got {exit_code}.\n"
            f"Checks: {[(c.check_name, c.status.value) for c in checks]}"
        )

    @given(checks=random_checks_combination())
    @settings(max_examples=100)
    def test_exit_code_matches_expected_for_any_combination(
        self, checks: List[CheckResult]
    ):
        """
        Property: Với bất kỳ tổ hợp CheckResults nào, exit code phải khớp với
        logic: lọc ERROR → kiểm tra FAIL (→2) → kiểm tra WARN (→1) → PASS (→0).

        **Validates: Requirements 6.2, 6.5**
        """
        report = _make_report(checks)
        exit_code = self.runner.get_exit_code(report)
        expected = _compute_expected_exit_code(checks)

        assert exit_code == expected, (
            f"Exit code mismatch: got {exit_code}, expected {expected}.\n"
            f"Checks: {[(c.check_name, c.status.value) for c in checks]}"
        )

    @given(
        base_checks=st.lists(
            check_result_with_status(CheckStatus.PASS),
            min_size=1,
            max_size=4,
        ),
        error_checks=st.lists(
            check_result_with_status(CheckStatus.ERROR),
            min_size=1,
            max_size=4,
        ),
    )
    @settings(max_examples=100)
    def test_error_checks_do_not_influence_exit_code(
        self, base_checks: List[CheckResult], error_checks: List[CheckResult]
    ):
        """
        Property: Thêm bất kỳ số lượng ERROR checks nào vào danh sách checks
        không thay đổi exit code. ERROR checks bị loại bỏ khỏi đánh giá.

        **Validates: Requirements 6.5**
        """
        # Exit code chỉ với base checks (không có ERROR)
        report_without_error = _make_report(base_checks)
        exit_code_without = self.runner.get_exit_code(report_without_error)

        # Exit code với base + error checks
        combined = base_checks + error_checks
        report_with_error = _make_report(combined)
        exit_code_with = self.runner.get_exit_code(report_with_error)

        assert exit_code_without == exit_code_with, (
            f"ERROR checks ảnh hưởng exit code: "
            f"without_error={exit_code_without}, with_error={exit_code_with}.\n"
            f"Base checks: {[(c.check_name, c.status.value) for c in base_checks]}\n"
            f"Error checks added: {len(error_checks)}"
        )
