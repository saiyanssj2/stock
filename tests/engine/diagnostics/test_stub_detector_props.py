# -*- coding: utf-8 -*-
"""
Property-based tests cho Stub Detector — Property 13: Stub keyword detection in source code.

Feature: train-backtest-verification, Property 13: Stub keyword detection in source code

**Validates: Requirements 5.2**

Properties:
1. Bất kỳ Python function nào có docstring chứa "STUB", "TODO", hoặc "placeholder"
   → phải được phát hiện là stub với reason="stub_keyword"
2. Bất kỳ Python function nào có body chỉ chứa logging calls (print(), logging.*, logger.*)
   và return → phải được phát hiện là stub với reason="no_computation"
"""

import textwrap

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.diagnostics.stub_detector import StubDetector


# ---------------------------------------------------------------------------
# Strategies — Sinh dữ liệu ngẫu nhiên hợp lệ
# ---------------------------------------------------------------------------

# Tên function hợp lệ (Python identifier)
function_name_strategy = st.from_regex(r"[a-z][a-z0-9_]{2,20}", fullmatch=True)

# File path giả
file_path_strategy = st.from_regex(r"engine/[a-z_]+\.py", fullmatch=True)

# Stub keywords phải xuất hiện trong docstring
stub_keyword_strategy = st.sampled_from(["STUB", "TODO", "placeholder"])

# Text ngẫu nhiên không chứa stub keywords (dùng làm padding)
safe_text_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Z"),
        max_codepoint=127,
    ),
    min_size=0,
    max_size=40,
).filter(
    lambda t: "STUB" not in t.upper()
    and "TODO" not in t.upper()
    and "PLACEHOLDER" not in t.upper()
)

# Logging calls hợp lệ
logging_call_strategy = st.sampled_from([
    'print("message")',
    'print(f"processing {x}")',
    "logging.info('started')",
    "logging.debug('debug info')",
    "logging.warning('warn')",
    "logging.error('error occurred')",
    "logger.info('processing')",
    "logger.debug('step done')",
    "logger.warning('slow')",
    "logger.error('failed')",
])

# Return statements cho logging-only body
return_statement_strategy = st.sampled_from([
    "return",
    "return None",
])


# ---------------------------------------------------------------------------
# Composite Strategies
# ---------------------------------------------------------------------------


@st.composite
def function_with_stub_keyword_in_docstring(draw):
    """
    Sinh source code Python function có docstring chứa stub keyword.

    Returns:
        tuple: (function_source, function_name)
    """
    func_name = draw(function_name_strategy)
    keyword = draw(stub_keyword_strategy)
    prefix_text = draw(safe_text_strategy)
    suffix_text = draw(safe_text_strategy)

    # Tạo docstring chứa keyword
    docstring_content = f"{prefix_text} {keyword} {suffix_text}".strip()

    # Body thực tế (có computation thật — để chắc chắn detection dựa vào docstring)
    source = textwrap.dedent(f'''\
def {func_name}(data):
    """{docstring_content}"""
    result = data * 2 + 1
    return result
''')
    return source, func_name


@st.composite
def function_with_logging_only_body(draw):
    """
    Sinh source code Python function có body chỉ chứa logging calls và return.

    Returns:
        tuple: (function_source, function_name)
    """
    func_name = draw(function_name_strategy)

    # Số lượng logging calls (ít nhất 1)
    num_calls = draw(st.integers(min_value=1, max_value=4))
    logging_calls = [draw(logging_call_strategy) for _ in range(num_calls)]

    # Có thể có return statement ở cuối hoặc không
    has_return = draw(st.booleans())

    # Có thể có docstring KHÔNG chứa stub keywords
    has_docstring = draw(st.booleans())
    docstring_text = draw(safe_text_strategy) if has_docstring else None

    # Xây dựng body
    body_lines = []
    if has_docstring and docstring_text:
        body_lines.append(f'    """{docstring_text}"""')

    for call in logging_calls:
        body_lines.append(f"    {call}")

    if has_return:
        ret_stmt = draw(return_statement_strategy)
        body_lines.append(f"    {ret_stmt}")

    body = "\n".join(body_lines)

    source = f"def {func_name}(data):\n{body}\n"
    return source, func_name


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestStubKeywordDetectionInDocstring:
    """
    Strategy 1: Function có docstring chứa stub keywords → phải detect là stub.

    Feature: train-backtest-verification, Property 13: Stub keyword detection in source code

    **Validates: Requirements 5.2**
    """

    @given(data=function_with_stub_keyword_in_docstring())
    @settings(max_examples=100)
    def test_function_with_stub_keyword_detected_as_stub(self, data):
        """
        Property: Bất kỳ function nào có docstring chứa "STUB", "TODO", hoặc
        "placeholder" → scan_function() phải trả về StubFinding với reason="stub_keyword".

        **Validates: Requirements 5.2**
        """
        source, func_name = data

        # Tạo detector mới cho mỗi test case (tránh accumulate findings)
        detector = StubDetector()

        finding = detector.scan_function(
            function_source=source,
            function_name=func_name,
            file_path="engine/test_module.py",
            is_critical_path=False,
        )

        # Phải phát hiện stub
        assert finding is not None, (
            f"scan_function() không detect stub cho function '{func_name}' "
            f"có docstring chứa stub keyword.\nSource:\n{source}"
        )
        # Reason phải là stub_keyword
        assert finding.reason == "stub_keyword", (
            f"Expected reason='stub_keyword', got '{finding.reason}' "
            f"cho function '{func_name}'.\nSource:\n{source}"
        )
        # Function name phải đúng
        assert finding.function_name == func_name

    @given(
        data=function_with_stub_keyword_in_docstring(),
        is_critical=st.booleans(),
    )
    @settings(max_examples=100)
    def test_stub_keyword_severity_depends_on_critical_path(self, data, is_critical):
        """
        Property: Severity phải là "critical" nếu is_critical_path=True,
        "warning" nếu is_critical_path=False.

        **Validates: Requirements 5.2**
        """
        source, func_name = data

        detector = StubDetector()

        finding = detector.scan_function(
            function_source=source,
            function_name=func_name,
            file_path="engine/test_module.py",
            is_critical_path=is_critical,
        )

        assert finding is not None
        expected_severity = "critical" if is_critical else "warning"
        assert finding.severity == expected_severity, (
            f"Expected severity='{expected_severity}', got '{finding.severity}' "
            f"với is_critical_path={is_critical}"
        )


class TestLoggingOnlyBodyDetection:
    """
    Strategy 2: Function có body chỉ chứa logging calls + return → phải detect là stub.

    Feature: train-backtest-verification, Property 13: Stub keyword detection in source code

    **Validates: Requirements 5.2**
    """

    @given(data=function_with_logging_only_body())
    @settings(max_examples=100)
    def test_function_with_logging_only_body_detected_as_stub(self, data):
        """
        Property: Bất kỳ function nào có body chỉ chứa print()/logging.*/logger.*
        và return → scan_function() phải trả về StubFinding với reason="no_computation".

        **Validates: Requirements 5.2**
        """
        source, func_name = data

        detector = StubDetector()

        finding = detector.scan_function(
            function_source=source,
            function_name=func_name,
            file_path="engine/test_module.py",
            is_critical_path=False,
        )

        # Phải phát hiện stub
        assert finding is not None, (
            f"scan_function() không detect stub cho function '{func_name}' "
            f"có body chỉ logging calls + return.\nSource:\n{source}"
        )
        # Reason phải là no_computation (hoặc stub_keyword nếu docstring vô tình match)
        assert finding.reason in ("no_computation", "stub_keyword"), (
            f"Expected reason='no_computation' hoặc 'stub_keyword', "
            f"got '{finding.reason}' cho function '{func_name}'.\nSource:\n{source}"
        )

    @given(
        data=function_with_logging_only_body(),
        is_critical=st.booleans(),
    )
    @settings(max_examples=100)
    def test_logging_only_severity_depends_on_critical_path(self, data, is_critical):
        """
        Property: Severity cho logging-only stub phải là "critical" nếu
        is_critical_path=True, "warning" nếu False.

        **Validates: Requirements 5.2**
        """
        source, func_name = data

        detector = StubDetector()

        finding = detector.scan_function(
            function_source=source,
            function_name=func_name,
            file_path="engine/test_module.py",
            is_critical_path=is_critical,
        )

        assert finding is not None
        expected_severity = "critical" if is_critical else "warning"
        assert finding.severity == expected_severity, (
            f"Expected severity='{expected_severity}', got '{finding.severity}' "
            f"với is_critical_path={is_critical}"
        )


# ---------------------------------------------------------------------------
# Property 14: Diagnostic report always generated
# ---------------------------------------------------------------------------


import json
import os
import tempfile

from engine.diagnostics.models import StubFinding


# Strategy tạo StubFinding ngẫu nhiên cho Property 14
_stub_finding_strategy = st.builds(
    StubFinding,
    function_name=st.from_regex(r"[a-z][a-z0-9_]{2,20}", fullmatch=True),
    file_path=st.from_regex(r"engine/[a-z_]+\.py", fullmatch=True),
    reason=st.sampled_from(["stub_keyword", "no_computation", "empty_body", "target_not_found"]),
    severity=st.sampled_from(["critical", "warning"]),
    line_number=st.one_of(st.none(), st.integers(min_value=1, max_value=10000)),
    details=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
)

# Strategy tạo danh sách stubs (0 đến nhiều)
_stub_findings_list_strategy = st.lists(_stub_finding_strategy, min_size=0, max_size=20)


class TestProperty14DiagnosticReportAlwaysGenerated:
    """
    Feature: train-backtest-verification, Property 14: Diagnostic report always generated

    For any scan completion (regardless of findings count), the stub_report.json
    SHALL always be created with valid schema including stubs array (possibly empty)
    and summary.total_stubs matching the array length.

    **Validates: Requirements 5.3, 5.5**
    """

    @given(findings=_stub_findings_list_strategy)
    @settings(max_examples=100)
    def test_report_always_created_with_valid_schema(
        self, findings: list
    ) -> None:
        """
        Property: Báo cáo luôn được tạo bất kể số lượng findings (0 hoặc nhiều).

        Verifies:
        - File stub_report.json luôn được tạo
        - JSON hợp lệ với stubs là array
        - summary.total_stubs == len(stubs)

        **Validates: Requirements 5.3, 5.5**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "stub_report.json")

            # Tạo detector với output path tạm
            detector = StubDetector(output_path=output_path)

            # Inject findings trực tiếp (mô phỏng scan scenarios khác nhau)
            detector.findings = list(findings)

            # Gọi generate_report
            report = detector.generate_report()

            # Verify 1: File luôn được tạo
            assert os.path.exists(output_path), (
                f"stub_report.json không được tạo khi có {len(findings)} findings"
            )

            # Verify 2: File chứa JSON hợp lệ
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Verify 3: stubs là array
            assert "stubs" in data, "Report thiếu trường 'stubs'"
            assert isinstance(data["stubs"], list), (
                f"'stubs' phải là array, nhận được {type(data['stubs'])}"
            )

            # Verify 4: summary tồn tại và có total_stubs
            assert "summary" in data, "Report thiếu trường 'summary'"
            assert "total_stubs" in data["summary"], (
                "summary thiếu trường 'total_stubs'"
            )

            # Verify 5: summary.total_stubs == len(stubs)
            assert data["summary"]["total_stubs"] == len(data["stubs"]), (
                f"summary.total_stubs ({data['summary']['total_stubs']}) "
                f"!= len(stubs) ({len(data['stubs'])})"
            )

            # Verify 6: Số findings khớp với input
            assert len(data["stubs"]) == len(findings), (
                f"Số stubs trong report ({len(data['stubs'])}) "
                f"!= số findings input ({len(findings)})"
            )
