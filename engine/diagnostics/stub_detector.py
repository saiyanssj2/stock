# -*- coding: utf-8 -*-
"""
Stub Detector — Phát hiện stub functions qua AST-based static analysis.

Quét source code để tìm functions không thực hiện computation thật:
- Thiếu torch operations (tensor creation, backward, step)
- Docstring chứa stub keywords (STUB, TODO, placeholder)
- Body chỉ có logging calls và return

References: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
"""

import ast
import inspect
import json
import logging
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from engine.diagnostics.models import StubFinding, StubReport

logger = logging.getLogger(__name__)

# Torch operations cần kiểm tra trong critical path
TORCH_OPERATIONS = [
    "torch.tensor",
    "torch.zeros",
    "torch.ones",
    "loss.backward",
    "optimizer.step",
    "model.forward",
    "model(",
    ".backward(",
    ".step(",
]

# Stub keywords trong docstring/comments
STUB_KEYWORDS = ["STUB", "TODO", "placeholder"]

# Logging-only patterns (body chỉ có logging → stub)
LOGGING_PATTERNS = ["print(", "logging.", "logger."]


class StubDetector:
    """Phát hiện stub functions qua static analysis.

    Sử dụng AST parsing để kiểm tra:
    - Function có chứa torch operations hay không
    - Docstring có chứa stub keywords hay không
    - Body chỉ có logging calls và return

    Attributes:
        output_path: Đường dẫn file báo cáo JSON.
        findings: Danh sách stub findings đã phát hiện.
    """

    def __init__(
        self, output_path: str = "data/engine/diagnostics/stub_report.json"
    ) -> None:
        """Khởi tạo StubDetector.

        Args:
            output_path: Đường dẫn file output cho stub report.
        """
        self.output_path = output_path
        self.findings: List[StubFinding] = []

    def scan_function(
        self,
        function_source: str,
        function_name: str,
        file_path: str,
        is_critical_path: bool = False,
    ) -> Optional[StubFinding]:
        """Quét một function cho stub patterns.

        Kiểm tra 3 điều kiện stub:
        1. Docstring chứa STUB_KEYWORDS → reason="stub_keyword"
        2. Body chỉ có logging + return → reason="no_computation"
        3. Critical path mà thiếu torch ops → reason="no_computation"

        Args:
            function_source: Source code của function.
            function_name: Tên function.
            file_path: Đường dẫn file chứa function.
            is_critical_path: True nếu function nằm trên critical training path.

        Returns:
            StubFinding nếu phát hiện stub, None nếu không.
        """
        try:
            tree = ast.parse(function_source)
        except SyntaxError:
            # Parse error → ghi finding với reason="parse_error"
            finding = StubFinding(
                function_name=function_name,
                file_path=file_path,
                reason="parse_error",
                severity="warning",
                details="Cannot parse function source code",
            )
            self.findings.append(finding)
            return finding

        # Tìm function definition trong AST
        func_node = self._find_function_node(tree, function_name)
        if func_node is None:
            # Nếu không tìm thấy function node, thử lấy node đầu tiên
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_node = node
                    break

        if func_node is None:
            return None

        # Lấy line number
        line_number = func_node.lineno

        # Kiểm tra 1: Docstring chứa stub keywords
        docstring = ast.get_docstring(func_node)
        if docstring and self._has_stub_keywords(docstring):
            severity = "critical" if is_critical_path else "warning"
            finding = StubFinding(
                function_name=function_name,
                file_path=file_path,
                reason="stub_keyword",
                severity=severity,
                line_number=line_number,
                details=f"Docstring contains stub keyword",
            )
            self.findings.append(finding)
            return finding

        # Kiểm tra 2: Body chỉ có logging + return
        if self._is_logging_only_body(func_node):
            severity = "critical" if is_critical_path else "warning"
            finding = StubFinding(
                function_name=function_name,
                file_path=file_path,
                reason="no_computation",
                severity=severity,
                line_number=line_number,
                details="Function body only contains logging calls and return",
            )
            self.findings.append(finding)
            return finding

        # Kiểm tra 3: Critical path mà thiếu torch operations
        if is_critical_path and not self._has_torch_operations(function_source):
            finding = StubFinding(
                function_name=function_name,
                file_path=file_path,
                reason="no_computation",
                severity="critical",
                line_number=line_number,
                details="No torch operations found (tensor creation, backward, step)",
            )
            self.findings.append(finding)
            return finding

        return None

    def scan_module(self, module_path: str) -> List[StubFinding]:
        """Quét toàn bộ module cho stubs.

        Parse file Python và kiểm tra tất cả functions/methods trong module.

        Args:
            module_path: Đường dẫn file Python cần quét.

        Returns:
            Danh sách StubFinding phát hiện được trong module.
        """
        module_findings: List[StubFinding] = []

        # Kiểm tra file tồn tại
        if not os.path.exists(module_path):
            finding = StubFinding(
                function_name="<module>",
                file_path=module_path,
                reason="target_not_found",
                severity="critical",
                details=f"Module file not found: {module_path}",
            )
            self.findings.append(finding)
            module_findings.append(finding)
            return module_findings

        try:
            source = Path(module_path).read_text(encoding="utf-8")
            tree = ast.parse(source)
        except SyntaxError as e:
            finding = StubFinding(
                function_name="<module>",
                file_path=module_path,
                reason="parse_error",
                severity="warning",
                details=f"Cannot parse module: {e}",
            )
            self.findings.append(finding)
            module_findings.append(finding)
            return module_findings
        except (OSError, IOError) as e:
            finding = StubFinding(
                function_name="<module>",
                file_path=module_path,
                reason="target_not_found",
                severity="critical",
                details=f"Cannot read module file: {e}",
            )
            self.findings.append(finding)
            module_findings.append(finding)
            return module_findings

        # Quét tất cả function definitions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Lấy source của function
                func_source = self._extract_function_source(source, node)
                func_name = node.name

                # Kiểm tra docstring cho stub keywords
                docstring = ast.get_docstring(node)
                if docstring and self._has_stub_keywords(docstring):
                    finding = StubFinding(
                        function_name=func_name,
                        file_path=module_path,
                        reason="stub_keyword",
                        severity="warning",
                        line_number=node.lineno,
                        details="Docstring contains stub keyword",
                    )
                    self.findings.append(finding)
                    module_findings.append(finding)
                    continue

                # Kiểm tra body chỉ có logging + return
                if self._is_logging_only_body(node):
                    finding = StubFinding(
                        function_name=func_name,
                        file_path=module_path,
                        reason="no_computation",
                        severity="warning",
                        line_number=node.lineno,
                        details="Function body only contains logging calls and return",
                    )
                    self.findings.append(finding)
                    module_findings.append(finding)

        return module_findings

    def scan_auto_learner(self) -> List[StubFinding]:
        """Quét AutoLearner._retrain_model() và các hàm liên quan.

        Tìm file auto_learner.py, parse AST, kiểm tra _retrain_model
        và các hàm được gọi trực tiếp từ _retrain_model.

        Returns:
            Danh sách StubFinding cho AutoLearner.
        """
        auto_learner_findings: List[StubFinding] = []

        # Tìm file auto_learner.py
        auto_learner_path = self._find_auto_learner_path()
        if auto_learner_path is None:
            finding = StubFinding(
                function_name="_retrain_model",
                file_path="engine/auto_learner.py",
                reason="target_not_found",
                severity="critical",
                details="Cannot locate AutoLearner module",
            )
            self.findings.append(finding)
            auto_learner_findings.append(finding)
            return auto_learner_findings

        # Đọc và parse source
        try:
            source = Path(auto_learner_path).read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, OSError, IOError) as e:
            finding = StubFinding(
                function_name="_retrain_model",
                file_path=str(auto_learner_path),
                reason="target_not_found",
                severity="critical",
                details=f"Cannot parse AutoLearner: {e}",
            )
            self.findings.append(finding)
            auto_learner_findings.append(finding)
            return auto_learner_findings

        # Tìm class AutoLearner và method _retrain_model
        retrain_node = None
        auto_learner_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "AutoLearner":
                auto_learner_class = node
                for item in node.body:
                    if (
                        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name == "_retrain_model"
                    ):
                        retrain_node = item
                        break
                break

        if retrain_node is None:
            # _retrain_model không tồn tại
            finding = StubFinding(
                function_name="_retrain_model",
                file_path=str(auto_learner_path),
                reason="target_not_found",
                severity="critical",
                details="Method _retrain_model not found in AutoLearner class",
            )
            self.findings.append(finding)
            auto_learner_findings.append(finding)
            return auto_learner_findings

        # Lấy source của _retrain_model
        retrain_source = self._extract_function_source(source, retrain_node)

        # Scan _retrain_model với is_critical_path=True
        finding = self.scan_function(
            function_source=retrain_source,
            function_name="_retrain_model",
            file_path=str(auto_learner_path),
            is_critical_path=True,
        )
        if finding:
            auto_learner_findings.append(finding)

        # Tìm và quét các hàm được gọi từ _retrain_model
        called_functions = self._get_called_functions(retrain_node)
        if auto_learner_class:
            for func_name in called_functions:
                # Tìm method trong AutoLearner class
                method_node = None
                for item in auto_learner_class.body:
                    if (
                        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name == func_name
                    ):
                        method_node = item
                        break

                if method_node:
                    method_source = self._extract_function_source(
                        source, method_node
                    )
                    sub_finding = self.scan_function(
                        function_source=method_source,
                        function_name=func_name,
                        file_path=str(auto_learner_path),
                        is_critical_path=False,
                    )
                    if sub_finding:
                        auto_learner_findings.append(sub_finding)

        return auto_learner_findings

    def generate_report(self) -> StubReport:
        """Tạo báo cáo stub detection. Luôn tạo dù không có stub.

        Xuất file JSON tại output_path với:
        - timestamp: ISO-8601
        - stubs: danh sách findings
        - summary: total_stubs, critical_count, warning_count

        Returns:
            StubReport chứa kết quả scan.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # Tính summary
        critical_count = sum(
            1 for f in self.findings if f.severity == "critical"
        )
        warning_count = sum(
            1 for f in self.findings if f.severity == "warning"
        )

        report = StubReport(
            timestamp=timestamp,
            stubs=self.findings,
            summary={
                "total_stubs": len(self.findings),
                "critical_count": critical_count,
                "warning_count": warning_count,
            },
        )

        # Ghi file JSON — tạo directory nếu chưa có
        self._write_report(report)

        return report

    # === Private helper methods ===

    def _find_function_node(
        self, tree: ast.Module, function_name: str
    ) -> Optional[ast.FunctionDef]:
        """Tìm function node theo tên trong AST tree.

        Args:
            tree: AST module đã parse.
            function_name: Tên function cần tìm.

        Returns:
            FunctionDef node hoặc None.
        """
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ):
                return node
        return None

    def _has_stub_keywords(self, text: str) -> bool:
        """Kiểm tra text có chứa stub keywords hay không.

        So sánh case-insensitive cho từng keyword.

        Args:
            text: Text cần kiểm tra (thường là docstring).

        Returns:
            True nếu chứa ít nhất 1 stub keyword.
        """
        text_upper = text.upper()
        for keyword in STUB_KEYWORDS:
            if keyword.upper() in text_upper:
                return True
        return False

    def _has_torch_operations(self, source: str) -> bool:
        """Kiểm tra source code có chứa torch operations hay không.

        Sử dụng AST analysis để kiểm tra chính xác hơn text matching.
        Fallback sang text matching cho các pattern đặc biệt.

        Args:
            source: Source code của function.

        Returns:
            True nếu tìm thấy ít nhất 1 torch operation.
        """
        # Text-based patterns cần kiểm tra cẩn thận
        # Chia thành 2 nhóm: exact prefix patterns và method call patterns
        exact_patterns = [
            "torch.tensor",
            "torch.zeros",
            "torch.ones",
            "loss.backward",
            "optimizer.step",
            "model.forward",
        ]

        # Method call patterns — cần kiểm tra trong context hợp lệ
        method_patterns = [
            ".backward(",
            ".step(",
        ]

        for pattern in exact_patterns:
            if pattern in source:
                return True

        for pattern in method_patterns:
            if pattern in source:
                return True

        # Kiểm tra model() call pattern — chỉ match khi là standalone call
        # ví dụ: model(x), self.model(x), nhưng không match train_model(x)
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # model(x) — Name node có id "model"
                    if isinstance(node.func, ast.Name) and node.func.id == "model":
                        return True
                    # self.model(x) — Attribute với attr "model"
                    if isinstance(node.func, ast.Attribute) and node.func.attr == "model":
                        return True
        except SyntaxError:
            # Nếu parse lỗi, fallback sang simple check
            pass

        return False

    def _is_logging_only_body(self, func_node: ast.FunctionDef) -> bool:
        """Kiểm tra function body chỉ có logging calls và return.

        Một function được coi là logging-only (stub) khi body:
        - Chỉ chứa logging calls (print, logging.*, logger.*)
        - Có thể có return/return None (bare return)
        - Có thể có pass
        - Có thể có if guard chỉ chứa logging

        Hàm KHÔNG phải logging-only nếu:
        - Có assignment (x = ...)
        - Có return với expression khác None
        - Có function calls không phải logging
        - Có vòng lặp, try/except, etc.

        Args:
            func_node: AST node của function.

        Returns:
            True nếu body chỉ có logging + bare return + pass.
        """
        body = func_node.body

        # Bỏ qua docstring (statement đầu tiên nếu là Expr Constant)
        start_idx = 0
        if body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, (ast.Constant, ast.Str)):
                start_idx = 1

        statements = body[start_idx:]

        # Body rỗng (chỉ docstring) → stub
        if not statements:
            return True

        # Nếu chỉ có pass → stub
        if len(statements) == 1 and isinstance(statements[0], ast.Pass):
            return True

        # Kiểm tra từng statement
        for stmt in statements:
            if isinstance(stmt, ast.Pass):
                continue
            elif isinstance(stmt, ast.Return):
                # Bare return hoặc return None → OK (vẫn logging-only)
                if stmt.value is None:
                    continue
                # return None literal
                if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
                    continue
                # return với expression → NOT logging-only
                return False
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                # Kiểm tra có phải logging call không
                if self._is_logging_call(stmt.value):
                    continue
                else:
                    return False
            elif isinstance(stmt, ast.If):
                # if statement với body chỉ logging → vẫn coi là logging-only
                if self._is_if_logging_only(stmt):
                    continue
                else:
                    return False
            else:
                # Bất kỳ statement khác (assignment, loop, etc.) → NOT logging-only
                return False

        return True

    def _is_logging_call(self, call_node: ast.Call) -> bool:
        """Kiểm tra call node có phải logging call hay không.

        Logging calls: print(), logging.*, logger.*

        Args:
            call_node: AST Call node.

        Returns:
            True nếu là logging call.
        """
        # print()
        if isinstance(call_node.func, ast.Name) and call_node.func.id == "print":
            return True

        # logging.info(), logger.info(), etc.
        if isinstance(call_node.func, ast.Attribute):
            if isinstance(call_node.func.value, ast.Name):
                obj_name = call_node.func.value.id
                if obj_name in ("logging", "logger"):
                    return True

        return False

    def _is_if_logging_only(self, if_node: ast.If) -> bool:
        """Kiểm tra if statement chỉ chứa logging calls.

        Args:
            if_node: AST If node.

        Returns:
            True nếu tất cả branches chỉ có logging/return/pass.
        """
        for stmt in if_node.body:
            if isinstance(stmt, ast.Pass) or isinstance(stmt, ast.Return):
                continue
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if not self._is_logging_call(stmt.value):
                    return False
            else:
                return False

        for stmt in if_node.orelse:
            if isinstance(stmt, ast.Pass) or isinstance(stmt, ast.Return):
                continue
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if not self._is_logging_call(stmt.value):
                    return False
            elif isinstance(stmt, ast.If):
                if not self._is_if_logging_only(stmt):
                    return False
            else:
                return False

        return True

    def _extract_function_source(
        self, full_source: str, func_node: ast.FunctionDef
    ) -> str:
        """Trích xuất source code của function từ full source.

        Tự động dedent để function source có thể parse standalone.

        Args:
            full_source: Toàn bộ source code file.
            func_node: AST node của function.

        Returns:
            Source code của function (đã dedent).
        """
        lines = full_source.splitlines()
        start_line = func_node.lineno - 1  # 0-indexed
        end_line = func_node.end_lineno if func_node.end_lineno else len(lines)
        func_lines = lines[start_line:end_line]
        raw_source = "\n".join(func_lines)
        return textwrap.dedent(raw_source)

    def _get_called_functions(self, func_node: ast.FunctionDef) -> List[str]:
        """Lấy danh sách tên hàm được gọi trong function body.

        Chỉ lấy các method calls dạng self.method_name().

        Args:
            func_node: AST node của function.

        Returns:
            Danh sách tên methods được gọi.
        """
        called: List[str] = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                # self.method_name()
                if isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "self"
                    ):
                        called.append(node.func.attr)
        return called

    def _find_auto_learner_path(self) -> Optional[str]:
        """Tìm đường dẫn file auto_learner.py.

        Tìm trong các vị trí phổ biến:
        1. engine/auto_learner.py (relative)
        2. Từ inspect module nếu import được

        Returns:
            Đường dẫn tuyệt đối hoặc None.
        """
        # Thử relative paths
        candidates = [
            "engine/auto_learner.py",
            os.path.join("engine", "auto_learner.py"),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        # Thử import module và lấy path
        try:
            import engine.auto_learner as al_module

            source_file = inspect.getfile(al_module)
            if os.path.exists(source_file):
                return source_file
        except (ImportError, TypeError, OSError):
            pass

        return None

    def _write_report(self, report: StubReport) -> None:
        """Ghi report ra file JSON.

        Tạo directory nếu chưa tồn tại.

        Args:
            report: StubReport cần ghi.
        """
        # Tạo output directory
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Serialize report
        report_dict = {
            "timestamp": report.timestamp,
            "stubs": [
                {
                    "function_name": f.function_name,
                    "file_path": f.file_path,
                    "reason": f.reason,
                    "severity": f.severity,
                    "line_number": f.line_number,
                    "details": f.details,
                }
                for f in report.stubs
            ],
            "summary": report.summary,
        }

        with open(self.output_path, "w", encoding="utf-8") as fp:
            json.dump(report_dict, fp, indent=2, ensure_ascii=False)
