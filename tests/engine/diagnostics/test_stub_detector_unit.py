# -*- coding: utf-8 -*-
"""
Unit tests cho Stub Detector.

Test scenarios:
1. scan_function() trên function chỉ có logging → detected stub "no_computation"
2. scan_function() trên function có torch ops → KHÔNG detected stub
3. scan_function() trên critical path function thiếu torch ops → severity="critical"
4. scan_module() trên file không tồn tại → reason="target_not_found", severity="critical"
5. generate_report() khi không có stubs → stubs=[], summary.total_stubs=0, file created
6. scan_auto_learner() test (mock file system)

**Validates: Requirements 5.1, 5.4, 5.5**
"""

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from engine.diagnostics.stub_detector import StubDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector(tmp_path):
    """Tạo StubDetector instance với output path trong tmp_path."""
    output_file = str(tmp_path / "stub_report.json")
    return StubDetector(output_path=output_file)


@pytest.fixture
def logging_only_function_source():
    """Source code hàm chỉ có logging → stub."""
    return textwrap.dedent("""\
        def _retrain_model(self, symbols):
            \"\"\"Retrain model cho danh sách symbols.\"\"\"
            logger.info(f"Starting retrain for {len(symbols)} symbols")
            logger.info("Retrain completed")
            return None
    """)


@pytest.fixture
def torch_function_source():
    """Source code hàm có torch operations → KHÔNG phải stub."""
    return textwrap.dedent("""\
        def _retrain_model(self, symbols):
            \"\"\"Retrain model thực sự.\"\"\"
            model = self.build_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(50):
                output = model(features)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            return model
    """)


@pytest.fixture
def critical_no_torch_source():
    """Source code hàm critical path nhưng không có torch ops."""
    return textwrap.dedent("""\
        def _retrain_model(self, symbols):
            \"\"\"Train the model.\"\"\"
            data = self.load_data(symbols)
            features = self.extract_features(data)
            results = self.compute_metrics(features)
            self.save_checkpoint(results)
            return results
    """)


@pytest.fixture
def stub_keyword_source():
    """Source code hàm có STUB keyword trong docstring."""
    return textwrap.dedent("""\
        def _retrain_model(self, symbols):
            \"\"\"STUB: placeholder cho retrain logic.\"\"\"
            pass
    """)


@pytest.fixture
def auto_learner_stub_file(tmp_path):
    """Tạo file auto_learner.py giả với _retrain_model là stub."""
    content = textwrap.dedent("""\
        import logging

        logger = logging.getLogger(__name__)


        class AutoLearner:
            \"\"\"AutoLearner manages training cycles.\"\"\"

            def _retrain_model(self, symbols):
                \"\"\"Retrain model cho symbols.\"\"\"
                logger.info(f"Starting retrain for {len(symbols)} symbols")
                logger.info("Retrain completed successfully")
                return None

            def _helper_method(self):
                \"\"\"Helper method.\"\"\"
                logger.debug("Helper called")
                return None
    """)
    file_path = tmp_path / "auto_learner.py"
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


@pytest.fixture
def auto_learner_real_file(tmp_path):
    """Tạo file auto_learner.py với _retrain_model thực sự."""
    content = textwrap.dedent("""\
        import torch
        import logging

        logger = logging.getLogger(__name__)


        class AutoLearner:
            \"\"\"AutoLearner with real training.\"\"\"

            def _retrain_model(self, symbols):
                \"\"\"Retrain model thực sự.\"\"\"
                model = self.build_model()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                for epoch in range(50):
                    output = model(self.features)
                    loss = self.criterion(output, self.labels)
                    loss.backward()
                    optimizer.step()
                return model

            def _load_data(self):
                \"\"\"Load data from disk.\"\"\"
                data = torch.tensor([1, 2, 3])
                return data
    """)
    file_path = tmp_path / "auto_learner.py"
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


# ---------------------------------------------------------------------------
# Test: scan_function() — logging-only → stub "no_computation" (Req 5.1)
# ---------------------------------------------------------------------------


class TestScanFunctionLoggingOnly:
    """Test scan_function() phát hiện stub khi body chỉ có logging."""

    def test_logging_only_detected_as_stub(self, detector, logging_only_function_source):
        """Hàm chỉ có logger.info() và return None → stub với reason="no_computation"."""
        finding = detector.scan_function(
            function_source=logging_only_function_source,
            function_name="_retrain_model",
            file_path="engine/auto_learner.py",
            is_critical_path=False,
        )

        assert finding is not None
        assert finding.function_name == "_retrain_model"
        assert finding.reason == "no_computation"
        assert finding.file_path == "engine/auto_learner.py"

    def test_logging_only_critical_path_severity(self, detector, logging_only_function_source):
        """Hàm logging-only trên critical path → severity="critical"."""
        finding = detector.scan_function(
            function_source=logging_only_function_source,
            function_name="_retrain_model",
            file_path="engine/auto_learner.py",
            is_critical_path=True,
        )

        assert finding is not None
        assert finding.severity == "critical"
        assert finding.reason == "no_computation"

    def test_logging_only_non_critical_path_severity(
        self, detector, logging_only_function_source
    ):
        """Hàm logging-only KHÔNG trên critical path → severity="warning"."""
        finding = detector.scan_function(
            function_source=logging_only_function_source,
            function_name="_retrain_model",
            file_path="engine/auto_learner.py",
            is_critical_path=False,
        )

        assert finding is not None
        assert finding.severity == "warning"


# ---------------------------------------------------------------------------
# Test: scan_function() — có torch ops → KHÔNG phải stub (Req 5.1)
# ---------------------------------------------------------------------------


class TestScanFunctionWithTorchOps:
    """Test scan_function() KHÔNG phát hiện stub khi có torch operations."""

    def test_torch_ops_not_detected_as_stub(self, detector, torch_function_source):
        """Hàm có loss.backward() và optimizer.step() → KHÔNG phải stub."""
        finding = detector.scan_function(
            function_source=torch_function_source,
            function_name="_retrain_model",
            file_path="engine/auto_learner.py",
            is_critical_path=True,
        )

        assert finding is None

    def test_torch_tensor_creation_not_stub(self, detector):
        """Hàm có torch.tensor() → KHÔNG phải stub."""
        source = textwrap.dedent("""\
            def compute(self, data):
                \"\"\"Compute features.\"\"\"
                tensor = torch.tensor(data)
                result = tensor.mean()
                return result
        """)
        finding = detector.scan_function(
            function_source=source,
            function_name="compute",
            file_path="engine/model.py",
            is_critical_path=True,
        )

        assert finding is None

    def test_model_forward_call_not_stub(self, detector):
        """Hàm có model() call → KHÔNG phải stub."""
        source = textwrap.dedent("""\
            def predict(self, features):
                \"\"\"Run prediction.\"\"\"
                output = model(features)
                return output
        """)
        finding = detector.scan_function(
            function_source=source,
            function_name="predict",
            file_path="engine/model.py",
            is_critical_path=True,
        )

        assert finding is None


# ---------------------------------------------------------------------------
# Test: scan_function() — critical path thiếu torch ops → severity="critical" (Req 5.1)
# ---------------------------------------------------------------------------


class TestScanFunctionCriticalPath:
    """Test scan_function() trên critical path function thiếu torch ops."""

    def test_critical_path_no_torch_ops_severity_critical(
        self, detector, critical_no_torch_source
    ):
        """Hàm critical path không có torch ops → severity="critical"."""
        finding = detector.scan_function(
            function_source=critical_no_torch_source,
            function_name="_retrain_model",
            file_path="engine/auto_learner.py",
            is_critical_path=True,
        )

        assert finding is not None
        assert finding.severity == "critical"
        assert finding.reason == "no_computation"

    def test_stub_keyword_in_docstring_critical(self, detector, stub_keyword_source):
        """Hàm có STUB keyword trong docstring trên critical path → severity="critical"."""
        finding = detector.scan_function(
            function_source=stub_keyword_source,
            function_name="_retrain_model",
            file_path="engine/auto_learner.py",
            is_critical_path=True,
        )

        assert finding is not None
        assert finding.reason == "stub_keyword"
        assert finding.severity == "critical"


# ---------------------------------------------------------------------------
# Test: scan_module() — file không tồn tại → target_not_found (Req 5.4)
# ---------------------------------------------------------------------------


class TestScanModuleTargetNotFound:
    """Test scan_module() khi file không tồn tại."""

    def test_nonexistent_file_returns_target_not_found(self, detector):
        """scan_module() trên file không tồn tại → reason="target_not_found"."""
        findings = detector.scan_module("/nonexistent/path/to/module.py")

        assert len(findings) == 1
        finding = findings[0]
        assert finding.reason == "target_not_found"
        assert finding.severity == "critical"
        assert finding.file_path == "/nonexistent/path/to/module.py"
        assert finding.function_name == "<module>"

    def test_nonexistent_file_continues_scanning(self, detector, tmp_path):
        """Sau target_not_found, detector vẫn tiếp tục scan target khác."""
        # Scan file không tồn tại
        detector.scan_module("/nonexistent/module.py")

        # Scan file tồn tại (tạo file Python hợp lệ)
        valid_file = tmp_path / "valid_module.py"
        valid_file.write_text(
            textwrap.dedent("""\
                def real_function():
                    \"\"\"Hàm bình thường.\"\"\"
                    x = 1 + 2
                    return x
            """),
            encoding="utf-8",
        )
        findings_valid = detector.scan_module(str(valid_file))

        # File không tồn tại có 1 finding, file valid có thể có 0
        assert len(detector.findings) >= 1
        # Quá trình scan không bị dừng
        assert findings_valid is not None


# ---------------------------------------------------------------------------
# Test: generate_report() — empty results (Req 5.5)
# ---------------------------------------------------------------------------


class TestGenerateReportEmpty:
    """Test generate_report() khi không phát hiện stub nào."""

    def test_empty_report_created(self, detector, tmp_path):
        """generate_report() luôn tạo file dù stubs=[]."""
        report = detector.generate_report()

        assert report.stubs == []
        assert report.summary["total_stubs"] == 0
        assert report.summary["critical_count"] == 0
        assert report.summary["warning_count"] == 0

    def test_empty_report_file_exists(self, detector, tmp_path):
        """File JSON phải tồn tại sau generate_report() dù không có stubs."""
        detector.generate_report()

        assert os.path.exists(detector.output_path)

    def test_empty_report_json_valid(self, detector, tmp_path):
        """File JSON phải có schema hợp lệ với stubs=[] và summary."""
        detector.generate_report()

        with open(detector.output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "stubs" in data
        assert "summary" in data
        assert data["stubs"] == []
        assert data["summary"]["total_stubs"] == 0

    def test_report_with_findings_has_correct_counts(self, detector, logging_only_function_source):
        """Report với findings phải có summary counts chính xác."""
        # Tạo 1 finding
        detector.scan_function(
            function_source=logging_only_function_source,
            function_name="_retrain_model",
            file_path="engine/auto_learner.py",
            is_critical_path=True,
        )

        report = detector.generate_report()

        assert report.summary["total_stubs"] == 1
        assert report.summary["critical_count"] == 1
        assert report.summary["warning_count"] == 0
        assert len(report.stubs) == 1


# ---------------------------------------------------------------------------
# Test: scan_auto_learner() (Req 5.1)
# ---------------------------------------------------------------------------


class TestScanAutoLearner:
    """Test scan_auto_learner() phát hiện stub trong AutoLearner."""

    def test_scan_auto_learner_stub_detected(self, detector, auto_learner_stub_file):
        """scan_auto_learner() phát hiện _retrain_model là stub."""
        with patch.object(detector, "_find_auto_learner_path", return_value=auto_learner_stub_file):
            findings = detector.scan_auto_learner()

        # Phải phát hiện ít nhất _retrain_model là stub
        retrain_findings = [f for f in findings if f.function_name == "_retrain_model"]
        assert len(retrain_findings) >= 1
        assert retrain_findings[0].severity == "critical"
        assert retrain_findings[0].reason == "no_computation"

    def test_scan_auto_learner_real_not_stub(self, detector, auto_learner_real_file):
        """scan_auto_learner() KHÔNG phát hiện stub khi có torch ops."""
        with patch.object(detector, "_find_auto_learner_path", return_value=auto_learner_real_file):
            findings = detector.scan_auto_learner()

        # _retrain_model có torch ops → KHÔNG bị đánh dấu stub
        retrain_findings = [f for f in findings if f.function_name == "_retrain_model"]
        assert len(retrain_findings) == 0

    def test_scan_auto_learner_not_found(self, detector):
        """scan_auto_learner() khi không tìm thấy file → target_not_found."""
        with patch.object(detector, "_find_auto_learner_path", return_value=None):
            findings = detector.scan_auto_learner()

        assert len(findings) == 1
        assert findings[0].reason == "target_not_found"
        assert findings[0].severity == "critical"
        assert findings[0].function_name == "_retrain_model"

    def test_scan_auto_learner_scans_called_methods(self, detector, auto_learner_stub_file):
        """scan_auto_learner() cũng quét các method được gọi từ _retrain_model."""
        with patch.object(detector, "_find_auto_learner_path", return_value=auto_learner_stub_file):
            findings = detector.scan_auto_learner()

        # _retrain_model trong stub file không gọi self.method nào trực tiếp
        # nhưng nếu có, chúng cũng sẽ được scan
        assert len(findings) >= 1  # Ít nhất _retrain_model bị phát hiện
