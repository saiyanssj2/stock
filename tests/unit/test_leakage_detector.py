# -*- coding: utf-8 -*-
"""
Unit tests cho Data Leakage Detector.

Test các scenario cụ thể:
- Req 4.1: Signal future leak detection
- Req 4.2: Chronological split overlap detection
- Req 4.3: Label look-ahead detection
- Req 4.4: Accuracy anomaly detection
- Req 4.5: Report output format (leakage_report.json)

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import json
from pathlib import Path

import numpy as np
import pytest

from engine.diagnostics.leakage_detector import DataLeakageDetector
from engine.diagnostics.models import LeakageViolation, ViolationType


# === Fixtures ===


@pytest.fixture
def detector(tmp_path):
    """Tạo DataLeakageDetector với output_path trong tmp_path."""
    output_file = tmp_path / "leakage_report.json"
    return DataLeakageDetector(output_path=str(output_file))


@pytest.fixture
def detector_with_path(tmp_path):
    """Tạo detector và trả về cả path để verify output."""
    output_file = tmp_path / "leakage_report.json"
    d = DataLeakageDetector(output_path=str(output_file))
    return d, output_file


# === Test check_signal_access (Req 4.1) ===


class TestCheckSignalAccess:
    """Tests cho signal future leak detection."""

    def test_no_violation_when_access_within_bounds(self, detector):
        """Truy cập dữ liệu trong phạm vi cho phép → không có violation."""
        result = detector.check_signal_access(
            df_length=100, access_index=50, max_allowed_index=50
        )
        assert result is None

    def test_no_violation_when_access_below_max(self, detector):
        """Truy cập index thấp hơn max allowed → không có violation."""
        result = detector.check_signal_access(
            df_length=100, access_index=30, max_allowed_index=50
        )
        assert result is None

    def test_violation_when_access_exceeds_max(self, detector):
        """Truy cập future data → SIGNAL_FUTURE_LEAK violation."""
        result = detector.check_signal_access(
            df_length=100, access_index=55, max_allowed_index=50
        )
        assert result is not None
        assert result.violation_type == ViolationType.SIGNAL_FUTURE_LEAK
        assert result.details["rows_leaked"] == 5
        assert result.details["index"] == 50
        assert result.details["access_index"] == 55

    def test_violation_one_row_leaked(self, detector):
        """Truy cập đúng 1 row tương lai → rows_leaked = 1."""
        result = detector.check_signal_access(
            df_length=100, access_index=51, max_allowed_index=50
        )
        assert result is not None
        assert result.details["rows_leaked"] == 1

    def test_violation_stored_in_internal_list(self, detector):
        """Violation được lưu vào internal list để dùng khi generate report."""
        detector.check_signal_access(df_length=100, access_index=55, max_allowed_index=50)
        assert len(detector._violations) == 1

    def test_single_row_dataset_no_violation(self, detector):
        """Dataset có 1 row, access_index = 0 = max_allowed → không violation."""
        result = detector.check_signal_access(
            df_length=1, access_index=0, max_allowed_index=0
        )
        assert result is None

    def test_single_row_dataset_violation(self, detector):
        """Dataset có 1 row, access_index > max_allowed → violation."""
        result = detector.check_signal_access(
            df_length=1, access_index=1, max_allowed_index=0
        )
        assert result is not None
        assert result.violation_type == ViolationType.SIGNAL_FUTURE_LEAK
        assert result.details["rows_leaked"] == 1

    def test_zero_length_dataset_no_violation(self, detector):
        """Dataset rỗng (df_length=0), access và max đều 0 → không violation."""
        result = detector.check_signal_access(
            df_length=0, access_index=0, max_allowed_index=0
        )
        assert result is None


# === Test check_chronological_split (Req 4.2) ===


class TestCheckChronologicalSplit:
    """Tests cho chronological split overlap detection."""

    def test_no_violation_when_properly_ordered(self, detector):
        """Split đúng thứ tự chronological → không có violation."""
        train = np.array([0, 1, 2, 3, 4])
        val = np.array([5, 6, 7])
        test = np.array([8, 9, 10])
        result = detector.check_chronological_split(train, val, test)
        assert result is None

    def test_violation_when_train_val_overlap(self, detector):
        """Train và val có overlap → SPLIT_OVERLAP violation."""
        train = np.array([0, 1, 2, 5])
        val = np.array([4, 5, 6])
        test = np.array([8, 9, 10])
        result = detector.check_chronological_split(train, val, test)
        assert result is not None
        assert result.violation_type == ViolationType.SPLIT_OVERLAP
        assert "train-val" in result.details["split_pairs"]

    def test_violation_when_val_test_overlap(self, detector):
        """Val và test có overlap → SPLIT_OVERLAP violation."""
        train = np.array([0, 1, 2])
        val = np.array([3, 4, 7])
        test = np.array([6, 7, 8])
        result = detector.check_chronological_split(train, val, test)
        assert result is not None
        assert result.violation_type == ViolationType.SPLIT_OVERLAP

    def test_violation_ordering_without_direct_overlap(self, detector):
        """max(train) >= min(val) nhưng không overlap trực tiếp."""
        train = np.array([0, 1, 2, 6])
        val = np.array([5, 7, 8])
        test = np.array([9, 10, 11])
        result = detector.check_chronological_split(train, val, test)
        assert result is not None
        assert result.violation_type == ViolationType.SPLIT_OVERLAP

    def test_no_violation_when_empty_split(self, detector):
        """Split rỗng → không thể validate, trả về None."""
        train = np.array([0, 1, 2])
        val = np.array([])
        test = np.array([5, 6, 7])
        result = detector.check_chronological_split(train, val, test)
        assert result is None

    def test_no_violation_when_train_empty(self, detector):
        """Train rỗng → không thể validate, trả về None."""
        train = np.array([])
        val = np.array([3, 4, 5])
        test = np.array([6, 7, 8])
        result = detector.check_chronological_split(train, val, test)
        assert result is None

    def test_no_violation_when_test_empty(self, detector):
        """Test rỗng → không thể validate, trả về None."""
        train = np.array([0, 1, 2])
        val = np.array([3, 4, 5])
        test = np.array([])
        result = detector.check_chronological_split(train, val, test)
        assert result is None

    def test_no_violation_when_all_empty(self, detector):
        """Tất cả splits rỗng → không thể validate, trả về None."""
        train = np.array([])
        val = np.array([])
        test = np.array([])
        result = detector.check_chronological_split(train, val, test)
        assert result is None

    def test_single_row_per_split_valid(self, detector):
        """Mỗi split chỉ có 1 phần tử, đúng thứ tự → không violation."""
        train = np.array([0])
        val = np.array([1])
        test = np.array([2])
        result = detector.check_chronological_split(train, val, test)
        assert result is None

    def test_single_row_per_split_invalid(self, detector):
        """Mỗi split chỉ có 1 phần tử, sai thứ tự → violation."""
        train = np.array([5])
        val = np.array([3])
        test = np.array([7])
        result = detector.check_chronological_split(train, val, test)
        assert result is not None
        assert result.violation_type == ViolationType.SPLIT_OVERLAP


# === Test check_label_range (Req 4.3) ===


class TestCheckLabelRange:
    """Tests cho label look-ahead detection."""

    def test_no_violation_when_within_horizon(self, detector):
        """Label dùng dữ liệu trong [i, i+horizon] → không có violation."""
        result = detector.check_label_range(
            row_index=10, actual_data_range=(10, 15), horizon=5
        )
        assert result is None

    def test_violation_when_looks_beyond_horizon(self, detector):
        """Label dùng dữ liệu ngoài horizon → LABEL_LOOK_AHEAD violation."""
        result = detector.check_label_range(
            row_index=10, actual_data_range=(10, 16), horizon=5
        )
        assert result is not None
        assert result.violation_type == ViolationType.LABEL_LOOK_AHEAD
        assert result.details["row_index"] == 10
        assert result.details["actual_range"] == [10, 16]
        assert result.details["allowed_range"] == [10, 15]

    def test_violation_when_looks_before_row(self, detector):
        """Label dùng dữ liệu trước row index → LABEL_LOOK_AHEAD violation."""
        result = detector.check_label_range(
            row_index=10, actual_data_range=(9, 15), horizon=5
        )
        assert result is not None
        assert result.violation_type == ViolationType.LABEL_LOOK_AHEAD

    def test_default_horizon(self, detector):
        """Dùng DEFAULT_HORIZON khi không truyền horizon."""
        result = detector.check_label_range(
            row_index=10, actual_data_range=(10, 15)
        )
        assert result is None

    def test_exact_boundary_allowed(self, detector):
        """actual_end == allowed_end (row_index + horizon) → hợp lệ."""
        result = detector.check_label_range(
            row_index=0, actual_data_range=(0, 5), horizon=5
        )
        assert result is None


# === Test check_accuracy_anomaly (Req 4.4) ===


class TestCheckAccuracyAnomaly:
    """Tests cho accuracy anomaly detection."""

    def test_no_violation_when_below_threshold(self, detector):
        """Accuracy dưới 80% → không có violation."""
        result = detector.check_accuracy_anomaly("VNM", [0.5, 0.6, 0.7])
        assert result is None

    def test_violation_three_consecutive_above_threshold(self, detector):
        """3 lần liên tiếp > 80% → POSSIBLE_LEAKAGE violation."""
        result = detector.check_accuracy_anomaly("VNM", [0.85, 0.82, 0.81])
        assert result is not None
        assert result.violation_type == ViolationType.POSSIBLE_LEAKAGE
        assert result.symbol == "VNM"
        assert result.details["consecutive_count"] == 3

    def test_no_violation_when_interrupted(self, detector):
        """Chuỗi bị gián đoạn bởi giá trị thấp → không có violation."""
        result = detector.check_accuracy_anomaly("VNM", [0.85, 0.82, 0.5, 0.85, 0.82])
        assert result is None

    def test_no_violation_when_less_than_three(self, detector):
        """Chỉ có 2 lần > 80% → chưa đủ ngưỡng."""
        result = detector.check_accuracy_anomaly("VNM", [0.85, 0.82])
        assert result is None

    def test_violation_with_exact_threshold(self, detector):
        """Accuracy đúng bằng 80% → KHÔNG vi phạm (cần > 80%, không >=)."""
        result = detector.check_accuracy_anomaly("VNM", [0.80, 0.80, 0.80])
        assert result is None

    def test_violation_avg_accuracy_correct(self, detector):
        """avg_accuracy được tính đúng từ 3 giá trị consecutive."""
        result = detector.check_accuracy_anomaly("VNM", [0.90, 0.85, 0.81])
        assert result is not None
        expected_avg = round((0.90 + 0.85 + 0.81) / 3, 4)
        assert result.details["avg_accuracy"] == expected_avg

    def test_empty_history(self, detector):
        """History rỗng → không có violation."""
        result = detector.check_accuracy_anomaly("VNM", [])
        assert result is None


# === Test generate_report (Req 4.5) ===


class TestGenerateReport:
    """Tests cho report generation."""

    def test_report_pass_when_no_violations(self, detector_with_path):
        """Không có violation → overall_status = 'pass'."""
        detector, output_file = detector_with_path
        report = detector.generate_report("VNM")
        assert report.overall_status == "pass"
        assert report.symbol == "VNM"
        assert sum(report.violation_counts.values()) == 0

    def test_report_fail_when_violations_exist(self, detector_with_path):
        """Có violation → overall_status = 'fail'."""
        detector, output_file = detector_with_path
        detector.check_signal_access(100, 55, 50)
        report = detector.generate_report("VNM")
        assert report.overall_status == "fail"

    def test_report_violation_counts_correct(self, detector_with_path):
        """violation_counts đếm đúng theo loại."""
        detector, output_file = detector_with_path
        detector.check_signal_access(100, 55, 50)
        detector.check_signal_access(100, 60, 50)
        detector.check_label_range(10, (10, 20), 5)
        report = detector.generate_report("VNM")
        assert report.violation_counts["SIGNAL_FUTURE_LEAK"] == 2
        assert report.violation_counts["LABEL_LOOK_AHEAD"] == 1
        assert report.violation_counts["SPLIT_OVERLAP"] == 0
        assert report.violation_counts["POSSIBLE_LEAKAGE"] == 0

    def test_report_written_to_json_file(self, detector_with_path):
        """Report được ghi ra JSON file đúng format."""
        detector, output_file = detector_with_path
        detector.check_signal_access(100, 55, 50)
        detector.generate_report("VNM")

        assert output_file.exists()
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["symbol"] == "VNM"
        assert data["overall_status"] == "fail"
        assert "timestamp" in data
        assert "violations" in data
        assert "violation_counts" in data
        assert len(data["violations"]) == 1

    def test_report_json_has_all_violation_types_in_counts(self, detector_with_path):
        """violation_counts luôn chứa tất cả 4 loại violation."""
        detector, output_file = detector_with_path
        detector.generate_report("VNM")

        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected_types = [
            "SIGNAL_FUTURE_LEAK",
            "SPLIT_OVERLAP",
            "LABEL_LOOK_AHEAD",
            "POSSIBLE_LEAKAGE",
        ]
        for vtype in expected_types:
            assert vtype in data["violation_counts"]

    def test_report_creates_output_directory(self, tmp_path):
        """Tạo thư mục output nếu chưa tồn tại."""
        nested_path = tmp_path / "nested" / "dir" / "report.json"
        detector = DataLeakageDetector(output_path=str(nested_path))
        detector.generate_report("VNM")
        assert nested_path.exists()
