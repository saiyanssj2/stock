"""
Tests cho engine/overfitting_detector.py

Kiểm tra OverfittingDetector: detect(), classify_severity(),
_compute_ks_statistic(), _compute_distribution_stats(), warning messages.
"""

import numpy as np
import pytest

from engine.evaluation_config import (
    EvaluationConfig,
    OverfittingResult,
    OverfittingSeverity,
)
from engine.overfitting_detector import OverfittingDetector


# ==============================================================================
# classify_severity tests
# ==============================================================================


class TestClassifySeverity:
    """Tests cho phương thức classify_severity."""

    def setup_method(self):
        """Khởi tạo detector với config mặc định."""
        self.detector = OverfittingDetector()

    def test_severity_none_below_mild_threshold(self):
        """Ratio < 1.5 → NONE."""
        assert self.detector.classify_severity(1.0) == OverfittingSeverity.NONE
        assert self.detector.classify_severity(1.2) == OverfittingSeverity.NONE
        assert self.detector.classify_severity(1.49) == OverfittingSeverity.NONE

    def test_severity_mild_at_boundary(self):
        """Ratio == 1.5 → MILD."""
        assert self.detector.classify_severity(1.5) == OverfittingSeverity.MILD

    def test_severity_mild_between_thresholds(self):
        """1.5 <= ratio < 2.0 → MILD."""
        assert self.detector.classify_severity(1.7) == OverfittingSeverity.MILD
        assert self.detector.classify_severity(1.99) == OverfittingSeverity.MILD

    def test_severity_moderate_at_boundary(self):
        """Ratio == 2.0 → MODERATE."""
        assert self.detector.classify_severity(2.0) == OverfittingSeverity.MODERATE

    def test_severity_moderate_between_thresholds(self):
        """2.0 <= ratio < 3.0 → MODERATE."""
        assert self.detector.classify_severity(2.5) == OverfittingSeverity.MODERATE
        assert self.detector.classify_severity(2.99) == OverfittingSeverity.MODERATE

    def test_severity_severe_at_boundary(self):
        """Ratio == 3.0 → SEVERE."""
        assert self.detector.classify_severity(3.0) == OverfittingSeverity.SEVERE

    def test_severity_severe_above_boundary(self):
        """Ratio > 3.0 → SEVERE."""
        assert self.detector.classify_severity(5.0) == OverfittingSeverity.SEVERE
        assert self.detector.classify_severity(10.0) == OverfittingSeverity.SEVERE

    def test_severity_with_custom_thresholds(self):
        """Kiểm tra classify_severity với config tùy chỉnh."""
        config = EvaluationConfig(
            overfitting_mild=2.0,
            overfitting_moderate=3.0,
            overfitting_severe=5.0,
        )
        detector = OverfittingDetector(config)
        assert detector.classify_severity(1.9) == OverfittingSeverity.NONE
        assert detector.classify_severity(2.0) == OverfittingSeverity.MILD
        assert detector.classify_severity(3.0) == OverfittingSeverity.MODERATE
        assert detector.classify_severity(5.0) == OverfittingSeverity.SEVERE

    def test_severity_infinity(self):
        """Ratio = infinity → SEVERE."""
        assert self.detector.classify_severity(float("inf")) == OverfittingSeverity.SEVERE


# ==============================================================================
# detect tests
# ==============================================================================


class TestDetect:
    """Tests cho phương thức detect."""

    def setup_method(self):
        """Khởi tạo detector với config mặc định."""
        self.detector = OverfittingDetector()

    def test_no_overfitting(self):
        """Loss ratio thấp, distributions tương tự → NONE."""
        train_preds = np.random.default_rng(42).uniform(-0.5, 0.5, 100)
        test_preds = np.random.default_rng(43).uniform(-0.5, 0.5, 100)

        result = self.detector.detect(
            train_loss=0.5,
            test_loss=0.6,
            train_predictions=train_preds,
            test_predictions=test_preds,
        )

        assert result.loss_ratio == pytest.approx(1.2)
        assert result.severity == OverfittingSeverity.NONE
        assert result.warning_message is None

    def test_mild_overfitting(self):
        """Loss ratio trong khoảng mild."""
        train_preds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        test_preds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result = self.detector.detect(
            train_loss=0.4,
            test_loss=0.7,  # ratio = 1.75
            train_predictions=train_preds,
            test_predictions=test_preds,
        )

        assert result.loss_ratio == pytest.approx(1.75)
        assert result.severity == OverfittingSeverity.MILD
        assert result.warning_message is None

    def test_moderate_overfitting_has_warning(self):
        """Loss ratio moderate → có warning message."""
        train_preds = np.array([0.1, 0.2, 0.3])
        test_preds = np.array([-0.5, -0.3, -0.1])

        result = self.detector.detect(
            train_loss=0.3,
            test_loss=0.75,  # ratio = 2.5
            train_predictions=train_preds,
            test_predictions=test_preds,
        )

        assert result.loss_ratio == pytest.approx(2.5)
        assert result.severity == OverfittingSeverity.MODERATE
        assert result.warning_message is not None
        assert "MODERATE" in result.warning_message

    def test_severe_overfitting_has_warning(self):
        """Loss ratio severe → có warning message."""
        train_preds = np.array([0.8, 0.9, 0.7])
        test_preds = np.array([-0.5, -0.6, -0.4])

        result = self.detector.detect(
            train_loss=0.1,
            test_loss=0.5,  # ratio = 5.0
            train_predictions=train_preds,
            test_predictions=test_preds,
        )

        assert result.loss_ratio == pytest.approx(5.0)
        assert result.severity == OverfittingSeverity.SEVERE
        assert result.warning_message is not None
        assert "SEVERE" in result.warning_message

    def test_train_loss_zero_test_loss_zero(self):
        """Edge case: cả train và test loss đều = 0."""
        train_preds = np.array([0.1, 0.2, 0.3])
        test_preds = np.array([0.1, 0.2, 0.3])

        result = self.detector.detect(
            train_loss=0.0,
            test_loss=0.0,
            train_predictions=train_preds,
            test_predictions=test_preds,
        )

        # Khi cả hai đều 0, ratio = 1.0 → NONE
        assert result.loss_ratio == 1.0
        assert result.severity == OverfittingSeverity.NONE

    def test_train_loss_zero_test_loss_positive(self):
        """Edge case: train_loss = 0, test_loss > 0 → ratio = inf → SEVERE."""
        train_preds = np.array([0.1, 0.2, 0.3])
        test_preds = np.array([0.1, 0.2, 0.3])

        result = self.detector.detect(
            train_loss=0.0,
            test_loss=0.5,
            train_predictions=train_preds,
            test_predictions=test_preds,
        )

        assert result.loss_ratio == float("inf")
        assert result.severity == OverfittingSeverity.SEVERE
        assert result.warning_message is not None

    def test_result_contains_distribution_stats(self):
        """Kết quả chứa distribution stats đầy đủ."""
        train_preds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        test_preds = np.array([-0.1, 0.0, 0.1, 0.2, 0.3])

        result = self.detector.detect(
            train_loss=0.5,
            test_loss=0.5,
            train_predictions=train_preds,
            test_predictions=test_preds,
        )

        # Kiểm tra train distribution
        assert result.train_distribution.mean == pytest.approx(0.3)
        assert result.train_distribution.min == pytest.approx(0.1)
        assert result.train_distribution.max == pytest.approx(0.5)
        assert result.train_distribution.std > 0

        # Kiểm tra test distribution
        assert result.test_distribution.mean == pytest.approx(0.1)
        assert result.test_distribution.min == pytest.approx(-0.1)
        assert result.test_distribution.max == pytest.approx(0.3)

    def test_result_contains_ks_statistic(self):
        """Kết quả chứa KS statistic hợp lệ."""
        train_preds = np.array([0.8, 0.9, 0.7, 0.85, 0.95])
        test_preds = np.array([-0.5, -0.3, -0.6, -0.4, -0.2])

        result = self.detector.detect(
            train_loss=0.5,
            test_loss=0.5,
            train_predictions=train_preds,
            test_predictions=test_preds,
        )

        # KS statistic phải nằm trong [0, 1]
        assert 0.0 <= result.ks_statistic <= 1.0
        # P-value phải nằm trong [0, 1]
        assert 0.0 <= result.ks_p_value <= 1.0
        # Distributions khác nhau nhiều → KS statistic cao
        assert result.ks_statistic > 0.5


# ==============================================================================
# _compute_ks_statistic tests
# ==============================================================================


class TestComputeKsStatistic:
    """Tests cho _compute_ks_statistic."""

    def setup_method(self):
        """Khởi tạo detector."""
        self.detector = OverfittingDetector()

    def test_identical_arrays_ks_zero(self):
        """Cùng array → KS statistic = 0."""
        preds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        stat, p_value = self.detector._compute_ks_statistic(preds, preds)
        assert stat == pytest.approx(0.0, abs=1e-10)

    def test_completely_different_distributions(self):
        """Distributions hoàn toàn khác → KS statistic cao."""
        train = np.array([0.8, 0.9, 0.7, 0.85, 0.95])
        test = np.array([-0.8, -0.9, -0.7, -0.85, -0.95])
        stat, p_value = self.detector._compute_ks_statistic(train, test)
        assert stat == 1.0

    def test_empty_train_array(self):
        """Train array rỗng → trả về (0.0, 1.0)."""
        stat, p_value = self.detector._compute_ks_statistic(
            np.array([]), np.array([0.1, 0.2])
        )
        assert stat == 0.0
        assert p_value == 1.0

    def test_empty_test_array(self):
        """Test array rỗng → trả về (0.0, 1.0)."""
        stat, p_value = self.detector._compute_ks_statistic(
            np.array([0.1, 0.2]), np.array([])
        )
        assert stat == 0.0
        assert p_value == 1.0


# ==============================================================================
# _compute_distribution_stats tests
# ==============================================================================


class TestComputeDistributionStats:
    """Tests cho _compute_distribution_stats."""

    def setup_method(self):
        """Khởi tạo detector."""
        self.detector = OverfittingDetector()

    def test_basic_stats(self):
        """Kiểm tra tính toán mean, std, min, max đúng."""
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = self.detector._compute_distribution_stats(preds)
        assert stats.mean == pytest.approx(3.0)
        assert stats.std == pytest.approx(np.std(preds))
        assert stats.min == pytest.approx(1.0)
        assert stats.max == pytest.approx(5.0)

    def test_single_element(self):
        """Array 1 phần tử → std = 0."""
        preds = np.array([0.5])
        stats = self.detector._compute_distribution_stats(preds)
        assert stats.mean == pytest.approx(0.5)
        assert stats.std == pytest.approx(0.0)
        assert stats.min == pytest.approx(0.5)
        assert stats.max == pytest.approx(0.5)

    def test_empty_array(self):
        """Array rỗng → DistributionStats mặc định."""
        stats = self.detector._compute_distribution_stats(np.array([]))
        assert stats.mean == 0.0
        assert stats.std == 0.0
        assert stats.min == 0.0
        assert stats.max == 0.0

    def test_negative_values(self):
        """Array với giá trị âm."""
        preds = np.array([-0.5, -0.3, -0.1, 0.1, 0.3])
        stats = self.detector._compute_distribution_stats(preds)
        assert stats.mean == pytest.approx(-0.1)
        assert stats.min == pytest.approx(-0.5)
        assert stats.max == pytest.approx(0.3)


# ==============================================================================
# Warning message tests
# ==============================================================================


class TestWarningMessage:
    """Tests cho warning message generation."""

    def setup_method(self):
        """Khởi tạo detector."""
        self.detector = OverfittingDetector()

    def test_no_warning_for_none_severity(self):
        """NONE severity → không có warning."""
        result = self.detector.detect(
            train_loss=1.0,
            test_loss=1.0,
            train_predictions=np.array([0.1, 0.2]),
            test_predictions=np.array([0.1, 0.2]),
        )
        assert result.warning_message is None

    def test_no_warning_for_mild_severity(self):
        """MILD severity → không có warning."""
        result = self.detector.detect(
            train_loss=0.4,
            test_loss=0.7,  # ratio = 1.75
            train_predictions=np.array([0.1, 0.2]),
            test_predictions=np.array([0.1, 0.2]),
        )
        assert result.warning_message is None

    def test_warning_contains_metrics_for_moderate(self):
        """MODERATE warning chứa metrics chi tiết."""
        result = self.detector.detect(
            train_loss=0.3,
            test_loss=0.75,  # ratio = 2.5
            train_predictions=np.array([0.1, 0.2, 0.3]),
            test_predictions=np.array([-0.1, 0.0, 0.1]),
        )
        assert result.warning_message is not None
        assert "loss_ratio" in result.warning_message
        assert "KS_statistic" in result.warning_message
        assert "train_mean" in result.warning_message
        assert "test_mean" in result.warning_message

    def test_warning_contains_severity_label(self):
        """Warning chứa label severity đúng."""
        result = self.detector.detect(
            train_loss=0.1,
            test_loss=0.5,  # ratio = 5.0
            train_predictions=np.array([0.1, 0.2, 0.3]),
            test_predictions=np.array([0.1, 0.2, 0.3]),
        )
        assert "SEVERE" in result.warning_message
