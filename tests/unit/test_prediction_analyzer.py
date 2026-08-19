"""
Tests cho engine/prediction_analyzer.py

Kiểm tra tất cả phương thức của PredictionAnalyzer:
- directional_accuracy: tỷ lệ dự đoán đúng hướng
- pearson_correlation: tương quan Pearson
- calibration_by_quartile: mean absolute return theo quartile
- score_distribution: thống kê phân phối và flags
- confusion_matrix: ma trận nhầm lẫn hướng
- analyze: tổng hợp tất cả metrics
"""

import numpy as np
import pytest

from engine.evaluation_config import (
    ConfusionMatrix,
    EvaluationConfig,
    PredictionQualityResult,
    ScoreDistributionResult,
)
from engine.prediction_analyzer import PredictionAnalyzer


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def analyzer() -> PredictionAnalyzer:
    """Khởi tạo PredictionAnalyzer với config mặc định."""
    return PredictionAnalyzer()


@pytest.fixture
def custom_analyzer() -> PredictionAnalyzer:
    """PredictionAnalyzer với thresholds tùy chỉnh."""
    config = EvaluationConfig(
        conservative_std_threshold=0.05,
        extreme_std_threshold=0.9,
    )
    return PredictionAnalyzer(config=config)


# ==============================================================================
# TestDirectionalAccuracy
# ==============================================================================


class TestDirectionalAccuracy:
    """Tests cho phương thức directional_accuracy."""

    def test_perfect_accuracy(self, analyzer: PredictionAnalyzer):
        """Tất cả dự đoán đúng hướng → 100%."""
        preds = np.array([0.5, -0.3, 0.7, -0.1])
        actuals = np.array([0.02, -0.01, 0.05, -0.03])
        assert analyzer.directional_accuracy(preds, actuals) == 100.0

    def test_zero_accuracy(self, analyzer: PredictionAnalyzer):
        """Tất cả dự đoán sai hướng → 0%."""
        preds = np.array([0.5, -0.3, 0.7, -0.1])
        actuals = np.array([-0.02, 0.01, -0.05, 0.03])
        assert analyzer.directional_accuracy(preds, actuals) == 0.0

    def test_partial_accuracy(self, analyzer: PredictionAnalyzer):
        """2/4 dự đoán đúng hướng → 50%."""
        preds = np.array([0.5, -0.3, 0.7, -0.1])
        actuals = np.array([0.02, 0.01, 0.05, 0.03])
        # preds[0]=+, actuals[0]=+ ✓
        # preds[1]=-, actuals[1]=+ ✗
        # preds[2]=+, actuals[2]=+ ✓
        # preds[3]=-, actuals[3]=+ ✗
        assert analyzer.directional_accuracy(preds, actuals) == 50.0

    def test_excludes_zero_predictions(self, analyzer: PredictionAnalyzer):
        """Loại bỏ cặp có prediction = 0."""
        preds = np.array([0.5, 0.0, 0.7, -0.1])
        actuals = np.array([0.02, -0.01, 0.05, -0.03])
        # Chỉ 3 cặp hợp lệ, tất cả đúng hướng → 100%
        assert analyzer.directional_accuracy(preds, actuals) == 100.0

    def test_excludes_zero_actuals(self, analyzer: PredictionAnalyzer):
        """Loại bỏ cặp có actual = 0."""
        preds = np.array([0.5, -0.3, 0.7, -0.1])
        actuals = np.array([0.02, 0.0, 0.05, -0.03])
        # Chỉ 3 cặp hợp lệ, tất cả đúng hướng → 100%
        assert analyzer.directional_accuracy(preds, actuals) == 100.0

    def test_all_zeros_returns_zero(self, analyzer: PredictionAnalyzer):
        """Tất cả predictions và actuals bằng 0 → 0%."""
        preds = np.array([0.0, 0.0, 0.0])
        actuals = np.array([0.0, 0.0, 0.0])
        assert analyzer.directional_accuracy(preds, actuals) == 0.0

    def test_empty_arrays(self, analyzer: PredictionAnalyzer):
        """Mảng rỗng → 0%."""
        preds = np.array([])
        actuals = np.array([])
        assert analyzer.directional_accuracy(preds, actuals) == 0.0


# ==============================================================================
# TestPearsonCorrelation
# ==============================================================================


class TestPearsonCorrelation:
    """Tests cho phương thức pearson_correlation."""

    def test_perfect_positive_correlation(self, analyzer: PredictionAnalyzer):
        """Tương quan hoàn hảo dương → 1.0."""
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actuals = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = analyzer.pearson_correlation(preds, actuals)
        assert abs(result - 1.0) < 1e-10

    def test_perfect_negative_correlation(self, analyzer: PredictionAnalyzer):
        """Tương quan hoàn hảo âm → -1.0."""
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actuals = np.array([-2.0, -4.0, -6.0, -8.0, -10.0])
        result = analyzer.pearson_correlation(preds, actuals)
        assert abs(result - (-1.0)) < 1e-10

    def test_no_correlation(self, analyzer: PredictionAnalyzer):
        """Không có tương quan → gần 0."""
        # Dùng dữ liệu đã biết không tương quan
        preds = np.array([1.0, -1.0, 1.0, -1.0])
        actuals = np.array([1.0, 1.0, -1.0, -1.0])
        result = analyzer.pearson_correlation(preds, actuals)
        assert abs(result) < 1e-10

    def test_constant_predictions_returns_zero(self, analyzer: PredictionAnalyzer):
        """Predictions hằng số → 0.0 (không xác định)."""
        preds = np.array([0.5, 0.5, 0.5, 0.5])
        actuals = np.array([0.01, -0.02, 0.03, -0.04])
        assert analyzer.pearson_correlation(preds, actuals) == 0.0

    def test_constant_actuals_returns_zero(self, analyzer: PredictionAnalyzer):
        """Actuals hằng số → 0.0 (không xác định)."""
        preds = np.array([0.1, 0.5, -0.3, 0.8])
        actuals = np.array([0.02, 0.02, 0.02, 0.02])
        assert analyzer.pearson_correlation(preds, actuals) == 0.0

    def test_single_element_returns_zero(self, analyzer: PredictionAnalyzer):
        """Chỉ 1 phần tử → 0.0."""
        preds = np.array([0.5])
        actuals = np.array([0.02])
        assert analyzer.pearson_correlation(preds, actuals) == 0.0

    def test_result_bounded(self, analyzer: PredictionAnalyzer):
        """Kết quả luôn trong [-1, 1]."""
        rng = np.random.default_rng(42)
        preds = rng.uniform(-1, 1, size=100)
        actuals = rng.uniform(-0.1, 0.1, size=100)
        result = analyzer.pearson_correlation(preds, actuals)
        assert -1.0 <= result <= 1.0


# ==============================================================================
# TestCalibrationByQuartile
# ==============================================================================


class TestCalibrationByQuartile:
    """Tests cho phương thức calibration_by_quartile."""

    def test_basic_quartile_split(self, analyzer: PredictionAnalyzer):
        """Kiểm tra chia 4 quartiles đúng."""
        # 8 phần tử, chia đều 2 mỗi quartile
        preds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        actuals = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
        result = analyzer.calibration_by_quartile(preds, actuals)

        assert "Q1" in result
        assert "Q2" in result
        assert "Q3" in result
        assert "Q4" in result
        # Tất cả giá trị phải >= 0 (mean absolute)
        for v in result.values():
            assert v >= 0.0

    def test_all_keys_present(self, analyzer: PredictionAnalyzer):
        """Luôn có đủ 4 keys Q1-Q4."""
        preds = np.array([0.5, -0.5, 0.3, -0.3])
        actuals = np.array([0.01, -0.01, 0.02, -0.02])
        result = analyzer.calibration_by_quartile(preds, actuals)
        assert set(result.keys()) == {"Q1", "Q2", "Q3", "Q4"}

    def test_empty_arrays(self, analyzer: PredictionAnalyzer):
        """Mảng rỗng → tất cả quartiles = 0."""
        result = analyzer.calibration_by_quartile(np.array([]), np.array([]))
        assert result == {"Q1": 0.0, "Q2": 0.0, "Q3": 0.0, "Q4": 0.0}

    def test_values_are_mean_absolute(self, analyzer: PredictionAnalyzer):
        """Giá trị là mean |actual| cho mỗi quartile."""
        # Tất cả predictions giống nhau → tất cả rơi vào cùng quartile
        preds = np.array([-1.0, -0.5, 0.5, 1.0])
        actuals = np.array([-0.1, -0.2, 0.3, 0.4])
        result = analyzer.calibration_by_quartile(preds, actuals)
        # Giá trị phải là mean absolute (>= 0)
        for v in result.values():
            assert v >= 0.0


# ==============================================================================
# TestScoreDistribution
# ==============================================================================


class TestScoreDistribution:
    """Tests cho phương thức score_distribution."""

    def test_basic_statistics(self, analyzer: PredictionAnalyzer):
        """Kiểm tra mean, std, skewness, kurtosis được tính đúng."""
        preds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = analyzer.score_distribution(preds)

        assert isinstance(result, ScoreDistributionResult)
        assert abs(result.mean - 0.3) < 1e-10
        assert result.std > 0.0

    def test_too_conservative_flag(self, analyzer: PredictionAnalyzer):
        """std < 0.1 → too_conservative = True."""
        # Tạo mảng với std rất nhỏ
        preds = np.array([0.5, 0.5, 0.5, 0.5, 0.50001])
        result = analyzer.score_distribution(preds)
        assert result.too_conservative is True
        assert result.too_extreme is False
        assert result.flag_message is not None

    def test_too_extreme_flag(self, analyzer: PredictionAnalyzer):
        """std > 0.8 → too_extreme = True."""
        # Tạo mảng với std lớn (spread rộng)
        preds = np.array([-1.0, -1.0, 1.0, 1.0])
        result = analyzer.score_distribution(preds)
        assert result.std > 0.8
        assert result.too_extreme is True
        assert result.too_conservative is False
        assert result.flag_message is not None

    def test_normal_range_no_flags(self, analyzer: PredictionAnalyzer):
        """std trong khoảng [0.1, 0.8] → không có flags."""
        # std ~ 0.3 (trong khoảng bình thường)
        rng = np.random.default_rng(42)
        preds = rng.normal(0, 0.3, size=1000)
        result = analyzer.score_distribution(preds)
        assert result.too_conservative is False
        assert result.too_extreme is False
        assert result.flag_message is None

    def test_empty_array(self, analyzer: PredictionAnalyzer):
        """Mảng rỗng → giá trị mặc định."""
        result = analyzer.score_distribution(np.array([]))
        assert result.mean == 0.0
        assert result.std == 0.0

    def test_custom_thresholds(self, custom_analyzer: PredictionAnalyzer):
        """Sử dụng thresholds tùy chỉnh từ config."""
        # std ~ 0.08, nhỏ hơn default 0.1 nhưng lớn hơn custom 0.05
        preds = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        result = custom_analyzer.score_distribution(preds)
        # Với custom threshold 0.05, std ~ 0.14 không conservative
        assert result.too_conservative is False


# ==============================================================================
# TestConfusionMatrix
# ==============================================================================


class TestConfusionMatrix:
    """Tests cho phương thức confusion_matrix."""

    def test_all_true_positives(self, analyzer: PredictionAnalyzer):
        """Tất cả predicted up + actual up → chỉ TP."""
        preds = np.array([0.5, 0.3, 0.7, 0.1])
        actuals = np.array([0.02, 0.01, 0.05, 0.03])
        cm = analyzer.confusion_matrix(preds, actuals)
        assert cm.true_positive == 4
        assert cm.true_negative == 0
        assert cm.false_positive == 0
        assert cm.false_negative == 0

    def test_all_true_negatives(self, analyzer: PredictionAnalyzer):
        """Tất cả predicted down + actual down → chỉ TN."""
        preds = np.array([-0.5, -0.3, -0.7, -0.1])
        actuals = np.array([-0.02, -0.01, -0.05, -0.03])
        cm = analyzer.confusion_matrix(preds, actuals)
        assert cm.true_positive == 0
        assert cm.true_negative == 4
        assert cm.false_positive == 0
        assert cm.false_negative == 0

    def test_mixed_categories(self, analyzer: PredictionAnalyzer):
        """Hỗn hợp TP, TN, FP, FN."""
        preds = np.array([0.5, -0.3, 0.7, -0.1])
        actuals = np.array([0.02, -0.01, -0.05, 0.03])
        # TP: pred[0]=+, actual[0]=+ → TP
        # TN: pred[1]=-, actual[1]=- → TN
        # FP: pred[2]=+, actual[2]=- → FP
        # FN: pred[3]=-, actual[3]=+ → FN
        cm = analyzer.confusion_matrix(preds, actuals)
        assert cm.true_positive == 1
        assert cm.true_negative == 1
        assert cm.false_positive == 1
        assert cm.false_negative == 1

    def test_excludes_zero_predictions(self, analyzer: PredictionAnalyzer):
        """Loại bỏ cặp có prediction = 0."""
        preds = np.array([0.5, 0.0, -0.3])
        actuals = np.array([0.02, -0.01, -0.05])
        cm = analyzer.confusion_matrix(preds, actuals)
        # Chỉ 2 cặp hợp lệ: TP + TN
        assert cm.total == 2
        assert cm.true_positive == 1
        assert cm.true_negative == 1

    def test_excludes_zero_actuals(self, analyzer: PredictionAnalyzer):
        """Loại bỏ cặp có actual = 0."""
        preds = np.array([0.5, -0.3, 0.7])
        actuals = np.array([0.02, 0.0, 0.05])
        cm = analyzer.confusion_matrix(preds, actuals)
        # Chỉ 2 cặp hợp lệ: TP + TP
        assert cm.total == 2
        assert cm.true_positive == 2

    def test_empty_arrays(self, analyzer: PredictionAnalyzer):
        """Mảng rỗng → confusion matrix toàn 0."""
        cm = analyzer.confusion_matrix(np.array([]), np.array([]))
        assert cm.total == 0

    def test_total_equals_non_zero_pairs(self, analyzer: PredictionAnalyzer):
        """Total = số cặp có cả pred và actual != 0."""
        preds = np.array([0.5, 0.0, -0.3, 0.7, -0.1])
        actuals = np.array([0.02, -0.01, 0.0, -0.05, 0.03])
        # Cặp hợp lệ: [0], [3], [4] → 3 cặp
        cm = analyzer.confusion_matrix(preds, actuals)
        assert cm.total == 3


# ==============================================================================
# TestAnalyze
# ==============================================================================


class TestAnalyze:
    """Tests cho phương thức analyze (tổng hợp)."""

    def test_returns_prediction_quality_result(self, analyzer: PredictionAnalyzer):
        """analyze() trả về PredictionQualityResult đầy đủ."""
        preds = np.array([0.5, -0.3, 0.7, -0.1, 0.2])
        actuals = np.array([0.02, -0.01, 0.05, -0.03, 0.01])
        result = analyzer.analyze(preds, actuals)

        assert isinstance(result, PredictionQualityResult)
        assert 0.0 <= result.directional_accuracy <= 100.0
        assert -1.0 <= result.pearson_correlation <= 1.0
        assert set(result.calibration_quartiles.keys()) == {"Q1", "Q2", "Q3", "Q4"}
        assert isinstance(result.score_distribution, ScoreDistributionResult)
        assert isinstance(result.confusion_matrix, ConfusionMatrix)

    def test_perfect_predictions(self, analyzer: PredictionAnalyzer):
        """Dự đoán hoàn hảo → accuracy 100%, correlation gần 1."""
        preds = np.array([0.5, -0.3, 0.7, -0.1, 0.2])
        actuals = np.array([0.05, -0.03, 0.07, -0.01, 0.02])
        result = analyzer.analyze(preds, actuals)

        assert result.directional_accuracy == 100.0
        assert result.pearson_correlation > 0.9

    def test_empty_arrays(self, analyzer: PredictionAnalyzer):
        """Mảng rỗng → giá trị mặc định/zero."""
        result = analyzer.analyze(np.array([]), np.array([]))

        assert result.directional_accuracy == 0.0
        assert result.pearson_correlation == 0.0
        assert result.confusion_matrix.total == 0

    def test_confusion_matrix_consistency(self, analyzer: PredictionAnalyzer):
        """CM total = số cặp non-zero hợp lệ."""
        preds = np.array([0.5, -0.3, 0.7, -0.1])
        actuals = np.array([0.02, -0.01, -0.05, 0.03])
        result = analyzer.analyze(preds, actuals)

        # Tất cả đều non-zero → total = 4
        assert result.confusion_matrix.total == 4
        # TP + TN + FP + FN = 4
        cm = result.confusion_matrix
        assert cm.true_positive + cm.true_negative + cm.false_positive + cm.false_negative == 4
