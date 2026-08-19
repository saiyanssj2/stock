"""
Tests cho engine/evaluation_config.py

Kiểm tra khởi tạo dataclasses, giá trị mặc định, enum values,
và các computed properties.
"""

import pytest

from engine.evaluation_config import (
    ConfusionMatrix,
    DistributionStats,
    EvaluationConfig,
    EvaluationResult,
    ModelMetadata,
    OverfittingResult,
    OverfittingSeverity,
    PredictionQualityResult,
    ScoreDistributionResult,
    SymbolEvaluationResult,
)


# ==============================================================================
# EvaluationConfig tests
# ==============================================================================


class TestEvaluationConfig:
    """Tests cho EvaluationConfig dataclass."""

    def test_default_signal_thresholds(self):
        """Kiểm tra giá trị mặc định của signal thresholds."""
        config = EvaluationConfig()
        assert config.buy_threshold == 0.3
        assert config.sell_threshold == -0.3

    def test_default_baseline_params(self):
        """Kiểm tra giá trị mặc định cho baseline comparison."""
        config = EvaluationConfig()
        assert config.random_seed == 42
        assert config.sma_short == 20
        assert config.sma_long == 50

    def test_default_overfitting_thresholds(self):
        """Kiểm tra giá trị mặc định cho overfitting thresholds."""
        config = EvaluationConfig()
        assert config.overfitting_mild == 1.5
        assert config.overfitting_moderate == 2.0
        assert config.overfitting_severe == 3.0

    def test_default_score_distribution_thresholds(self):
        """Kiểm tra giá trị mặc định cho score distribution flags."""
        config = EvaluationConfig()
        assert config.conservative_std_threshold == 0.1
        assert config.extreme_std_threshold == 0.8

    def test_default_report_config(self):
        """Kiểm tra giá trị mặc định cho report output."""
        config = EvaluationConfig()
        assert config.report_dir == "engine/reports"
        assert config.json_filename == "evaluation_report.json"
        assert config.html_filename == "evaluation_report.html"

    def test_custom_thresholds(self):
        """Kiểm tra khởi tạo với giá trị tùy chỉnh."""
        config = EvaluationConfig(
            buy_threshold=0.5,
            sell_threshold=-0.5,
            random_seed=123,
        )
        assert config.buy_threshold == 0.5
        assert config.sell_threshold == -0.5
        assert config.random_seed == 123


# ==============================================================================
# OverfittingSeverity tests
# ==============================================================================


class TestOverfittingSeverity:
    """Tests cho OverfittingSeverity enum."""

    def test_enum_values(self):
        """Kiểm tra tất cả enum values."""
        assert OverfittingSeverity.NONE.value == "none"
        assert OverfittingSeverity.MILD.value == "mild"
        assert OverfittingSeverity.MODERATE.value == "moderate"
        assert OverfittingSeverity.SEVERE.value == "severe"

    def test_enum_count(self):
        """Kiểm tra đủ 4 mức severity."""
        assert len(OverfittingSeverity) == 4


# ==============================================================================
# DistributionStats tests
# ==============================================================================


class TestDistributionStats:
    """Tests cho DistributionStats dataclass."""

    def test_default_values(self):
        """Kiểm tra giá trị mặc định."""
        stats = DistributionStats()
        assert stats.mean == 0.0
        assert stats.std == 0.0
        assert stats.min == 0.0
        assert stats.max == 0.0

    def test_custom_values(self):
        """Kiểm tra khởi tạo với giá trị tùy chỉnh."""
        stats = DistributionStats(mean=0.5, std=0.2, min=-0.8, max=0.9)
        assert stats.mean == 0.5
        assert stats.std == 0.2
        assert stats.min == -0.8
        assert stats.max == 0.9


# ==============================================================================
# ConfusionMatrix tests
# ==============================================================================


class TestConfusionMatrix:
    """Tests cho ConfusionMatrix dataclass."""

    def test_default_values(self):
        """Kiểm tra giá trị mặc định (tất cả = 0)."""
        cm = ConfusionMatrix()
        assert cm.true_positive == 0
        assert cm.true_negative == 0
        assert cm.false_positive == 0
        assert cm.false_negative == 0

    def test_total_property(self):
        """Kiểm tra tính tổng đúng."""
        cm = ConfusionMatrix(
            true_positive=10, true_negative=5, false_positive=3, false_negative=2
        )
        assert cm.total == 20

    def test_accuracy_property(self):
        """Kiểm tra tính accuracy đúng."""
        cm = ConfusionMatrix(
            true_positive=10, true_negative=5, false_positive=3, false_negative=2
        )
        # (10 + 5) / 20 * 100 = 75.0
        assert cm.accuracy == 75.0

    def test_accuracy_zero_total(self):
        """Kiểm tra accuracy khi total = 0."""
        cm = ConfusionMatrix()
        assert cm.accuracy == 0.0

    def test_perfect_accuracy(self):
        """Kiểm tra accuracy 100% khi không có false predictions."""
        cm = ConfusionMatrix(true_positive=50, true_negative=50)
        assert cm.accuracy == 100.0


# ==============================================================================
# ScoreDistributionResult tests
# ==============================================================================


class TestScoreDistributionResult:
    """Tests cho ScoreDistributionResult dataclass."""

    def test_default_values(self):
        """Kiểm tra giá trị mặc định."""
        sdr = ScoreDistributionResult()
        assert sdr.mean == 0.0
        assert sdr.std == 0.0
        assert sdr.skewness == 0.0
        assert sdr.kurtosis == 0.0
        assert sdr.too_conservative is False
        assert sdr.too_extreme is False
        assert sdr.flag_message is None

    def test_flags_set(self):
        """Kiểm tra flags được set đúng."""
        sdr = ScoreDistributionResult(
            std=0.05,
            too_conservative=True,
            flag_message="Model too conservative",
        )
        assert sdr.too_conservative is True
        assert sdr.too_extreme is False
        assert sdr.flag_message == "Model too conservative"


# ==============================================================================
# PredictionQualityResult tests
# ==============================================================================


class TestPredictionQualityResult:
    """Tests cho PredictionQualityResult dataclass."""

    def test_default_values(self):
        """Kiểm tra giá trị mặc định."""
        pqr = PredictionQualityResult()
        assert pqr.directional_accuracy == 0.0
        assert pqr.pearson_correlation == 0.0
        assert pqr.calibration_quartiles == {}
        assert isinstance(pqr.score_distribution, ScoreDistributionResult)
        assert isinstance(pqr.confusion_matrix, ConfusionMatrix)


# ==============================================================================
# OverfittingResult tests
# ==============================================================================


class TestOverfittingResult:
    """Tests cho OverfittingResult dataclass."""

    def test_default_values(self):
        """Kiểm tra giá trị mặc định."""
        result = OverfittingResult()
        assert result.loss_ratio == 0.0
        assert result.severity == OverfittingSeverity.NONE
        assert result.ks_statistic == 0.0
        assert result.ks_p_value == 0.0
        assert isinstance(result.train_distribution, DistributionStats)
        assert isinstance(result.test_distribution, DistributionStats)
        assert result.warning_message is None

    def test_with_warning(self):
        """Kiểm tra khi có warning message."""
        result = OverfittingResult(
            loss_ratio=2.5,
            severity=OverfittingSeverity.MODERATE,
            warning_message="Overfitting detected: loss ratio = 2.5",
        )
        assert result.severity == OverfittingSeverity.MODERATE
        assert result.warning_message is not None


# ==============================================================================
# SymbolEvaluationResult tests
# ==============================================================================


class TestSymbolEvaluationResult:
    """Tests cho SymbolEvaluationResult dataclass."""

    def test_default_values(self):
        """Kiểm tra giá trị mặc định."""
        ser = SymbolEvaluationResult()
        assert ser.symbol == ""
        assert ser.backtest_result is None
        assert ser.prediction_quality is None

    def test_with_symbol(self):
        """Kiểm tra khởi tạo với symbol."""
        ser = SymbolEvaluationResult(symbol="VNM")
        assert ser.symbol == "VNM"


# ==============================================================================
# ModelMetadata tests
# ==============================================================================


class TestModelMetadata:
    """Tests cho ModelMetadata dataclass."""

    def test_default_values(self):
        """Kiểm tra giá trị mặc định."""
        meta = ModelMetadata()
        assert meta.architecture == "StockEvalNet (TCN + Attention)"
        assert meta.parameter_count == 0
        assert meta.training_timestamp is None
        assert meta.model_path is None
        assert meta.symbols_trained == []

    def test_custom_metadata(self):
        """Kiểm tra khởi tạo với metadata tùy chỉnh."""
        meta = ModelMetadata(
            parameter_count=1_500_000,
            training_timestamp="2024-01-15T10:30:00",
            model_path="engine/models/stock_eval_net.pt",
            symbols_trained=["VNM", "FPT", "VIC"],
        )
        assert meta.parameter_count == 1_500_000
        assert meta.symbols_trained == ["VNM", "FPT", "VIC"]


# ==============================================================================
# EvaluationResult tests
# ==============================================================================


class TestEvaluationResult:
    """Tests cho EvaluationResult dataclass."""

    def test_default_values(self):
        """Kiểm tra giá trị mặc định."""
        result = EvaluationResult()
        assert isinstance(result.model_metadata, ModelMetadata)
        assert result.evaluation_timestamp is not None
        assert result.symbol_results == {}
        assert result.aggregated_backtest is None
        assert result.comparison_result is None
        assert result.overfitting_result is None
        assert result.aggregated_prediction_quality is None

    def test_evaluation_timestamp_format(self):
        """Kiểm tra timestamp có format ISO."""
        result = EvaluationResult()
        # ISO format: YYYY-MM-DDTHH:MM:SS...
        assert "T" in result.evaluation_timestamp
        assert len(result.evaluation_timestamp) >= 19

    def test_symbol_results_isolation(self):
        """Kiểm tra mỗi instance có dict riêng (no shared state)."""
        result1 = EvaluationResult()
        result2 = EvaluationResult()
        result1.symbol_results["VNM"] = SymbolEvaluationResult(symbol="VNM")
        assert "VNM" not in result2.symbol_results
