"""
Tests cho engine/report_generator.py

Kiểm tra ReportGenerator: generate_json(), generate_html(),
_serialize_equity_curve(), graceful degradation khi HTML fail.
"""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from engine.config import BacktestResult, ComparisonResult, Trade
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
from engine.report_generator import ReportGenerator


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def tmp_report_dir(tmp_path):
    """Tạo thư mục tạm cho report output."""
    return str(tmp_path / "reports")


@pytest.fixture
def config(tmp_report_dir):
    """Config với report_dir trỏ đến thư mục tạm."""
    return EvaluationConfig(
        report_dir=tmp_report_dir,
        json_filename="test_report.json",
        html_filename="test_report.html",
    )


@pytest.fixture
def generator(config):
    """ReportGenerator instance với config tạm."""
    return ReportGenerator(config)


@pytest.fixture
def sample_equity_curve():
    """Tạo equity curve mẫu với DatetimeIndex."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    values = [100_000_000.0, 101_000_000.0, 99_500_000.0, 102_000_000.0, 103_000_000.0]
    return pd.Series(values, index=dates)


@pytest.fixture
def sample_backtest_result(sample_equity_curve):
    """BacktestResult mẫu."""
    return BacktestResult(
        total_return_pct=3.0,
        annualized_return_pct=15.5,
        win_rate=60.0,
        max_drawdown=-5.2,
        sharpe_ratio=1.25,
        equity_curve=sample_equity_curve,
        trades=[
            Trade(
                entry_date=pd.Timestamp("2024-01-02"),
                exit_date=pd.Timestamp("2024-01-04"),
                entry_price=25000.0,
                exit_price=26000.0,
                shares=100,
                pnl=100000.0,
                pnl_pct=4.0,
            )
        ],
    )


@pytest.fixture
def sample_comparison_result(sample_backtest_result):
    """ComparisonResult mẫu."""
    return ComparisonResult(
        results={
            "model": sample_backtest_result,
            "buy_and_hold": BacktestResult(total_return_pct=2.0, sharpe_ratio=0.8),
            "random": BacktestResult(total_return_pct=-1.0, sharpe_ratio=-0.3),
            "sma_crossover": BacktestResult(total_return_pct=1.5, sharpe_ratio=0.6),
        },
        date_range_start="2024-01-01",
        date_range_end="2024-03-31",
        initial_capital=100_000_000.0,
    )


@pytest.fixture
def sample_overfitting_result():
    """OverfittingResult mẫu."""
    return OverfittingResult(
        loss_ratio=1.8,
        severity=OverfittingSeverity.MILD,
        ks_statistic=0.12,
        ks_p_value=0.45,
        train_distribution=DistributionStats(mean=0.1, std=0.3, min=-0.8, max=0.9),
        test_distribution=DistributionStats(mean=0.05, std=0.35, min=-0.7, max=0.85),
        warning_message=None,
    )


@pytest.fixture
def sample_prediction_quality():
    """PredictionQualityResult mẫu."""
    return PredictionQualityResult(
        directional_accuracy=55.0,
        pearson_correlation=0.32,
        calibration_quartiles={"Q1": 0.01, "Q2": 0.02, "Q3": 0.03, "Q4": 0.05},
        score_distribution=ScoreDistributionResult(
            mean=0.05, std=0.3, skewness=0.1, kurtosis=2.5
        ),
        confusion_matrix=ConfusionMatrix(
            true_positive=30, true_negative=25, false_positive=20, false_negative=25
        ),
    )


@pytest.fixture
def sample_evaluation_result(
    sample_backtest_result,
    sample_comparison_result,
    sample_overfitting_result,
    sample_prediction_quality,
):
    """EvaluationResult hoàn chỉnh mẫu."""
    return EvaluationResult(
        model_metadata=ModelMetadata(
            architecture="StockEvalNet (TCN + Attention)",
            parameter_count=150000,
            training_timestamp="2024-01-01T10:00:00",
            model_path="engine/models/stock_eval_net.pt",
            symbols_trained=["VNM", "FPT", "VIC"],
        ),
        evaluation_timestamp="2024-01-15T14:30:00",
        symbol_results={
            "VNM": SymbolEvaluationResult(
                symbol="VNM",
                backtest_result=sample_backtest_result,
                prediction_quality=sample_prediction_quality,
            ),
            "FPT": SymbolEvaluationResult(
                symbol="FPT",
                backtest_result=BacktestResult(total_return_pct=5.0, win_rate=65.0),
                prediction_quality=None,
            ),
        },
        aggregated_backtest=sample_backtest_result,
        comparison_result=sample_comparison_result,
        overfitting_result=sample_overfitting_result,
        aggregated_prediction_quality=sample_prediction_quality,
    )


# ==============================================================================
# _serialize_equity_curve tests
# ==============================================================================


class TestSerializeEquityCurve:
    """Tests cho _serialize_equity_curve."""

    def test_serialize_normal_equity_curve(self, generator, sample_equity_curve):
        """Serialize equity curve bình thường → list of [timestamp_iso, value]."""
        result = generator._serialize_equity_curve(sample_equity_curve)

        assert len(result) == 5
        # Kiểm tra cấu trúc mỗi phần tử: [str, float]
        for item in result:
            assert isinstance(item, list)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_serialize_preserves_values(self, generator, sample_equity_curve):
        """Giá trị sau serialize phải khớp với giá trị gốc."""
        result = generator._serialize_equity_curve(sample_equity_curve)

        for i, (timestamp, value) in enumerate(result):
            assert value == sample_equity_curve.iloc[i]

    def test_serialize_timestamps_are_iso_format(self, generator, sample_equity_curve):
        """Timestamps phải ở định dạng ISO."""
        result = generator._serialize_equity_curve(sample_equity_curve)

        for item in result:
            # Parse ISO string phải thành công
            parsed = datetime.fromisoformat(item[0])
            assert parsed is not None

    def test_serialize_none_curve(self, generator):
        """None equity curve → list rỗng."""
        result = generator._serialize_equity_curve(None)
        assert result == []

    def test_serialize_empty_curve(self, generator):
        """Empty pd.Series → list rỗng."""
        empty_curve = pd.Series([], dtype=float)
        result = generator._serialize_equity_curve(empty_curve)
        assert result == []

    def test_serialize_single_point(self, generator):
        """Equity curve với 1 điểm duy nhất."""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        curve = pd.Series([100_000_000.0], index=dates)

        result = generator._serialize_equity_curve(curve)
        assert len(result) == 1
        assert result[0][1] == 100_000_000.0


# ==============================================================================
# generate_json tests
# ==============================================================================


class TestGenerateJson:
    """Tests cho generate_json."""

    def test_creates_report_dir(self, generator, config):
        """Tạo thư mục report_dir nếu chưa tồn tại."""
        # Đảm bảo thư mục chưa tồn tại
        assert not os.path.exists(config.report_dir)

        result = EvaluationResult()
        generator.generate_json(result)

        assert os.path.exists(config.report_dir)

    def test_returns_absolute_path(self, generator):
        """Trả về đường dẫn tuyệt đối đến file JSON."""
        result = EvaluationResult()
        path = generator.generate_json(result)

        assert os.path.isabs(path)
        assert path.endswith(".json")

    def test_json_file_is_valid(self, generator, sample_evaluation_result):
        """File JSON phải parse được."""
        path = generator.generate_json(sample_evaluation_result)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_json_contains_required_keys(self, generator, sample_evaluation_result):
        """JSON phải chứa tất cả required keys."""
        path = generator.generate_json(sample_evaluation_result)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = [
            "backtest_metrics",
            "comparison_metrics",
            "overfitting_metrics",
            "prediction_quality",
            "model_metadata",
            "evaluation_timestamp",
            "per_symbol",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_json_per_symbol_count_matches(self, generator, sample_evaluation_result):
        """Số lượng entries trong per_symbol phải khớp với symbol_results."""
        path = generator.generate_json(sample_evaluation_result)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data["per_symbol"]) == len(
            sample_evaluation_result.symbol_results
        )

    def test_json_equity_curve_format(self, generator, sample_evaluation_result):
        """Equity curve trong JSON phải là list of [str, float]."""
        path = generator.generate_json(sample_evaluation_result)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        equity_curve = data["backtest_metrics"]["equity_curve"]
        assert isinstance(equity_curve, list)
        assert len(equity_curve) > 0
        for item in equity_curve:
            assert isinstance(item, list)
            assert len(item) == 2

    def test_json_enum_serialized_as_value(self, generator, sample_evaluation_result):
        """Enum values phải được serialize thành string value."""
        path = generator.generate_json(sample_evaluation_result)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        severity = data["overfitting_metrics"]["severity"]
        assert severity == "mild"  # OverfittingSeverity.MILD.value

    def test_json_model_metadata(self, generator, sample_evaluation_result):
        """Model metadata phải được serialize đầy đủ."""
        path = generator.generate_json(sample_evaluation_result)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data["model_metadata"]
        assert meta["architecture"] == "StockEvalNet (TCN + Attention)"
        assert meta["parameter_count"] == 150000
        assert meta["symbols_trained"] == ["VNM", "FPT", "VIC"]

    def test_json_with_none_fields(self, generator):
        """JSON với None fields (không có backtest, comparison, etc.)."""
        result = EvaluationResult()
        path = generator.generate_json(result)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["backtest_metrics"] is None
        assert data["comparison_metrics"] is None
        assert data["overfitting_metrics"] is None
        assert data["prediction_quality"] is None
        assert data["per_symbol"] == {}


# ==============================================================================
# generate_html tests
# ==============================================================================


class TestGenerateHtml:
    """Tests cho generate_html."""

    def test_creates_report_dir(self, generator, config):
        """Tạo thư mục report_dir nếu chưa tồn tại."""
        assert not os.path.exists(config.report_dir)

        result = EvaluationResult()
        generator.generate_html(result)

        assert os.path.exists(config.report_dir)

    def test_returns_absolute_path(self, generator, sample_evaluation_result):
        """Trả về đường dẫn tuyệt đối đến file HTML."""
        path = generator.generate_html(sample_evaluation_result)

        assert path is not None
        assert os.path.isabs(path)
        assert path.endswith(".html")

    def test_html_contains_doctype(self, generator, sample_evaluation_result):
        """HTML phải bắt đầu bằng <!DOCTYPE html>."""
        path = generator.generate_html(sample_evaluation_result)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        assert content.startswith("<!DOCTYPE html>")

    def test_html_contains_metric_tables(self, generator, sample_evaluation_result):
        """HTML phải chứa các metric tables."""
        path = generator.generate_html(sample_evaluation_result)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Kiểm tra các section chính
        assert "Kết quả Backtest" in content
        assert "So sánh với Baseline" in content
        assert "Phát hiện Overfitting" in content
        assert "Chất lượng dự đoán" in content

    def test_html_contains_confusion_matrix(self, generator, sample_evaluation_result):
        """HTML phải chứa confusion matrix."""
        path = generator.generate_html(sample_evaluation_result)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Confusion Matrix" in content
        assert "(TP)" in content
        assert "(FP)" in content
        assert "(FN)" in content
        assert "(TN)" in content

    def test_html_contains_equity_curve_data(self, generator, sample_evaluation_result):
        """HTML phải chứa equity curve data embedded."""
        path = generator.generate_html(sample_evaluation_result)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "equity-curve-data" in content

    def test_html_contains_per_symbol(self, generator, sample_evaluation_result):
        """HTML phải chứa per-symbol section."""
        path = generator.generate_html(sample_evaluation_result)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "VNM" in content
        assert "FPT" in content

    def test_html_graceful_degradation(self, generator, sample_evaluation_result, config):
        """Khi HTML generation fail, trả None và không raise exception."""
        # Giả lập lỗi bằng cách patch open để raise exception
        with patch("builtins.open", side_effect=IOError("Disk full")):
            result = generator.generate_html(sample_evaluation_result)

        assert result is None

    def test_html_with_none_sections(self, generator):
        """HTML với result rỗng (tất cả None) vẫn phải render thành công."""
        result = EvaluationResult()
        path = generator.generate_html(result)

        assert path is not None
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "<!DOCTYPE html>" in content
        assert "Không có dữ liệu" in content

    def test_html_overfitting_severity_displayed(
        self, generator, sample_evaluation_result
    ):
        """Overfitting severity phải hiển thị trong HTML."""
        path = generator.generate_html(sample_evaluation_result)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "MILD" in content


# ==============================================================================
# Constructor tests
# ==============================================================================


class TestReportGeneratorInit:
    """Tests cho ReportGenerator constructor."""

    def test_default_config(self):
        """Khởi tạo với config mặc định."""
        gen = ReportGenerator()
        assert gen.config.report_dir == "engine/reports"
        assert gen.config.json_filename == "evaluation_report.json"
        assert gen.config.html_filename == "evaluation_report.html"

    def test_custom_config(self, config):
        """Khởi tạo với custom config."""
        gen = ReportGenerator(config)
        assert gen.config == config
