"""
Module tạo báo cáo đánh giá mô hình dưới dạng JSON và HTML.

Chứa class ReportGenerator với các phương thức:
- generate_json(): serialize EvaluationResult thành JSON file
- generate_html(): tạo HTML report với metric tables, equity curve, confusion matrix
- _serialize_equity_curve(): convert pd.Series (DatetimeIndex) → list of [timestamp_iso, value]
"""

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from engine.config import BacktestResult, ComparisonResult
from engine.evaluation_config import (
    EvaluationConfig,
    EvaluationResult,
    ModelMetadata,
    OverfittingResult,
    PredictionQualityResult,
    SymbolEvaluationResult,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Tạo báo cáo đánh giá HTML và JSON."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Khởi tạo ReportGenerator.

        Args:
            config: Cấu hình đánh giá, sử dụng mặc định nếu None
        """
        self.config = config or EvaluationConfig()

    def generate_json(self, result: EvaluationResult) -> str:
        """
        Tạo JSON report chứa tất cả metrics.

        Serialize EvaluationResult thành JSON file với các keys:
        backtest_metrics, comparison_metrics, overfitting_metrics,
        prediction_quality, model_metadata, evaluation_timestamp, per_symbol.

        Args:
            result: Kết quả đánh giá tổng hợp

        Returns:
            Đường dẫn tuyệt đối đến file JSON đã tạo
        """
        # Tạo thư mục report nếu chưa tồn tại
        os.makedirs(self.config.report_dir, exist_ok=True)

        # Xây dựng dictionary cho JSON report
        report_data = self._build_report_dict(result)

        # Ghi file JSON
        json_path = os.path.join(self.config.report_dir, self.config.json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"JSON report đã được tạo tại: {json_path}")
        return os.path.abspath(json_path)

    def generate_html(self, result: EvaluationResult) -> Optional[str]:
        """
        Tạo HTML report với formatted metric tables, equity curve data,
        và directional confusion matrix.

        Args:
            result: Kết quả đánh giá tổng hợp

        Returns:
            Đường dẫn tuyệt đối đến file HTML, hoặc None nếu generation thất bại

        Note:
            Nếu HTML generation fail, error được log và trả None (graceful degradation).
        """
        try:
            # Tạo thư mục report nếu chưa tồn tại
            os.makedirs(self.config.report_dir, exist_ok=True)

            # Render HTML template
            html_content = self._render_html_template(result)

            # Ghi file HTML
            html_path = os.path.join(self.config.report_dir, self.config.html_filename)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"HTML report đã được tạo tại: {html_path}")
            return os.path.abspath(html_path)

        except Exception as e:
            logger.error(f"HTML report generation thất bại: {e}")
            return None

    def _serialize_equity_curve(self, curve: Optional[pd.Series]) -> List[List]:
        """
        Serialize equity curve thành list of [timestamp_iso, value] pairs.

        Args:
            curve: pd.Series với DatetimeIndex chứa giá trị equity theo thời gian

        Returns:
            List of [timestamp_iso_str, float_value] pairs.
            Trả về list rỗng nếu curve là None hoặc rỗng.
        """
        if curve is None or curve.empty:
            return []

        result = []
        for timestamp, value in curve.items():
            # Chuyển timestamp sang ISO string
            if isinstance(timestamp, pd.Timestamp):
                ts_str = timestamp.isoformat()
            else:
                ts_str = str(timestamp)
            result.append([ts_str, float(value)])

        return result

    def _build_report_dict(self, result: EvaluationResult) -> Dict[str, Any]:
        """
        Xây dựng dictionary chứa tất cả dữ liệu cho JSON report.

        Args:
            result: Kết quả đánh giá tổng hợp

        Returns:
            Dictionary với cấu trúc report hoàn chỉnh
        """
        report = {}

        # Model metadata
        report["model_metadata"] = self._serialize_dataclass(result.model_metadata)

        # Evaluation timestamp
        report["evaluation_timestamp"] = result.evaluation_timestamp

        # Backtest metrics
        report["backtest_metrics"] = self._serialize_backtest_result(
            result.aggregated_backtest
        )

        # Comparison metrics
        report["comparison_metrics"] = self._serialize_comparison_result(
            result.comparison_result
        )

        # Overfitting metrics
        report["overfitting_metrics"] = self._serialize_dataclass(
            result.overfitting_result
        )

        # Prediction quality
        report["prediction_quality"] = self._serialize_dataclass(
            result.aggregated_prediction_quality
        )

        # Per-symbol breakdown
        report["per_symbol"] = self._serialize_per_symbol(result.symbol_results)

        return report

    def _serialize_dataclass(self, obj: Any) -> Optional[Dict[str, Any]]:
        """
        Serialize một dataclass thành dictionary, xử lý Enum values.

        Args:
            obj: Dataclass object hoặc None

        Returns:
            Dictionary đã serialize hoặc None
        """
        if obj is None:
            return None

        data = asdict(obj)
        return self._convert_enums(data)

    def _convert_enums(self, data: Any) -> Any:
        """
        Đệ quy chuyển Enum values thành .value string.

        Args:
            data: Dữ liệu cần xử lý (dict, list, hoặc giá trị đơn)

        Returns:
            Dữ liệu đã chuyển đổi Enum
        """
        if isinstance(data, dict):
            return {key: self._convert_enums(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_enums(item) for item in data]
        elif isinstance(data, Enum):
            return data.value
        return data

    def _serialize_backtest_result(
        self, backtest: Optional[BacktestResult]
    ) -> Optional[Dict[str, Any]]:
        """
        Serialize BacktestResult, bao gồm equity curve dạng list of pairs.

        Args:
            backtest: Kết quả backtest hoặc None

        Returns:
            Dictionary chứa metrics và equity_curve đã serialize
        """
        if backtest is None:
            return None

        return {
            "total_return_pct": backtest.total_return_pct,
            "annualized_return_pct": backtest.annualized_return_pct,
            "win_rate": backtest.win_rate,
            "max_drawdown": backtest.max_drawdown,
            "sharpe_ratio": backtest.sharpe_ratio,
            "equity_curve": self._serialize_equity_curve(backtest.equity_curve),
            "trades_count": len(backtest.trades),
        }

    def _serialize_comparison_result(
        self, comparison: Optional[ComparisonResult]
    ) -> Optional[Dict[str, Any]]:
        """
        Serialize ComparisonResult với tất cả strategy results.

        Args:
            comparison: Kết quả so sánh strategies hoặc None

        Returns:
            Dictionary chứa comparison metrics
        """
        if comparison is None:
            return None

        strategies = {}
        for name, br in comparison.results.items():
            strategies[name] = self._serialize_backtest_result(br)

        return {
            "strategies": strategies,
            "date_range_start": comparison.date_range_start,
            "date_range_end": comparison.date_range_end,
            "initial_capital": comparison.initial_capital,
            "excluded_strategies": comparison.excluded_strategies,
            "exclusion_reasons": comparison.exclusion_reasons,
        }

    def _serialize_per_symbol(
        self, symbol_results: Dict[str, SymbolEvaluationResult]
    ) -> Dict[str, Any]:
        """
        Serialize per-symbol results.

        Args:
            symbol_results: Dict mapping symbol → SymbolEvaluationResult

        Returns:
            Dictionary chứa per-symbol breakdown
        """
        per_symbol = {}
        for symbol, sym_result in symbol_results.items():
            entry: Dict[str, Any] = {"symbol": sym_result.symbol}

            # Backtest result cho symbol
            entry["backtest_result"] = self._serialize_backtest_result(
                sym_result.backtest_result
            )

            # Prediction quality cho symbol
            entry["prediction_quality"] = self._serialize_dataclass(
                sym_result.prediction_quality
            )

            per_symbol[symbol] = entry

        return per_symbol

    def _render_html_template(self, result: EvaluationResult) -> str:
        """
        Render HTML report sử dụng inline template (string formatting).

        Args:
            result: Kết quả đánh giá tổng hợp

        Returns:
            Chuỗi HTML hoàn chỉnh
        """
        # Metadata section
        metadata_html = self._render_metadata_section(result)

        # Backtest metrics section
        backtest_html = self._render_backtest_section(result.aggregated_backtest)

        # Comparison section
        comparison_html = self._render_comparison_section(result.comparison_result)

        # Overfitting section
        overfitting_html = self._render_overfitting_section(result.overfitting_result)

        # Prediction quality section
        prediction_html = self._render_prediction_section(
            result.aggregated_prediction_quality
        )

        # Per-symbol section
        per_symbol_html = self._render_per_symbol_section(result.symbol_results)

        # Equity curve data (JSON cho JavaScript visualization)
        equity_data_json = json.dumps(
            self._serialize_equity_curve(
                result.aggregated_backtest.equity_curve
                if result.aggregated_backtest
                else None
            ),
            default=str,
        )

        html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo cáo đánh giá mô hình - Post-Training Evaluation</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-bottom: 1px solid #ecf0f1; padding-bottom: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px 15px; text-align: left; border: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .metric-value {{ font-weight: bold; color: #2c3e50; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .warning {{ color: #f39c12; font-weight: bold; }}
        .severity-none {{ color: #27ae60; }}
        .severity-mild {{ color: #f39c12; }}
        .severity-moderate {{ color: #e67e22; }}
        .severity-severe {{ color: #e74c3c; }}
        .confusion-matrix {{ width: auto; margin: 15px 0; }}
        .confusion-matrix th {{ background-color: #8e44ad; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
        #equity-curve-data {{ display: none; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Báo cáo đánh giá mô hình sau huấn luyện</h1>
        <p class="timestamp">Thời gian đánh giá: {result.evaluation_timestamp}</p>

        {metadata_html}
        {backtest_html}
        {comparison_html}
        {overfitting_html}
        {prediction_html}
        {per_symbol_html}

        <!-- Dữ liệu equity curve cho downstream visualization -->
        <div id="equity-curve-data" data-equity='{equity_data_json}'></div>
    </div>
</body>
</html>"""

        return html

    def _render_metadata_section(self, result: EvaluationResult) -> str:
        """Render phần metadata của model."""
        meta = result.model_metadata
        return f"""
        <h2>🔧 Thông tin mô hình</h2>
        <table>
            <tr><th>Thuộc tính</th><th>Giá trị</th></tr>
            <tr><td>Kiến trúc</td><td class="metric-value">{meta.architecture}</td></tr>
            <tr><td>Số lượng tham số</td><td class="metric-value">{meta.parameter_count:,}</td></tr>
            <tr><td>Thời gian huấn luyện</td><td>{meta.training_timestamp or 'N/A'}</td></tr>
            <tr><td>Đường dẫn model</td><td>{meta.model_path or 'N/A'}</td></tr>
            <tr><td>Symbols đã huấn luyện</td><td>{', '.join(meta.symbols_trained) if meta.symbols_trained else 'N/A'}</td></tr>
        </table>"""

    def _render_backtest_section(
        self, backtest: Optional[BacktestResult]
    ) -> str:
        """Render phần backtest metrics."""
        if backtest is None:
            return "<h2>📈 Kết quả Backtest</h2><p>Không có dữ liệu backtest.</p>"

        return_class = "positive" if backtest.total_return_pct >= 0 else "negative"
        return f"""
        <h2>📈 Kết quả Backtest</h2>
        <table>
            <tr><th>Metric</th><th>Giá trị</th></tr>
            <tr><td>Tổng lợi nhuận</td><td class="metric-value {return_class}">{backtest.total_return_pct:.2f}%</td></tr>
            <tr><td>Lợi nhuận hàng năm</td><td class="metric-value">{backtest.annualized_return_pct:.2f}%</td></tr>
            <tr><td>Win Rate</td><td class="metric-value">{backtest.win_rate:.2f}%</td></tr>
            <tr><td>Max Drawdown</td><td class="metric-value negative">{backtest.max_drawdown:.2f}%</td></tr>
            <tr><td>Sharpe Ratio</td><td class="metric-value">{backtest.sharpe_ratio:.4f}</td></tr>
            <tr><td>Số lệnh giao dịch</td><td class="metric-value">{len(backtest.trades)}</td></tr>
        </table>"""

    def _render_comparison_section(
        self, comparison: Optional[ComparisonResult]
    ) -> str:
        """Render phần so sánh strategies."""
        if comparison is None:
            return "<h2>⚖️ So sánh với Baseline</h2><p>Không có dữ liệu so sánh.</p>"

        rows = ""
        for name, br in comparison.results.items():
            return_class = "positive" if br.total_return_pct >= 0 else "negative"
            rows += f"""
            <tr>
                <td class="metric-value">{name}</td>
                <td class="{return_class}">{br.total_return_pct:.2f}%</td>
                <td>{br.annualized_return_pct:.2f}%</td>
                <td>{br.win_rate:.2f}%</td>
                <td>{br.max_drawdown:.2f}%</td>
                <td>{br.sharpe_ratio:.4f}</td>
            </tr>"""

        return f"""
        <h2>⚖️ So sánh với Baseline</h2>
        <table>
            <tr>
                <th>Strategy</th><th>Tổng lợi nhuận</th><th>Hàng năm</th>
                <th>Win Rate</th><th>Max DD</th><th>Sharpe</th>
            </tr>
            {rows}
        </table>"""

    def _render_overfitting_section(
        self, overfitting: Optional[OverfittingResult]
    ) -> str:
        """Render phần overfitting detection."""
        if overfitting is None:
            return "<h2>🔍 Phát hiện Overfitting</h2><p>Không có dữ liệu overfitting.</p>"

        severity_class = f"severity-{overfitting.severity.value}"
        warning_html = ""
        if overfitting.warning_message:
            warning_html = f'<tr><td>Cảnh báo</td><td class="warning">{overfitting.warning_message}</td></tr>'

        return f"""
        <h2>🔍 Phát hiện Overfitting</h2>
        <table>
            <tr><th>Metric</th><th>Giá trị</th></tr>
            <tr><td>Loss Ratio (test/train)</td><td class="metric-value">{overfitting.loss_ratio:.4f}</td></tr>
            <tr><td>Mức độ</td><td class="metric-value {severity_class}">{overfitting.severity.value.upper()}</td></tr>
            <tr><td>KS Statistic</td><td class="metric-value">{overfitting.ks_statistic:.4f}</td></tr>
            <tr><td>KS P-value</td><td class="metric-value">{overfitting.ks_p_value:.4f}</td></tr>
            {warning_html}
        </table>"""

    def _render_prediction_section(
        self, prediction: Optional[PredictionQualityResult]
    ) -> str:
        """Render phần prediction quality bao gồm confusion matrix."""
        if prediction is None:
            return "<h2>🎯 Chất lượng dự đoán</h2><p>Không có dữ liệu prediction quality.</p>"

        # Confusion matrix HTML
        cm = prediction.confusion_matrix
        confusion_html = f"""
        <h3>Ma trận nhầm lẫn (Confusion Matrix)</h3>
        <table class="confusion-matrix">
            <tr><th></th><th>Actual Up</th><th>Actual Down</th></tr>
            <tr><td><strong>Predicted Up</strong></td><td class="positive">{cm.true_positive} (TP)</td><td class="negative">{cm.false_positive} (FP)</td></tr>
            <tr><td><strong>Predicted Down</strong></td><td class="negative">{cm.false_negative} (FN)</td><td class="positive">{cm.true_negative} (TN)</td></tr>
        </table>
        <p>Accuracy: <span class="metric-value">{cm.accuracy:.2f}%</span> | Total: {cm.total}</p>"""

        # Score distribution info
        sd = prediction.score_distribution
        flag_html = ""
        if sd.flag_message:
            flag_html = f'<tr><td>Cảnh báo phân phối</td><td class="warning">{sd.flag_message}</td></tr>'

        return f"""
        <h2>🎯 Chất lượng dự đoán</h2>
        <table>
            <tr><th>Metric</th><th>Giá trị</th></tr>
            <tr><td>Directional Accuracy</td><td class="metric-value">{prediction.directional_accuracy:.2f}%</td></tr>
            <tr><td>Pearson Correlation</td><td class="metric-value">{prediction.pearson_correlation:.4f}</td></tr>
            <tr><td>Score Mean</td><td>{sd.mean:.4f}</td></tr>
            <tr><td>Score Std</td><td>{sd.std:.4f}</td></tr>
            <tr><td>Score Skewness</td><td>{sd.skewness:.4f}</td></tr>
            <tr><td>Score Kurtosis</td><td>{sd.kurtosis:.4f}</td></tr>
            {flag_html}
        </table>
        {confusion_html}"""

    def _render_per_symbol_section(
        self, symbol_results: Dict[str, SymbolEvaluationResult]
    ) -> str:
        """Render phần per-symbol breakdown."""
        if not symbol_results:
            return "<h2>📋 Kết quả theo Symbol</h2><p>Không có dữ liệu per-symbol.</p>"

        rows = ""
        for symbol, sym_result in symbol_results.items():
            br = sym_result.backtest_result
            pq = sym_result.prediction_quality

            total_return = f"{br.total_return_pct:.2f}%" if br else "N/A"
            win_rate = f"{br.win_rate:.2f}%" if br else "N/A"
            dir_acc = f"{pq.directional_accuracy:.2f}%" if pq else "N/A"
            correlation = f"{pq.pearson_correlation:.4f}" if pq else "N/A"

            rows += f"""
            <tr>
                <td class="metric-value">{symbol}</td>
                <td>{total_return}</td>
                <td>{win_rate}</td>
                <td>{dir_acc}</td>
                <td>{correlation}</td>
            </tr>"""

        return f"""
        <h2>📋 Kết quả theo Symbol</h2>
        <table>
            <tr>
                <th>Symbol</th><th>Tổng lợi nhuận</th><th>Win Rate</th>
                <th>Directional Acc.</th><th>Correlation</th>
            </tr>
            {rows}
        </table>"""
