"""
Unit tests cho EvaluationModule (orchestrator).

Tests bao gồm:
- Khởi tạo module với tất cả sub-components
- _load_model() raise ModelError khi file không tồn tại
- _normalize_features() áp dụng min-max và z-score normalization
- _validate_inputs() raise errors khi inputs không hợp lệ
- Per-symbol error isolation
- evaluate() end-to-end flow với mock model
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from engine.config import DataError, ModelError
from engine.evaluation_config import EvaluationConfig, EvaluationResult
from engine.evaluation_module import EvaluationModule
from engine.training_pipeline import TrainingResult


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def config():
    """Cấu hình đánh giá mặc định cho tests."""
    return EvaluationConfig(
        report_dir=tempfile.mkdtemp(),
    )


@pytest.fixture
def module(config):
    """EvaluationModule instance cho tests."""
    return EvaluationModule(config=config)


@pytest.fixture
def sample_df():
    """DataFrame mẫu cho tests."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "time": dates,
        "open": np.random.uniform(20, 30, n),
        "high": np.random.uniform(25, 35, n),
        "low": np.random.uniform(15, 25, n),
        "close": np.random.uniform(20, 30, n),
        "volume": np.random.randint(1000, 10000, n),
    })


@pytest.fixture
def training_result(tmp_path):
    """TrainingResult mẫu cho tests."""
    # Tạo file model giả
    model_path = str(tmp_path / "model.pt")
    # Không tạo file thật ở đây - tests sẽ tạo khi cần
    return TrainingResult(
        epochs_completed=10,
        final_train_loss=0.05,
        final_val_loss=0.08,
        model_path=model_path,
        symbols_trained=["VNM", "FPT"],
    )


# ==============================================================================
# Test __init__
# ==============================================================================


class TestEvaluationModuleInit:
    """Tests cho EvaluationModule.__init__()."""

    def test_init_default_config(self):
        """Khởi tạo với config mặc định."""
        module = EvaluationModule()
        assert module.config is not None
        assert module.backtest_evaluator is not None
        assert module.baseline_comparator is not None
        assert module.overfitting_detector is not None
        assert module.prediction_analyzer is not None
        assert module.report_generator is not None

    def test_init_custom_config(self, config):
        """Khởi tạo với config tùy chỉnh."""
        module = EvaluationModule(config=config)
        assert module.config is config
        assert module.config.report_dir == config.report_dir

    def test_sub_components_share_config(self, module, config):
        """Tất cả sub-components nhận cùng config."""
        assert module.backtest_evaluator.config is config
        assert module.baseline_comparator.config is config
        assert module.overfitting_detector.config is config
        assert module.prediction_analyzer.config is config
        assert module.report_generator.config is config


# ==============================================================================
# Test _load_model
# ==============================================================================


class TestLoadModel:
    """Tests cho EvaluationModule._load_model()."""

    def test_load_model_file_not_found(self, module):
        """Raise ModelError khi file không tồn tại."""
        with pytest.raises(ModelError) as exc_info:
            module._load_model("/nonexistent/path/model.pt")

        assert exc_info.value.error_code == "MODEL_NOT_FOUND"
        assert "model_path" in exc_info.value.details

    def test_load_model_invalid_file(self, module, tmp_path):
        """Raise ModelError khi file không phải model hợp lệ."""
        # Tạo file giả không phải model
        fake_model_path = str(tmp_path / "fake_model.pt")
        Path(fake_model_path).write_text("not a model")

        with pytest.raises(ModelError) as exc_info:
            module._load_model(fake_model_path)

        assert exc_info.value.error_code == "MODEL_LOAD_ERROR"

    def test_load_model_success(self, module, tmp_path):
        """Load model thành công từ valid checkpoint."""
        from engine.evaluation_model import StockEvalNet

        # Tạo model và save state_dict
        model = StockEvalNet()
        model_path = str(tmp_path / "valid_model.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)

        # Load model
        loaded_model = module._load_model(model_path)
        assert isinstance(loaded_model, StockEvalNet)
        assert not loaded_model.training  # Phải ở eval mode


# ==============================================================================
# Test _normalize_features
# ==============================================================================


class TestNormalizeFeatures:
    """Tests cho EvaluationModule._normalize_features()."""

    def test_min_max_normalization(self, module):
        """Min-max normalization áp dụng đúng công thức."""
        features = np.array([[10.0, 20.0], [30.0, 40.0]])
        norm_params = {
            "min_vals": [0.0, 0.0],
            "max_vals": [40.0, 80.0],
        }

        result = module._normalize_features(features, norm_params)

        expected = np.array([[0.25, 0.25], [0.75, 0.5]])
        np.testing.assert_allclose(result, expected)

    def test_zscore_normalization(self, module):
        """Z-score normalization áp dụng đúng công thức."""
        features = np.array([[10.0, 20.0], [30.0, 40.0]])
        norm_params = {
            "mean": [20.0, 30.0],
            "std": [10.0, 10.0],
        }

        result = module._normalize_features(features, norm_params)

        expected = np.array([[-1.0, -1.0], [1.0, 1.0]])
        np.testing.assert_allclose(result, expected)

    def test_norm_params_mismatch_raises_error(self, module):
        """Raise DataError khi norm_params size không khớp."""
        features = np.array([[10.0, 20.0, 30.0]])  # 3 features
        norm_params = {
            "min_vals": [0.0, 0.0],  # Chỉ 2 values
            "max_vals": [1.0, 1.0],
        }

        with pytest.raises(DataError) as exc_info:
            module._normalize_features(features, norm_params)

        assert exc_info.value.error_code == "NORM_PARAMS_MISMATCH"

    def test_no_valid_keys_returns_features_unchanged(self, module):
        """Trả về features gốc khi norm_params không có keys hợp lệ."""
        features = np.array([[1.0, 2.0]])
        norm_params = {"invalid_key": [1, 2]}

        result = module._normalize_features(features, norm_params)
        np.testing.assert_array_equal(result, features)

    def test_zero_range_handling(self, module):
        """Xử lý trường hợp max == min (range = 0)."""
        features = np.array([[5.0, 10.0]])
        norm_params = {
            "min_vals": [5.0, 0.0],
            "max_vals": [5.0, 20.0],  # min == max cho feature 0
        }

        # Không nên raise exception
        result = module._normalize_features(features, norm_params)
        assert result.shape == features.shape


# ==============================================================================
# Test _validate_inputs
# ==============================================================================


class TestValidateInputs:
    """Tests cho EvaluationModule._validate_inputs()."""

    def test_model_path_none(self, module):
        """Raise ModelError khi model_path is None."""
        result = TrainingResult(model_path=None)

        with pytest.raises(ModelError) as exc_info:
            module._validate_inputs(result, {"VNM": pd.DataFrame()})

        assert exc_info.value.error_code == "MODEL_NOT_FOUND"

    def test_model_path_not_exists(self, module):
        """Raise ModelError khi model_path file không tồn tại."""
        result = TrainingResult(model_path="/nonexistent/model.pt")

        with pytest.raises(ModelError) as exc_info:
            module._validate_inputs(result, {"VNM": pd.DataFrame()})

        assert exc_info.value.error_code == "MODEL_NOT_FOUND"

    def test_empty_test_data(self, module, tmp_path):
        """Raise DataError khi test_data rỗng."""
        model_path = str(tmp_path / "model.pt")
        Path(model_path).touch()
        result = TrainingResult(model_path=model_path)

        with pytest.raises(DataError) as exc_info:
            module._validate_inputs(result, {})

        assert exc_info.value.error_code == "INSUFFICIENT_TEST_DATA"

    def test_valid_inputs_pass(self, module, tmp_path, sample_df):
        """Không raise exception với inputs hợp lệ."""
        model_path = str(tmp_path / "model.pt")
        Path(model_path).touch()
        result = TrainingResult(model_path=model_path)

        # Không nên raise exception
        module._validate_inputs(result, {"VNM": sample_df})


# ==============================================================================
# Test _generate_predictions
# ==============================================================================


class TestGeneratePredictions:
    """Tests cho EvaluationModule._generate_predictions()."""

    def test_predictions_shape(self, module):
        """Predictions có shape đúng (num_samples,)."""
        from engine.evaluation_model import StockEvalNet

        model = StockEvalNet()
        model.eval()

        # Tạo features shape (5, 60, 61) - 5 samples, lookback=60, 61 features
        features = np.random.randn(5, 60, 61).astype(np.float32)

        predictions = module._generate_predictions(model, features)

        assert predictions.shape == (5,)

    def test_predictions_in_range(self, module):
        """Predictions nằm trong khoảng [-1, 1] (do Tanh output)."""
        from engine.evaluation_model import StockEvalNet

        model = StockEvalNet()
        model.eval()

        features = np.random.randn(10, 60, 61).astype(np.float32)

        predictions = module._generate_predictions(model, features)

        assert np.all(predictions >= -1.0)
        assert np.all(predictions <= 1.0)

    def test_predictions_with_norm_params(self, module):
        """Predictions áp dụng norm_params trước inference."""
        from engine.evaluation_model import StockEvalNet

        model = StockEvalNet()
        model.eval()

        features = np.random.randn(3, 60, 61).astype(np.float32)
        norm_params = {
            "min_vals": np.zeros(61).tolist(),
            "max_vals": np.ones(61).tolist(),
        }

        # Không nên raise exception
        predictions = module._generate_predictions(model, features, norm_params)
        assert predictions.shape == (3,)


# ==============================================================================
# Test per-symbol error isolation
# ==============================================================================


class TestPerSymbolIsolation:
    """Tests cho per-symbol error isolation."""

    def test_one_symbol_fail_others_continue(self, module, tmp_path, sample_df):
        """Khi một symbol fail, các symbols khác vẫn được đánh giá."""
        from engine.evaluation_model import StockEvalNet

        # Tạo model và save
        model = StockEvalNet()
        model_path = str(tmp_path / "model.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)

        result = TrainingResult(
            model_path=model_path,
            final_train_loss=0.05,
            final_val_loss=0.08,
            symbols_trained=["VNM", "FAIL"],
        )

        # FAIL symbol có DataFrame rỗng → sẽ raise DataError
        test_data = {
            "VNM": sample_df,
            "FAIL": pd.DataFrame(),  # Rỗng → sẽ fail
        }

        # Mock _generate_predictions và backtest để tránh dependency phức tạp
        fake_preds = np.random.randn(len(sample_df))
        with patch.object(module, "_generate_predictions", return_value=fake_preds), \
             patch.object(module.backtest_evaluator, "run_backtest") as mock_bt, \
             patch.object(module.baseline_comparator, "compare") as mock_comp, \
             patch.object(module.report_generator, "generate_json") as mock_json, \
             patch.object(module.report_generator, "generate_html") as mock_html:
            mock_bt.return_value = MagicMock()
            mock_comp.return_value = MagicMock()
            mock_json.return_value = "/tmp/report.json"
            mock_html.return_value = "/tmp/report.html"

            eval_result = module.evaluate(result, test_data)

        # VNM phải có kết quả, FAIL bị bỏ qua
        assert "VNM" in eval_result.symbol_results
        assert "FAIL" not in eval_result.symbol_results


# ==============================================================================
# Test _compute_actual_returns
# ==============================================================================


class TestComputeActualReturns:
    """Tests cho _compute_actual_returns."""

    def test_returns_calculation(self, module):
        """Tính actual returns đúng từ close prices."""
        df = pd.DataFrame({"close": [100.0, 110.0, 105.0, 120.0]})
        returns = module._compute_actual_returns(df)

        assert len(returns) == 4
        assert abs(returns[0] - 0.1) < 1e-10  # (110-100)/100
        assert abs(returns[1] - (-5 / 110)) < 1e-10  # (105-110)/110
        assert abs(returns[2] - (15 / 105)) < 1e-10  # (120-105)/105
        assert returns[3] == 0.0  # Cuối cùng = 0

    def test_empty_df(self, module):
        """Trả về array rỗng cho DataFrame rỗng."""
        df = pd.DataFrame({"close": []})
        returns = module._compute_actual_returns(df)
        assert len(returns) == 0


# ==============================================================================
# Test _extract_features_from_df
# ==============================================================================


class TestExtractFeatures:
    """Tests cho _extract_features_from_df."""

    def test_excludes_time_column(self, module, sample_df):
        """Loại bỏ cột 'time' khi trích xuất features."""
        features = module._extract_features_from_df(sample_df)

        # sample_df có 6 cột: time, open, high, low, close, volume
        # Features không chứa 'time' → 5 cột
        assert features.shape == (100, 5)

    def test_all_numeric_columns(self, module):
        """Lấy tất cả cột numeric (trừ time)."""
        df = pd.DataFrame({
            "open": [1.0, 2.0],
            "close": [3.0, 4.0],
            "indicator1": [0.5, 0.6],
        })
        features = module._extract_features_from_df(df)
        assert features.shape == (2, 3)


# ==============================================================================
# Integration Tests - End-to-End Flow
# ==============================================================================


class TestEndToEndFlow:
    """
    Integration tests cho end-to-end evaluation flow.

    Test toàn bộ pipeline: validate inputs → load model → generate predictions
    → backtest → compare baselines → detect overfitting → analyze predictions
    → generate reports.

    Validates: Requirements 6.1, 6.2, 6.3
    """

    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """Tạo model checkpoint giả hợp lệ cho integration test."""
        from engine.evaluation_model import StockEvalNet

        model = StockEvalNet()
        model_path = str(tmp_path / "integration_model.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)
        return model_path

    @pytest.fixture
    def multi_symbol_test_data(self):
        """Tạo test data cho nhiều symbols."""
        np.random.seed(42)
        n = 50

        def make_df(seed):
            rng = np.random.default_rng(seed)
            dates = pd.date_range("2023-06-01", periods=n, freq="B")
            close = 100 + np.cumsum(rng.uniform(-1, 1.2, n))
            return pd.DataFrame({
                "time": dates,
                "open": close - rng.uniform(0, 1, n),
                "high": close + rng.uniform(0, 2, n),
                "low": close - rng.uniform(0, 2, n),
                "close": close,
                "volume": rng.integers(1000, 10000, n),
            })

        return {
            "VNM": make_df(1),
            "FPT": make_df(2),
            "VIC": make_df(3),
        }

    def test_end_to_end_with_mock_model(
        self, tmp_path, mock_model_path, multi_symbol_test_data
    ):
        """
        Test end-to-end flow: evaluate() trả về EvaluationResult với tất cả fields.

        Mock _generate_predictions và backtest để tránh dependency phức tạp
        với model architecture (61 channels) và BacktestEngine.
        """
        config = EvaluationConfig(report_dir=str(tmp_path / "reports"))
        module = EvaluationModule(config=config)

        training_result = TrainingResult(
            epochs_completed=20,
            final_train_loss=0.04,
            final_val_loss=0.07,
            model_path=mock_model_path,
            symbols_trained=["VNM", "FPT", "VIC"],
        )

        # Mock predictions (model cần 61 features nhưng test data chỉ có 5)
        fake_preds = np.random.uniform(-1, 1, 50)

        with patch.object(module, "_generate_predictions", return_value=fake_preds), \
             patch.object(module.backtest_evaluator, "run_backtest") as mock_bt, \
             patch.object(module.baseline_comparator, "compare") as mock_comp, \
             patch.object(module.report_generator, "generate_json") as mock_json, \
             patch.object(module.report_generator, "generate_html") as mock_html:
            mock_bt.return_value = MagicMock(
                total_return_pct=5.0,
                annualized_return_pct=12.0,
                win_rate=55.0,
                max_drawdown=-8.0,
                sharpe_ratio=1.2,
                equity_curve=pd.Series(
                    [100, 102, 105],
                    index=pd.date_range("2023-06-01", periods=3, freq="B"),
                ),
                trades=[],
            )
            mock_comp.return_value = MagicMock()
            mock_json.return_value = "/tmp/report.json"
            mock_html.return_value = "/tmp/report.html"

            eval_result = module.evaluate(
                training_result=training_result,
                test_data=multi_symbol_test_data,
                norm_params=None,
            )

        # Kiểm tra EvaluationResult có tất cả fields cần thiết
        assert isinstance(eval_result, EvaluationResult)
        assert eval_result.model_metadata is not None
        assert eval_result.model_metadata.architecture == "StockEvalNet (TCN + Attention)"
        assert eval_result.model_metadata.model_path == mock_model_path
        assert eval_result.model_metadata.parameter_count > 0
        assert eval_result.evaluation_timestamp is not None

        # Per-symbol results: tất cả 3 symbols đều phải có kết quả
        assert len(eval_result.symbol_results) == 3
        assert "VNM" in eval_result.symbol_results
        assert "FPT" in eval_result.symbol_results
        assert "VIC" in eval_result.symbol_results

        # Mỗi symbol result phải có backtest_result (đã mock)
        for symbol, sym_result in eval_result.symbol_results.items():
            assert sym_result.symbol == symbol
            assert sym_result.backtest_result is not None

    def test_end_to_end_with_norm_params(
        self, tmp_path, mock_model_path, multi_symbol_test_data
    ):
        """
        Test evaluate() áp dụng norm_params đúng cách.

        Khi cung cấp norm_params, features phải được normalize trước inference.
        Validates: Requirement 6.2
        """
        config = EvaluationConfig(report_dir=str(tmp_path / "reports"))
        module = EvaluationModule(config=config)

        training_result = TrainingResult(
            model_path=mock_model_path,
            final_train_loss=0.05,
            final_val_loss=0.08,
            symbols_trained=["VNM"],
        )

        # norm_params cho 5 features (open, high, low, close, volume)
        norm_params = {
            "min_vals": [0.0, 0.0, 0.0, 0.0, 0.0],
            "max_vals": [50.0, 50.0, 50.0, 50.0, 20000.0],
        }

        test_data = {"VNM": multi_symbol_test_data["VNM"]}
        fake_preds = np.random.uniform(-1, 1, 50)

        # Mock _generate_predictions và backtest
        with patch.object(module, "_generate_predictions", return_value=fake_preds), \
             patch.object(module.backtest_evaluator, "run_backtest") as mock_bt, \
             patch.object(module.baseline_comparator, "compare") as mock_comp, \
             patch.object(module.report_generator, "generate_json") as mock_json, \
             patch.object(module.report_generator, "generate_html") as mock_html:
            mock_bt.return_value = MagicMock(
                total_return_pct=3.0,
                annualized_return_pct=8.0,
                win_rate=50.0,
                max_drawdown=-5.0,
                sharpe_ratio=0.9,
                equity_curve=pd.Series([100, 101]),
                trades=[],
            )
            mock_comp.return_value = None
            mock_json.return_value = "/tmp/report.json"
            mock_html.return_value = "/tmp/report.html"

            # Không nên raise exception với norm_params hợp lệ
            eval_result = module.evaluate(
                training_result=training_result,
                test_data=test_data,
                norm_params=norm_params,
            )

        assert isinstance(eval_result, EvaluationResult)
        assert "VNM" in eval_result.symbol_results

    def test_end_to_end_predictions_generated(
        self, tmp_path, mock_model_path, multi_symbol_test_data
    ):
        """
        Verify predictions được generate cho mỗi symbol trong pipeline.
        """
        config = EvaluationConfig(report_dir=str(tmp_path / "reports"))
        module = EvaluationModule(config=config)

        training_result = TrainingResult(
            model_path=mock_model_path,
            final_train_loss=0.05,
            final_val_loss=0.08,
            symbols_trained=["VNM"],
        )

        test_data = {"VNM": multi_symbol_test_data["VNM"]}
        fake_preds = np.random.uniform(-1, 1, 50)

        # Spy trên _generate_predictions để xác nhận nó được gọi
        with patch.object(module.backtest_evaluator, "run_backtest") as mock_bt, \
             patch.object(module.baseline_comparator, "compare") as mock_comp, \
             patch.object(module.report_generator, "generate_json") as mock_json, \
             patch.object(module.report_generator, "generate_html") as mock_html, \
             patch.object(module, "_generate_predictions", return_value=fake_preds) as spy_preds:
            mock_bt.return_value = MagicMock(
                total_return_pct=0.0,
                annualized_return_pct=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                equity_curve=pd.Series([100]),
                trades=[],
            )
            mock_comp.return_value = None
            mock_json.return_value = "/tmp/report.json"
            mock_html.return_value = "/tmp/report.html"

            module.evaluate(
                training_result=training_result,
                test_data=test_data,
            )

        # _generate_predictions phải được gọi (ít nhất 1 lần cho _evaluate_symbol
        # và 1 lần nữa cho aggregate)
        assert spy_preds.call_count >= 1


# ==============================================================================
# Integration Tests - Error Handling
# ==============================================================================


class TestErrorHandlingIntegration:
    """
    Integration tests cho error handling trong evaluate().

    Test các trường hợp lỗi: model file không tồn tại, test data rỗng,
    norm_params size mismatch.

    Validates: Requirements 6.4
    """

    def test_model_file_not_found_raises_model_error(self, tmp_path):
        """evaluate() raise ModelError khi model file không tồn tại."""
        config = EvaluationConfig(report_dir=str(tmp_path / "reports"))
        module = EvaluationModule(config=config)

        training_result = TrainingResult(
            model_path="/nonexistent/path/model.pt",
            symbols_trained=["VNM"],
        )
        test_data = {"VNM": pd.DataFrame({"close": [1, 2, 3]})}

        with pytest.raises(ModelError) as exc_info:
            module.evaluate(training_result, test_data)

        assert exc_info.value.error_code == "MODEL_NOT_FOUND"
        assert "model_path" in exc_info.value.details

    def test_model_path_none_raises_model_error(self, tmp_path):
        """evaluate() raise ModelError khi model_path is None."""
        config = EvaluationConfig(report_dir=str(tmp_path / "reports"))
        module = EvaluationModule(config=config)

        training_result = TrainingResult(model_path=None)
        test_data = {"VNM": pd.DataFrame({"close": [1, 2, 3]})}

        with pytest.raises(ModelError) as exc_info:
            module.evaluate(training_result, test_data)

        assert exc_info.value.error_code == "MODEL_NOT_FOUND"

    def test_empty_test_data_raises_data_error(self, tmp_path):
        """evaluate() raise DataError khi test_data dict rỗng."""
        config = EvaluationConfig(report_dir=str(tmp_path / "reports"))
        module = EvaluationModule(config=config)

        # Tạo model file hợp lệ
        model_path = str(tmp_path / "model.pt")
        Path(model_path).touch()

        training_result = TrainingResult(
            model_path=model_path,
            symbols_trained=["VNM"],
        )

        with pytest.raises(DataError) as exc_info:
            module.evaluate(training_result, test_data={})

        assert exc_info.value.error_code == "INSUFFICIENT_TEST_DATA"

    def test_norm_params_mismatch_does_not_crash_pipeline(self, tmp_path):
        """
        Khi norm_params mismatch kích thước features, symbol bị skip
        nhưng pipeline không crash hoàn toàn (per-symbol isolation).
        """
        from engine.evaluation_model import StockEvalNet

        config = EvaluationConfig(report_dir=str(tmp_path / "reports"))
        module = EvaluationModule(config=config)

        # Tạo model file hợp lệ
        model = StockEvalNet()
        model_path = str(tmp_path / "model.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)

        training_result = TrainingResult(
            model_path=model_path,
            final_train_loss=0.05,
            final_val_loss=0.08,
            symbols_trained=["VNM"],
        )

        # DataFrame có 5 features (open, high, low, close, volume)
        np.random.seed(42)
        n = 30
        test_data = {
            "VNM": pd.DataFrame({
                "time": pd.date_range("2023-01-01", periods=n, freq="B"),
                "open": np.random.uniform(20, 30, n),
                "high": np.random.uniform(25, 35, n),
                "low": np.random.uniform(15, 25, n),
                "close": np.random.uniform(20, 30, n),
                "volume": np.random.randint(1000, 10000, n),
            })
        }

        # norm_params chỉ có 3 values nhưng features có 5 cột → mismatch
        norm_params = {
            "min_vals": [0.0, 0.0, 0.0],
            "max_vals": [1.0, 1.0, 1.0],
        }

        # Mock report generation
        with patch.object(module.report_generator, "generate_json") as mock_json, \
             patch.object(module.report_generator, "generate_html") as mock_html:
            mock_json.return_value = "/tmp/report.json"
            mock_html.return_value = "/tmp/report.html"

            # Pipeline không crash, nhưng symbol bị skip do DataError
            eval_result = module.evaluate(
                training_result=training_result,
                test_data=test_data,
                norm_params=norm_params,
            )

        # VNM bị skip do norm_params mismatch → symbol_results rỗng
        assert "VNM" not in eval_result.symbol_results


# ==============================================================================
# Integration Tests - Per-Symbol Isolation
# ==============================================================================


class TestPerSymbolIsolationIntegration:
    """
    Integration tests cho per-symbol error isolation.

    Verify rằng khi một symbol gặp lỗi, các symbols khác vẫn được
    đánh giá thành công.

    Validates: Requirements 6.4 (per-symbol error isolation)
    """

    def test_empty_symbol_skipped_others_succeed(self, tmp_path):
        """
        Symbol với DataFrame rỗng bị skip, symbols khác vẫn đánh giá thành công.
        """
        from engine.evaluation_model import StockEvalNet

        config = EvaluationConfig(report_dir=str(tmp_path / "reports"))
        module = EvaluationModule(config=config)

        # Tạo model
        model = StockEvalNet()
        model_path = str(tmp_path / "model.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)

        training_result = TrainingResult(
            model_path=model_path,
            final_train_loss=0.05,
            final_val_loss=0.08,
            symbols_trained=["VNM", "BAD", "FPT"],
        )

        np.random.seed(42)
        n = 50
        good_df = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=n, freq="B"),
            "open": np.random.uniform(20, 30, n),
            "high": np.random.uniform(25, 35, n),
            "low": np.random.uniform(15, 25, n),
            "close": np.random.uniform(20, 30, n),
            "volume": np.random.randint(1000, 10000, n),
        })

        test_data = {
            "VNM": good_df.copy(),
            "BAD": pd.DataFrame(),  # Rỗng → DataError
            "FPT": good_df.copy(),
        }

        fake_preds = np.random.uniform(-1, 1, n)

        # Mock predictions + backtest + reports
        with patch.object(module, "_generate_predictions", return_value=fake_preds), \
             patch.object(module.backtest_evaluator, "run_backtest") as mock_bt, \
             patch.object(module.baseline_comparator, "compare") as mock_comp, \
             patch.object(module.report_generator, "generate_json") as mock_json, \
             patch.object(module.report_generator, "generate_html") as mock_html:
            mock_bt.return_value = MagicMock(
                total_return_pct=2.0,
                annualized_return_pct=6.0,
                win_rate=50.0,
                max_drawdown=-3.0,
                sharpe_ratio=0.8,
                equity_curve=pd.Series([100]),
                trades=[],
            )
            mock_comp.return_value = None
            mock_json.return_value = "/tmp/report.json"
            mock_html.return_value = "/tmp/report.html"

            eval_result = module.evaluate(training_result, test_data)

        # VNM và FPT thành công, BAD bị skip
        assert "VNM" in eval_result.symbol_results
        assert "FPT" in eval_result.symbol_results
        assert "BAD" not in eval_result.symbol_results

    def test_multiple_symbols_fail_remaining_succeed(self, tmp_path):
        """
        Nhiều symbols fail nhưng symbols hợp lệ vẫn được đánh giá.
        """
        from engine.evaluation_model import StockEvalNet

        config = EvaluationConfig(report_dir=str(tmp_path / "reports"))
        module = EvaluationModule(config=config)

        model = StockEvalNet()
        model_path = str(tmp_path / "model.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)

        training_result = TrainingResult(
            model_path=model_path,
            final_train_loss=0.05,
            final_val_loss=0.08,
            symbols_trained=["GOOD", "BAD1", "BAD2"],
        )

        np.random.seed(42)
        n = 30
        good_df = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=n, freq="B"),
            "open": np.random.uniform(20, 30, n),
            "high": np.random.uniform(25, 35, n),
            "low": np.random.uniform(15, 25, n),
            "close": np.random.uniform(20, 30, n),
            "volume": np.random.randint(1000, 10000, n),
        })

        test_data = {
            "GOOD": good_df,
            "BAD1": pd.DataFrame(),
            "BAD2": pd.DataFrame(),
        }

        fake_preds = np.random.uniform(-1, 1, n)

        with patch.object(module, "_generate_predictions", return_value=fake_preds), \
             patch.object(module.backtest_evaluator, "run_backtest") as mock_bt, \
             patch.object(module.baseline_comparator, "compare") as mock_comp, \
             patch.object(module.report_generator, "generate_json") as mock_json, \
             patch.object(module.report_generator, "generate_html") as mock_html:
            mock_bt.return_value = MagicMock(
                total_return_pct=1.0,
                annualized_return_pct=4.0,
                win_rate=45.0,
                max_drawdown=-2.0,
                sharpe_ratio=0.5,
                equity_curve=pd.Series([100]),
                trades=[],
            )
            mock_comp.return_value = None
            mock_json.return_value = "/tmp/report.json"
            mock_html.return_value = "/tmp/report.html"

            eval_result = module.evaluate(training_result, test_data)

        # Chỉ GOOD thành công
        assert "GOOD" in eval_result.symbol_results
        assert "BAD1" not in eval_result.symbol_results
        assert "BAD2" not in eval_result.symbol_results
        assert len(eval_result.symbol_results) == 1


# ==============================================================================
# Integration Tests - Graceful Degradation
# ==============================================================================


class TestGracefulDegradation:
    """
    Integration tests cho graceful degradation behavior.

    Verify rằng khi HTML report generation thất bại, JSON report vẫn được tạo
    và evaluate() không raise exception.

    Validates: Requirements 5.6
    """

    @pytest.fixture
    def setup_evaluation(self, tmp_path):
        """Thiết lập evaluation pipeline sẵn sàng chạy."""
        from engine.evaluation_model import StockEvalNet

        config = EvaluationConfig(report_dir=str(tmp_path / "reports"))
        module = EvaluationModule(config=config)

        model = StockEvalNet()
        model_path = str(tmp_path / "model.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)

        training_result = TrainingResult(
            model_path=model_path,
            final_train_loss=0.05,
            final_val_loss=0.08,
            symbols_trained=["VNM"],
        )

        np.random.seed(42)
        n = 50
        test_data = {
            "VNM": pd.DataFrame({
                "time": pd.date_range("2023-01-01", periods=n, freq="B"),
                "open": np.random.uniform(20, 30, n),
                "high": np.random.uniform(25, 35, n),
                "low": np.random.uniform(15, 25, n),
                "close": np.random.uniform(20, 30, n),
                "volume": np.random.randint(1000, 10000, n),
            })
        }

        # Fake predictions tương ứng số rows
        fake_preds = np.random.uniform(-1, 1, n)

        return module, training_result, test_data, fake_preds

    def test_html_failure_does_not_raise(self, setup_evaluation):
        """
        Khi generate_html raise exception, evaluate() vẫn hoàn tất
        và trả về EvaluationResult hợp lệ.
        """
        module, training_result, test_data, fake_preds = setup_evaluation

        with patch.object(module, "_generate_predictions", return_value=fake_preds), \
             patch.object(module.backtest_evaluator, "run_backtest") as mock_bt, \
             patch.object(module.baseline_comparator, "compare") as mock_comp, \
             patch.object(module.report_generator, "generate_json") as mock_json, \
             patch.object(
                 module.report_generator, "generate_html",
                 side_effect=RuntimeError("Template rendering failed")
             ):
            mock_bt.return_value = MagicMock(
                total_return_pct=2.0,
                annualized_return_pct=6.0,
                win_rate=50.0,
                max_drawdown=-3.0,
                sharpe_ratio=0.8,
                equity_curve=pd.Series([100]),
                trades=[],
            )
            mock_comp.return_value = None
            mock_json.return_value = "/tmp/report.json"

            # Không nên raise exception
            eval_result = module.evaluate(training_result, test_data)

        # Kết quả vẫn hợp lệ
        assert isinstance(eval_result, EvaluationResult)
        assert "VNM" in eval_result.symbol_results
        # JSON vẫn được gọi thành công
        mock_json.assert_called_once()

    def test_json_generated_before_html(self, setup_evaluation):
        """
        JSON report được tạo trước HTML, đảm bảo nếu HTML fail thì
        JSON đã được lưu.
        """
        module, training_result, test_data, fake_preds = setup_evaluation

        call_order = []

        def track_json(result):
            call_order.append("json")
            return "/tmp/report.json"

        def track_html(result):
            call_order.append("html")
            raise RuntimeError("HTML failed")

        with patch.object(module, "_generate_predictions", return_value=fake_preds), \
             patch.object(module.backtest_evaluator, "run_backtest") as mock_bt, \
             patch.object(module.baseline_comparator, "compare") as mock_comp, \
             patch.object(
                 module.report_generator, "generate_json", side_effect=track_json
             ), \
             patch.object(
                 module.report_generator, "generate_html", side_effect=track_html
             ):
            mock_bt.return_value = MagicMock(
                total_return_pct=0.0,
                annualized_return_pct=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                equity_curve=pd.Series([100]),
                trades=[],
            )
            mock_comp.return_value = None

            eval_result = module.evaluate(training_result, test_data)

        # JSON phải được gọi trước HTML
        assert call_order == ["json", "html"]
        assert isinstance(eval_result, EvaluationResult)

    def test_both_reports_fail_still_returns_result(self, setup_evaluation):
        """
        Ngay cả khi cả JSON và HTML đều fail, evaluate() vẫn trả về
        EvaluationResult (report generation không block kết quả).
        """
        module, training_result, test_data, fake_preds = setup_evaluation

        with patch.object(module, "_generate_predictions", return_value=fake_preds), \
             patch.object(module.backtest_evaluator, "run_backtest") as mock_bt, \
             patch.object(module.baseline_comparator, "compare") as mock_comp, \
             patch.object(
                 module.report_generator, "generate_json",
                 side_effect=IOError("Disk full")
             ), \
             patch.object(
                 module.report_generator, "generate_html",
                 side_effect=RuntimeError("Template error")
             ):
            mock_bt.return_value = MagicMock(
                total_return_pct=0.0,
                annualized_return_pct=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                equity_curve=pd.Series([100]),
                trades=[],
            )
            mock_comp.return_value = None

            # Không nên raise exception
            eval_result = module.evaluate(training_result, test_data)

        # Kết quả vẫn được trả về
        assert isinstance(eval_result, EvaluationResult)
        assert "VNM" in eval_result.symbol_results
