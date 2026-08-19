"""
Module chính điều phối toàn bộ quy trình đánh giá mô hình sau huấn luyện.

EvaluationModule orchestrate flow:
1. Validate inputs (model_path tồn tại, test_data không rỗng)
2. Load model từ checkpoint
3. Cho mỗi symbol: generate predictions → backtest → analyze predictions
4. Compare baselines
5. Detect overfitting
6. Generate reports (JSON + HTML graceful degradation)
7. Trả về EvaluationResult
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from engine.backtest_evaluator import BacktestEvaluator, ModelStrategy
from engine.baseline_comparator import BaselineComparator
from engine.config import DataError, ModelError
from engine.evaluation_config import (
    EvaluationConfig,
    EvaluationResult,
    ModelMetadata,
    SymbolEvaluationResult,
)
from engine.evaluation_model import StockEvalNet
from engine.overfitting_detector import OverfittingDetector
from engine.prediction_analyzer import PredictionAnalyzer
from engine.report_generator import ReportGenerator
from engine.training_pipeline import TrainingResult

logger = logging.getLogger(__name__)


class EvaluationModule:
    """
    Module chính điều phối toàn bộ quy trình đánh giá sau huấn luyện.

    Khởi tạo tất cả sub-components và orchestrate evaluation flow:
    validate inputs → load model → generate predictions → backtest →
    compare baselines → detect overfitting → analyze predictions → generate reports.

    Parameters
    ----------
    config : EvaluationConfig, optional
        Cấu hình đánh giá. Sử dụng giá trị mặc định nếu không cung cấp.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        """Khởi tạo EvaluationModule với tất cả sub-components."""
        self.config = config or EvaluationConfig()
        self.backtest_evaluator = BacktestEvaluator(self.config)
        self.baseline_comparator = BaselineComparator(self.config)
        self.overfitting_detector = OverfittingDetector(self.config)
        self.prediction_analyzer = PredictionAnalyzer(self.config)
        self.report_generator = ReportGenerator(self.config)

    def _load_model(self, model_path: str) -> StockEvalNet:
        """
        Load StockEvalNet từ model_path.

        Kiểm tra file tồn tại trước khi load. Sử dụng torch.load() với
        weights_only=False để load toàn bộ model state.

        Parameters
        ----------
        model_path : str
            Đường dẫn đến file checkpoint (.pt) của model.

        Returns
        -------
        StockEvalNet
            Model đã load và chuyển sang eval mode.

        Raises
        ------
        ModelError
            Nếu file không tồn tại (MODEL_NOT_FOUND) hoặc không load được (MODEL_LOAD_ERROR).
        """
        # Kiểm tra file tồn tại
        if not Path(model_path).exists():
            raise ModelError(
                f"Model file not found: {model_path}",
                error_code="MODEL_NOT_FOUND",
                details={"model_path": model_path},
            )

        # Load model từ checkpoint
        try:
            checkpoint = torch.load(
                model_path, map_location="cpu", weights_only=False
            )

            # Tạo model instance và load state_dict
            model = StockEvalNet()

            # Checkpoint có thể chứa state_dict trực tiếp hoặc trong key "model_state_dict"
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                # Thử load trực tiếp nếu checkpoint là state_dict
                model.load_state_dict(checkpoint)

            model.eval()
            logger.info(f"Model đã load thành công từ: {model_path}")
            return model

        except ModelError:
            raise
        except Exception as e:
            raise ModelError(
                f"Failed to load model from {model_path}: {e}",
                error_code="MODEL_LOAD_ERROR",
                details={"model_path": model_path, "error": str(e)},
            )

    def _normalize_features(
        self, features: np.ndarray, norm_params: Dict
    ) -> np.ndarray:
        """
        Áp dụng normalization parameters từ training session.

        Sử dụng min-max normalization: (features - min) / (max - min)
        hoặc mean/std normalization tùy vào cấu trúc norm_params.

        Parameters
        ----------
        features : np.ndarray
            Raw feature array, shape (num_samples, num_features) hoặc
            (num_samples, lookback, num_features).
        norm_params : Dict
            Normalization parameters từ training session.
            Có thể chứa {"min_vals": [...], "max_vals": [...]}
            hoặc {"mean": [...], "std": [...]}.

        Returns
        -------
        np.ndarray
            Normalized features cùng shape với input.

        Raises
        ------
        DataError
            Nếu norm_params không khớp feature dimensions.
        """
        if "min_vals" in norm_params and "max_vals" in norm_params:
            # Min-max normalization
            min_vals = np.array(norm_params["min_vals"], dtype=np.float64)
            max_vals = np.array(norm_params["max_vals"], dtype=np.float64)

            # Kiểm tra kích thước khớp
            num_features = features.shape[-1]
            if len(min_vals) != num_features:
                raise DataError(
                    f"norm_params size mismatch: expected {num_features} features, "
                    f"got {len(min_vals)}",
                    error_code="NORM_PARAMS_MISMATCH",
                    details={"expected": num_features, "got": len(min_vals)},
                )

            # Tránh chia cho 0
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1.0

            normalized = (features - min_vals) / range_vals
            return normalized

        elif "mean" in norm_params and "std" in norm_params:
            # Z-score normalization
            mean = np.array(norm_params["mean"], dtype=np.float64)
            std = np.array(norm_params["std"], dtype=np.float64)

            # Kiểm tra kích thước khớp
            num_features = features.shape[-1]
            if len(mean) != num_features:
                raise DataError(
                    f"norm_params size mismatch: expected {num_features} features, "
                    f"got {len(mean)}",
                    error_code="NORM_PARAMS_MISMATCH",
                    details={"expected": num_features, "got": len(mean)},
                )

            # Tránh chia cho 0
            std_safe = std.copy()
            std_safe[std_safe == 0] = 1.0

            normalized = (features - mean) / std_safe
            return normalized

        else:
            # Không có norm_params hợp lệ → trả về features gốc
            logger.warning("norm_params không có key 'min_vals'/'max_vals' hoặc 'mean'/'std', "
                           "bỏ qua normalization")
            return features

    def _generate_predictions(
        self,
        model: StockEvalNet,
        features: np.ndarray,
        norm_params: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Chạy inference với model + normalized features.

        Normalize features nếu norm_params được cung cấp, chuyển sang tensor,
        chạy forward pass (no_grad), trả về ndarray Position_Score.

        Parameters
        ----------
        model : StockEvalNet
            Model đã load và ở eval mode.
        features : np.ndarray
            Feature array, shape (num_samples, lookback, num_features)
            hoặc (num_samples, num_features).
        norm_params : Dict, optional
            Normalization parameters. Nếu None, sử dụng features gốc.

        Returns
        -------
        np.ndarray
            Array Position_Score shape (num_samples,) với giá trị trong [-1, 1].
        """
        # Áp dụng normalization nếu có norm_params
        if norm_params is not None:
            features = self._normalize_features(features, norm_params)

        # Chuyển sang tensor
        tensor_input = torch.tensor(features, dtype=torch.float32)

        # Đảm bảo shape đúng cho model (batch, lookback, num_features)
        if tensor_input.dim() == 2:
            # Nếu là (num_samples, num_features), thêm lookback dimension
            tensor_input = tensor_input.unsqueeze(1)

        # Chạy inference với no_grad
        with torch.no_grad():
            output = model(tensor_input)

        # Chuyển output về numpy, flatten thành 1D
        predictions = output.cpu().numpy().flatten()

        return predictions

    def _extract_features_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Trích xuất feature matrix từ DataFrame OHLCV.

        Lấy tất cả cột numeric (trừ 'time') làm features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu OHLCV và indicators.

        Returns
        -------
        np.ndarray
            Feature array shape (num_rows, num_features).
        """
        # Loại bỏ cột time nếu có
        feature_cols = [col for col in df.columns if col != "time"]
        features = df[feature_cols].values.astype(np.float64)
        return features

    def _compute_actual_returns(self, df: pd.DataFrame) -> np.ndarray:
        """
        Tính actual returns từ close prices.

        Sử dụng percentage change: (close[t+1] - close[t]) / close[t].

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame chứa cột 'close'.

        Returns
        -------
        np.ndarray
            Array actual returns, cùng length với df (padding 0 ở cuối).
        """
        close = df["close"].values.astype(np.float64)
        returns = np.zeros(len(close))

        if len(close) > 1:
            # Forward returns: return tại t = (close[t+1] - close[t]) / close[t]
            returns[:-1] = (close[1:] - close[:-1]) / np.where(
                close[:-1] != 0, close[:-1], 1.0
            )

        return returns

    def evaluate(
        self,
        training_result: TrainingResult,
        test_data: Dict[str, pd.DataFrame],
        train_predictions: Optional[np.ndarray] = None,
        norm_params: Optional[Dict] = None,
    ) -> EvaluationResult:
        """
        Thực hiện đánh giá toàn diện mô hình sau huấn luyện.

        Orchestrate toàn bộ flow: validate inputs → load model → generate predictions
        → backtest → compare baselines → detect overfitting → analyze predictions
        → generate reports.

        Parameters
        ----------
        training_result : TrainingResult
            Kết quả từ train_full() hoặc train_incremental().
        test_data : Dict[str, pd.DataFrame]
            Dict mapping symbol → DataFrame cho tập test (15% cuối).
        train_predictions : np.ndarray, optional
            Predictions trên tập train (cho overfitting detection).
        norm_params : Dict, optional
            Normalization parameters từ training session.
            Nếu None, sẽ không áp dụng normalization.

        Returns
        -------
        EvaluationResult
            Kết quả tổng hợp toàn bộ đánh giá.

        Raises
        ------
        ModelError
            Nếu model file không tồn tại hoặc không load được.
        DataError
            Nếu test_data rỗng hoặc không hợp lệ.
        """
        # === 1. Validate inputs ===
        self._validate_inputs(training_result, test_data)

        # === 2. Load model ===
        model = self._load_model(training_result.model_path)
        logger.info(f"Model loaded, bắt đầu evaluation cho {len(test_data)} symbols")

        # === 3. Per-symbol evaluation ===
        symbol_results: Dict[str, SymbolEvaluationResult] = {}
        all_predictions: list = []
        all_actual_returns: list = []
        first_symbol_strategy = None
        first_test_df = None
        first_start_date = None
        first_end_date = None

        for symbol, df in test_data.items():
            try:
                sym_result = self._evaluate_symbol(
                    symbol=symbol,
                    df=df,
                    model=model,
                    norm_params=norm_params,
                )
                symbol_results[symbol] = sym_result

                # Thu thập predictions cho aggregation
                features = self._extract_features_from_df(df)
                preds = self._generate_predictions(model, features, norm_params)
                actual_rets = self._compute_actual_returns(df)

                # Cắt để cùng length
                min_len = min(len(preds), len(actual_rets))
                all_predictions.append(preds[:min_len])
                all_actual_returns.append(actual_rets[:min_len])

                # Lưu thông tin cho baseline comparison (dùng symbol đầu tiên)
                if first_symbol_strategy is None:
                    first_test_df = df
                    dates = self._extract_dates(df)
                    first_start_date = str(dates.min().date())
                    first_end_date = str(dates.max().date())

                    # Tạo ModelStrategy cho baseline comparison
                    index_mapping = {i: i for i in range(len(preds))}
                    first_symbol_strategy = ModelStrategy(
                        predictions=preds,
                        index_mapping=index_mapping,
                        buy_threshold=self.config.buy_threshold,
                        sell_threshold=self.config.sell_threshold,
                    )

            except (DataError, Exception) as e:
                # Per-symbol error isolation: log warning và tiếp tục
                logger.warning(f"Bỏ qua symbol {symbol}: {e}")
                continue

        # === 4. Compare baselines (dùng symbol đầu tiên) ===
        comparison_result = None
        if first_symbol_strategy is not None and first_test_df is not None:
            try:
                comparison_result = self.baseline_comparator.compare(
                    model_strategy=first_symbol_strategy,
                    test_df=first_test_df,
                    start_date=first_start_date,
                    end_date=first_end_date,
                )
            except Exception as e:
                logger.warning(f"Baseline comparison thất bại: {e}")

        # === 5. Detect overfitting ===
        overfitting_result = None
        if all_predictions:
            test_preds_concat = np.concatenate(all_predictions)
            train_preds_for_check = (
                train_predictions
                if train_predictions is not None
                else test_preds_concat  # Fallback: dùng test predictions nếu không có train
            )

            try:
                overfitting_result = self.overfitting_detector.detect(
                    train_loss=training_result.final_train_loss,
                    test_loss=training_result.final_val_loss,
                    train_predictions=train_preds_for_check,
                    test_predictions=test_preds_concat,
                )
            except Exception as e:
                logger.warning(f"Overfitting detection thất bại: {e}")

        # === 6. Aggregate prediction quality ===
        aggregated_prediction_quality = None
        if all_predictions and all_actual_returns:
            try:
                concat_preds = np.concatenate(all_predictions)
                concat_actuals = np.concatenate(all_actual_returns)
                aggregated_prediction_quality = self.prediction_analyzer.analyze(
                    concat_preds, concat_actuals
                )
            except Exception as e:
                logger.warning(f"Prediction analysis aggregation thất bại: {e}")

        # === 7. Tổng hợp kết quả ===
        # Lấy aggregated backtest từ symbol đầu tiên (nếu có)
        aggregated_backtest = None
        for sym_result in symbol_results.values():
            if sym_result.backtest_result is not None:
                aggregated_backtest = sym_result.backtest_result
                break

        # Model metadata
        model_metadata = ModelMetadata(
            architecture="StockEvalNet (TCN + Attention)",
            parameter_count=sum(p.numel() for p in model.parameters()),
            training_timestamp=None,
            model_path=training_result.model_path,
            symbols_trained=training_result.symbols_trained,
        )

        eval_result = EvaluationResult(
            model_metadata=model_metadata,
            symbol_results=symbol_results,
            aggregated_backtest=aggregated_backtest,
            comparison_result=comparison_result,
            overfitting_result=overfitting_result,
            aggregated_prediction_quality=aggregated_prediction_quality,
        )

        # === 8. Generate reports ===
        try:
            self.report_generator.generate_json(eval_result)
        except Exception as e:
            logger.error(f"JSON report generation thất bại: {e}")

        # HTML với graceful degradation
        try:
            self.report_generator.generate_html(eval_result)
        except Exception as e:
            logger.error(f"HTML report generation thất bại: {e}")

        logger.info(
            f"Evaluation hoàn tất: {len(symbol_results)} symbols đánh giá thành công"
        )

        return eval_result

    def _validate_inputs(
        self,
        training_result: TrainingResult,
        test_data: Dict[str, pd.DataFrame],
    ) -> None:
        """
        Validate inputs trước khi bắt đầu evaluation.

        Kiểm tra model_path tồn tại và test_data không rỗng.

        Raises
        ------
        ModelError
            Nếu model_path không tồn tại.
        DataError
            Nếu test_data rỗng.
        """
        # Kiểm tra model_path
        if training_result.model_path is None:
            raise ModelError(
                "model_path is None in TrainingResult",
                error_code="MODEL_NOT_FOUND",
                details={"model_path": None},
            )

        if not Path(training_result.model_path).exists():
            raise ModelError(
                f"Model file not found: {training_result.model_path}",
                error_code="MODEL_NOT_FOUND",
                details={"model_path": training_result.model_path},
            )

        # Kiểm tra test_data không rỗng
        if not test_data:
            raise DataError(
                "test_data is empty: không có symbol nào để đánh giá",
                error_code="INSUFFICIENT_TEST_DATA",
                details={"symbols_count": 0},
            )

    def _evaluate_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        model: StockEvalNet,
        norm_params: Optional[Dict] = None,
    ) -> SymbolEvaluationResult:
        """
        Đánh giá model cho một symbol.

        Gồm: generate predictions → backtest → analyze predictions.

        Parameters
        ----------
        symbol : str
            Mã cổ phiếu.
        df : pd.DataFrame
            DataFrame OHLCV cho symbol.
        model : StockEvalNet
            Model đã load.
        norm_params : Dict, optional
            Normalization parameters.

        Returns
        -------
        SymbolEvaluationResult
            Kết quả đánh giá cho symbol.

        Raises
        ------
        DataError
            Nếu DataFrame không đủ dữ liệu.
        """
        if df.empty:
            raise DataError(
                f"Test data for symbol {symbol} is empty",
                error_code="INSUFFICIENT_TEST_DATA",
                details={"symbol": symbol, "rows": 0},
            )

        logger.info(f"Đánh giá symbol {symbol} với {len(df)} rows")

        # Generate predictions
        features = self._extract_features_from_df(df)
        predictions = self._generate_predictions(model, features, norm_params)

        # Tính actual returns
        actual_returns = self._compute_actual_returns(df)

        # Cắt để cùng length
        min_len = min(len(predictions), len(actual_returns))
        predictions = predictions[:min_len]
        actual_returns = actual_returns[:min_len]

        # Chạy backtest
        backtest_result = None
        try:
            backtest_result = self.backtest_evaluator.run_backtest(
                predictions=predictions,
                test_df=df,
            )
        except Exception as e:
            logger.warning(f"Backtest thất bại cho {symbol}: {e}")

        # Analyze predictions
        prediction_quality = None
        try:
            prediction_quality = self.prediction_analyzer.analyze(
                predictions=predictions,
                actual_returns=actual_returns,
            )
        except Exception as e:
            logger.warning(f"Prediction analysis thất bại cho {symbol}: {e}")

        return SymbolEvaluationResult(
            symbol=symbol,
            backtest_result=backtest_result,
            prediction_quality=prediction_quality,
        )

    def _extract_dates(self, df: pd.DataFrame) -> pd.Series:
        """Trích xuất chuỗi ngày từ DataFrame."""
        if "time" in df.columns:
            return pd.to_datetime(df["time"])
        elif isinstance(df.index, pd.DatetimeIndex):
            return df.index.to_series()
        else:
            # Fallback: tạo date range giả
            return pd.Series(pd.date_range("2020-01-01", periods=len(df), freq="B"))
