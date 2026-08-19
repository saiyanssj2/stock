"""
Phân tích chất lượng dự đoán Position_Score.

Module cung cấp class PredictionAnalyzer với các phương thức đo lường:
- Directional accuracy (tỷ lệ dự đoán đúng hướng)
- Pearson correlation (tương quan tuyến tính)
- Calibration by quartile (hiệu chỉnh theo phân vị)
- Score distribution (phân phối điểm dự đoán)
- Confusion matrix (ma trận nhầm lẫn hướng)
"""

from typing import Dict

import numpy as np
from scipy import stats

from engine.evaluation_config import (
    ConfusionMatrix,
    EvaluationConfig,
    PredictionQualityResult,
    ScoreDistributionResult,
)


class PredictionAnalyzer:
    """Phân tích chất lượng dự đoán Position_Score."""

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        """
        Khởi tạo PredictionAnalyzer.

        Args:
            config: Cấu hình đánh giá, dùng mặc định nếu không truyền.
        """
        self.config = config or EvaluationConfig()

    def directional_accuracy(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
    ) -> float:
        """
        Tính tỷ lệ dự đoán đúng hướng (sign match).

        Loại bỏ các cặp mà prediction hoặc actual bằng 0.

        Args:
            predictions: Mảng Position_Score dự đoán.
            actual_returns: Mảng lợi nhuận thực tế tương ứng.

        Returns:
            Percentage [0, 100] dự đoán đúng hướng.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        actual_returns = np.asarray(actual_returns, dtype=np.float64)

        # Loại bỏ cặp có prediction hoặc actual bằng 0
        mask = (predictions != 0) & (actual_returns != 0)
        filtered_preds = predictions[mask]
        filtered_actuals = actual_returns[mask]

        if len(filtered_preds) == 0:
            return 0.0

        # So sánh dấu: sign(prediction) == sign(actual)
        sign_match = np.sign(filtered_preds) == np.sign(filtered_actuals)
        accuracy = np.sum(sign_match) / len(filtered_preds) * 100.0

        return float(accuracy)

    def pearson_correlation(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
    ) -> float:
        """
        Tính Pearson correlation giữa predictions và actual returns.

        Args:
            predictions: Mảng Position_Score dự đoán.
            actual_returns: Mảng lợi nhuận thực tế tương ứng.

        Returns:
            Hệ số tương quan Pearson trong khoảng [-1, 1].
            Trả về 0.0 nếu không tính được (ví dụ: mảng hằng số).
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        actual_returns = np.asarray(actual_returns, dtype=np.float64)

        if len(predictions) < 2:
            return 0.0

        # Kiểm tra mảng hằng số (std = 0) → correlation không xác định
        if np.std(predictions) == 0 or np.std(actual_returns) == 0:
            return 0.0

        corr, _ = stats.pearsonr(predictions, actual_returns)

        # Xử lý NaN (trường hợp hiếm)
        if np.isnan(corr):
            return 0.0

        return float(corr)

    def calibration_by_quartile(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        Tính mean absolute return cho mỗi quartile của predictions.

        Chia predictions thành 4 nhóm dựa trên percentile (25, 50, 75)
        và tính trung bình |actual_return| cho mỗi nhóm.

        Args:
            predictions: Mảng Position_Score dự đoán.
            actual_returns: Mảng lợi nhuận thực tế tương ứng.

        Returns:
            Dict với keys 'Q1', 'Q2', 'Q3', 'Q4' → mean absolute return.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        actual_returns = np.asarray(actual_returns, dtype=np.float64)

        if len(predictions) == 0:
            return {"Q1": 0.0, "Q2": 0.0, "Q3": 0.0, "Q4": 0.0}

        # Tính percentiles để chia thành 4 nhóm
        p25, p50, p75 = np.percentile(predictions, [25, 50, 75])

        # Phân nhóm predictions vào quartiles
        q1_mask = predictions <= p25
        q2_mask = (predictions > p25) & (predictions <= p50)
        q3_mask = (predictions > p50) & (predictions <= p75)
        q4_mask = predictions > p75

        # Tính mean absolute return cho mỗi quartile
        result: Dict[str, float] = {}
        for label, mask in [
            ("Q1", q1_mask),
            ("Q2", q2_mask),
            ("Q3", q3_mask),
            ("Q4", q4_mask),
        ]:
            if np.any(mask):
                result[label] = float(np.mean(np.abs(actual_returns[mask])))
            else:
                result[label] = 0.0

        return result

    def score_distribution(
        self,
        predictions: np.ndarray,
    ) -> ScoreDistributionResult:
        """
        Tính thống kê phân phối Position_Score.

        Bao gồm mean, std, skewness, kurtosis. Đặt flag:
        - too_conservative nếu std < conservative_std_threshold (0.1)
        - too_extreme nếu std > extreme_std_threshold (0.8)

        Args:
            predictions: Mảng Position_Score dự đoán.

        Returns:
            ScoreDistributionResult với các thống kê và flags.
        """
        predictions = np.asarray(predictions, dtype=np.float64)

        if len(predictions) == 0:
            return ScoreDistributionResult()

        mean_val = float(np.mean(predictions))
        std_val = float(np.std(predictions, ddof=0))
        skewness_val = float(stats.skew(predictions))
        kurtosis_val = float(stats.kurtosis(predictions))

        # Đánh giá flags dựa trên thresholds
        too_conservative = std_val < self.config.conservative_std_threshold
        too_extreme = std_val > self.config.extreme_std_threshold

        # Tạo flag message nếu cần
        flag_message = None
        if too_conservative:
            flag_message = (
                f"Model quá bảo thủ: std={std_val:.4f} "
                f"< {self.config.conservative_std_threshold}"
            )
        elif too_extreme:
            flag_message = (
                f"Model quá cực đoan: std={std_val:.4f} "
                f"> {self.config.extreme_std_threshold}"
            )

        return ScoreDistributionResult(
            mean=mean_val,
            std=std_val,
            skewness=skewness_val,
            kurtosis=kurtosis_val,
            too_conservative=too_conservative,
            too_extreme=too_extreme,
            flag_message=flag_message,
        )

    def confusion_matrix(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
    ) -> ConfusionMatrix:
        """
        Tính confusion matrix hướng dự đoán.

        Phân loại:
        - TP: predicted up (>0), actual up (>0)
        - TN: predicted down (<0), actual down (<0)
        - FP: predicted up (>0), actual down (<0)
        - FN: predicted down (<0), actual up (>0)

        Loại bỏ các cặp mà prediction hoặc actual bằng 0.

        Args:
            predictions: Mảng Position_Score dự đoán.
            actual_returns: Mảng lợi nhuận thực tế tương ứng.

        Returns:
            ConfusionMatrix với TP, TN, FP, FN.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        actual_returns = np.asarray(actual_returns, dtype=np.float64)

        # Loại bỏ cặp có prediction hoặc actual bằng 0
        mask = (predictions != 0) & (actual_returns != 0)
        filtered_preds = predictions[mask]
        filtered_actuals = actual_returns[mask]

        if len(filtered_preds) == 0:
            return ConfusionMatrix()

        # Phân loại theo dấu
        pred_up = filtered_preds > 0
        pred_down = filtered_preds < 0
        actual_up = filtered_actuals > 0
        actual_down = filtered_actuals < 0

        tp = int(np.sum(pred_up & actual_up))
        tn = int(np.sum(pred_down & actual_down))
        fp = int(np.sum(pred_up & actual_down))
        fn = int(np.sum(pred_down & actual_up))

        return ConfusionMatrix(
            true_positive=tp,
            true_negative=tn,
            false_positive=fp,
            false_negative=fn,
        )

    def analyze(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
    ) -> PredictionQualityResult:
        """
        Phân tích toàn diện chất lượng dự đoán.

        Gọi tất cả methods phân tích và tổng hợp kết quả.

        Args:
            predictions: Mảng Position_Score dự đoán.
            actual_returns: Mảng lợi nhuận thực tế tương ứng.

        Returns:
            PredictionQualityResult chứa tất cả metrics.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        actual_returns = np.asarray(actual_returns, dtype=np.float64)

        return PredictionQualityResult(
            directional_accuracy=self.directional_accuracy(predictions, actual_returns),
            pearson_correlation=self.pearson_correlation(predictions, actual_returns),
            calibration_quartiles=self.calibration_by_quartile(
                predictions, actual_returns
            ),
            score_distribution=self.score_distribution(predictions),
            confusion_matrix=self.confusion_matrix(predictions, actual_returns),
        )
