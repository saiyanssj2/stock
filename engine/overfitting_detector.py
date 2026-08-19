"""
Phát hiện và đánh giá mức độ overfitting của mô hình.

Sử dụng loss ratio (test_loss / train_loss) và Kolmogorov-Smirnov statistic
để đo distribution shift giữa train predictions và test predictions.
Phân loại overfitting severity: none, mild, moderate, severe.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.stats import ks_2samp

from engine.evaluation_config import (
    DistributionStats,
    EvaluationConfig,
    OverfittingResult,
    OverfittingSeverity,
)

logger = logging.getLogger(__name__)


class OverfittingDetector:
    """Phát hiện và đánh giá mức độ overfitting."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Khởi tạo OverfittingDetector.

        Args:
            config: Cấu hình đánh giá, sử dụng mặc định nếu không cung cấp.
        """
        self.config = config or EvaluationConfig()

    def detect(
        self,
        train_loss: float,
        test_loss: float,
        train_predictions: np.ndarray,
        test_predictions: np.ndarray,
    ) -> OverfittingResult:
        """
        Phân tích overfitting dựa trên loss ratio và distribution shift.

        Args:
            train_loss: Final training loss
            test_loss: Loss trên test set
            train_predictions: Position_Score trên train set
            test_predictions: Position_Score trên test set

        Returns:
            OverfittingResult với severity, metrics, warnings
        """
        # Tính loss ratio, xử lý edge case train_loss = 0
        if train_loss == 0:
            # Nếu train_loss = 0, coi như không có overfitting nếu test_loss cũng = 0
            # Ngược lại, ratio = infinity → severe
            if test_loss == 0:
                loss_ratio = 1.0
            else:
                loss_ratio = float("inf")
        else:
            loss_ratio = test_loss / train_loss

        # Phân loại severity dựa trên loss ratio
        severity = self.classify_severity(loss_ratio)

        # Tính KS statistic để đo distribution shift
        ks_stat, ks_p_value = self._compute_ks_statistic(
            train_predictions, test_predictions
        )

        # Tính distribution stats cho train và test predictions
        train_dist = self._compute_distribution_stats(train_predictions)
        test_dist = self._compute_distribution_stats(test_predictions)

        # Tạo warning message nếu severity là moderate hoặc severe
        warning_message = self._generate_warning(
            severity, loss_ratio, ks_stat, ks_p_value, train_dist, test_dist
        )

        return OverfittingResult(
            loss_ratio=loss_ratio,
            severity=severity,
            ks_statistic=ks_stat,
            ks_p_value=ks_p_value,
            train_distribution=train_dist,
            test_distribution=test_dist,
            warning_message=warning_message,
        )

    def classify_severity(self, ratio: float) -> OverfittingSeverity:
        """
        Phân loại severity dựa trên loss ratio.

        Thresholds (từ config):
        - none: ratio < overfitting_mild (default 1.5)
        - mild: overfitting_mild <= ratio < overfitting_moderate (default 2.0)
        - moderate: overfitting_moderate <= ratio < overfitting_severe (default 3.0)
        - severe: ratio >= overfitting_severe (default 3.0)

        Args:
            ratio: Loss ratio (test_loss / train_loss)

        Returns:
            OverfittingSeverity tương ứng
        """
        if ratio >= self.config.overfitting_severe:
            return OverfittingSeverity.SEVERE
        elif ratio >= self.config.overfitting_moderate:
            return OverfittingSeverity.MODERATE
        elif ratio >= self.config.overfitting_mild:
            return OverfittingSeverity.MILD
        else:
            return OverfittingSeverity.NONE

    def _compute_ks_statistic(
        self,
        train_preds: np.ndarray,
        test_preds: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Tính KS statistic giữa hai distributions.

        Args:
            train_preds: Predictions trên train set
            test_preds: Predictions trên test set

        Returns:
            Tuple (ks_statistic, p_value)
        """
        # ks_2samp yêu cầu array 1D không rỗng
        train_flat = np.asarray(train_preds).flatten()
        test_flat = np.asarray(test_preds).flatten()

        if len(train_flat) == 0 or len(test_flat) == 0:
            return 0.0, 1.0

        statistic, p_value = ks_2samp(train_flat, test_flat)
        return float(statistic), float(p_value)

    def _compute_distribution_stats(self, preds: np.ndarray) -> DistributionStats:
        """
        Tính thống kê phân phối cho một tập predictions.

        Args:
            preds: Array predictions

        Returns:
            DistributionStats với mean, std, min, max
        """
        arr = np.asarray(preds).flatten()

        if len(arr) == 0:
            return DistributionStats()

        return DistributionStats(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
        )

    def _generate_warning(
        self,
        severity: OverfittingSeverity,
        loss_ratio: float,
        ks_stat: float,
        ks_p_value: float,
        train_dist: DistributionStats,
        test_dist: DistributionStats,
    ) -> Optional[str]:
        """
        Tạo warning message khi severity là moderate hoặc severe.

        Args:
            severity: Mức độ overfitting
            loss_ratio: Loss ratio
            ks_stat: KS statistic
            ks_p_value: KS p-value
            train_dist: Distribution stats của train predictions
            test_dist: Distribution stats của test predictions

        Returns:
            Warning message hoặc None nếu severity thấp
        """
        if severity not in (OverfittingSeverity.MODERATE, OverfittingSeverity.SEVERE):
            return None

        severity_label = severity.value.upper()
        message = (
            f"[{severity_label}] Overfitting detected: "
            f"loss_ratio={loss_ratio:.3f}, "
            f"KS_statistic={ks_stat:.4f} (p={ks_p_value:.4f}), "
            f"train_mean={train_dist.mean:.4f}, train_std={train_dist.std:.4f}, "
            f"test_mean={test_dist.mean:.4f}, test_std={test_dist.std:.4f}"
        )

        logger.warning(message)
        return message
