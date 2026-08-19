"""
Label Corrector cho Mistake-Driven Learning.

Sửa training labels dựa trên actual trade outcomes:
- Trade lỗ nặng (< -5%): flip hướng signal (±0.8)
- Trade lỗ vừa (-5% đến -2%): sửa thành HOLD (0.0)
- Trade lỗ nhẹ (> -2%): không sửa

Chỉ sửa khi original signal đủ confident (|signal| > threshold).
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from engine.mistake_learning.models import CorrectedLabels, TradeRecord


class LabelCorrector:
    """Sửa training labels dựa trên actual trade outcomes.

    Khi một trade bị lỗ, label gốc (ví dụ BUY +0.8) được sửa thành
    SELL (-0.8) hoặc HOLD (0.0) tùy mức độ lỗ. Chỉ sửa khi original
    signal đủ confident (vượt ngưỡng min_confidence_for_correction).
    """

    def __init__(
        self,
        correction_threshold: float = -0.05,
        min_confidence_for_correction: float = 0.5,
    ) -> None:
        """Khởi tạo LabelCorrector.

        Args:
            correction_threshold: Ngưỡng PnL để trigger correction (mặc định -5%).
            min_confidence_for_correction: Ngưỡng |signal| tối thiểu để sửa label.
        """
        self.correction_threshold = correction_threshold
        self.min_confidence_for_correction = min_confidence_for_correction

    def correct_labels(
        self,
        bad_trades: List[TradeRecord],
        original_labels: np.ndarray,
        dates: pd.DatetimeIndex,
    ) -> CorrectedLabels:
        """Sửa training labels dựa trên actual trade outcomes.

        Duyệt qua danh sách bad trades, tìm ngày entry tương ứng trong
        training dates, và áp dụng correction nếu thỏa điều kiện.

        Args:
            bad_trades: Danh sách trades bị lỗ cần sửa label.
            original_labels: Mảng labels gốc, shape (len(dates),).
            dates: DatetimeIndex của training data (tương ứng với labels).

        Returns:
            CorrectedLabels chứa labels đã sửa, mask, và count.
        """
        corrected = original_labels.copy()
        mask = np.zeros(len(original_labels), dtype=bool)

        for trade in bad_trades:
            # Tìm index của entry_date trong training dates
            idx = self._find_date_index(trade.entry_date, dates)
            if idx is None:
                continue

            # Chỉ sửa khi original signal đủ confident
            if abs(original_labels[idx]) <= self.min_confidence_for_correction:
                continue

            # Không sửa nếu loss quá nhỏ (> -2%) - Requirement 4.8
            if trade.pnl_pct > -0.02:
                continue

            # Tính label mới dựa trên outcome thực tế
            new_label = self.compute_corrected_label(trade)

            corrected[idx] = new_label
            mask[idx] = True

        return CorrectedLabels(
            original_labels=original_labels,
            corrected_labels=corrected,
            correction_mask=mask,
            correction_count=int(mask.sum()),
        )

    def compute_corrected_label(self, trade: TradeRecord) -> float:
        """Tính label đã sửa dựa trên kết quả trade thực tế.

        Logic:
        - Lỗ nặng (pnl_pct < -5%): flip hướng signal
          - Signal dương (BUY) → -0.8 (nên SELL)
          - Signal âm (SELL) → +0.8 (nên BUY)
          - Signal = 0 → 0.0
        - Lỗ vừa (-5% <= pnl_pct <= -2%): → 0.0 (nên HOLD)
        - Lỗ nhẹ (pnl_pct > -2%): giữ nguyên (không sửa)

        Args:
            trade: TradeRecord chứa pnl_pct và signal_strength.

        Returns:
            Label đã sửa, giá trị trong [-1.0, 1.0].
        """
        if trade.pnl_pct < -0.05:
            # Lỗ nặng: flip hướng signal
            if trade.signal_strength > 0:
                return -0.8  # Đang BUY, nên SELL
            elif trade.signal_strength < 0:
                return 0.8  # Đang SELL, nên BUY
            else:
                return 0.0  # Signal = 0, nên HOLD
        elif trade.pnl_pct <= -0.02:
            # Lỗ vừa (-5% đến -2%): nên HOLD
            return 0.0
        else:
            # Lỗ nhẹ (> -2%): không cần sửa, giữ signal gốc
            return trade.signal_strength

    def _find_date_index(
        self,
        entry_date: Optional[pd.Timestamp],
        dates: pd.DatetimeIndex,
    ) -> Optional[int]:
        """Tìm index của entry_date trong dates (so sánh theo ngày).

        Normalize cả hai về date-only để so sánh, tránh vấn đề timezone
        và thời gian trong ngày.

        Args:
            entry_date: Ngày entry của trade.
            dates: DatetimeIndex cần tìm.

        Returns:
            Index nếu tìm thấy, None nếu không.
        """
        if entry_date is None:
            return None

        # Normalize entry_date về date-only
        entry_date_normalized = pd.Timestamp(entry_date).normalize()

        # Normalize dates để so sánh
        normalized_dates = dates.normalize()

        # Tìm vị trí khớp
        matches = normalized_dates == entry_date_normalized
        if matches.any():
            # Lấy index đầu tiên khớp
            return int(np.where(matches)[0][0])

        return None
