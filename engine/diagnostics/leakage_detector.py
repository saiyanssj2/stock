"""
Data Leakage Detector — Kiểm tra tính toàn vẹn chronological trong backtest và training.

Phát hiện các loại vi phạm:
- SIGNAL_FUTURE_LEAK: Signal truy cập future data (Req 4.1)
- SPLIT_OVERLAP: Chronological split bị overlap (Req 4.2)
- LABEL_LOOK_AHEAD: Label sử dụng dữ liệu ngoài [i, i+horizon] (Req 4.3)
- POSSIBLE_LEAKAGE: Accuracy bất thường cao liên tiếp (Req 4.4)
- Report xuất ra leakage_report.json (Req 4.5)
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.diagnostics.models import (
    LeakageReport,
    LeakageViolation,
    ViolationType,
)


class DataLeakageDetector:
    """Kiểm tra data leakage trong backtest và training.

    Attributes:
        ACCURACY_THRESHOLD: Ngưỡng accuracy nghi ngờ leakage (80%).
        CONSECUTIVE_THRESHOLD: Số lần liên tiếp vượt ngưỡng để cảnh báo.
        DEFAULT_HORIZON: Horizon mặc định cho label range check.
    """

    ACCURACY_THRESHOLD: float = 0.80
    CONSECUTIVE_THRESHOLD: int = 3
    DEFAULT_HORIZON: int = 5

    def __init__(
        self, output_path: str = "data/engine/diagnostics/leakage_report.json"
    ):
        """Khởi tạo DataLeakageDetector.

        Args:
            output_path: Đường dẫn file output cho leakage report.
        """
        self._output_path = output_path
        self._violations: List[LeakageViolation] = []

    def check_signal_access(
        self,
        df_length: int,
        access_index: int,
        max_allowed_index: int,
    ) -> Optional[LeakageViolation]:
        """Kiểm tra signal generation không truy cập future data.

        Verify rằng signal chỉ truy cập dữ liệu từ row 0 đến max_allowed_index.
        Nếu access_index > max_allowed_index → vi phạm SIGNAL_FUTURE_LEAK.

        Args:
            df_length: Tổng số rows trong DataFrame.
            access_index: Index mà signal đang truy cập.
            max_allowed_index: Index tối đa được phép truy cập.

        Returns:
            LeakageViolation nếu phát hiện vi phạm, None nếu hợp lệ.
        """
        if access_index > max_allowed_index:
            # Tính số rows tương lai bị lộ
            rows_leaked = access_index - max_allowed_index
            violation = LeakageViolation(
                violation_type=ViolationType.SIGNAL_FUTURE_LEAK,
                symbol="",
                details={
                    "index": max_allowed_index,
                    "access_index": access_index,
                    "rows_leaked": rows_leaked,
                    "df_length": df_length,
                },
            )
            self._violations.append(violation)
            return violation
        return None

    def check_chronological_split(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> Optional[LeakageViolation]:
        """Verify max(train) < min(val) < min(test), no overlap.

        Kiểm tra chronological split đảm bảo:
        1. max(train_indices) < min(val_indices)
        2. max(val_indices) < min(test_indices)
        3. Không có index nào xuất hiện trong nhiều hơn một split.

        Args:
            train_indices: Mảng indices cho tập train.
            val_indices: Mảng indices cho tập validation.
            test_indices: Mảng indices cho tập test.

        Returns:
            LeakageViolation nếu phát hiện overlap, None nếu hợp lệ.
        """
        # Chuyển sang numpy array nếu chưa phải
        train_arr = np.asarray(train_indices)
        val_arr = np.asarray(val_indices)
        test_arr = np.asarray(test_indices)

        # Kiểm tra mảng rỗng — nếu bất kỳ split nào rỗng, không thể validate
        if train_arr.size == 0 or val_arr.size == 0 or test_arr.size == 0:
            return None

        # Kiểm tra ordering: max(train) < min(val) < min(test)
        ordering_violated = (
            np.max(train_arr) >= np.min(val_arr)
            or np.max(val_arr) >= np.min(test_arr)
        )

        # Kiểm tra overlap: bất kỳ index nào xuất hiện trong nhiều hơn một split
        train_set = set(train_arr.tolist())
        val_set = set(val_arr.tolist())
        test_set = set(test_arr.tolist())

        overlap_train_val = train_set & val_set
        overlap_val_test = val_set & test_set
        overlap_train_test = train_set & test_set

        total_overlap = overlap_train_val | overlap_val_test | overlap_train_test

        if ordering_violated or len(total_overlap) > 0:
            # Xác định split pairs bị overlap
            split_pairs = []
            if overlap_train_val:
                split_pairs.append("train-val")
            if overlap_val_test:
                split_pairs.append("val-test")
            if overlap_train_test:
                split_pairs.append("train-test")
            if ordering_violated and not total_overlap:
                # Ordering bị vi phạm nhưng không có overlap trực tiếp
                if np.max(train_arr) >= np.min(val_arr):
                    split_pairs.append("train-val")
                if np.max(val_arr) >= np.min(test_arr):
                    split_pairs.append("val-test")

            violation = LeakageViolation(
                violation_type=ViolationType.SPLIT_OVERLAP,
                symbol="",
                details={
                    "overlapping_indices_count": len(total_overlap),
                    "split_pairs": split_pairs,
                    "max_train": int(np.max(train_arr)),
                    "min_val": int(np.min(val_arr)),
                    "max_val": int(np.max(val_arr)),
                    "min_test": int(np.min(test_arr)),
                },
            )
            self._violations.append(violation)
            return violation
        return None

    def check_label_range(
        self,
        row_index: int,
        actual_data_range: Tuple[int, int],
        horizon: int = DEFAULT_HORIZON,
    ) -> Optional[LeakageViolation]:
        """Verify label tại row i chỉ dùng dữ liệu trong [i, i+horizon].

        Kiểm tra rằng label computation tại row_index chỉ sử dụng close prices
        trong phạm vi [row_index, row_index + horizon]. Nếu actual_data_range
        nằm ngoài phạm vi → vi phạm LABEL_LOOK_AHEAD.

        Args:
            row_index: Index của row đang compute label.
            actual_data_range: Tuple (start, end) phạm vi dữ liệu thực tế được sử dụng.
            horizon: Số sessions cho phép label nhìn về phía trước.

        Returns:
            LeakageViolation nếu phát hiện vi phạm, None nếu hợp lệ.
        """
        allowed_start = row_index
        allowed_end = row_index + horizon

        actual_start, actual_end = actual_data_range

        # Kiểm tra phạm vi thực tế có nằm trong ranh giới cho phép không
        if actual_start < allowed_start or actual_end > allowed_end:
            violation = LeakageViolation(
                violation_type=ViolationType.LABEL_LOOK_AHEAD,
                symbol="",
                details={
                    "row_index": row_index,
                    "allowed_range": [allowed_start, allowed_end],
                    "actual_range": [actual_start, actual_end],
                    "horizon": horizon,
                },
            )
            self._violations.append(violation)
            return violation
        return None

    def check_accuracy_anomaly(
        self,
        symbol: str,
        accuracy_history: List[float],
    ) -> Optional[LeakageViolation]:
        """Kiểm tra accuracy > 80% trong 3 lần liên tiếp.

        Nếu phát hiện CONSECUTIVE_THRESHOLD (3) lần liên tiếp accuracy vượt
        ACCURACY_THRESHOLD (80%) → cảnh báo POSSIBLE_LEAKAGE.

        Args:
            symbol: Mã chứng khoán.
            accuracy_history: Danh sách accuracy qua các lần backtest.

        Returns:
            LeakageViolation nếu phát hiện anomaly, None nếu hợp lệ.
        """
        if len(accuracy_history) < self.CONSECUTIVE_THRESHOLD:
            return None

        # Tìm chuỗi liên tiếp vượt ngưỡng
        consecutive_count = 0
        consecutive_values: List[float] = []

        for accuracy in accuracy_history:
            if accuracy > self.ACCURACY_THRESHOLD:
                consecutive_count += 1
                consecutive_values.append(accuracy)
                if consecutive_count >= self.CONSECUTIVE_THRESHOLD:
                    # Tính accuracy trung bình của chuỗi vi phạm
                    avg_accuracy = sum(
                        consecutive_values[-self.CONSECUTIVE_THRESHOLD:]
                    ) / self.CONSECUTIVE_THRESHOLD
                    violation = LeakageViolation(
                        violation_type=ViolationType.POSSIBLE_LEAKAGE,
                        symbol=symbol,
                        details={
                            "symbol": symbol,
                            "avg_accuracy": round(avg_accuracy, 4),
                            "consecutive_count": consecutive_count,
                            "threshold": self.ACCURACY_THRESHOLD,
                        },
                    )
                    self._violations.append(violation)
                    return violation
            else:
                consecutive_count = 0
                consecutive_values = []

        return None

    def generate_report(self, symbol: str) -> LeakageReport:
        """Tạo báo cáo leakage cho một symbol và xuất ra JSON file.

        Tổng hợp tất cả violations đã phát hiện, đếm theo từng loại,
        và xuất ra leakage_report.json.

        Args:
            symbol: Mã chứng khoán được kiểm tra.

        Returns:
            LeakageReport chứa kết quả kiểm tra.
        """
        # Đếm violations theo từng loại
        violation_counts: Dict[str, int] = {
            ViolationType.SIGNAL_FUTURE_LEAK.value: 0,
            ViolationType.SPLIT_OVERLAP.value: 0,
            ViolationType.LABEL_LOOK_AHEAD.value: 0,
            ViolationType.POSSIBLE_LEAKAGE.value: 0,
        }

        for violation in self._violations:
            violation_counts[violation.violation_type.value] += 1

        # Xác định overall_status
        total_violations = sum(violation_counts.values())
        overall_status = "pass" if total_violations == 0 else "fail"

        # Tạo report
        report = LeakageReport(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            + "Z",
            symbol=symbol,
            violations=list(self._violations),
            violation_counts=violation_counts,
            overall_status=overall_status,
        )

        # Xuất ra JSON file
        self._write_report(report)

        return report

    def _write_report(self, report: LeakageReport) -> None:
        """Ghi report ra file JSON.

        Args:
            report: LeakageReport cần ghi.
        """
        # Tạo thư mục output nếu chưa tồn tại
        output_dir = os.path.dirname(self._output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Serialize report sang dict
        report_dict = {
            "timestamp": report.timestamp,
            "symbol": report.symbol,
            "violations": [
                {
                    "violation_type": v.violation_type.value,
                    "symbol": v.symbol,
                    "details": v.details,
                }
                for v in report.violations
            ],
            "violation_counts": report.violation_counts,
            "overall_status": report.overall_status,
        }

        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
