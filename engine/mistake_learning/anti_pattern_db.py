"""
Anti-Pattern Database cho Mistake-Driven Learning.

Tích lũy patterns dẫn đến lỗ qua nhiều cycles. Merge patterns tương tự
(cosine similarity > 0.8), expire patterns cũ, và cap tại max_patterns.
Persist ra file JSON tại engine/models/anti_patterns.json.
"""

import json
import logging
import math
import os
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

from engine.mistake_learning.models import AntiPattern, TradeRecord

logger = logging.getLogger(__name__)

_FEATURE_VECTOR_LENGTH = 61


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Tính cosine similarity giữa 2 vectors.

    Trả về 0.0 nếu một trong hai vector có norm = 0.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _is_valid_feature_vector(feature_vector: Optional[List[float]]) -> bool:
    """Kiểm tra feature vector có hợp lệ không.

    Hợp lệ khi: không None, đúng 61 phần tử, không chứa NaN.
    """
    if feature_vector is None:
        return False
    if len(feature_vector) != _FEATURE_VECTOR_LENGTH:
        return False
    for val in feature_vector:
        if not math.isfinite(val):
            return False
    return True


class AntiPatternDatabase:
    """Tích lũy và quản lý anti-patterns qua các cycles.

    Merge patterns có cosine similarity > 0.8, expire patterns cũ
    (không thấy trong 20 cycles gần nhất), cap tại 100 patterns.
    """

    PATTERNS_FILE = "engine/models/anti_patterns.json"

    def __init__(self, base_dir: str = ".", max_patterns: int = 100) -> None:
        """Khởi tạo database, load từ file nếu tồn tại.

        Args:
            base_dir: Thư mục gốc chứa file persistence.
            max_patterns: Số lượng patterns tối đa lưu trữ.
        """
        self._base_dir = base_dir
        self._max_patterns = max_patterns
        self._similarity_threshold = 0.8
        self._expiry_cycles = 20
        self._patterns: List[AntiPattern] = []
        self._load()

    def update_patterns(
        self, bad_trades: List[TradeRecord], cycle_number: int
    ) -> None:
        """Cập nhật database với bad trades mới.

        Cho mỗi trade:
        - Skip nếu feature_vector không hợp lệ (None, != 61 dims, chứa NaN)
        - Nếu cosine similarity > 0.8 với pattern hiện có → merge
        - Ngược lại → tạo pattern mới

        Sau khi xử lý tất cả trades:
        - Expire patterns cũ (current_cycle - last_seen_cycle > 20)
        - Merge patterns nội bộ nếu similarity > 0.8
        - Cap tại max_patterns, giữ patterns có confidence cao nhất
        - Recalculate confidence cho tất cả patterns

        Args:
            bad_trades: Danh sách bad trades cần xử lý.
            cycle_number: Số cycle hiện tại.
        """
        for trade in bad_trades:
            if not _is_valid_feature_vector(trade.feature_vector):
                continue

            feature_vec = np.array(trade.feature_vector, dtype=np.float64)

            # Tìm pattern có similarity cao nhất
            best_idx, best_sim = self._find_best_match(feature_vec)

            if best_idx is not None and best_sim > self._similarity_threshold:
                # Merge vào pattern hiện có
                self._merge_into_pattern(
                    self._patterns[best_idx], feature_vec, trade.pnl_pct, cycle_number
                )
            else:
                # Tạo pattern mới
                new_pattern = AntiPattern(
                    pattern_id=str(uuid.uuid4()),
                    indicator_conditions={},
                    centroid=trade.feature_vector[:],
                    outcome_avg_pnl=trade.pnl_pct,
                    frequency=1,
                    first_seen_cycle=cycle_number,
                    last_seen_cycle=cycle_number,
                    confidence=0.0,
                )
                self._patterns.append(new_pattern)

        # Expire patterns cũ
        self._expire_patterns(cycle_number)

        # Merge nội bộ: hai patterns còn lại có similarity > 0.8
        self._merge_internal_patterns(cycle_number)

        # Recalculate confidence trước khi cap
        for p in self._patterns:
            cycles_observed = cycle_number - p.first_seen_cycle + 1
            p.confidence = p.frequency / max(cycles_observed, 1)

        # Cap tại max_patterns, giữ highest confidence
        if len(self._patterns) > self._max_patterns:
            self._patterns.sort(key=lambda p: p.confidence, reverse=True)
            self._patterns = self._patterns[: self._max_patterns]

        self._save()

    def get_active_patterns(self) -> List[AntiPattern]:
        """Trả về bản copy danh sách patterns hiện tại."""
        return deepcopy(self._patterns)

    def match_pattern(
        self, feature_vector: np.ndarray, threshold: float = 0.8
    ) -> Optional[AntiPattern]:
        """Tìm pattern có similarity cao nhất với feature_vector.

        Trả về pattern có cosine similarity cao nhất nếu > threshold,
        ngược lại trả về None.

        Args:
            feature_vector: Vector 61 chiều cần so khớp.
            threshold: Ngưỡng similarity tối thiểu.

        Returns:
            AntiPattern có similarity cao nhất hoặc None.
        """
        if len(self._patterns) == 0:
            return None

        best_pattern: Optional[AntiPattern] = None
        best_sim = -1.0

        for pattern in self._patterns:
            centroid = np.array(pattern.centroid, dtype=np.float64)
            sim = _cosine_similarity(feature_vector, centroid)
            if sim > threshold and sim > best_sim:
                best_sim = sim
                best_pattern = pattern

        if best_pattern is not None:
            return deepcopy(best_pattern)
        return None

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Trả về thống kê tổng hợp về database.

        Returns:
            Dict chứa total_patterns, avg_frequency, avg_confidence,
            max_frequency, min_confidence, max_confidence.
        """
        if not self._patterns:
            return {
                "total_patterns": 0,
                "avg_frequency": 0.0,
                "avg_confidence": 0.0,
                "max_frequency": 0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
            }

        frequencies = [p.frequency for p in self._patterns]
        confidences = [p.confidence for p in self._patterns]

        return {
            "total_patterns": len(self._patterns),
            "avg_frequency": sum(frequencies) / len(frequencies),
            "avg_confidence": sum(confidences) / len(confidences),
            "max_frequency": max(frequencies),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
        }

    # =========================================================================
    # Private methods
    # =========================================================================

    def _find_best_match(
        self, feature_vec: np.ndarray
    ) -> tuple:
        """Tìm pattern có cosine similarity cao nhất với feature_vec.

        Returns:
            Tuple (index, similarity) hoặc (None, 0.0) nếu không tìm thấy.
        """
        if not self._patterns:
            return None, 0.0

        best_idx: Optional[int] = None
        best_sim = -1.0

        for i, pattern in enumerate(self._patterns):
            centroid = np.array(pattern.centroid, dtype=np.float64)
            sim = _cosine_similarity(feature_vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        return best_idx, best_sim

    def _merge_into_pattern(
        self,
        pattern: AntiPattern,
        feature_vec: np.ndarray,
        pnl_pct: float,
        cycle_number: int,
    ) -> None:
        """Merge trade mới vào pattern hiện có.

        Cập nhật: frequency, last_seen_cycle, outcome_avg_pnl (running avg),
        centroid (running mean weighted by frequency).
        """
        old_freq = pattern.frequency
        pattern.frequency += 1
        pattern.last_seen_cycle = cycle_number

        # Running average PnL
        pattern.outcome_avg_pnl = (
            pattern.outcome_avg_pnl * old_freq + pnl_pct
        ) / pattern.frequency

        # Running mean centroid
        old_centroid = np.array(pattern.centroid, dtype=np.float64)
        new_centroid = (old_centroid * old_freq + feature_vec) / pattern.frequency
        pattern.centroid = new_centroid.tolist()

    def _expire_patterns(self, current_cycle: int) -> None:
        """Xóa patterns không được thấy trong 20 cycles gần nhất."""
        self._patterns = [
            p
            for p in self._patterns
            if (current_cycle - p.last_seen_cycle) <= self._expiry_cycles
        ]

    def _merge_internal_patterns(self, cycle_number: int) -> None:
        """Merge các patterns nội bộ có similarity > 0.8.

        Lặp cho đến khi không còn cặp nào cần merge.
        """
        merged = True
        while merged:
            merged = False
            i = 0
            while i < len(self._patterns):
                j = i + 1
                while j < len(self._patterns):
                    centroid_i = np.array(
                        self._patterns[i].centroid, dtype=np.float64
                    )
                    centroid_j = np.array(
                        self._patterns[j].centroid, dtype=np.float64
                    )
                    sim = _cosine_similarity(centroid_i, centroid_j)
                    if sim > self._similarity_threshold:
                        # Merge j vào i
                        self._merge_two_patterns(i, j)
                        merged = True
                        # Không tăng j vì list đã thay đổi
                    else:
                        j += 1
                i += 1

    def _merge_two_patterns(self, idx_a: int, idx_b: int) -> None:
        """Merge pattern tại idx_b vào pattern tại idx_a.

        Kết quả: frequency cộng dồn, centroid weighted average,
        first_seen_cycle lấy sớm nhất, last_seen_cycle lấy muộn nhất.
        """
        pa = self._patterns[idx_a]
        pb = self._patterns[idx_b]

        total_freq = pa.frequency + pb.frequency

        # Weighted average centroid
        centroid_a = np.array(pa.centroid, dtype=np.float64)
        centroid_b = np.array(pb.centroid, dtype=np.float64)
        new_centroid = (
            centroid_a * pa.frequency + centroid_b * pb.frequency
        ) / total_freq

        # Weighted average PnL
        new_avg_pnl = (
            pa.outcome_avg_pnl * pa.frequency + pb.outcome_avg_pnl * pb.frequency
        ) / total_freq

        pa.frequency = total_freq
        pa.centroid = new_centroid.tolist()
        pa.outcome_avg_pnl = new_avg_pnl
        pa.first_seen_cycle = min(pa.first_seen_cycle, pb.first_seen_cycle)
        pa.last_seen_cycle = max(pa.last_seen_cycle, pb.last_seen_cycle)

        # Xóa pattern b
        self._patterns.pop(idx_b)

    def _get_file_path(self) -> str:
        """Trả về đường dẫn đầy đủ tới file persistence."""
        return os.path.join(self._base_dir, self.PATTERNS_FILE)

    def _load(self) -> None:
        """Load patterns từ file JSON.

        Nếu file không tồn tại → bắt đầu với list rỗng.
        Nếu file corrupted (invalid JSON) → log warning, reset empty.
        """
        file_path = self._get_file_path()
        if not os.path.exists(file_path):
            self._patterns = []
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._patterns = []
            for item in data:
                pattern = AntiPattern(
                    pattern_id=item.get("pattern_id", ""),
                    indicator_conditions=item.get("indicator_conditions", {}),
                    centroid=item.get("centroid", []),
                    outcome_avg_pnl=item.get("outcome_avg_pnl", 0.0),
                    frequency=item.get("frequency", 0),
                    first_seen_cycle=item.get("first_seen_cycle", 0),
                    last_seen_cycle=item.get("last_seen_cycle", 0),
                    confidence=item.get("confidence", 0.0),
                )
                self._patterns.append(pattern)

        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logger.warning(
                "Anti-pattern database file bị corrupted, reset về empty: %s", e
            )
            self._patterns = []

    def _save(self) -> None:
        """Serialize patterns ra file JSON.

        Tạo thư mục nếu chưa tồn tại.
        """
        file_path = self._get_file_path()
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        data = []
        for p in self._patterns:
            data.append(
                {
                    "pattern_id": p.pattern_id,
                    "indicator_conditions": p.indicator_conditions,
                    "centroid": p.centroid,
                    "outcome_avg_pnl": p.outcome_avg_pnl,
                    "frequency": p.frequency,
                    "first_seen_cycle": p.first_seen_cycle,
                    "last_seen_cycle": p.last_seen_cycle,
                    "confidence": p.confidence,
                }
            )

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
