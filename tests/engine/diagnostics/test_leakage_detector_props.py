# -*- coding: utf-8 -*-
"""
Property-based tests cho Data Leakage Detector — Property 12: Accuracy anomaly detection.

Feature: train-backtest-verification, Property 12: Accuracy anomaly detection

**Validates: Requirements 4.4**

Properties:
1. Bất kỳ sequence accuracy nào KHÔNG có 3 giá trị liên tiếp > 0.80 → check_accuracy_anomaly() trả về None
2. Bất kỳ sequence accuracy nào CÓ ít nhất 3 giá trị liên tiếp > 0.80 → check_accuracy_anomaly() trả về violation POSSIBLE_LEAKAGE
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.diagnostics.leakage_detector import DataLeakageDetector
from engine.diagnostics.models import ViolationType


# ---------------------------------------------------------------------------
# Strategies — Sinh dữ liệu ngẫu nhiên hợp lệ
# ---------------------------------------------------------------------------

# Symbol hợp lệ (mã chứng khoán Việt Nam: 3 ký tự viết hoa)
symbol_strategy = st.from_regex(r"[A-Z]{3}", fullmatch=True)

# Giá trị accuracy hợp lệ [0.0, 1.0]
accuracy_value_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

# Giá trị accuracy <= 0.80 (KHÔNG trigger — bao gồm cả giá trị đúng bằng 0.80)
non_triggering_accuracy = st.floats(
    min_value=0.0, max_value=0.80, allow_nan=False, allow_infinity=False
)

# Giá trị accuracy > 0.80 (trigger condition)
triggering_accuracy = st.floats(
    min_value=0.80, max_value=1.0, allow_nan=False, allow_infinity=False,
    exclude_min=True,  # Strict > 0.80, loại bỏ chính xác 0.80
)


# ---------------------------------------------------------------------------
# Composite Strategies
# ---------------------------------------------------------------------------


@st.composite
def sequence_without_3_consecutive_high(draw):
    """
    Sinh sequence accuracy KHÔNG có 3 giá trị liên tiếp > 0.80.

    Thuật toán: tạo sequence bằng cách đảm bảo sau mỗi 2 giá trị > 0.80 liên tiếp,
    giá trị tiếp theo phải <= 0.80.

    Returns:
        list[float]: Sequence accuracy values không trigger cảnh báo.
    """
    length = draw(st.integers(min_value=0, max_value=30))
    sequence = []
    consecutive_high = 0

    for _ in range(length):
        if consecutive_high >= 2:
            # Buộc phải chọn giá trị <= 0.80 để phá chuỗi
            val = draw(non_triggering_accuracy)
            sequence.append(val)
            consecutive_high = 0
        else:
            # Có thể chọn bất kỳ giá trị nào
            val = draw(accuracy_value_strategy)
            sequence.append(val)
            if val > 0.80:
                consecutive_high += 1
            else:
                consecutive_high = 0

    return sequence


@st.composite
def sequence_with_3_consecutive_high(draw):
    """
    Sinh sequence accuracy CÓ ít nhất 3 giá trị liên tiếp > 0.80.

    Thuật toán: tạo prefix ngẫu nhiên (không trigger), sau đó chèn đúng 3+
    giá trị > 0.80 liên tiếp, rồi thêm suffix ngẫu nhiên.

    Returns:
        list[float]: Sequence accuracy values trigger cảnh báo POSSIBLE_LEAKAGE.
    """
    # Prefix: các giá trị ngẫu nhiên (có thể rỗng)
    prefix_len = draw(st.integers(min_value=0, max_value=10))
    prefix = []
    for _ in range(prefix_len):
        prefix.append(draw(non_triggering_accuracy))

    # Phần liên tiếp > 0.80: ít nhất 3 giá trị
    consecutive_count = draw(st.integers(min_value=3, max_value=8))
    consecutive_high_values = [draw(triggering_accuracy) for _ in range(consecutive_count)]

    # Suffix: các giá trị ngẫu nhiên (có thể rỗng)
    suffix_len = draw(st.integers(min_value=0, max_value=10))
    suffix = []
    for _ in range(suffix_len):
        suffix.append(draw(accuracy_value_strategy))

    sequence = prefix + consecutive_high_values + suffix
    return sequence


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestAccuracyAnomalyNoTrigger:
    """
    Strategy 1: Sequence KHÔNG có 3 giá trị liên tiếp > 0.80 → verify None.

    Feature: train-backtest-verification, Property 12: Accuracy anomaly detection

    **Validates: Requirements 4.4**
    """

    @given(
        symbol=symbol_strategy,
        accuracy_history=sequence_without_3_consecutive_high(),
    )
    @settings(max_examples=100)
    def test_no_consecutive_high_returns_none(self, symbol, accuracy_history):
        """
        Property: Bất kỳ sequence accuracy nào KHÔNG có 3 giá trị liên tiếp > 0.80
        → check_accuracy_anomaly() phải trả về None.

        Ngưỡng STRICT > 0.80 (giá trị đúng bằng 0.80 KHÔNG trigger).

        **Validates: Requirements 4.4**
        """
        detector = DataLeakageDetector()

        result = detector.check_accuracy_anomaly(
            symbol=symbol,
            accuracy_history=accuracy_history,
        )

        assert result is None, (
            f"check_accuracy_anomaly() trả về violation cho sequence "
            f"KHÔNG có 3 giá trị liên tiếp > 0.80.\n"
            f"Symbol: {symbol}\n"
            f"Sequence: {accuracy_history}\n"
            f"Result: {result}"
        )


class TestAccuracyAnomalyTrigger:
    """
    Strategy 2: Sequence CÓ ít nhất 3 giá trị liên tiếp > 0.80 → verify violation.

    Feature: train-backtest-verification, Property 12: Accuracy anomaly detection

    **Validates: Requirements 4.4**
    """

    @given(
        symbol=symbol_strategy,
        accuracy_history=sequence_with_3_consecutive_high(),
    )
    @settings(max_examples=100)
    def test_consecutive_high_returns_violation(self, symbol, accuracy_history):
        """
        Property: Bất kỳ sequence accuracy nào CÓ ít nhất 3 giá trị liên tiếp > 0.80
        → check_accuracy_anomaly() phải trả về LeakageViolation với type POSSIBLE_LEAKAGE.

        **Validates: Requirements 4.4**
        """
        detector = DataLeakageDetector()

        result = detector.check_accuracy_anomaly(
            symbol=symbol,
            accuracy_history=accuracy_history,
        )

        # Phải trả về violation
        assert result is not None, (
            f"check_accuracy_anomaly() trả về None cho sequence "
            f"CÓ ít nhất 3 giá trị liên tiếp > 0.80.\n"
            f"Symbol: {symbol}\n"
            f"Sequence: {accuracy_history}"
        )

        # Violation type phải là POSSIBLE_LEAKAGE
        assert result.violation_type == ViolationType.POSSIBLE_LEAKAGE, (
            f"Expected violation_type=POSSIBLE_LEAKAGE, "
            f"got {result.violation_type}.\n"
            f"Symbol: {symbol}\n"
            f"Sequence: {accuracy_history}"
        )

        # Symbol phải đúng
        assert result.symbol == symbol, (
            f"Expected symbol='{symbol}', got '{result.symbol}'"
        )

        # Details phải chứa thông tin cần thiết
        assert "avg_accuracy" in result.details, (
            "Violation details thiếu trường 'avg_accuracy'"
        )
        assert "consecutive_count" in result.details, (
            "Violation details thiếu trường 'consecutive_count'"
        )
        assert result.details["consecutive_count"] >= 3, (
            f"consecutive_count phải >= 3, got {result.details['consecutive_count']}"
        )


# ---------------------------------------------------------------------------
# Property 11: Label generation range compliance
# ---------------------------------------------------------------------------


@st.composite
def valid_label_range(draw):
    """
    Sinh (row_index, actual_data_range, horizon) với actual_range nằm trong [i, i+H].

    Đảm bảo actual_start >= row_index và actual_end <= row_index + horizon.

    Returns:
        tuple: (row_index, actual_data_range, horizon)
    """
    row_index = draw(st.integers(min_value=0, max_value=10000))
    horizon = draw(st.integers(min_value=1, max_value=100))

    allowed_start = row_index
    allowed_end = row_index + horizon

    # Sinh actual_start trong [allowed_start, allowed_end]
    actual_start = draw(st.integers(min_value=allowed_start, max_value=allowed_end))

    # Sinh actual_end trong [actual_start, allowed_end]
    actual_end = draw(st.integers(min_value=actual_start, max_value=allowed_end))

    return row_index, (actual_start, actual_end), horizon


@st.composite
def invalid_label_range(draw):
    """
    Sinh (row_index, actual_data_range, horizon) với actual_range vượt ngoài [i, i+H].

    Đảm bảo actual_start < row_index HOẶC actual_end > row_index + horizon.

    Returns:
        tuple: (row_index, actual_data_range, horizon)
    """
    horizon = draw(st.integers(min_value=1, max_value=100))

    # Chọn loại vi phạm: trước allowed_start, sau allowed_end, hoặc cả hai
    violation_type = draw(st.sampled_from(["before", "after", "both"]))

    if violation_type == "before":
        # Cần row_index >= 1 để có thể sinh actual_start < row_index
        row_index = draw(st.integers(min_value=1, max_value=10000))
        allowed_start = row_index
        allowed_end = row_index + horizon

        # actual_start < allowed_start (truy cập dữ liệu ngoài phạm vi bên trái)
        actual_start = draw(
            st.integers(min_value=0, max_value=allowed_start - 1)
        )
        # actual_end có thể hợp lệ hoặc không
        actual_end = draw(st.integers(min_value=actual_start, max_value=allowed_end))
    elif violation_type == "after":
        # Bất kỳ row_index nào đều có thể có actual_end > allowed_end
        row_index = draw(st.integers(min_value=0, max_value=10000))
        allowed_start = row_index
        allowed_end = row_index + horizon

        # actual_end > allowed_end (truy cập dữ liệu tương lai ngoài horizon)
        actual_start = draw(st.integers(min_value=allowed_start, max_value=allowed_end))
        actual_end = draw(
            st.integers(min_value=allowed_end + 1, max_value=allowed_end + 100)
        )
    else:  # both
        # Cần row_index >= 1 để có thể sinh actual_start < row_index
        row_index = draw(st.integers(min_value=1, max_value=10000))
        allowed_start = row_index
        allowed_end = row_index + horizon

        # Cả hai đều vi phạm
        actual_start = draw(
            st.integers(min_value=0, max_value=allowed_start - 1)
        )
        actual_end = draw(
            st.integers(min_value=allowed_end + 1, max_value=allowed_end + 100)
        )

    return row_index, (actual_start, actual_end), horizon


class TestProperty11LabelRangeNoViolation:
    """
    Strategy 1: actual_data_range nằm trong [i, i+H] → verify None (no violation).

    Feature: train-backtest-verification, Property 11: Label generation range compliance

    **Validates: Requirements 4.3**
    """

    @given(data=valid_label_range())
    @settings(max_examples=100)
    def test_valid_range_returns_no_violation(self, data):
        """
        Property: Khi actual_data_range nằm hoàn toàn trong phạm vi cho phép
        [row_index, row_index + horizon], check_label_range() phải trả về None.

        **Validates: Requirements 4.3**
        """
        row_index, actual_data_range, horizon = data

        # Tạo detector mới cho mỗi test case (tránh accumulate violations)
        detector = DataLeakageDetector()

        result = detector.check_label_range(
            row_index=row_index,
            actual_data_range=actual_data_range,
            horizon=horizon,
        )

        # Không phát hiện violation khi range hợp lệ
        assert result is None, (
            f"check_label_range() trả về violation khi actual_range "
            f"{actual_data_range} nằm trong phạm vi cho phép "
            f"[{row_index}, {row_index + horizon}].\n"
            f"  row_index={row_index}, horizon={horizon}"
        )


class TestProperty11LabelRangeViolation:
    """
    Strategy 2: actual_data_range vượt ngoài [i, i+H] → verify LABEL_LOOK_AHEAD violation.

    Feature: train-backtest-verification, Property 11: Label generation range compliance

    **Validates: Requirements 4.3**
    """

    @given(data=invalid_label_range())
    @settings(max_examples=100)
    def test_invalid_range_returns_label_look_ahead_violation(self, data):
        """
        Property: Khi actual_data_range vượt ngoài phạm vi cho phép
        [row_index, row_index + horizon], check_label_range() phải trả về
        LeakageViolation với violation_type == LABEL_LOOK_AHEAD.

        **Validates: Requirements 4.3**
        """
        row_index, actual_data_range, horizon = data

        # Tạo detector mới cho mỗi test case
        detector = DataLeakageDetector()

        result = detector.check_label_range(
            row_index=row_index,
            actual_data_range=actual_data_range,
            horizon=horizon,
        )

        # Phải phát hiện violation
        assert result is not None, (
            f"check_label_range() không phát hiện violation khi actual_range "
            f"{actual_data_range} vượt ngoài phạm vi cho phép "
            f"[{row_index}, {row_index + horizon}].\n"
            f"  row_index={row_index}, horizon={horizon}"
        )

        # Violation type phải là LABEL_LOOK_AHEAD
        assert result.violation_type == ViolationType.LABEL_LOOK_AHEAD, (
            f"Expected violation_type=LABEL_LOOK_AHEAD, "
            f"got '{result.violation_type.value}'.\n"
            f"  row_index={row_index}, actual_range={actual_data_range}, "
            f"horizon={horizon}"
        )

        # Details phải chứa thông tin chính xác
        assert result.details["row_index"] == row_index, (
            f"details['row_index'] = {result.details['row_index']}, "
            f"expected {row_index}"
        )
        assert result.details["horizon"] == horizon, (
            f"details['horizon'] = {result.details['horizon']}, "
            f"expected {horizon}"
        )
        assert result.details["allowed_range"] == [row_index, row_index + horizon], (
            f"details['allowed_range'] = {result.details['allowed_range']}, "
            f"expected [{row_index}, {row_index + horizon}]"
        )
        assert result.details["actual_range"] == list(actual_data_range), (
            f"details['actual_range'] = {result.details['actual_range']}, "
            f"expected {list(actual_data_range)}"
        )


# ---------------------------------------------------------------------------
# Property 10: Chronological split ordering invariant
# ---------------------------------------------------------------------------

import numpy as np


# ---------------------------------------------------------------------------
# Strategies cho Property 10
# ---------------------------------------------------------------------------


@st.composite
def valid_chronological_split(draw):
    """
    Sinh 3 mảng indices thoả mãn:
    - max(train) < min(val) < min(test)
    - Không có index nào xuất hiện trong nhiều hơn một split

    Returns:
        tuple: (train_indices, val_indices, test_indices) dưới dạng np.ndarray
    """
    # Tổng số data points (ít nhất 3 để mỗi split có ít nhất 1 phần tử)
    total_size = draw(st.integers(min_value=3, max_value=500))

    # Chia tỷ lệ cho 3 splits, đảm bảo mỗi split ít nhất 1 phần tử
    train_size = draw(st.integers(min_value=1, max_value=total_size - 2))
    remaining = total_size - train_size
    val_size = draw(st.integers(min_value=1, max_value=remaining - 1))
    test_size = remaining - val_size

    # Tạo indices liên tiếp không overlap, bắt đầu từ offset ngẫu nhiên
    start_offset = draw(st.integers(min_value=0, max_value=100))

    train_indices = np.arange(start_offset, start_offset + train_size)
    val_indices = np.arange(start_offset + train_size, start_offset + train_size + val_size)
    test_indices = np.arange(
        start_offset + train_size + val_size,
        start_offset + train_size + val_size + test_size,
    )

    # Shuffle trong mỗi split (thứ tự bên trong split không quan trọng)
    if draw(st.booleans()):
        rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**32 - 1)))
        rng.shuffle(train_indices)
    if draw(st.booleans()):
        rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**32 - 1)))
        rng.shuffle(val_indices)
    if draw(st.booleans()):
        rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**32 - 1)))
        rng.shuffle(test_indices)

    return train_indices, val_indices, test_indices


@st.composite
def split_with_ordering_violation(draw):
    """
    Sinh 3 mảng indices vi phạm ordering:
    - max(train) >= min(val) HOẶC max(val) >= min(test)

    Returns:
        tuple: (train_indices, val_indices, test_indices) dưới dạng np.ndarray
    """
    # Chọn loại vi phạm ordering
    violation_kind = draw(st.sampled_from(["train_val", "val_test", "both"]))

    train_size = draw(st.integers(min_value=2, max_value=15))
    val_size = draw(st.integers(min_value=2, max_value=15))
    test_size = draw(st.integers(min_value=2, max_value=15))

    if violation_kind == "train_val":
        # max(train) >= min(val): đặt train overlap với val range
        val_start = draw(st.integers(min_value=10, max_value=50))
        # Train kết thúc tại hoặc sau val_start
        train_start = val_start - train_size + 1 + draw(st.integers(min_value=0, max_value=5))
        train_indices = np.arange(train_start, train_start + train_size)
        val_indices = np.arange(val_start, val_start + val_size)
        # Test sau val (hợp lệ)
        test_start = val_start + val_size + draw(st.integers(min_value=1, max_value=5))
        test_indices = np.arange(test_start, test_start + test_size)
    elif violation_kind == "val_test":
        # max(val) >= min(test)
        train_indices = np.arange(0, train_size)
        val_start = train_size + draw(st.integers(min_value=1, max_value=5))
        val_indices = np.arange(val_start, val_start + val_size)
        # Test bắt đầu tại hoặc trước val kết thúc
        test_start = val_start + val_size - draw(st.integers(min_value=1, max_value=val_size))
        test_indices = np.arange(test_start, test_start + test_size)
    else:  # both
        # Cả train_val lẫn val_test vi phạm
        val_start = draw(st.integers(min_value=10, max_value=30))
        train_start = val_start - train_size + 1 + draw(st.integers(min_value=0, max_value=3))
        train_indices = np.arange(train_start, train_start + train_size)
        val_indices = np.arange(val_start, val_start + val_size)
        # Test bắt đầu trước val kết thúc
        test_start = val_start + val_size - draw(st.integers(min_value=1, max_value=val_size))
        test_indices = np.arange(test_start, test_start + test_size)

    # Đảm bảo thực sự có vi phạm ordering
    assume(
        np.max(train_indices) >= np.min(val_indices)
        or np.max(val_indices) >= np.min(test_indices)
    )

    # Loại bỏ các trường hợp có mảng rỗng
    assume(train_indices.size > 0 and val_indices.size > 0 and test_indices.size > 0)

    return train_indices, val_indices, test_indices


@st.composite
def split_with_overlap_violation(draw):
    """
    Sinh 3 mảng indices có overlap (index xuất hiện trong nhiều hơn 1 split).

    Returns:
        tuple: (train_indices, val_indices, test_indices) dưới dạng np.ndarray
    """
    # Chọn loại overlap
    overlap_pair = draw(st.sampled_from(["train_val", "val_test", "train_test"]))

    # Tạo base split hợp lệ trước
    train_size = draw(st.integers(min_value=3, max_value=20))
    val_size = draw(st.integers(min_value=3, max_value=20))
    test_size = draw(st.integers(min_value=3, max_value=20))

    base_start = draw(st.integers(min_value=0, max_value=50))
    train_indices = np.arange(base_start, base_start + train_size)
    val_indices = np.arange(base_start + train_size, base_start + train_size + val_size)
    test_indices = np.arange(
        base_start + train_size + val_size,
        base_start + train_size + val_size + test_size,
    )

    # Thêm overlap
    if overlap_pair == "train_val":
        # Copy một vài indices từ val vào train
        num_overlap = draw(st.integers(min_value=1, max_value=min(3, val_size)))
        overlap_indices = val_indices[:num_overlap]
        train_indices = np.concatenate([train_indices, overlap_indices])
    elif overlap_pair == "val_test":
        # Copy một vài indices từ test vào val
        num_overlap = draw(st.integers(min_value=1, max_value=min(3, test_size)))
        overlap_indices = test_indices[:num_overlap]
        val_indices = np.concatenate([val_indices, overlap_indices])
    else:  # train_test
        # Copy một vài indices từ test vào train
        num_overlap = draw(st.integers(min_value=1, max_value=min(3, test_size)))
        overlap_indices = test_indices[:num_overlap]
        train_indices = np.concatenate([train_indices, overlap_indices])

    # Xác nhận overlap tồn tại
    train_set = set(train_indices.tolist())
    val_set = set(val_indices.tolist())
    test_set = set(test_indices.tolist())
    total_overlap = (train_set & val_set) | (val_set & test_set) | (train_set & test_set)
    assume(len(total_overlap) > 0)

    return train_indices, val_indices, test_indices


# ---------------------------------------------------------------------------
# Property 10 Tests
# ---------------------------------------------------------------------------


class TestChronologicalSplitValidInput:
    """
    Strategy 1: Splits hợp lệ (properly ordered, non-overlapping) → verify returns None.

    Feature: train-backtest-verification, Property 10: Chronological split ordering invariant

    **Validates: Requirements 4.2**
    """

    @given(data=valid_chronological_split())
    @settings(max_examples=100)
    def test_valid_split_returns_none(self, data):
        """
        Property: Với bất kỳ split hợp lệ (max(train) < min(val) < min(test), no overlap),
        check_chronological_split() phải trả về None (không có vi phạm).

        **Validates: Requirements 4.2**
        """
        train_indices, val_indices, test_indices = data

        # Xác nhận preconditions (split thực sự hợp lệ)
        assert np.max(train_indices) < np.min(val_indices), "Precondition failed: train/val ordering"
        assert np.max(val_indices) < np.min(test_indices), "Precondition failed: val/test ordering"

        train_set = set(train_indices.tolist())
        val_set = set(val_indices.tolist())
        test_set = set(test_indices.tolist())
        assert len(train_set & val_set) == 0, "Precondition failed: train/val overlap"
        assert len(val_set & test_set) == 0, "Precondition failed: val/test overlap"
        assert len(train_set & test_set) == 0, "Precondition failed: train/test overlap"

        # Gọi method
        detector = DataLeakageDetector()
        result = detector.check_chronological_split(train_indices, val_indices, test_indices)

        # Phải trả về None (không có vi phạm)
        assert result is None, (
            f"check_chronological_split() trả về vi phạm cho split hợp lệ.\n"
            f"train: [{np.min(train_indices)}..{np.max(train_indices)}], "
            f"val: [{np.min(val_indices)}..{np.max(val_indices)}], "
            f"test: [{np.min(test_indices)}..{np.max(test_indices)}]"
        )


class TestChronologicalSplitOrderingViolation:
    """
    Strategy 2: Splits vi phạm ordering hoặc overlap → verify returns violation.

    Feature: train-backtest-verification, Property 10: Chronological split ordering invariant

    **Validates: Requirements 4.2**
    """

    @given(data=split_with_ordering_violation())
    @settings(max_examples=100)
    def test_ordering_violation_returns_violation(self, data):
        """
        Property: Với bất kỳ split vi phạm ordering (max(train) >= min(val)
        hoặc max(val) >= min(test)), check_chronological_split() phải trả về
        LeakageViolation với violation_type=SPLIT_OVERLAP.

        **Validates: Requirements 4.2**
        """
        train_indices, val_indices, test_indices = data

        detector = DataLeakageDetector()
        result = detector.check_chronological_split(train_indices, val_indices, test_indices)

        # Phải trả về vi phạm
        assert result is not None, (
            f"check_chronological_split() trả về None cho split có ordering violation.\n"
            f"max(train)={np.max(train_indices)}, min(val)={np.min(val_indices)}, "
            f"max(val)={np.max(val_indices)}, min(test)={np.min(test_indices)}"
        )

        # Loại vi phạm phải đúng
        assert result.violation_type == ViolationType.SPLIT_OVERLAP, (
            f"Expected violation_type=SPLIT_OVERLAP, got {result.violation_type}"
        )

        # Details phải chứa thông tin split
        assert "max_train" in result.details
        assert "min_val" in result.details
        assert "split_pairs" in result.details

    @given(data=split_with_overlap_violation())
    @settings(max_examples=100)
    def test_overlap_violation_returns_violation(self, data):
        """
        Property: Với bất kỳ split có overlap (index xuất hiện trong nhiều hơn 1 split),
        check_chronological_split() phải trả về LeakageViolation.

        **Validates: Requirements 4.2**
        """
        train_indices, val_indices, test_indices = data

        detector = DataLeakageDetector()
        result = detector.check_chronological_split(train_indices, val_indices, test_indices)

        # Phải trả về vi phạm
        assert result is not None, (
            f"check_chronological_split() trả về None cho split có overlap.\n"
            f"train: {sorted(train_indices.tolist())}\n"
            f"val: {sorted(val_indices.tolist())}\n"
            f"test: {sorted(test_indices.tolist())}"
        )

        # Loại vi phạm phải đúng
        assert result.violation_type == ViolationType.SPLIT_OVERLAP, (
            f"Expected violation_type=SPLIT_OVERLAP, got {result.violation_type}"
        )

        # Details phải chứa overlapping_indices_count > 0
        assert result.details.get("overlapping_indices_count", 0) > 0, (
            f"Expected overlapping_indices_count > 0, got {result.details}"
        )

        # split_pairs phải có ít nhất 1 cặp bị overlap
        assert len(result.details.get("split_pairs", [])) > 0, (
            f"Expected split_pairs non-empty, got {result.details.get('split_pairs')}"
        )
