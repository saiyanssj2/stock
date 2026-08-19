# -*- coding: utf-8 -*-
"""
Property-based tests cho Cycle History — persistence round-trip và improvement trend detection.

# Feature: stock-trading-platform-refactor, Properties 6, 7

**Validates: Requirements 3.4, 3.5**

Property 6:
    Với bất kỳ CycleResult hợp lệ nào, persist_cycle() rồi load_all_cycles()
    phải bảo toàn toàn bộ dữ liệu gốc mà không mất mát hay hỏng.

Property 7:
    Với bất kỳ chuỗi CycleResult nào, detect_improvement_trend() trả về True
    khi và chỉ khi 3+ cycles cuối có Sharpe ratio strictly increasing.
"""

import tempfile
from datetime import datetime

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.cycle_history import (
    detect_improvement_trend,
    load_all_cycles,
    persist_cycle,
)
from models.data_models import CycleResult


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Phase: các phase training hợp lệ
phase_strategy = st.sampled_from(["PHASE_C", "PHASE_B", "PHASE_A"])

# Sharpe ratio: giá trị hữu hạn, thực tế
sharpe_strategy = st.floats(
    min_value=-5.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
)

# Win rate: 0-100%
win_rate_strategy = st.floats(
    min_value=0.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
)

# Total return: %
total_return_strategy = st.floats(
    min_value=-100.0,
    max_value=1000.0,
    allow_nan=False,
    allow_infinity=False,
)

# Strategies beaten: 0-4
strategies_beaten_strategy = st.integers(min_value=0, max_value=4)

# Validation loss: giá trị dương
validation_loss_strategy = st.floats(
    min_value=0.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
)

# Duration: giây
duration_strategy = st.floats(
    min_value=0.0,
    max_value=3600.0,
    allow_nan=False,
    allow_infinity=False,
)

# Notes: text ngắn
notes_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=0,
    max_size=50,
)

# Datetime: tránh microsecond khác biệt khi serialize (ISO format bảo toàn đến microsecond)
datetime_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
)


@st.composite
def cycle_result_strategy(draw, cycle_number=None):
    """
    Sinh CycleResult ngẫu nhiên với tất cả fields hợp lệ.

    Args:
        cycle_number: Nếu None, generate ngẫu nhiên. Nếu chỉ định, dùng giá trị đó.
    """
    if cycle_number is None:
        cycle_number = draw(st.integers(min_value=1, max_value=1000))

    phase = draw(phase_strategy)
    sharpe_ratio = draw(sharpe_strategy)
    win_rate = draw(win_rate_strategy)
    total_return = draw(total_return_strategy)
    strategies_beaten = draw(strategies_beaten_strategy)
    validation_loss = draw(validation_loss_strategy)
    is_improving = draw(st.booleans())
    timestamp = draw(datetime_strategy)
    duration_seconds = draw(duration_strategy)
    notes = draw(notes_strategy)

    return CycleResult(
        cycle_number=cycle_number,
        phase=phase,
        sharpe_ratio=sharpe_ratio,
        win_rate=win_rate,
        total_return=total_return,
        strategies_beaten=strategies_beaten,
        validation_loss=validation_loss,
        is_improving=is_improving,
        timestamp=timestamp,
        duration_seconds=duration_seconds,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Property 6: Cycle history persistence round-trip
# ---------------------------------------------------------------------------


class TestCycleHistoryPersistenceRoundTrip:
    """Property 6: Cycle history persistence round-trip."""

    @given(cycle=cycle_result_strategy())
    @settings(max_examples=100)
    def test_persist_then_load_preserves_all_fields(self, cycle: CycleResult):
        """
        Property: Với bất kỳ CycleResult hợp lệ, persist_cycle() → load_all_cycles()
        phải bảo toàn toàn bộ dữ liệu gốc.

        # Feature: stock-trading-platform-refactor, Property 6: Cycle history persistence round-trip
        **Validates: Requirements 3.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Persist cycle
            persist_cycle(cycle, history_dir=tmp_dir)

            # Load lại
            loaded_cycles = load_all_cycles(history_dir=tmp_dir)

            # Phải có đúng 1 cycle
            assert len(loaded_cycles) == 1, (
                f"Expected 1 cycle, got {len(loaded_cycles)}"
            )

            loaded = loaded_cycles[0]

            # Assert tất cả fields bảo toàn
            assert loaded.cycle_number == cycle.cycle_number, (
                f"cycle_number: {loaded.cycle_number} != {cycle.cycle_number}"
            )
            assert loaded.phase == cycle.phase, (
                f"phase: {loaded.phase!r} != {cycle.phase!r}"
            )
            assert loaded.sharpe_ratio == cycle.sharpe_ratio, (
                f"sharpe_ratio: {loaded.sharpe_ratio} != {cycle.sharpe_ratio}"
            )
            assert loaded.win_rate == cycle.win_rate, (
                f"win_rate: {loaded.win_rate} != {cycle.win_rate}"
            )
            assert loaded.total_return == cycle.total_return, (
                f"total_return: {loaded.total_return} != {cycle.total_return}"
            )
            assert loaded.strategies_beaten == cycle.strategies_beaten, (
                f"strategies_beaten: {loaded.strategies_beaten} != {cycle.strategies_beaten}"
            )
            assert loaded.validation_loss == cycle.validation_loss, (
                f"validation_loss: {loaded.validation_loss} != {cycle.validation_loss}"
            )
            assert loaded.is_improving == cycle.is_improving, (
                f"is_improving: {loaded.is_improving} != {cycle.is_improving}"
            )
            assert loaded.timestamp == cycle.timestamp, (
                f"timestamp: {loaded.timestamp} != {cycle.timestamp}"
            )
            assert loaded.duration_seconds == cycle.duration_seconds, (
                f"duration_seconds: {loaded.duration_seconds} != {cycle.duration_seconds}"
            )
            assert loaded.notes == cycle.notes, (
                f"notes: {loaded.notes!r} != {cycle.notes!r}"
            )

    @given(
        cycles=st.lists(
            cycle_result_strategy(),
            min_size=2,
            max_size=10,
        )
    )
    @settings(max_examples=50)
    def test_multiple_cycles_persist_and_load_sorted(self, cycles: list):
        """
        Property: Nhiều CycleResult persist riêng lẻ → load_all_cycles trả về
        sorted theo cycle_number, bảo toàn tất cả dữ liệu.

        # Feature: stock-trading-platform-refactor, Property 6: Cycle history persistence round-trip
        **Validates: Requirements 3.4**
        """
        # Đảm bảo cycle_number unique để mỗi file khác nhau
        seen_numbers = set()
        unique_cycles = []
        for c in cycles:
            if c.cycle_number not in seen_numbers:
                seen_numbers.add(c.cycle_number)
                unique_cycles.append(c)

        assume(len(unique_cycles) >= 2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Persist tất cả
            for c in unique_cycles:
                persist_cycle(c, history_dir=tmp_dir)

            # Load lại
            loaded_cycles = load_all_cycles(history_dir=tmp_dir)

            # Kiểm tra số lượng
            assert len(loaded_cycles) == len(unique_cycles), (
                f"Expected {len(unique_cycles)} cycles, got {len(loaded_cycles)}"
            )

            # Kiểm tra sorted theo cycle_number
            for i in range(1, len(loaded_cycles)):
                assert loaded_cycles[i].cycle_number >= loaded_cycles[i - 1].cycle_number, (
                    "Loaded cycles không sorted theo cycle_number"
                )


# ---------------------------------------------------------------------------
# Property 7: Improvement trend detection
# ---------------------------------------------------------------------------


class TestImprovementTrendDetection:
    """Property 7: Improvement trend detection."""

    @given(
        sharpe_values=st.lists(
            st.floats(min_value=-5.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=20,
        )
    )
    @settings(max_examples=100)
    def test_improving_iff_last_3_strictly_increasing(self, sharpe_values: list):
        """
        Property: detect_improvement_trend trả về True khi và chỉ khi
        3+ cycles cuối cùng có Sharpe ratio strictly increasing.

        # Feature: stock-trading-platform-refactor, Property 7: Improvement trend detection
        **Validates: Requirements 3.4, 3.5**
        """
        # Tạo CycleResult list từ sharpe_values
        cycles = []
        for i, sharpe in enumerate(sharpe_values):
            cycles.append(CycleResult(
                cycle_number=i + 1,
                phase="PHASE_C",
                sharpe_ratio=sharpe,
                win_rate=50.0,
                total_return=0.0,
                strategies_beaten=0,
                validation_loss=1.0,
                is_improving=False,
                timestamp=datetime(2024, 1, 1),
                duration_seconds=10.0,
                notes="",
            ))

        result = detect_improvement_trend(cycles, min_consecutive=3)

        # Tính expected: last 3 cycles có sharpe strictly increasing
        last_3 = sharpe_values[-3:]
        expected = all(last_3[i] < last_3[i + 1] for i in range(len(last_3) - 1))

        assert result == expected, (
            f"detect_improvement_trend={result} nhưng expected={expected}. "
            f"Last 3 sharpe values: {last_3}"
        )

    @given(
        sharpe_values=st.lists(
            st.floats(min_value=-5.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=0,
            max_size=2,
        )
    )
    @settings(max_examples=50)
    def test_fewer_than_3_cycles_returns_false(self, sharpe_values: list):
        """
        Property: Khi có ít hơn 3 cycles, detect_improvement_trend luôn trả về False.

        # Feature: stock-trading-platform-refactor, Property 7: Improvement trend detection
        **Validates: Requirements 3.4, 3.5**
        """
        cycles = []
        for i, sharpe in enumerate(sharpe_values):
            cycles.append(CycleResult(
                cycle_number=i + 1,
                phase="PHASE_C",
                sharpe_ratio=sharpe,
                win_rate=50.0,
                total_return=0.0,
                strategies_beaten=0,
                validation_loss=1.0,
                is_improving=False,
                timestamp=datetime(2024, 1, 1),
                duration_seconds=10.0,
                notes="",
            ))

        result = detect_improvement_trend(cycles, min_consecutive=3)
        assert result is False, (
            f"Expected False khi chỉ có {len(cycles)} cycles, got True"
        )

    @given(
        base=st.floats(min_value=-3.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        increments=st.lists(
            st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=5,
        ),
    )
    @settings(max_examples=50)
    def test_strictly_increasing_sequence_returns_true(self, base: float, increments: list):
        """
        Property: Chuỗi strictly increasing Sharpe (3+ cycles) luôn trả về True.

        # Feature: stock-trading-platform-refactor, Property 7: Improvement trend detection
        **Validates: Requirements 3.4, 3.5**
        """
        # Tạo chuỗi strictly increasing từ base + cumulative increments
        sharpe_values = [base]
        current = base
        for inc in increments:
            current += inc
            sharpe_values.append(current)

        # Cần ít nhất 3 giá trị
        assume(len(sharpe_values) >= 3)

        cycles = []
        for i, sharpe in enumerate(sharpe_values):
            cycles.append(CycleResult(
                cycle_number=i + 1,
                phase="PHASE_C",
                sharpe_ratio=sharpe,
                win_rate=50.0,
                total_return=0.0,
                strategies_beaten=0,
                validation_loss=1.0,
                is_improving=False,
                timestamp=datetime(2024, 1, 1),
                duration_seconds=10.0,
                notes="",
            ))

        result = detect_improvement_trend(cycles, min_consecutive=3)
        assert result is True, (
            f"Expected True cho strictly increasing sharpe: {sharpe_values}"
        )
