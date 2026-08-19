# -*- coding: utf-8 -*-
"""
Unit tests cho engine/workers/phase_transition.py.

Kiểm tra logic chuyển phase:
- Phase C → B: strategies_beaten >= threshold × N consecutive cycles
- Phase B → A: validation loss stable × N consecutive cycles
- check_phase_transition entry point

References: Req 3.6, 4.4, 4.5
"""

from datetime import datetime

import pytest

from engine.workers.phase_transition import (
    check_phase_transition,
    evaluate_phase_b,
    evaluate_phase_c,
    is_loss_stable,
)
from models.data_models import CycleResult
from models.training_models import TrainingPhase


# --- Helpers ---


def make_cycle(
    cycle_number: int = 1,
    phase: str = "phase_c",
    strategies_beaten: int = 0,
    validation_loss: float = 0.5,
    sharpe_ratio: float = 1.0,
) -> CycleResult:
    """Tạo CycleResult cho test."""
    return CycleResult(
        cycle_number=cycle_number,
        phase=phase,
        sharpe_ratio=sharpe_ratio,
        win_rate=50.0,
        total_return=5.0,
        strategies_beaten=strategies_beaten,
        validation_loss=validation_loss,
        is_improving=True,
        timestamp=datetime(2024, 1, 1),
        duration_seconds=100.0,
    )


# --- Tests cho is_loss_stable ---


class TestIsLossStable:
    """Tests cho hàm is_loss_stable."""

    def test_empty_list_returns_false(self):
        """Danh sách rỗng → không stable."""
        assert is_loss_stable([]) is False

    def test_single_value_is_stable(self):
        """Chỉ 1 giá trị → variance = 0 → stable."""
        assert is_loss_stable([0.5]) is True

    def test_identical_values_are_stable(self):
        """Tất cả giá trị giống nhau → variance = 0 → stable."""
        assert is_loss_stable([0.3, 0.3, 0.3, 0.3, 0.3]) is True

    def test_very_close_values_are_stable(self):
        """Giá trị gần nhau → variance nhỏ → stable."""
        # Variance sẽ rất nhỏ (<0.01)
        losses = [0.30, 0.31, 0.29, 0.30, 0.31]
        assert is_loss_stable(losses, threshold=0.01) is True

    def test_high_variance_is_not_stable(self):
        """Giá trị dao động mạnh → variance lớn → not stable."""
        losses = [0.1, 0.9, 0.2, 0.8, 0.3]
        assert is_loss_stable(losses, threshold=0.01) is False

    def test_custom_threshold(self):
        """Ngưỡng threshold tùy chỉnh."""
        losses = [1.0, 1.5, 1.0, 1.5, 1.0]
        # variance = 0.06 → vượt ngưỡng 0.01
        assert is_loss_stable(losses, threshold=0.01) is False
        # variance = 0.06 → dưới ngưỡng 0.1
        assert is_loss_stable(losses, threshold=0.1) is True


# --- Tests cho evaluate_phase_c ---


class TestEvaluatePhaseC:
    """Tests cho logic chuyển Phase C → Phase B."""

    def test_not_enough_cycles_returns_false(self):
        """Không đủ số cycle → không chuyển."""
        cycles = [make_cycle(strategies_beaten=3) for _ in range(2)]
        assert evaluate_phase_c(cycles, threshold=2, required_consecutive=3) is False

    def test_all_cycles_beat_threshold_returns_true(self):
        """3 cycle liên tiếp đều beat >= 2 strategies → chuyển phase."""
        cycles = [make_cycle(strategies_beaten=3) for _ in range(3)]
        assert evaluate_phase_c(cycles, threshold=2, required_consecutive=3) is True

    def test_exactly_at_threshold_returns_true(self):
        """Beat đúng 2 strategies (= threshold) → tính là đủ."""
        cycles = [make_cycle(strategies_beaten=2) for _ in range(3)]
        assert evaluate_phase_c(cycles, threshold=2, required_consecutive=3) is True

    def test_one_cycle_below_threshold_returns_false(self):
        """1 cycle trong 3 cycle gần nhất không đạt → không chuyển."""
        cycles = [
            make_cycle(strategies_beaten=3),
            make_cycle(strategies_beaten=1),  # Dưới threshold
            make_cycle(strategies_beaten=3),
        ]
        assert evaluate_phase_c(cycles, threshold=2, required_consecutive=3) is False

    def test_only_recent_cycles_matter(self):
        """Chỉ xét N cycle gần nhất, cycle cũ không ảnh hưởng."""
        cycles = [
            make_cycle(strategies_beaten=0),  # Cycle cũ, thấp
            make_cycle(strategies_beaten=0),  # Cycle cũ, thấp
            make_cycle(strategies_beaten=3),  # 3 cycle gần nhất
            make_cycle(strategies_beaten=3),
            make_cycle(strategies_beaten=2),
        ]
        assert evaluate_phase_c(cycles, threshold=2, required_consecutive=3) is True

    def test_empty_list_returns_false(self):
        """Không có cycle nào → không chuyển."""
        assert evaluate_phase_c([], threshold=2, required_consecutive=3) is False

    def test_custom_required_consecutive(self):
        """Custom required_consecutive = 5."""
        cycles = [make_cycle(strategies_beaten=3) for _ in range(5)]
        assert evaluate_phase_c(cycles, threshold=2, required_consecutive=5) is True

        cycles_short = [make_cycle(strategies_beaten=3) for _ in range(4)]
        assert evaluate_phase_c(cycles_short, threshold=2, required_consecutive=5) is False


# --- Tests cho evaluate_phase_b ---


class TestEvaluatePhaseB:
    """Tests cho logic chuyển Phase B → Phase A."""

    def test_not_enough_cycles_returns_false(self):
        """Không đủ 5 cycle → không chuyển."""
        cycles = [make_cycle(validation_loss=0.3) for _ in range(4)]
        assert evaluate_phase_b(cycles, required_consecutive=5) is False

    def test_stable_loss_returns_true(self):
        """5 cycle liên tiếp có loss ổn định → chuyển phase."""
        cycles = [make_cycle(validation_loss=0.30 + i * 0.001) for i in range(5)]
        assert evaluate_phase_b(cycles, required_consecutive=5, stability_threshold=0.01) is True

    def test_unstable_loss_returns_false(self):
        """Loss dao động mạnh → không chuyển."""
        cycles = [
            make_cycle(validation_loss=0.1),
            make_cycle(validation_loss=0.9),
            make_cycle(validation_loss=0.1),
            make_cycle(validation_loss=0.9),
            make_cycle(validation_loss=0.1),
        ]
        assert evaluate_phase_b(cycles, required_consecutive=5, stability_threshold=0.01) is False

    def test_identical_loss_is_stable(self):
        """Loss không đổi → variance = 0 → chuyển phase."""
        cycles = [make_cycle(validation_loss=0.25) for _ in range(5)]
        assert evaluate_phase_b(cycles, required_consecutive=5) is True

    def test_only_recent_cycles_matter(self):
        """Chỉ xét 5 cycle gần nhất."""
        cycles = [
            make_cycle(validation_loss=0.9),  # Cycle cũ, loss cao
            make_cycle(validation_loss=0.1),  # Cycle cũ, loss thấp
            # 5 cycle gần nhất ổn định
            make_cycle(validation_loss=0.30),
            make_cycle(validation_loss=0.31),
            make_cycle(validation_loss=0.29),
            make_cycle(validation_loss=0.30),
            make_cycle(validation_loss=0.31),
        ]
        assert evaluate_phase_b(cycles, required_consecutive=5, stability_threshold=0.01) is True

    def test_empty_list_returns_false(self):
        """Không có cycle → không chuyển."""
        assert evaluate_phase_b([], required_consecutive=5) is False

    def test_custom_stability_threshold(self):
        """Ngưỡng stability tùy chỉnh."""
        # Losses có variance ~0.06
        cycles = [
            make_cycle(validation_loss=1.0),
            make_cycle(validation_loss=1.5),
            make_cycle(validation_loss=1.0),
            make_cycle(validation_loss=1.5),
            make_cycle(validation_loss=1.0),
        ]
        # Ngưỡng 0.01 → not stable
        assert evaluate_phase_b(cycles, required_consecutive=5, stability_threshold=0.01) is False
        # Ngưỡng 0.1 → stable
        assert evaluate_phase_b(cycles, required_consecutive=5, stability_threshold=0.1) is True


# --- Tests cho check_phase_transition ---


class TestCheckPhaseTransition:
    """Tests cho entry point check_phase_transition."""

    def test_phase_c_transitions_to_b(self):
        """Phase C đủ điều kiện → chuyển sang Phase B."""
        cycles = [make_cycle(strategies_beaten=3) for _ in range(3)]
        result = check_phase_transition(TrainingPhase.PHASE_C, cycles)
        assert result == TrainingPhase.PHASE_B

    def test_phase_c_no_transition(self):
        """Phase C không đủ điều kiện → None."""
        cycles = [make_cycle(strategies_beaten=1) for _ in range(3)]
        result = check_phase_transition(TrainingPhase.PHASE_C, cycles)
        assert result is None

    def test_phase_b_transitions_to_a(self):
        """Phase B đủ điều kiện → chuyển sang Phase A."""
        cycles = [make_cycle(validation_loss=0.30) for _ in range(5)]
        result = check_phase_transition(TrainingPhase.PHASE_B, cycles)
        assert result == TrainingPhase.PHASE_A

    def test_phase_b_no_transition(self):
        """Phase B không đủ điều kiện → None."""
        cycles = [
            make_cycle(validation_loss=0.1),
            make_cycle(validation_loss=0.9),
            make_cycle(validation_loss=0.1),
            make_cycle(validation_loss=0.9),
            make_cycle(validation_loss=0.1),
        ]
        result = check_phase_transition(TrainingPhase.PHASE_B, cycles)
        assert result is None

    def test_phase_a_never_transitions(self):
        """Phase A là phase cuối → không chuyển tiếp."""
        cycles = [make_cycle(strategies_beaten=4, validation_loss=0.30) for _ in range(10)]
        result = check_phase_transition(TrainingPhase.PHASE_A, cycles)
        assert result is None

    def test_empty_cycles_no_transition(self):
        """Không có history → không chuyển phase nào."""
        assert check_phase_transition(TrainingPhase.PHASE_C, []) is None
        assert check_phase_transition(TrainingPhase.PHASE_B, []) is None
        assert check_phase_transition(TrainingPhase.PHASE_A, []) is None
