# -*- coding: utf-8 -*-
"""
Phase transition logic cho Auto-Learner.

Kiểm tra điều kiện chuyển phase training:
- Phase C → Phase B: Sharpe ratio beat >= threshold strategies × N consecutive cycles
- Phase B → Phase A: Validation loss stable × N consecutive cycles

References: Req 3.6, 4.4, 4.5
"""

from typing import List, Optional

from models.data_models import CycleResult
from models.training_models import TrainingPhase


def check_phase_transition(
    phase: TrainingPhase, cycle_results: List[CycleResult]
) -> Optional[TrainingPhase]:
    """
    Kiểm tra xem có nên chuyển phase hay không dựa trên kết quả cycle history.

    Args:
        phase: Phase hiện tại (PHASE_C, PHASE_B, PHASE_A)
        cycle_results: Danh sách kết quả các cycle đã chạy (theo thứ tự thời gian)

    Returns:
        Phase mới nếu đủ điều kiện chuyển, None nếu không chuyển.
    """
    if phase == TrainingPhase.PHASE_C:
        if evaluate_phase_c(cycle_results):
            return TrainingPhase.PHASE_B
    elif phase == TrainingPhase.PHASE_B:
        if evaluate_phase_b(cycle_results):
            return TrainingPhase.PHASE_A
    # Phase A không chuyển tiếp nữa
    return None


def evaluate_phase_c(
    cycle_results: List[CycleResult],
    threshold: int = 2,
    required_consecutive: int = 3,
) -> bool:
    """
    Kiểm tra điều kiện chuyển Phase C → Phase B.

    Điều kiện: N cycle liên tiếp gần nhất đều có strategies_beaten >= threshold.

    Args:
        cycle_results: Danh sách kết quả cycle (theo thứ tự thời gian)
        threshold: Số benchmark strategies cần beat (mặc định 2 trên 4)
        required_consecutive: Số cycle liên tiếp cần đạt (mặc định 3)

    Returns:
        True nếu đủ điều kiện chuyển phase, False nếu không.
    """
    if len(cycle_results) < required_consecutive:
        return False

    # Lấy N cycle gần nhất
    recent_cycles = cycle_results[-required_consecutive:]

    return all(
        cycle.strategies_beaten >= threshold for cycle in recent_cycles
    )


def evaluate_phase_b(
    cycle_results: List[CycleResult],
    required_consecutive: int = 5,
    stability_threshold: float = 0.01,
) -> bool:
    """
    Kiểm tra điều kiện chuyển Phase B → Phase A.

    Điều kiện: N cycle liên tiếp gần nhất có validation loss ổn định
    (variance < stability_threshold).

    Args:
        cycle_results: Danh sách kết quả cycle (theo thứ tự thời gian)
        required_consecutive: Số cycle liên tiếp cần ổn định (mặc định 5)
        stability_threshold: Ngưỡng variance tối đa cho "ổn định" (mặc định 0.01)

    Returns:
        True nếu đủ điều kiện chuyển phase, False nếu không.
    """
    if len(cycle_results) < required_consecutive:
        return False

    # Lấy N cycle gần nhất
    recent_cycles = cycle_results[-required_consecutive:]
    losses = [cycle.validation_loss for cycle in recent_cycles]

    return is_loss_stable(losses, stability_threshold)


def is_loss_stable(losses: List[float], threshold: float = 0.01) -> bool:
    """
    Kiểm tra chuỗi loss có "ổn định" hay không (variance thấp).

    Args:
        losses: Danh sách giá trị validation loss
        threshold: Ngưỡng variance tối đa (mặc định 0.01)

    Returns:
        True nếu variance < threshold, False nếu không.
    """
    if not losses:
        return False

    n = len(losses)
    if n < 2:
        # Chỉ 1 giá trị → variance = 0 → stable
        return True

    mean = sum(losses) / n
    variance = sum((x - mean) ** 2 for x in losses) / n

    return variance < threshold
