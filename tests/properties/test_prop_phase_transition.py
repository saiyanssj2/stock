# -*- coding: utf-8 -*-
"""
Property-based tests cho Phase Transition logic.

# Feature: stock-trading-platform-refactor, Property 8: Phase C to B transition criteria
# Feature: stock-trading-platform-refactor, Property 9: Phase B to A transition criteria

**Validates: Requirements 3.6, 4.4, 4.5**

Properties:
8. Phase C → B: transition iff strategies_beaten >= 2 cho 3 cycle liên tiếp gần nhất
9. Phase B → A: transition iff validation loss stable (variance < threshold) cho 5 cycle liên tiếp gần nhất
"""

from datetime import datetime, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from engine.workers.phase_transition import evaluate_phase_b, evaluate_phase_c
from models.data_models import CycleResult


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Threshold mặc định cho Phase C: beat >= 2 strategies
PHASE_C_THRESHOLD = 2
# Số cycle liên tiếp cần đạt cho Phase C
PHASE_C_REQUIRED_CONSECUTIVE = 3
# Số cycle liên tiếp cần đạt cho Phase B
PHASE_B_REQUIRED_CONSECUTIVE = 5
# Ngưỡng variance cho Phase B
PHASE_B_STABILITY_THRESHOLD = 0.01


@st.composite
def cycle_result_strategy(
    draw,
    strategies_beaten=None,
    validation_loss=None,
):
    """
    Sinh CycleResult ngẫu nhiên với strategies_beaten và validation_loss tuỳ chọn.

    Parameters
    ----------
    strategies_beaten : int hoặc None
        Số strategies đã beat. Nếu None sẽ random [0, 4].
    validation_loss : float hoặc None
        Validation loss. Nếu None sẽ random.
    """
    cycle_number = draw(st.integers(min_value=1, max_value=100))
    phase = draw(st.sampled_from(["phase_c", "phase_b", "phase_a"]))
    sharpe_ratio = draw(st.floats(min_value=-2.0, max_value=5.0, allow_nan=False))
    win_rate = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    total_return = draw(st.floats(min_value=-50.0, max_value=200.0, allow_nan=False))

    if strategies_beaten is None:
        strategies_beaten = draw(st.integers(min_value=0, max_value=4))
    if validation_loss is None:
        validation_loss = draw(st.floats(
            min_value=0.001, max_value=5.0, allow_nan=False, allow_infinity=False
        ))

    is_improving = draw(st.booleans())
    duration_seconds = draw(st.floats(min_value=1.0, max_value=3600.0, allow_nan=False))

    return CycleResult(
        cycle_number=cycle_number,
        phase=phase,
        sharpe_ratio=sharpe_ratio,
        win_rate=win_rate,
        total_return=total_return,
        strategies_beaten=strategies_beaten,
        validation_loss=validation_loss,
        is_improving=is_improving,
        timestamp=datetime(2024, 6, 15, 12, 0, 0),
        duration_seconds=duration_seconds,
    )


@st.composite
def cycle_results_meeting_phase_c(draw, num_prefix=None):
    """
    Sinh danh sách CycleResult trong đó 3 cycle cuối đều có strategies_beaten >= 2.
    Prefix có thể có bất kỳ giá trị nào.
    """
    if num_prefix is None:
        num_prefix = draw(st.integers(min_value=0, max_value=10))

    # Prefix cycles: bất kỳ strategies_beaten nào
    prefix = [draw(cycle_result_strategy()) for _ in range(num_prefix)]

    # 3 cycle cuối đều beat >= threshold
    beaten_value = draw(st.integers(min_value=PHASE_C_THRESHOLD, max_value=4))
    tail = [
        draw(cycle_result_strategy(strategies_beaten=beaten_value))
        for _ in range(PHASE_C_REQUIRED_CONSECUTIVE)
    ]

    return prefix + tail


@st.composite
def cycle_results_not_meeting_phase_c(draw):
    """
    Sinh danh sách CycleResult KHÔNG đủ điều kiện Phase C→B.
    Đảm bảo ít nhất 1 trong 3 cycle cuối có strategies_beaten < 2.
    """
    total_length = draw(st.integers(min_value=PHASE_C_REQUIRED_CONSECUTIVE, max_value=15))

    # Tạo danh sách cycle ngẫu nhiên
    cycles = [draw(cycle_result_strategy()) for _ in range(total_length)]

    # Đảm bảo ít nhất 1 trong 3 cycle cuối KHÔNG đạt threshold
    fail_index = draw(st.integers(min_value=0, max_value=PHASE_C_REQUIRED_CONSECUTIVE - 1))
    fail_beaten = draw(st.integers(min_value=0, max_value=PHASE_C_THRESHOLD - 1))

    # Thay thế cycle tại vị trí fail_index (tính từ cuối) bằng cycle không đạt
    target_idx = total_length - PHASE_C_REQUIRED_CONSECUTIVE + fail_index
    cycles[target_idx] = draw(cycle_result_strategy(strategies_beaten=fail_beaten))

    return cycles


@st.composite
def cycle_results_meeting_phase_b(draw, num_prefix=None):
    """
    Sinh danh sách CycleResult trong đó 5 cycle cuối có validation loss ổn định
    (variance < 0.01).
    """
    if num_prefix is None:
        num_prefix = draw(st.integers(min_value=0, max_value=10))

    # Prefix cycles: bất kỳ validation_loss nào
    prefix = [draw(cycle_result_strategy()) for _ in range(num_prefix)]

    # 5 cycle cuối có loss ổn định: chọn base loss, thêm nhiễu nhỏ
    base_loss = draw(st.floats(min_value=0.1, max_value=3.0, allow_nan=False))

    # Variance < threshold → std < sqrt(threshold) ≈ 0.1
    # Dùng perturbation nhỏ để đảm bảo variance < 0.01
    max_perturbation = 0.05  # Đảm bảo variance nhỏ hơn 0.01
    tail = []
    for _ in range(PHASE_B_REQUIRED_CONSECUTIVE):
        perturbation = draw(st.floats(
            min_value=-max_perturbation,
            max_value=max_perturbation,
            allow_nan=False,
        ))
        loss_val = base_loss + perturbation
        # Đảm bảo loss > 0
        loss_val = max(loss_val, 0.001)
        tail.append(draw(cycle_result_strategy(validation_loss=loss_val)))

    # Xác nhận variance thực sự < threshold
    losses = [c.validation_loss for c in tail]
    mean = sum(losses) / len(losses)
    variance = sum((x - mean) ** 2 for x in losses) / len(losses)

    # Nếu variance >= threshold (do rounding), ép lại bằng constant loss
    if variance >= PHASE_B_STABILITY_THRESHOLD:
        tail = [
            draw(cycle_result_strategy(validation_loss=base_loss))
            for _ in range(PHASE_B_REQUIRED_CONSECUTIVE)
        ]

    return prefix + tail


@st.composite
def cycle_results_not_meeting_phase_b(draw):
    """
    Sinh danh sách CycleResult KHÔNG đủ điều kiện Phase B→A.
    Variance của 5 cycle cuối >= 0.01.
    """
    total_length = draw(st.integers(min_value=PHASE_B_REQUIRED_CONSECUTIVE, max_value=15))

    # Tạo base loss và tạo biến động lớn trong 5 cycle cuối
    base_loss = draw(st.floats(min_value=0.5, max_value=2.0, allow_nan=False))

    # Prefix: bất kỳ
    prefix_length = total_length - PHASE_B_REQUIRED_CONSECUTIVE
    prefix = [draw(cycle_result_strategy()) for _ in range(prefix_length)]

    # 5 cycle cuối: tạo loss có variance >= threshold
    # Dùng cách tạo 1 giá trị outlier đủ xa
    tail_losses = []
    for i in range(PHASE_B_REQUIRED_CONSECUTIVE):
        if i == 0:
            # Giá trị outlier, cách xa base_loss đủ để variance >= 0.01
            outlier_offset = draw(st.floats(min_value=0.3, max_value=2.0, allow_nan=False))
            tail_losses.append(base_loss + outlier_offset)
        else:
            tail_losses.append(base_loss)

    # Xác nhận variance >= threshold
    mean = sum(tail_losses) / len(tail_losses)
    variance = sum((x - mean) ** 2 for x in tail_losses) / len(tail_losses)

    if variance < PHASE_B_STABILITY_THRESHOLD:
        # Tăng outlier để chắc chắn variance đủ lớn
        tail_losses[0] = base_loss + 1.0

    tail = [
        draw(cycle_result_strategy(validation_loss=loss))
        for loss in tail_losses
    ]

    return prefix + tail


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestPhaseCToBTransition:
    """Property 8: Phase C to B transition criteria."""

    @given(data=cycle_results_meeting_phase_c())
    @settings(max_examples=100)
    def test_transition_when_criteria_met(self, data):
        """
        Property: Khi 3 cycle liên tiếp gần nhất đều có strategies_beaten >= 2,
        evaluate_phase_c phải trả về True.

        # Feature: stock-trading-platform-refactor, Property 8: Phase C to B transition criteria
        **Validates: Requirements 3.6, 4.4**
        """
        result = evaluate_phase_c(
            data,
            threshold=PHASE_C_THRESHOLD,
            required_consecutive=PHASE_C_REQUIRED_CONSECUTIVE,
        )

        assert result is True, (
            f"Đủ điều kiện Phase C→B (3 cycle cuối beat >= 2) nhưng evaluate_phase_c "
            f"trả về False. Strategies beaten cuối: "
            f"{[c.strategies_beaten for c in data[-PHASE_C_REQUIRED_CONSECUTIVE:]]}"
        )

    @given(data=cycle_results_not_meeting_phase_c())
    @settings(max_examples=100)
    def test_no_transition_when_criteria_not_met(self, data):
        """
        Property: Khi ít nhất 1 trong 3 cycle cuối có strategies_beaten < 2,
        evaluate_phase_c phải trả về False.

        # Feature: stock-trading-platform-refactor, Property 8: Phase C to B transition criteria
        **Validates: Requirements 3.6, 4.4**
        """
        result = evaluate_phase_c(
            data,
            threshold=PHASE_C_THRESHOLD,
            required_consecutive=PHASE_C_REQUIRED_CONSECUTIVE,
        )

        assert result is False, (
            f"KHÔNG đủ điều kiện Phase C→B nhưng evaluate_phase_c trả về True. "
            f"Strategies beaten cuối: "
            f"{[c.strategies_beaten for c in data[-PHASE_C_REQUIRED_CONSECUTIVE:]]}"
        )

    @given(
        cycles=st.lists(cycle_result_strategy(), min_size=0, max_size=2),
    )
    @settings(max_examples=50)
    def test_insufficient_cycles_never_transitions(self, cycles):
        """
        Property: Khi có ít hơn 3 cycle, evaluate_phase_c luôn trả về False
        bất kể strategies_beaten.

        # Feature: stock-trading-platform-refactor, Property 8: Phase C to B transition criteria
        **Validates: Requirements 3.6, 4.4**
        """
        result = evaluate_phase_c(
            cycles,
            threshold=PHASE_C_THRESHOLD,
            required_consecutive=PHASE_C_REQUIRED_CONSECUTIVE,
        )

        assert result is False, (
            f"Chỉ có {len(cycles)} cycle (< 3 required) nhưng evaluate_phase_c "
            f"trả về True."
        )

    @given(
        beaten=st.integers(min_value=PHASE_C_THRESHOLD, max_value=4),
        prefix_length=st.integers(min_value=0, max_value=10),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_transition_iff_all_recent_cycles_meet_threshold(
        self, beaten, prefix_length, data
    ):
        """
        Property (biconditional): evaluate_phase_c trả về True iff TẤT CẢ
        3 cycle gần nhất có strategies_beaten >= threshold.

        # Feature: stock-trading-platform-refactor, Property 8: Phase C to B transition criteria
        **Validates: Requirements 3.6, 4.4**
        """
        # Tạo prefix bất kỳ
        prefix = [
            data.draw(cycle_result_strategy())
            for _ in range(prefix_length)
        ]

        # Tạo tail với beaten cố định (đủ điều kiện)
        tail = [
            data.draw(cycle_result_strategy(strategies_beaten=beaten))
            for _ in range(PHASE_C_REQUIRED_CONSECUTIVE)
        ]

        cycles = prefix + tail
        result = evaluate_phase_c(
            cycles,
            threshold=PHASE_C_THRESHOLD,
            required_consecutive=PHASE_C_REQUIRED_CONSECUTIVE,
        )

        # Kiểm tra biconditional: tất cả 3 cycle cuối >= threshold → True
        recent = cycles[-PHASE_C_REQUIRED_CONSECUTIVE:]
        all_meet = all(c.strategies_beaten >= PHASE_C_THRESHOLD for c in recent)

        assert result == all_meet, (
            f"Biconditional violated: all_meet={all_meet}, result={result}. "
            f"Beaten values: {[c.strategies_beaten for c in recent]}"
        )


class TestPhaseBToATransition:
    """Property 9: Phase B to A transition criteria."""

    @given(data=cycle_results_meeting_phase_b())
    @settings(max_examples=100)
    def test_transition_when_loss_stable(self, data):
        """
        Property: Khi 5 cycle liên tiếp gần nhất có validation loss ổn định
        (variance < 0.01), evaluate_phase_b phải trả về True.

        # Feature: stock-trading-platform-refactor, Property 9: Phase B to A transition criteria
        **Validates: Requirements 4.5**
        """
        result = evaluate_phase_b(
            data,
            required_consecutive=PHASE_B_REQUIRED_CONSECUTIVE,
            stability_threshold=PHASE_B_STABILITY_THRESHOLD,
        )

        # Tính variance để debug
        recent = data[-PHASE_B_REQUIRED_CONSECUTIVE:]
        losses = [c.validation_loss for c in recent]
        mean = sum(losses) / len(losses)
        variance = sum((x - mean) ** 2 for x in losses) / len(losses)

        assert result is True, (
            f"Đủ điều kiện Phase B→A (loss stable, variance={variance:.6f} < "
            f"{PHASE_B_STABILITY_THRESHOLD}) nhưng evaluate_phase_b trả về False. "
            f"Losses: {losses}"
        )

    @given(data=cycle_results_not_meeting_phase_b())
    @settings(max_examples=100)
    def test_no_transition_when_loss_unstable(self, data):
        """
        Property: Khi variance validation loss của 5 cycle cuối >= 0.01,
        evaluate_phase_b phải trả về False.

        # Feature: stock-trading-platform-refactor, Property 9: Phase B to A transition criteria
        **Validates: Requirements 4.5**
        """
        result = evaluate_phase_b(
            data,
            required_consecutive=PHASE_B_REQUIRED_CONSECUTIVE,
            stability_threshold=PHASE_B_STABILITY_THRESHOLD,
        )

        # Tính variance để debug
        recent = data[-PHASE_B_REQUIRED_CONSECUTIVE:]
        losses = [c.validation_loss for c in recent]
        mean = sum(losses) / len(losses)
        variance = sum((x - mean) ** 2 for x in losses) / len(losses)

        assert result is False, (
            f"KHÔNG đủ điều kiện Phase B→A (variance={variance:.6f} >= "
            f"{PHASE_B_STABILITY_THRESHOLD}) nhưng evaluate_phase_b trả về True. "
            f"Losses: {losses}"
        )

    @given(
        cycles=st.lists(cycle_result_strategy(), min_size=0, max_size=4),
    )
    @settings(max_examples=50)
    def test_insufficient_cycles_never_transitions(self, cycles):
        """
        Property: Khi có ít hơn 5 cycle, evaluate_phase_b luôn trả về False
        bất kể validation loss.

        # Feature: stock-trading-platform-refactor, Property 9: Phase B to A transition criteria
        **Validates: Requirements 4.5**
        """
        result = evaluate_phase_b(
            cycles,
            required_consecutive=PHASE_B_REQUIRED_CONSECUTIVE,
            stability_threshold=PHASE_B_STABILITY_THRESHOLD,
        )

        assert result is False, (
            f"Chỉ có {len(cycles)} cycle (< 5 required) nhưng evaluate_phase_b "
            f"trả về True."
        )

    @given(
        base_loss=st.floats(min_value=0.1, max_value=3.0, allow_nan=False),
        prefix_length=st.integers(min_value=0, max_value=5),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_constant_loss_always_transitions(self, base_loss, prefix_length, data):
        """
        Property: Khi 5 cycle cuối có cùng một giá trị validation loss (variance = 0),
        evaluate_phase_b luôn trả về True.

        # Feature: stock-trading-platform-refactor, Property 9: Phase B to A transition criteria
        **Validates: Requirements 4.5**
        """
        # Prefix ngẫu nhiên
        prefix = [
            data.draw(cycle_result_strategy())
            for _ in range(prefix_length)
        ]

        # Tail: 5 cycle có cùng loss (variance = 0 < 0.01)
        tail = [
            data.draw(cycle_result_strategy(validation_loss=base_loss))
            for _ in range(PHASE_B_REQUIRED_CONSECUTIVE)
        ]

        cycles = prefix + tail
        result = evaluate_phase_b(
            cycles,
            required_consecutive=PHASE_B_REQUIRED_CONSECUTIVE,
            stability_threshold=PHASE_B_STABILITY_THRESHOLD,
        )

        assert result is True, (
            f"5 cycle cuối có cùng loss={base_loss} (variance=0) nhưng "
            f"evaluate_phase_b trả về False."
        )
