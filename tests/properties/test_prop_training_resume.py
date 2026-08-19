# -*- coding: utf-8 -*-
"""
Property-based tests cho training resume - đảm bảo resume chỉ train pending symbols.

# Feature: stock-trading-platform-refactor, Property 10: Training resume skips completed symbols

**Validates: Requirements 4.10**

Properties:
1. resume_training_symbols trả về đúng danh sách pending_symbols từ checkpoint
2. resume_training_symbols KHÔNG bao gồm bất kỳ symbol nào trong completed_symbols
3. Thứ tự pending_symbols được bảo toàn
"""

from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from engine.workers.training_checkpoint import resume_training_symbols
from models.training_models import SessionCheckpoint, TrainingPhase


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Tạo symbol hợp lệ (3 ký tự uppercase) giống mã chứng khoán VN
symbol_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu",)),
    min_size=2,
    max_size=5,
).filter(lambda s: len(s) >= 2)

# Phase training ngẫu nhiên
phase_strategy = st.sampled_from(list(TrainingPhase))


@st.composite
def session_checkpoint_strategy(draw):
    """
    Sinh SessionCheckpoint ngẫu nhiên với completed_symbols và pending_symbols
    không trùng nhau (disjoint).

    Đảm bảo:
    - completed_symbols và pending_symbols không có phần tử chung
    - Mỗi list có ít nhất 0 phần tử, tối đa 20
    """
    # Sinh danh sách symbols duy nhất
    all_symbols = draw(st.lists(
        symbol_strategy,
        min_size=1,
        max_size=30,
        unique=True,
    ))

    # Chia thành completed và pending (không overlap)
    split_point = draw(st.integers(min_value=0, max_value=len(all_symbols)))
    completed_symbols = all_symbols[:split_point]
    pending_symbols = all_symbols[split_point:]

    phase = draw(phase_strategy)
    cycle_number = draw(st.integers(min_value=1, max_value=100))
    session_id = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=4,
        max_size=16,
    ))

    checkpoint = SessionCheckpoint(
        session_id=session_id,
        phase=phase,
        cycle_number=cycle_number,
        completed_symbols=completed_symbols,
        pending_symbols=pending_symbols,
        current_symbol=None,
        current_epoch=0,
        total_epochs=draw(st.integers(min_value=1, max_value=50)),
        model_path=None,
        optimizer_state_path=None,
        created_at=datetime(2024, 6, 15, 12, 0, 0),
        metadata={},
    )

    return checkpoint


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestTrainingResumeSkipsCompleted:
    """Property 10: Training resume skips completed symbols."""

    @given(checkpoint=session_checkpoint_strategy())
    @settings(max_examples=100)
    def test_resume_returns_exactly_pending_symbols(self, checkpoint):
        """
        Property: resume_training_symbols trả về đúng danh sách pending_symbols
        từ checkpoint — không thừa, không thiếu.

        # Feature: stock-trading-platform-refactor, Property 10: Training resume skips completed symbols
        **Validates: Requirements 4.10**
        """
        result = resume_training_symbols(checkpoint)

        assert result == checkpoint.pending_symbols, (
            f"resume_training_symbols phải trả về đúng pending_symbols.\n"
            f"Expected: {checkpoint.pending_symbols}\n"
            f"Got: {result}"
        )

    @given(checkpoint=session_checkpoint_strategy())
    @settings(max_examples=100)
    def test_resume_excludes_all_completed_symbols(self, checkpoint):
        """
        Property: resume_training_symbols KHÔNG chứa bất kỳ symbol nào
        từ completed_symbols — completed symbols phải bị skip hoàn toàn.

        # Feature: stock-trading-platform-refactor, Property 10: Training resume skips completed symbols
        **Validates: Requirements 4.10**
        """
        result = resume_training_symbols(checkpoint)
        result_set = set(result)
        completed_set = set(checkpoint.completed_symbols)

        # Giao giữa result và completed phải rỗng
        overlap = result_set & completed_set

        assert len(overlap) == 0, (
            f"resume_training_symbols chứa completed symbols!\n"
            f"Completed: {checkpoint.completed_symbols}\n"
            f"Result: {result}\n"
            f"Overlap: {overlap}"
        )

    @given(checkpoint=session_checkpoint_strategy())
    @settings(max_examples=100)
    def test_resume_preserves_pending_order(self, checkpoint):
        """
        Property: resume_training_symbols bảo toàn thứ tự của pending_symbols
        từ checkpoint (không sắp xếp lại).

        # Feature: stock-trading-platform-refactor, Property 10: Training resume skips completed symbols
        **Validates: Requirements 4.10**
        """
        result = resume_training_symbols(checkpoint)

        # So sánh thứ tự chính xác
        assert result == list(checkpoint.pending_symbols), (
            f"Thứ tự pending_symbols phải được bảo toàn.\n"
            f"Expected order: {checkpoint.pending_symbols}\n"
            f"Got order: {result}"
        )

    @given(
        completed=st.lists(symbol_strategy, min_size=1, max_size=15, unique=True),
        phase=phase_strategy,
    )
    @settings(max_examples=100)
    def test_resume_with_empty_pending_returns_empty(self, completed, phase):
        """
        Property: Khi pending_symbols rỗng (tất cả đã train xong),
        resume_training_symbols trả về list rỗng.

        # Feature: stock-trading-platform-refactor, Property 10: Training resume skips completed symbols
        **Validates: Requirements 4.10**
        """
        checkpoint = SessionCheckpoint(
            session_id="test-session",
            phase=phase,
            cycle_number=1,
            completed_symbols=completed,
            pending_symbols=[],
            current_symbol=None,
            current_epoch=0,
            total_epochs=10,
            created_at=datetime(2024, 6, 15, 12, 0, 0),
            metadata={},
        )

        result = resume_training_symbols(checkpoint)

        assert result == [], (
            f"Khi pending_symbols rỗng, result phải rỗng.\n"
            f"Completed: {completed}\n"
            f"Got: {result}"
        )
