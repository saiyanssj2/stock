# -*- coding: utf-8 -*-
"""
Property-based tests cho checkpoint persistence round-trip.

# Feature: stock-trading-platform-refactor, Property 3: Checkpoint persistence round-trip

**Validates: Requirements 1.8, 4.6, 4.9**

Property:
    Với bất kỳ SessionCheckpoint hợp lệ nào, save_checkpoint() rồi load_checkpoint()
    phải trả về checkpoint tương đương với tất cả fields được bảo toàn.
"""

import tempfile
from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from engine.workers.training_checkpoint import load_checkpoint, save_checkpoint
from models.training_models import SessionCheckpoint, TrainingPhase


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Session ID: alphanumeric, 4-20 ký tự
session_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=4,
    max_size=20,
).filter(lambda s: len(s) > 0)

# TrainingPhase ngẫu nhiên
training_phase_strategy = st.sampled_from(list(TrainingPhase))

# Cycle number: 0-1000
cycle_number_strategy = st.integers(min_value=0, max_value=1000)

# Danh sách symbols: mỗi symbol 2-5 ký tự uppercase
symbol_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu",)),
    min_size=2,
    max_size=5,
).filter(lambda s: len(s) >= 2)

symbol_list_strategy = st.lists(symbol_strategy, min_size=0, max_size=15)

# Datetime cho created_at (tránh microsecond precision loss khi serialize ISO)
datetime_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
)

# Optional string fields
optional_string_strategy = st.one_of(st.none(), st.text(min_size=1, max_size=50))

# Epoch numbers
epoch_strategy = st.integers(min_value=0, max_value=500)

# Metadata: dict string → string
metadata_strategy = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=15,
    ),
    values=st.text(min_size=0, max_size=50),
    max_size=5,
)


@st.composite
def session_checkpoint_strategy(draw):
    """
    Sinh SessionCheckpoint ngẫu nhiên với tất cả fields hợp lệ.

    Đảm bảo:
    - session_id là alphanumeric
    - phase là TrainingPhase hợp lệ
    - completed_symbols và pending_symbols là list string
    - current_symbol là optional string
    - current_epoch, total_epochs >= 0
    - model_path, optimizer_state_path là optional string
    - created_at là datetime hợp lệ
    - metadata là dict[str, str]
    """
    session_id = draw(session_id_strategy)
    phase = draw(training_phase_strategy)
    cycle_number = draw(cycle_number_strategy)
    completed_symbols = draw(symbol_list_strategy)
    pending_symbols = draw(symbol_list_strategy)
    current_symbol = draw(optional_string_strategy)
    current_epoch = draw(epoch_strategy)
    total_epochs = draw(epoch_strategy)
    model_path = draw(optional_string_strategy)
    optimizer_state_path = draw(optional_string_strategy)
    created_at = draw(datetime_strategy)
    metadata = draw(metadata_strategy)

    return SessionCheckpoint(
        session_id=session_id,
        phase=phase,
        cycle_number=cycle_number,
        completed_symbols=completed_symbols,
        pending_symbols=pending_symbols,
        current_symbol=current_symbol,
        current_epoch=current_epoch,
        total_epochs=total_epochs,
        model_path=model_path,
        optimizer_state_path=optimizer_state_path,
        created_at=created_at,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    """Property 3: Checkpoint persistence round-trip."""

    @given(checkpoint=session_checkpoint_strategy())
    @settings(max_examples=100)
    def test_save_then_load_preserves_all_fields(self, checkpoint: SessionCheckpoint):
        """
        Property: Với bất kỳ SessionCheckpoint hợp lệ nào,
        save_checkpoint() → load_checkpoint() phải trả về checkpoint
        tương đương với tất cả fields được bảo toàn.

        # Feature: stock-trading-platform-refactor, Property 3: Checkpoint persistence round-trip
        **Validates: Requirements 1.8, 4.6, 4.9**
        """
        # Dùng temp directory để tránh side-effects giữa các test runs
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save checkpoint
            save_checkpoint(checkpoint, checkpoint_dir=tmp_dir)

            # Load checkpoint
            loaded = load_checkpoint(checkpoint.session_id, checkpoint_dir=tmp_dir)

            # Assert loaded thành công
            assert loaded is not None, (
                f"load_checkpoint trả về None cho session_id='{checkpoint.session_id}'"
            )

            # Assert tất cả fields được bảo toàn
            assert loaded.session_id == checkpoint.session_id, (
                f"session_id mismatch: {loaded.session_id!r} != {checkpoint.session_id!r}"
            )
            assert loaded.phase == checkpoint.phase, (
                f"phase mismatch: {loaded.phase} != {checkpoint.phase}"
            )
            assert loaded.cycle_number == checkpoint.cycle_number, (
                f"cycle_number mismatch: {loaded.cycle_number} != {checkpoint.cycle_number}"
            )
            assert loaded.completed_symbols == checkpoint.completed_symbols, (
                f"completed_symbols mismatch: {loaded.completed_symbols} != {checkpoint.completed_symbols}"
            )
            assert loaded.pending_symbols == checkpoint.pending_symbols, (
                f"pending_symbols mismatch: {loaded.pending_symbols} != {checkpoint.pending_symbols}"
            )
            assert loaded.current_symbol == checkpoint.current_symbol, (
                f"current_symbol mismatch: {loaded.current_symbol!r} != {checkpoint.current_symbol!r}"
            )
            assert loaded.current_epoch == checkpoint.current_epoch, (
                f"current_epoch mismatch: {loaded.current_epoch} != {checkpoint.current_epoch}"
            )
            assert loaded.total_epochs == checkpoint.total_epochs, (
                f"total_epochs mismatch: {loaded.total_epochs} != {checkpoint.total_epochs}"
            )
            assert loaded.model_path == checkpoint.model_path, (
                f"model_path mismatch: {loaded.model_path!r} != {checkpoint.model_path!r}"
            )
            assert loaded.optimizer_state_path == checkpoint.optimizer_state_path, (
                f"optimizer_state_path mismatch: "
                f"{loaded.optimizer_state_path!r} != {checkpoint.optimizer_state_path!r}"
            )
            # Datetime: so sánh đến microsecond (ISO format bảo toàn microsecond)
            assert loaded.created_at == checkpoint.created_at, (
                f"created_at mismatch: {loaded.created_at} != {checkpoint.created_at}"
            )
            assert loaded.metadata == checkpoint.metadata, (
                f"metadata mismatch: {loaded.metadata} != {checkpoint.metadata}"
            )
