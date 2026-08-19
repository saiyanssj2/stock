# -*- coding: utf-8 -*-
"""
Property-based tests cho Training Progress Percentage.

# Feature: stock-trading-platform-refactor, Property 21: Training progress percentage

**Validates: Requirements 9.2**

Property:
    Với mọi training session có T total symbols và C symbols completed (0 ≤ C ≤ T, T > 0),
    progress_pct phải bằng (C / T) × 100.
"""

from datetime import datetime
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from models.task_models import TaskState, TaskStatus, TaskType
from models.training_models import TrainingPhase, TrainingProgress
from engine.workers.training_worker import TrainingEngineWorker


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Total symbols: ít nhất 1, tối đa 200 (phản ánh thực tế VN30 + watchlist)
total_symbols_strategy = st.integers(min_value=1, max_value=200)

# Phase ngẫu nhiên
phase_strategy = st.sampled_from(list(TrainingPhase))


@st.composite
def completed_total_strategy(draw):
    """
    Sinh cặp (completed, total) hợp lệ: 0 <= completed <= total, total > 0.
    """
    total = draw(total_symbols_strategy)
    completed = draw(st.integers(min_value=0, max_value=total))
    return completed, total


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestTrainingProgressPercentage:
    """Property 21: Training progress percentage."""

    @given(data=completed_total_strategy(), phase=phase_strategy)
    @settings(max_examples=200)
    def test_progress_pct_equals_completed_over_total_times_100(self, data, phase):
        """
        Property: progress_pct == (completed / total) × 100
        cho mọi cặp (completed, total) hợp lệ.

        Kiểm tra trực tiếp qua TrainingEngineWorker.report_progress() bằng cách
        capture giá trị progress_pct được ghi vào status file.

        # Feature: stock-trading-platform-refactor, Property 21: Training progress percentage
        **Validates: Requirements 9.2**
        """
        completed, total = data
        expected_pct = (completed / total) * 100.0

        # Thiết lập worker với completed/total đã cho
        worker = TrainingEngineWorker(task_id="test-progress-001")
        worker.phase = phase
        worker.symbols_completed = completed
        worker.symbols_total = total
        worker.current_symbol = "FPT"
        worker._cycle_number = 1
        worker._current_epoch = 5
        worker._total_epochs = 10
        worker._current_loss = 0.5
        worker._started_at = datetime(2024, 6, 15, 10, 0, 0)

        # Tạo TrainingProgress object
        progress = TrainingProgress(
            phase=phase,
            cycle_number=1,
            current_symbol="FPT",
            symbols_completed=completed,
            symbols_total=total,
            current_epoch=5,
            total_epochs=10,
            current_loss=0.5,
            eta_seconds=120.0,
        )

        # Capture giá trị progress_pct ghi vào write_status
        captured_status = {}

        def mock_write_status(status: TaskStatus):
            captured_status["progress_pct"] = status.progress_pct

        with patch(
            "engine.workers.training_worker.write_status",
            side_effect=mock_write_status,
        ):
            worker.report_progress(progress)

        assert "progress_pct" in captured_status, (
            "write_status không được gọi — report_progress() không ghi status"
        )

        actual_pct = captured_status["progress_pct"]

        assert actual_pct == expected_pct, (
            f"Progress sai: completed={completed}, total={total}, "
            f"expected={expected_pct:.4f}%, got={actual_pct:.4f}%"
        )

    @given(phase=phase_strategy)
    @settings(max_examples=50)
    def test_progress_zero_when_no_symbols_completed(self, phase):
        """
        Property: Khi completed == 0 và total > 0, progress_pct phải == 0.0.

        # Feature: stock-trading-platform-refactor, Property 21: Training progress percentage
        **Validates: Requirements 9.2**
        """
        worker = TrainingEngineWorker(task_id="test-progress-zero")
        worker.phase = phase
        worker.symbols_completed = 0
        worker.symbols_total = 30  # VN30
        worker.current_symbol = "VNM"
        worker._started_at = datetime(2024, 6, 15, 10, 0, 0)

        progress = TrainingProgress(
            phase=phase,
            cycle_number=1,
            current_symbol="VNM",
            symbols_completed=0,
            symbols_total=30,
            current_epoch=1,
            total_epochs=10,
            current_loss=1.0,
            eta_seconds=300.0,
        )

        captured_status = {}

        def mock_write_status(status: TaskStatus):
            captured_status["progress_pct"] = status.progress_pct

        with patch(
            "engine.workers.training_worker.write_status",
            side_effect=mock_write_status,
        ):
            worker.report_progress(progress)

        assert captured_status["progress_pct"] == 0.0, (
            f"Progress phải == 0.0 khi chưa hoàn thành symbol nào, "
            f"got={captured_status['progress_pct']}"
        )

    @given(total=total_symbols_strategy, phase=phase_strategy)
    @settings(max_examples=50)
    def test_progress_100_when_all_symbols_completed(self, total, phase):
        """
        Property: Khi completed == total, progress_pct phải == 100.0.

        # Feature: stock-trading-platform-refactor, Property 21: Training progress percentage
        **Validates: Requirements 9.2**
        """
        worker = TrainingEngineWorker(task_id="test-progress-full")
        worker.phase = phase
        worker.symbols_completed = total
        worker.symbols_total = total
        worker.current_symbol = "HPG"
        worker._started_at = datetime(2024, 6, 15, 10, 0, 0)

        progress = TrainingProgress(
            phase=phase,
            cycle_number=1,
            current_symbol="HPG",
            symbols_completed=total,
            symbols_total=total,
            current_epoch=10,
            total_epochs=10,
            current_loss=0.1,
            eta_seconds=0.0,
        )

        captured_status = {}

        def mock_write_status(status: TaskStatus):
            captured_status["progress_pct"] = status.progress_pct

        with patch(
            "engine.workers.training_worker.write_status",
            side_effect=mock_write_status,
        ):
            worker.report_progress(progress)

        assert captured_status["progress_pct"] == 100.0, (
            f"Progress phải == 100.0 khi tất cả {total} symbols hoàn thành, "
            f"got={captured_status['progress_pct']}"
        )

    @given(data=completed_total_strategy())
    @settings(max_examples=100)
    def test_progress_pct_bounded_0_to_100(self, data):
        """
        Property: progress_pct luôn nằm trong khoảng [0.0, 100.0].

        # Feature: stock-trading-platform-refactor, Property 21: Training progress percentage
        **Validates: Requirements 9.2**
        """
        completed, total = data
        expected_pct = (completed / total) * 100.0

        # Kiểm tra công thức luôn cho kết quả trong [0, 100]
        assert 0.0 <= expected_pct <= 100.0, (
            f"Progress ngoài khoảng [0, 100]: completed={completed}, "
            f"total={total}, pct={expected_pct}"
        )

        # Kiểm tra qua worker
        worker = TrainingEngineWorker(task_id="test-progress-bounded")
        worker.phase = TrainingPhase.PHASE_C
        worker.symbols_completed = completed
        worker.symbols_total = total
        worker.current_symbol = "FPT"
        worker._started_at = datetime(2024, 6, 15, 10, 0, 0)

        progress = TrainingProgress(
            phase=TrainingPhase.PHASE_C,
            cycle_number=1,
            current_symbol="FPT",
            symbols_completed=completed,
            symbols_total=total,
            current_epoch=1,
            total_epochs=10,
            current_loss=0.5,
            eta_seconds=60.0,
        )

        captured_status = {}

        def mock_write_status(status: TaskStatus):
            captured_status["progress_pct"] = status.progress_pct

        with patch(
            "engine.workers.training_worker.write_status",
            side_effect=mock_write_status,
        ):
            worker.report_progress(progress)

        actual_pct = captured_status["progress_pct"]
        assert 0.0 <= actual_pct <= 100.0, (
            f"progress_pct ngoài [0, 100]: {actual_pct}"
        )
