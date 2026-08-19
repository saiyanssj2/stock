# -*- coding: utf-8 -*-
"""
Unit tests cho ResourceManager.

Kiểm tra:
- GPU memory fraction cap enforcement (max 0.7 per task)
- Tổng GPU allocation không vượt 1.0
- Multiple concurrent tasks allocation
- Release behavior
- CPU threads allocation proportional
- check_availability logic

References: Req 1.4, 2.7
"""

import os
from unittest.mock import patch

import pytest

from models.task_models import ResourceAllocation, TaskType
from orchestrator.resource_manager import ResourceManager, _DEFAULT_GPU_ALLOCATIONS


class TestAllocateResources:
    """Test allocate_resources() - phân bổ tài nguyên cho task."""

    def test_training_task_gets_0_7_gpu(self) -> None:
        """TRAINING task nhận 0.7 GPU fraction (max allowed)."""
        rm = ResourceManager()
        alloc = rm.allocate_resources(TaskType.TRAINING)
        assert alloc.gpu_memory_fraction == 0.7
        assert alloc.can_start is True

    def test_backtest_task_gets_0_3_gpu(self) -> None:
        """BACKTEST task nhận 0.3 GPU fraction."""
        rm = ResourceManager()
        alloc = rm.allocate_resources(TaskType.BACKTEST)
        assert alloc.gpu_memory_fraction == 0.3
        assert alloc.can_start is True

    def test_analysis_task_gets_0_2_gpu(self) -> None:
        """ANALYSIS task nhận 0.2 GPU fraction."""
        rm = ResourceManager()
        alloc = rm.allocate_resources(TaskType.ANALYSIS)
        assert alloc.gpu_memory_fraction == 0.2
        assert alloc.can_start is True

    def test_gpu_fraction_never_exceeds_0_7(self) -> None:
        """GPU fraction per task không bao giờ vượt 0.7 (GPU_MAX_FRACTION)."""
        rm = ResourceManager()
        for task_type in TaskType:
            alloc = rm.allocate_resources(task_type)
            assert alloc.gpu_memory_fraction <= 0.7

    def test_allocation_returns_resource_allocation_type(self) -> None:
        """allocate_resources trả về ResourceAllocation dataclass."""
        rm = ResourceManager()
        alloc = rm.allocate_resources(TaskType.ANALYSIS)
        assert isinstance(alloc, ResourceAllocation)

    def test_cpu_threads_positive(self) -> None:
        """CPU threads luôn >= 1 khi can_start=True."""
        rm = ResourceManager()
        alloc = rm.allocate_resources(TaskType.TRAINING)
        assert alloc.cpu_threads >= 1

    @patch("os.cpu_count", return_value=8)
    def test_cpu_threads_proportional_to_gpu(self, _mock_cpu: object) -> None:
        """CPU threads được phân bổ proportionally theo GPU fraction."""
        rm = ResourceManager()

        # TRAINING: 0.7 * 8 = 5.6 → 5 threads
        alloc_train = rm.allocate_resources(TaskType.TRAINING, task_id="t1")
        assert alloc_train.cpu_threads == 5

        # Giải phóng để test riêng
        rm.release_resources("t1")

        # ANALYSIS: 0.2 * 8 = 1.6 → 1 thread
        alloc_analysis = rm.allocate_resources(TaskType.ANALYSIS)
        assert alloc_analysis.cpu_threads == 1

    def test_allocation_with_task_id_tracks_internally(self) -> None:
        """Khi cung cấp task_id, allocation được track vào _active_allocations."""
        rm = ResourceManager()
        rm.allocate_resources(TaskType.BACKTEST, task_id="bt-001")
        assert "bt-001" in rm._active_allocations
        assert rm.get_active_task_count() == 1

    def test_allocation_without_task_id_not_tracked(self) -> None:
        """Khi không có task_id, allocation không được track."""
        rm = ResourceManager()
        rm.allocate_resources(TaskType.BACKTEST)
        assert rm.get_active_task_count() == 0


class TestTotalGpuCapacity:
    """Test tổng GPU allocation không vượt quá 1.0."""

    def test_reject_when_total_exceeds_1_0(self) -> None:
        """Từ chối task mới khi tổng GPU allocation vượt 1.0."""
        rm = ResourceManager()
        # Phân bổ TRAINING (0.7)
        alloc1 = rm.allocate_resources(TaskType.TRAINING, task_id="t1")
        assert alloc1.can_start is True

        # Phân bổ BACKTEST (0.3) → tổng = 1.0, vẫn OK
        alloc2 = rm.allocate_resources(TaskType.BACKTEST, task_id="t2")
        assert alloc2.can_start is True

        # Phân bổ ANALYSIS (0.2) → tổng = 1.2, vượt quá → reject
        alloc3 = rm.allocate_resources(TaskType.ANALYSIS)
        assert alloc3.can_start is False
        assert alloc3.reason is not None

    def test_reject_provides_descriptive_reason(self) -> None:
        """Khi reject, reason mô tả rõ tình trạng GPU."""
        rm = ResourceManager()
        rm.allocate_resources(TaskType.TRAINING, task_id="t1")
        rm.allocate_resources(TaskType.BACKTEST, task_id="t2")

        alloc = rm.allocate_resources(TaskType.ANALYSIS)
        assert alloc.can_start is False
        assert "GPU" in alloc.reason

    def test_training_plus_training_exceeds_capacity(self) -> None:
        """Hai task TRAINING (0.7 + 0.7 = 1.4) vượt capacity."""
        rm = ResourceManager()
        alloc1 = rm.allocate_resources(TaskType.TRAINING, task_id="t1")
        assert alloc1.can_start is True

        alloc2 = rm.allocate_resources(TaskType.TRAINING)
        assert alloc2.can_start is False

    def test_exact_capacity_1_0_allowed(self) -> None:
        """Tổng allocation đúng = 1.0 thì vẫn được chấp nhận."""
        rm = ResourceManager()
        # 0.7 + 0.3 = 1.0 exactly
        alloc1 = rm.allocate_resources(TaskType.TRAINING, task_id="t1")
        assert alloc1.can_start is True

        alloc2 = rm.allocate_resources(TaskType.BACKTEST, task_id="t2")
        assert alloc2.can_start is True


class TestConcurrentTasks:
    """Test multiple concurrent tasks."""

    def test_multiple_analysis_tasks_allowed(self) -> None:
        """Nhiều ANALYSIS tasks (0.2 mỗi task) có thể chạy đồng thời."""
        rm = ResourceManager()
        # 5 analysis tasks * 0.2 = 1.0
        for i in range(5):
            alloc = rm.allocate_resources(TaskType.ANALYSIS, task_id=f"a{i}")
            assert alloc.can_start is True

        # Task thứ 6 sẽ bị reject (tổng = 1.2)
        alloc_reject = rm.allocate_resources(TaskType.ANALYSIS)
        assert alloc_reject.can_start is False

    def test_mixed_tasks_respect_capacity(self) -> None:
        """Mix nhiều loại task vẫn respect tổng capacity."""
        rm = ResourceManager()
        # BACKTEST (0.3) + ANALYSIS (0.2) + ANALYSIS (0.2) + ANALYSIS (0.2) = 0.9
        rm.allocate_resources(TaskType.BACKTEST, task_id="bt1")
        rm.allocate_resources(TaskType.ANALYSIS, task_id="a1")
        rm.allocate_resources(TaskType.ANALYSIS, task_id="a2")
        rm.allocate_resources(TaskType.ANALYSIS, task_id="a3")

        assert rm.get_current_gpu_usage() == pytest.approx(0.9)

        # Thêm ANALYSIS (0.2) → 1.1, reject
        alloc = rm.allocate_resources(TaskType.ANALYSIS)
        assert alloc.can_start is False

    def test_gpu_usage_tracks_accurately(self) -> None:
        """get_current_gpu_usage() trả về tổng chính xác."""
        rm = ResourceManager()
        rm.allocate_resources(TaskType.TRAINING, task_id="t1")
        assert rm.get_current_gpu_usage() == pytest.approx(0.7)

        rm.allocate_resources(TaskType.ANALYSIS, task_id="a1")
        assert rm.get_current_gpu_usage() == pytest.approx(0.9)


class TestReleaseResources:
    """Test release_resources() - giải phóng resource."""

    def test_release_existing_task(self) -> None:
        """Release task đang active trả về True và giải phóng resource."""
        rm = ResourceManager()
        rm.allocate_resources(TaskType.TRAINING, task_id="t1")
        assert rm.get_current_gpu_usage() == pytest.approx(0.7)

        result = rm.release_resources("t1")
        assert result is True
        assert rm.get_current_gpu_usage() == pytest.approx(0.0)
        assert rm.get_active_task_count() == 0

    def test_release_nonexistent_task(self) -> None:
        """Release task không tồn tại trả về False."""
        rm = ResourceManager()
        result = rm.release_resources("nonexistent")
        assert result is False

    def test_release_frees_capacity_for_new_tasks(self) -> None:
        """Sau khi release, resource freed cho task mới."""
        rm = ResourceManager()
        rm.allocate_resources(TaskType.TRAINING, task_id="t1")
        rm.allocate_resources(TaskType.BACKTEST, task_id="bt1")
        # Tổng = 1.0, full

        # ANALYSIS bị reject
        alloc = rm.allocate_resources(TaskType.ANALYSIS)
        assert alloc.can_start is False

        # Release BACKTEST → freed 0.3
        rm.release_resources("bt1")

        # Bây giờ ANALYSIS (0.2) fit được
        alloc2 = rm.allocate_resources(TaskType.ANALYSIS, task_id="a1")
        assert alloc2.can_start is True

    def test_double_release_returns_false(self) -> None:
        """Release cùng task_id lần 2 trả về False."""
        rm = ResourceManager()
        rm.allocate_resources(TaskType.ANALYSIS, task_id="a1")
        rm.release_resources("a1")
        result = rm.release_resources("a1")
        assert result is False


class TestCheckAvailability:
    """Test check_availability() - kiểm tra resource đủ cho task mới."""

    def test_available_when_empty(self) -> None:
        """Hệ thống rỗng → luôn available."""
        rm = ResourceManager()
        assert rm.check_availability() is True

    def test_available_with_partial_load(self) -> None:
        """Hệ thống chưa full → still available."""
        rm = ResourceManager()
        rm.allocate_resources(TaskType.TRAINING, task_id="t1")
        # 0.7 used, 0.3 remaining, min task = 0.2 → still fits
        assert rm.check_availability() is True

    def test_unavailable_when_full(self) -> None:
        """Hệ thống full → unavailable."""
        rm = ResourceManager()
        rm.allocate_resources(TaskType.TRAINING, task_id="t1")
        rm.allocate_resources(TaskType.BACKTEST, task_id="bt1")
        # 1.0 used, min task = 0.2, 0 remaining → không fit
        assert rm.check_availability() is False

    def test_available_after_release(self) -> None:
        """Sau khi release, check_availability lại = True."""
        rm = ResourceManager()
        rm.allocate_resources(TaskType.TRAINING, task_id="t1")
        rm.allocate_resources(TaskType.BACKTEST, task_id="bt1")
        assert rm.check_availability() is False

        rm.release_resources("bt1")
        assert rm.check_availability() is True


class TestGetAllocation:
    """Test get_allocation() - tra cứu allocation của task."""

    def test_get_existing_allocation(self) -> None:
        """Trả về allocation cho task đang active."""
        rm = ResourceManager()
        rm.allocate_resources(TaskType.TRAINING, task_id="t1")
        alloc = rm.get_allocation("t1")
        assert alloc is not None
        assert alloc.gpu_memory_fraction == 0.7
        assert alloc.can_start is True

    def test_get_nonexistent_returns_none(self) -> None:
        """Trả về None cho task không tồn tại."""
        rm = ResourceManager()
        assert rm.get_allocation("nonexistent") is None


class TestEdgeCases:
    """Test edge cases và boundary conditions."""

    def test_empty_manager_gpu_usage_zero(self) -> None:
        """Manager rỗng → GPU usage = 0."""
        rm = ResourceManager()
        assert rm.get_current_gpu_usage() == 0.0

    def test_active_task_count_accurate(self) -> None:
        """get_active_task_count() đếm chính xác."""
        rm = ResourceManager()
        assert rm.get_active_task_count() == 0

        rm.allocate_resources(TaskType.TRAINING, task_id="t1")
        assert rm.get_active_task_count() == 1

        rm.allocate_resources(TaskType.ANALYSIS, task_id="a1")
        assert rm.get_active_task_count() == 2

        rm.release_resources("t1")
        assert rm.get_active_task_count() == 1

    @patch("os.cpu_count", return_value=None)
    def test_cpu_count_none_defaults_to_4(self, _mock: object) -> None:
        """Khi os.cpu_count() trả None, dùng default 4 CPUs."""
        rm = ResourceManager()
        alloc = rm.allocate_resources(TaskType.TRAINING)
        # 0.7 * 4 = 2.8 → 2 threads
        assert alloc.cpu_threads == 2

    @patch("os.cpu_count", return_value=1)
    def test_min_one_cpu_thread(self, _mock: object) -> None:
        """Luôn phân bổ ít nhất 1 CPU thread."""
        rm = ResourceManager()
        alloc = rm.allocate_resources(TaskType.ANALYSIS)
        # 0.2 * 1 = 0.2 → max(1, 0) = 1
        assert alloc.cpu_threads >= 1
