# -*- coding: utf-8 -*-
"""
ResourceManager - Quản lý tài nguyên hệ thống (GPU, CPU) cho các task.

Chức năng chính:
- Kiểm tra resource availability trước khi start task mới
- Phân bổ GPU memory fraction theo loại task (cap 0.7 per task)
- Đảm bảo tổng GPU allocation không vượt quá 1.0
- Phân bổ CPU threads proportionally theo os.cpu_count()
- Giải phóng resource khi task hoàn thành

References: Req 1.4, 2.7
"""

import os
from typing import Dict, Optional

from config.settings import GPU_MAX_FRACTION
from models.task_models import ResourceAllocation, TaskType

# Phân bổ GPU memory mặc định cho từng loại task
_DEFAULT_GPU_ALLOCATIONS: Dict[TaskType, float] = {
    TaskType.TRAINING: 0.7,
    TaskType.BACKTEST: 0.3,
    TaskType.ANALYSIS: 0.2,
}

# Tổng GPU fraction tối đa trên toàn hệ thống
_TOTAL_GPU_CAPACITY: float = 1.0


class ResourceManager:
    """
    Quản lý phân bổ tài nguyên cho các task chạy song song.

    Theo dõi GPU memory fraction và CPU threads đã cấp cho các task đang active.
    Đảm bảo:
    - Mỗi task không vượt quá GPU_MAX_FRACTION (0.7)
    - Tổng GPU allocation không vượt 1.0
    - CPU threads được phân bổ proportionally

    Attributes
    ----------
    _active_allocations : Dict[str, ResourceAllocation]
        Mapping task_id → ResourceAllocation cho các task đang active.
    """

    def __init__(self) -> None:
        """Khởi tạo ResourceManager với danh sách allocation rỗng."""
        self._active_allocations: Dict[str, ResourceAllocation] = {}

    def allocate_resources(self, task_type: TaskType, task_id: Optional[str] = None) -> ResourceAllocation:
        """
        Kiểm tra availability và phân bổ tài nguyên cho task mới.

        Parameters
        ----------
        task_type : TaskType
            Loại task cần phân bổ resource.
        task_id : Optional[str]
            ID của task. Nếu cung cấp và allocation thành công,
            sẽ tự động track vào _active_allocations.

        Returns
        -------
        ResourceAllocation
            Kết quả phân bổ. can_start=False nếu không đủ resource.
        """
        # Lấy GPU fraction mặc định cho loại task, cap tại GPU_MAX_FRACTION
        requested_gpu = min(
            _DEFAULT_GPU_ALLOCATIONS.get(task_type, 0.2),
            GPU_MAX_FRACTION,
        )

        # Kiểm tra tổng GPU còn đủ không
        current_usage = self.get_current_gpu_usage()
        remaining_gpu = _TOTAL_GPU_CAPACITY - current_usage

        # Dùng epsilon tolerance để tránh floating point precision issues
        _EPSILON = 1e-9
        if requested_gpu > remaining_gpu + _EPSILON:
            return ResourceAllocation(
                gpu_memory_fraction=requested_gpu,
                cpu_threads=0,
                can_start=False,
                reason=(
                    f"Không đủ GPU memory. Yêu cầu: {requested_gpu:.1%}, "
                    f"còn trống: {remaining_gpu:.1%} "
                    f"(đang dùng: {current_usage:.1%})"
                ),
            )

        # Phân bổ CPU threads proportionally theo GPU fraction
        total_cpus = os.cpu_count() or 4
        cpu_threads = max(1, int(total_cpus * requested_gpu))

        allocation = ResourceAllocation(
            gpu_memory_fraction=requested_gpu,
            cpu_threads=cpu_threads,
            can_start=True,
        )

        # Track allocation nếu có task_id
        if task_id is not None:
            self._active_allocations[task_id] = allocation

        return allocation

    def check_availability(self) -> bool:
        """
        Kiểm tra hệ thống còn đủ resource cho ít nhất 1 task mới.

        Kiểm tra dựa trên allocation nhỏ nhất (ANALYSIS = 0.2).
        Nếu ngay cả task nhỏ nhất cũng không đủ resource → False.

        Returns
        -------
        bool
            True nếu hệ thống còn đủ resource cho task mới.
        """
        current_usage = self.get_current_gpu_usage()
        # Kiểm tra với allocation nhỏ nhất có thể
        min_allocation = min(_DEFAULT_GPU_ALLOCATIONS.values())
        _EPSILON = 1e-9
        return (current_usage + min_allocation) <= _TOTAL_GPU_CAPACITY + _EPSILON

    def get_current_gpu_usage(self) -> float:
        """
        Lấy tổng GPU memory fraction đang được sử dụng.

        Returns
        -------
        float
            Tổng GPU fraction đã cấp cho tất cả active tasks (0.0 - 1.0).
        """
        return sum(
            alloc.gpu_memory_fraction
            for alloc in self._active_allocations.values()
        )

    def release_resources(self, task_id: str) -> bool:
        """
        Giải phóng resource khi task hoàn thành hoặc bị dừng.

        Parameters
        ----------
        task_id : str
            ID của task cần giải phóng resource.

        Returns
        -------
        bool
            True nếu giải phóng thành công, False nếu task_id không tồn tại.
        """
        if task_id not in self._active_allocations:
            return False
        del self._active_allocations[task_id]
        return True

    def get_allocation(self, task_id: str) -> Optional[ResourceAllocation]:
        """
        Lấy allocation hiện tại của một task.

        Parameters
        ----------
        task_id : str
            ID của task cần tra cứu.

        Returns
        -------
        Optional[ResourceAllocation]
            ResourceAllocation nếu task đang active, None nếu không tìm thấy.
        """
        return self._active_allocations.get(task_id)

    def get_active_task_count(self) -> int:
        """Trả về số lượng task đang active."""
        return len(self._active_allocations)
