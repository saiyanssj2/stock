# -*- coding: utf-8 -*-
"""
Property-based tests cho ResourceManager - Resource allocation cap.

Kiểm tra property: GPU memory fraction không bao giờ vượt 0.70 (70%)
cho bất kỳ task nào được phân bổ bởi TaskOrchestrator.

# Feature: stock-trading-platform-refactor, Property 1: Resource allocation cap
"""

import uuid

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from orchestrator.resource_manager import ResourceManager
from models.task_models import TaskType


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy sinh ngẫu nhiên TaskType
task_type_strategy = st.sampled_from(list(TaskType))

# Strategy sinh chuỗi operations (allocate hoặc release)
# Mỗi operation là tuple: ("allocate", TaskType) hoặc ("release", task_id_index)
_operation_strategy = st.one_of(
    st.tuples(st.just("allocate"), task_type_strategy),
    st.tuples(st.just("release"), st.integers(min_value=0, max_value=99)),
)

# Strategy sinh chuỗi ngẫu nhiên các operations
operation_sequence_strategy = st.lists(
    _operation_strategy,
    min_size=1,
    max_size=50,
)


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestResourceAllocationCap:
    """
    Property tests đảm bảo GPU memory fraction cap 0.70 luôn được enforce.

    **Validates: Requirements 1.4**
    """

    @given(task_type=task_type_strategy)
    @settings(max_examples=100)
    def test_single_allocation_never_exceeds_cap(self, task_type: TaskType):
        """
        Property 1.1: Một allocation đơn lẻ không bao giờ vượt quá 0.70.

        Với bất kỳ TaskType nào, gpu_memory_fraction trả về từ
        allocate_resources() phải <= 0.70.

        **Validates: Requirements 1.4**
        """
        # Arrange
        rm = ResourceManager()

        # Act
        allocation = rm.allocate_resources(task_type)

        # Assert - GPU fraction luôn <= 0.70
        assert allocation.gpu_memory_fraction <= 0.70, (
            f"GPU fraction {allocation.gpu_memory_fraction} vượt cap 0.70 "
            f"cho task type {task_type.value}"
        )

    @given(operations=operation_sequence_strategy)
    @settings(max_examples=100)
    def test_sequence_of_operations_never_exceeds_cap(self, operations):
        """
        Property 1.2: Sau bất kỳ chuỗi allocate/release nào,
        mọi allocation thành công đều có gpu_memory_fraction <= 0.70.

        **Validates: Requirements 1.4**
        """
        # Arrange
        rm = ResourceManager()
        allocated_task_ids: list = []

        # Act & Assert
        for op_type, op_value in operations:
            if op_type == "allocate":
                task_type = op_value
                task_id = str(uuid.uuid4())
                allocation = rm.allocate_resources(task_type, task_id=task_id)

                # Mọi allocation (dù can_start=True hay False) phải có fraction <= 0.70
                assert allocation.gpu_memory_fraction <= 0.70, (
                    f"GPU fraction {allocation.gpu_memory_fraction} vượt cap 0.70 "
                    f"cho task type {task_type.value} sau chuỗi operations"
                )

                if allocation.can_start:
                    allocated_task_ids.append(task_id)

            elif op_type == "release":
                # Release task theo index (nếu có task để release)
                if allocated_task_ids:
                    idx = op_value % len(allocated_task_ids)
                    task_id_to_release = allocated_task_ids.pop(idx)
                    rm.release_resources(task_id_to_release)

    @given(
        task_types=st.lists(
            task_type_strategy,
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=100)
    def test_total_gpu_usage_never_exceeds_capacity(self, task_types):
        """
        Property 1.3: Tổng GPU usage cho các allocation được chấp nhận
        (can_start=True) không bao giờ vượt quá 1.0.

        **Validates: Requirements 1.4**
        """
        # Arrange
        rm = ResourceManager()

        # Act - Allocate nhiều tasks liên tiếp
        for task_type in task_types:
            task_id = str(uuid.uuid4())
            allocation = rm.allocate_resources(task_type, task_id=task_id)

            # Nếu được chấp nhận, kiểm tra tổng GPU không vượt 1.0
            if allocation.can_start:
                total_usage = rm.get_current_gpu_usage()
                assert total_usage <= 1.0 + 1e-9, (
                    f"Tổng GPU usage {total_usage} vượt capacity 1.0 "
                    f"sau khi allocate {task_type.value}"
                )

            # Mỗi allocation riêng lẻ vẫn phải <= 0.70
            assert allocation.gpu_memory_fraction <= 0.70, (
                f"GPU fraction {allocation.gpu_memory_fraction} vượt cap 0.70"
            )
