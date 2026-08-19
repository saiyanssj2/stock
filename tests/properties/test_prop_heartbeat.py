# -*- coding: utf-8 -*-
"""
Property-based tests cho HeartbeatMonitor - phát hiện task frozen dựa trên heartbeat timestamp.

# Feature: stock-trading-platform-refactor, Property 2: Heartbeat frozen detection

**Validates: Requirements 1.7**

Properties:
1. Task RUNNING với heartbeat_ts > 60s → is_task_frozen trả về True
2. Task RUNNING với heartbeat_ts <= 60s → is_task_frozen trả về False
3. Task không ở trạng thái RUNNING → is_task_frozen luôn trả về False bất kể heartbeat age
"""

from datetime import datetime, timedelta
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from config.settings import HEARTBEAT_TIMEOUT_THRESHOLD
from models.task_models import TaskState, TaskStatus, TaskType
from orchestrator.heartbeat_monitor import get_frozen_tasks, is_task_frozen


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Thời gian elapsed ngẫu nhiên từ 0-600 giây
elapsed_seconds_strategy = st.floats(min_value=0.0, max_value=600.0)

# Trạng thái task ngẫu nhiên
task_state_strategy = st.sampled_from(list(TaskState))

# Loại task ngẫu nhiên
task_type_strategy = st.sampled_from(list(TaskType))

# Trạng thái KHÔNG PHẢI RUNNING (IDLE, COMPLETED, ERROR, PAUSED)
non_running_state_strategy = st.sampled_from([
    TaskState.IDLE,
    TaskState.COMPLETED,
    TaskState.ERROR,
    TaskState.PAUSED,
])


@st.composite
def task_status_strategy(draw, state=None, elapsed_seconds=None):
    """
    Sinh TaskStatus ngẫu nhiên với state và elapsed_seconds cho trước hoặc random.

    Parameters
    ----------
    state : TaskState hoặc None
        Trạng thái task. Nếu None sẽ random.
    elapsed_seconds : float hoặc None
        Số giây từ heartbeat đến "hiện tại". Nếu None sẽ random.
    """
    if state is None:
        state = draw(task_state_strategy)
    if elapsed_seconds is None:
        elapsed_seconds = draw(elapsed_seconds_strategy)

    task_type = draw(task_type_strategy)
    task_id = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=4,
        max_size=12,
    ))
    progress = draw(st.floats(min_value=0.0, max_value=100.0))
    message = draw(st.text(min_size=0, max_size=30))

    # Thời điểm "hiện tại" cố định để mock
    now = datetime(2024, 6, 15, 12, 0, 0)
    heartbeat_ts = now - timedelta(seconds=elapsed_seconds)
    started_at = now - timedelta(hours=1)

    return {
        "status": TaskStatus(
            task_id=task_id,
            task_type=task_type,
            state=state,
            progress_pct=progress,
            message=message,
            heartbeat_ts=heartbeat_ts,
            started_at=started_at,
        ),
        "now": now,
        "elapsed_seconds": elapsed_seconds,
    }


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestHeartbeatFrozenDetection:
    """Property 2: Heartbeat frozen detection."""

    @given(
        elapsed=st.floats(
            min_value=HEARTBEAT_TIMEOUT_THRESHOLD + 0.001,
            max_value=600.0,
        ),
        task_type=task_type_strategy,
    )
    @settings(max_examples=100)
    def test_running_task_with_old_heartbeat_is_frozen(self, elapsed, task_type):
        """
        Property: Task RUNNING với heartbeat_ts > 60s từ thời điểm hiện tại
        → is_task_frozen phải trả về True.

        # Feature: stock-trading-platform-refactor, Property 2: Heartbeat frozen detection
        **Validates: Requirements 1.7**
        """
        now = datetime(2024, 6, 15, 12, 0, 0)
        heartbeat_ts = now - timedelta(seconds=elapsed)

        status = TaskStatus(
            task_id="test-task-001",
            task_type=task_type,
            state=TaskState.RUNNING,
            progress_pct=50.0,
            message="Training in progress",
            heartbeat_ts=heartbeat_ts,
            started_at=now - timedelta(hours=1),
        )

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now

            result = is_task_frozen(status)

            assert result is True, (
                f"Task RUNNING với heartbeat {elapsed:.2f}s trước phải bị frozen, "
                f"nhưng is_task_frozen trả về False"
            )

    @given(
        elapsed=st.floats(
            min_value=0.0,
            max_value=HEARTBEAT_TIMEOUT_THRESHOLD,
        ),
        task_type=task_type_strategy,
    )
    @settings(max_examples=100)
    def test_running_task_with_recent_heartbeat_not_frozen(self, elapsed, task_type):
        """
        Property: Task RUNNING với heartbeat_ts <= 60s từ thời điểm hiện tại
        → is_task_frozen phải trả về False.

        # Feature: stock-trading-platform-refactor, Property 2: Heartbeat frozen detection
        **Validates: Requirements 1.7**
        """
        now = datetime(2024, 6, 15, 12, 0, 0)
        heartbeat_ts = now - timedelta(seconds=elapsed)

        status = TaskStatus(
            task_id="test-task-002",
            task_type=task_type,
            state=TaskState.RUNNING,
            progress_pct=30.0,
            message="Backtest running",
            heartbeat_ts=heartbeat_ts,
            started_at=now - timedelta(hours=1),
        )

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now

            result = is_task_frozen(status)

            assert result is False, (
                f"Task RUNNING với heartbeat {elapsed:.2f}s trước KHÔNG bị frozen, "
                f"nhưng is_task_frozen trả về True"
            )

    @given(
        state=non_running_state_strategy,
        elapsed=elapsed_seconds_strategy,
        task_type=task_type_strategy,
    )
    @settings(max_examples=100)
    def test_non_running_task_never_frozen(self, state, elapsed, task_type):
        """
        Property: Task KHÔNG ở trạng thái RUNNING (IDLE, COMPLETED, ERROR, PAUSED)
        → is_task_frozen luôn trả về False bất kể heartbeat age.

        # Feature: stock-trading-platform-refactor, Property 2: Heartbeat frozen detection
        **Validates: Requirements 1.7**
        """
        now = datetime(2024, 6, 15, 12, 0, 0)
        heartbeat_ts = now - timedelta(seconds=elapsed)

        status = TaskStatus(
            task_id="test-task-003",
            task_type=task_type,
            state=state,
            progress_pct=75.0,
            message="Task not running",
            heartbeat_ts=heartbeat_ts,
            started_at=now - timedelta(hours=1),
        )

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now

            result = is_task_frozen(status)

            assert result is False, (
                f"Task ở trạng thái {state.value} không bao giờ bị frozen, "
                f"nhưng is_task_frozen trả về True (elapsed={elapsed:.2f}s)"
            )

    @given(
        data=st.data(),
        num_tasks=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50)
    def test_get_frozen_tasks_returns_only_frozen_ids(self, data, num_tasks):
        """
        Property: get_frozen_tasks trả về đúng danh sách task_ids bị frozen
        (RUNNING + heartbeat > 60s).

        # Feature: stock-trading-platform-refactor, Property 2: Heartbeat frozen detection
        **Validates: Requirements 1.7**
        """
        now = datetime(2024, 6, 15, 12, 0, 0)
        statuses = {}
        expected_frozen = []

        for i in range(num_tasks):
            task_id = f"task-{i:03d}"
            state = data.draw(task_state_strategy)
            elapsed = data.draw(elapsed_seconds_strategy)
            task_type = data.draw(task_type_strategy)

            heartbeat_ts = now - timedelta(seconds=elapsed)
            status = TaskStatus(
                task_id=task_id,
                task_type=task_type,
                state=state,
                progress_pct=50.0,
                message=f"Task {i}",
                heartbeat_ts=heartbeat_ts,
                started_at=now - timedelta(hours=1),
            )
            statuses[task_id] = status

            # Tính expected: chỉ RUNNING + elapsed > threshold mới frozen
            if state == TaskState.RUNNING and elapsed > HEARTBEAT_TIMEOUT_THRESHOLD:
                expected_frozen.append(task_id)

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now

            result = get_frozen_tasks(statuses)

            assert set(result) == set(expected_frozen), (
                f"Expected frozen: {expected_frozen}, got: {result}"
            )
