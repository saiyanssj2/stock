# -*- coding: utf-8 -*-
"""
Unit tests cho orchestrator/heartbeat_monitor.py

Mock datetime.now() để kiểm soát thời gian, đảm bảo test deterministic.
Covers: is_task_frozen, get_frozen_tasks, check_heartbeats.

References: Req 1.7
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from config.settings import HEARTBEAT_TIMEOUT_THRESHOLD
from models.task_models import TaskState, TaskStatus, TaskType
from orchestrator.heartbeat_monitor import (
    check_heartbeats,
    get_frozen_tasks,
    is_task_frozen,
)


def _make_status(
    task_id: str = "task-001",
    state: TaskState = TaskState.RUNNING,
    heartbeat_ts: datetime = None,
    task_type: TaskType = TaskType.TRAINING,
) -> TaskStatus:
    """Helper tạo TaskStatus cho test."""
    now = datetime.now()
    return TaskStatus(
        task_id=task_id,
        task_type=task_type,
        state=state,
        progress_pct=50.0,
        message="Testing",
        heartbeat_ts=heartbeat_ts or now,
        started_at=now - timedelta(minutes=10),
    )


class TestIsTaskFrozen:
    """Test is_task_frozen() - kiểm tra 1 task có frozen hay không."""

    def test_running_task_with_fresh_heartbeat_not_frozen(self):
        """Task RUNNING với heartbeat mới → không frozen."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        # Heartbeat cách đây 30s (trong ngưỡng 60s)
        status = _make_status(
            state=TaskState.RUNNING,
            heartbeat_ts=now - timedelta(seconds=30),
        )

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now
            assert is_task_frozen(status) is False

    def test_running_task_with_stale_heartbeat_is_frozen(self):
        """Task RUNNING với heartbeat > 60s → frozen."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        # Heartbeat cách đây 61s (vượt ngưỡng)
        status = _make_status(
            state=TaskState.RUNNING,
            heartbeat_ts=now - timedelta(seconds=61),
        )

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now
            assert is_task_frozen(status) is True

    def test_running_task_at_exact_threshold_not_frozen(self):
        """Task RUNNING với heartbeat đúng = 60s → không frozen (chỉ > mới frozen)."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        status = _make_status(
            state=TaskState.RUNNING,
            heartbeat_ts=now - timedelta(seconds=60),
        )

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now
            assert is_task_frozen(status) is False

    def test_completed_task_with_old_heartbeat_not_frozen(self):
        """Task COMPLETED dù heartbeat cũ → không frozen."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        status = _make_status(
            state=TaskState.COMPLETED,
            heartbeat_ts=now - timedelta(seconds=120),
        )

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now
            assert is_task_frozen(status) is False

    def test_idle_task_not_frozen(self):
        """Task IDLE dù heartbeat cũ → không frozen."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        status = _make_status(
            state=TaskState.IDLE,
            heartbeat_ts=now - timedelta(seconds=200),
        )

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now
            assert is_task_frozen(status) is False

    def test_error_task_not_frozen(self):
        """Task ERROR dù heartbeat cũ → không frozen."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        status = _make_status(
            state=TaskState.ERROR,
            heartbeat_ts=now - timedelta(seconds=300),
        )

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now
            assert is_task_frozen(status) is False

    def test_paused_task_not_frozen(self):
        """Task PAUSED dù heartbeat cũ → không frozen."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        status = _make_status(
            state=TaskState.PAUSED,
            heartbeat_ts=now - timedelta(seconds=150),
        )

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now
            assert is_task_frozen(status) is False

    def test_very_old_heartbeat_is_frozen(self):
        """Task RUNNING với heartbeat rất cũ (5 phút) → frozen."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        status = _make_status(
            state=TaskState.RUNNING,
            heartbeat_ts=now - timedelta(minutes=5),
        )

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now
            assert is_task_frozen(status) is True


class TestGetFrozenTasks:
    """Test get_frozen_tasks() - lọc frozen tasks từ dict."""

    def test_empty_dict_returns_empty_list(self):
        """Dict rỗng → danh sách trống."""
        result = get_frozen_tasks({})
        assert result == []

    def test_all_healthy_tasks_returns_empty(self):
        """Tất cả task healthy → danh sách trống."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        statuses = {
            "task-001": _make_status(
                task_id="task-001",
                state=TaskState.RUNNING,
                heartbeat_ts=now - timedelta(seconds=10),
            ),
            "task-002": _make_status(
                task_id="task-002",
                state=TaskState.RUNNING,
                heartbeat_ts=now - timedelta(seconds=20),
            ),
        }

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now
            result = get_frozen_tasks(statuses)
            assert result == []

    def test_one_frozen_task_detected(self):
        """Một task frozen trong nhiều task → trả về đúng task đó."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        statuses = {
            "task-001": _make_status(
                task_id="task-001",
                state=TaskState.RUNNING,
                heartbeat_ts=now - timedelta(seconds=10),
            ),
            "task-002": _make_status(
                task_id="task-002",
                state=TaskState.RUNNING,
                heartbeat_ts=now - timedelta(seconds=90),
            ),
        }

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now
            result = get_frozen_tasks(statuses)
            assert result == ["task-002"]

    def test_multiple_frozen_tasks_detected(self):
        """Nhiều task frozen → trả về tất cả."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        statuses = {
            "task-001": _make_status(
                task_id="task-001",
                state=TaskState.RUNNING,
                heartbeat_ts=now - timedelta(seconds=120),
            ),
            "task-002": _make_status(
                task_id="task-002",
                state=TaskState.RUNNING,
                heartbeat_ts=now - timedelta(seconds=90),
            ),
            "task-003": _make_status(
                task_id="task-003",
                state=TaskState.RUNNING,
                heartbeat_ts=now - timedelta(seconds=30),
            ),
        }

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now
            result = get_frozen_tasks(statuses)
            assert sorted(result) == ["task-001", "task-002"]

    def test_non_running_tasks_excluded(self):
        """Task không phải RUNNING bị loại trừ dù heartbeat cũ."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        statuses = {
            "task-001": _make_status(
                task_id="task-001",
                state=TaskState.COMPLETED,
                heartbeat_ts=now - timedelta(seconds=200),
            ),
            "task-002": _make_status(
                task_id="task-002",
                state=TaskState.ERROR,
                heartbeat_ts=now - timedelta(seconds=300),
            ),
            "task-003": _make_status(
                task_id="task-003",
                state=TaskState.IDLE,
                heartbeat_ts=now - timedelta(seconds=150),
            ),
        }

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now
            result = get_frozen_tasks(statuses)
            assert result == []


class TestCheckHeartbeats:
    """Test check_heartbeats() - integration với read_all_statuses."""

    @patch("orchestrator.heartbeat_monitor.read_all_statuses")
    def test_no_statuses_returns_empty(self, mock_read_all):
        """Không có status files → danh sách trống."""
        mock_read_all.return_value = {}
        result = check_heartbeats()
        assert result == []
        mock_read_all.assert_called_once()

    @patch("orchestrator.heartbeat_monitor.read_all_statuses")
    def test_returns_frozen_task_ids(self, mock_read_all):
        """Trả về task_ids bị frozen từ status files."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        mock_read_all.return_value = {
            "train-001": _make_status(
                task_id="train-001",
                state=TaskState.RUNNING,
                heartbeat_ts=now - timedelta(seconds=100),
                task_type=TaskType.TRAINING,
            ),
            "backtest-001": _make_status(
                task_id="backtest-001",
                state=TaskState.RUNNING,
                heartbeat_ts=now - timedelta(seconds=20),
                task_type=TaskType.BACKTEST,
            ),
        }

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now
            result = check_heartbeats()
            assert result == ["train-001"]

    @patch("orchestrator.heartbeat_monitor.read_all_statuses")
    def test_mixed_states_only_checks_running(self, mock_read_all):
        """Chỉ kiểm tra task RUNNING, bỏ qua các state khác."""
        now = datetime(2024, 1, 15, 10, 0, 0)
        mock_read_all.return_value = {
            "task-run": _make_status(
                task_id="task-run",
                state=TaskState.RUNNING,
                heartbeat_ts=now - timedelta(seconds=90),
            ),
            "task-done": _make_status(
                task_id="task-done",
                state=TaskState.COMPLETED,
                heartbeat_ts=now - timedelta(seconds=200),
            ),
            "task-err": _make_status(
                task_id="task-err",
                state=TaskState.ERROR,
                heartbeat_ts=now - timedelta(seconds=300),
            ),
        }

        with patch("orchestrator.heartbeat_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = now
            result = check_heartbeats()
            assert result == ["task-run"]
