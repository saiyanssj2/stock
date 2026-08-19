# -*- coding: utf-8 -*-
"""
Unit tests cho TaskOrchestrator.get_active_tasks_from_disk().

Kiểm tra:
- Trả về tasks có state=RUNNING và heartbeat fresh
- Lọc bỏ tasks không phải RUNNING (COMPLETED, ERROR, IDLE)
- Lọc bỏ tasks có heartbeat quá cũ (> HEARTBEAT_TIMEOUT_THRESHOLD)
- Hoạt động trên orchestrator instance mới (empty _processes)
- Trả về list rỗng khi không có active task nào

Requirements: 2.1, 2.2, 2.3
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from config.settings import HEARTBEAT_TIMEOUT_THRESHOLD
from models.task_models import TaskState, TaskStatus, TaskType
from orchestrator.task_orchestrator import TaskOrchestrator


class TestGetActiveTasksFromDisk:
    """Tests cho get_active_tasks_from_disk() method."""

    @patch("orchestrator.task_orchestrator.read_all_statuses")
    def test_returns_running_task_with_fresh_heartbeat(self, mock_read_all):
        """Trả về task có state=RUNNING và heartbeat trong ngưỡng."""
        now = datetime.now()
        running_task = TaskStatus(
            task_id="training_aabbccdd",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=45.0,
            message="Training epoch 5/100",
            heartbeat_ts=now - timedelta(seconds=10),  # 10s ago - còn fresh
            started_at=now - timedelta(minutes=5),
        )
        mock_read_all.return_value = {"training_aabbccdd": running_task}

        orchestrator = TaskOrchestrator()
        result = orchestrator.get_active_tasks_from_disk()

        assert len(result) == 1
        assert result[0].task_id == "training_aabbccdd"
        assert result[0].state == TaskState.RUNNING

    @patch("orchestrator.task_orchestrator.read_all_statuses")
    def test_filters_out_completed_tasks(self, mock_read_all):
        """Không trả về tasks có state != RUNNING."""
        now = datetime.now()
        mock_read_all.return_value = {
            "training_completed": TaskStatus(
                task_id="training_completed",
                task_type=TaskType.TRAINING,
                state=TaskState.COMPLETED,
                progress_pct=100.0,
                message="Done",
                heartbeat_ts=now - timedelta(seconds=5),
                started_at=now - timedelta(minutes=10),
            ),
            "backtest_error": TaskStatus(
                task_id="backtest_error",
                task_type=TaskType.BACKTEST,
                state=TaskState.ERROR,
                progress_pct=30.0,
                message="Crashed",
                heartbeat_ts=now - timedelta(seconds=5),
                started_at=now - timedelta(minutes=3),
                error="OOM error",
            ),
            "analysis_idle": TaskStatus(
                task_id="analysis_idle",
                task_type=TaskType.ANALYSIS,
                state=TaskState.IDLE,
                progress_pct=0.0,
                message="Not started",
                heartbeat_ts=now - timedelta(seconds=5),
                started_at=now,
            ),
        }

        orchestrator = TaskOrchestrator()
        result = orchestrator.get_active_tasks_from_disk()

        assert len(result) == 0

    @patch("orchestrator.task_orchestrator.read_all_statuses")
    def test_filters_out_stale_heartbeat(self, mock_read_all):
        """Không trả về task RUNNING nếu heartbeat quá cũ (frozen)."""
        now = datetime.now()
        # heartbeat cách đây > HEARTBEAT_TIMEOUT_THRESHOLD (60s)
        stale_task = TaskStatus(
            task_id="training_stale",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=20.0,
            message="Frozen",
            heartbeat_ts=now - timedelta(seconds=HEARTBEAT_TIMEOUT_THRESHOLD + 10),
            started_at=now - timedelta(minutes=5),
        )
        mock_read_all.return_value = {"training_stale": stale_task}

        orchestrator = TaskOrchestrator()
        result = orchestrator.get_active_tasks_from_disk()

        assert len(result) == 0

    @patch("orchestrator.task_orchestrator.read_all_statuses")
    def test_returns_empty_list_when_no_status_files(self, mock_read_all):
        """Trả về list rỗng khi không có status file nào."""
        mock_read_all.return_value = {}

        orchestrator = TaskOrchestrator()
        result = orchestrator.get_active_tasks_from_disk()

        assert result == []

    @patch("orchestrator.task_orchestrator.read_all_statuses")
    def test_works_on_fresh_orchestrator_instance(self, mock_read_all):
        """Hoạt động đúng trên orchestrator mới (empty _processes dict)."""
        now = datetime.now()
        running_task = TaskStatus(
            task_id="training_newinstance",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=60.0,
            message="Training running",
            heartbeat_ts=now - timedelta(seconds=5),
            started_at=now - timedelta(minutes=10),
        )
        mock_read_all.return_value = {"training_newinstance": running_task}

        # Tạo orchestrator mới — simulating new session (empty _processes)
        orchestrator = TaskOrchestrator()
        assert orchestrator._processes == {}

        result = orchestrator.get_active_tasks_from_disk()

        assert len(result) == 1
        assert result[0].task_id == "training_newinstance"

    @patch("orchestrator.task_orchestrator.read_all_statuses")
    def test_returns_multiple_active_tasks(self, mock_read_all):
        """Trả về nhiều active tasks khi có nhiều task RUNNING với heartbeat fresh."""
        now = datetime.now()
        mock_read_all.return_value = {
            "training_active1": TaskStatus(
                task_id="training_active1",
                task_type=TaskType.TRAINING,
                state=TaskState.RUNNING,
                progress_pct=30.0,
                message="Training",
                heartbeat_ts=now - timedelta(seconds=15),
                started_at=now - timedelta(minutes=5),
            ),
            "backtest_active2": TaskStatus(
                task_id="backtest_active2",
                task_type=TaskType.BACKTEST,
                state=TaskState.RUNNING,
                progress_pct=50.0,
                message="Backtesting",
                heartbeat_ts=now - timedelta(seconds=20),
                started_at=now - timedelta(minutes=3),
            ),
        }

        orchestrator = TaskOrchestrator()
        result = orchestrator.get_active_tasks_from_disk()

        assert len(result) == 2
        task_ids = [t.task_id for t in result]
        assert "training_active1" in task_ids
        assert "backtest_active2" in task_ids

    @patch("orchestrator.task_orchestrator.read_all_statuses")
    def test_mixed_statuses_only_returns_active(self, mock_read_all):
        """Lọc đúng trong mix nhiều trạng thái khác nhau."""
        now = datetime.now()
        mock_read_all.return_value = {
            "training_running": TaskStatus(
                task_id="training_running",
                task_type=TaskType.TRAINING,
                state=TaskState.RUNNING,
                progress_pct=40.0,
                message="Running",
                heartbeat_ts=now - timedelta(seconds=5),
                started_at=now - timedelta(minutes=2),
            ),
            "backtest_completed": TaskStatus(
                task_id="backtest_completed",
                task_type=TaskType.BACKTEST,
                state=TaskState.COMPLETED,
                progress_pct=100.0,
                message="Done",
                heartbeat_ts=now - timedelta(seconds=5),
                started_at=now - timedelta(minutes=10),
            ),
            "training_stale": TaskStatus(
                task_id="training_stale",
                task_type=TaskType.TRAINING,
                state=TaskState.RUNNING,
                progress_pct=10.0,
                message="Frozen",
                heartbeat_ts=now - timedelta(seconds=120),  # > 60s threshold
                started_at=now - timedelta(minutes=15),
            ),
            "analysis_error": TaskStatus(
                task_id="analysis_error",
                task_type=TaskType.ANALYSIS,
                state=TaskState.ERROR,
                progress_pct=0.0,
                message="Error",
                heartbeat_ts=now - timedelta(seconds=5),
                started_at=now - timedelta(minutes=1),
                error="Some error",
            ),
        }

        orchestrator = TaskOrchestrator()
        result = orchestrator.get_active_tasks_from_disk()

        # Chỉ training_running thỏa cả 2 điều kiện: RUNNING + heartbeat fresh
        assert len(result) == 1
        assert result[0].task_id == "training_running"

    @patch("orchestrator.task_orchestrator.read_all_statuses")
    def test_heartbeat_at_exact_threshold_excluded(self, mock_read_all):
        """Task có heartbeat đúng bằng HEARTBEAT_TIMEOUT_THRESHOLD bị loại bỏ."""
        now = datetime.now()
        # heartbeat_age == HEARTBEAT_TIMEOUT_THRESHOLD (không < threshold)
        borderline_task = TaskStatus(
            task_id="training_border",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=50.0,
            message="Borderline",
            heartbeat_ts=now - timedelta(seconds=HEARTBEAT_TIMEOUT_THRESHOLD),
            started_at=now - timedelta(minutes=5),
        )
        mock_read_all.return_value = {"training_border": borderline_task}

        orchestrator = TaskOrchestrator()
        result = orchestrator.get_active_tasks_from_disk()

        # heartbeat_age_seconds == HEARTBEAT_TIMEOUT_THRESHOLD → không < threshold → loại bỏ
        assert len(result) == 0

    @patch("orchestrator.task_orchestrator.read_all_statuses")
    def test_heartbeat_just_within_threshold_included(self, mock_read_all):
        """Task có heartbeat ngay dưới threshold được bao gồm."""
        now = datetime.now()
        # heartbeat cách 59s — nhỏ hơn threshold 60s
        fresh_task = TaskStatus(
            task_id="training_justfresh",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=70.0,
            message="Just fresh",
            heartbeat_ts=now - timedelta(seconds=HEARTBEAT_TIMEOUT_THRESHOLD - 1),
            started_at=now - timedelta(minutes=3),
        )
        mock_read_all.return_value = {"training_justfresh": fresh_task}

        orchestrator = TaskOrchestrator()
        result = orchestrator.get_active_tasks_from_disk()

        assert len(result) == 1
        assert result[0].task_id == "training_justfresh"

    @patch("orchestrator.task_orchestrator.read_all_statuses")
    def test_task_metadata_preserved_in_result(self, mock_read_all):
        """Metadata (task_id, task_type, start_time, progress) được giữ nguyên trong kết quả."""
        now = datetime.now()
        started = now - timedelta(minutes=10)
        task = TaskStatus(
            task_id="training_meta",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=55.5,
            message="Training epoch 10/20",
            heartbeat_ts=now - timedelta(seconds=3),
            started_at=started,
            details={"epochs": 20, "pid": 12345},
        )
        mock_read_all.return_value = {"training_meta": task}

        orchestrator = TaskOrchestrator()
        result = orchestrator.get_active_tasks_from_disk()

        assert len(result) == 1
        active = result[0]
        assert active.task_id == "training_meta"
        assert active.task_type == TaskType.TRAINING
        assert active.started_at == started
        assert active.progress_pct == 55.5
        assert active.details == {"epochs": 20, "pid": 12345}
