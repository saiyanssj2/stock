# -*- coding: utf-8 -*-
"""
Unit tests cho TaskOrchestrator - lifecycle cơ bản với mocked processes.

Kiểm tra:
- start_task() tạo task_id đúng format, ghi status, spawn process
- stop_task() dừng process gracefully, cập nhật status
- get_task_status() delegate đúng cho status_protocol
- get_all_statuses() trả về tất cả active tasks
- Failure isolation: một task crash không ảnh hưởng task khác
"""

import multiprocessing
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from models.task_models import TaskState, TaskStatus, TaskType
from orchestrator.task_orchestrator import TaskOrchestrator, worker_entry


class TestWorkerEntry:
    """Tests cho worker_entry function."""

    @patch("orchestrator.task_orchestrator.write_status")
    def test_worker_entry_writes_running_status(self, mock_write_status):
        """worker_entry ghi status RUNNING khi được gọi."""
        task_type = TaskType.TRAINING
        task_id = "training_abc12345"
        params = {"symbol": "VNM", "epochs": 100}

        worker_entry(task_type, task_id, params)

        mock_write_status.assert_called_once()
        written_status = mock_write_status.call_args[0][0]
        assert written_status.task_id == task_id
        assert written_status.task_type == task_type
        assert written_status.state == TaskState.RUNNING
        assert written_status.progress_pct == 0.0
        assert written_status.details == params

    @patch("orchestrator.task_orchestrator.write_status")
    def test_worker_entry_sets_heartbeat_timestamp(self, mock_write_status):
        """worker_entry ghi heartbeat_ts là thời điểm hiện tại."""
        before = datetime.now()
        worker_entry(TaskType.BACKTEST, "backtest_xyz", {})
        after = datetime.now()

        written_status = mock_write_status.call_args[0][0]
        assert before <= written_status.heartbeat_ts <= after
        assert before <= written_status.started_at <= after


class TestStartTask:
    """Tests cho TaskOrchestrator.start_task()."""

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("multiprocessing.Process")
    def test_start_task_returns_valid_task_id(self, mock_process_cls, mock_write_status):
        """start_task trả về task_id đúng format: {task_type}_{hex8}."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.is_alive.return_value = True
        mock_process_cls.return_value = mock_proc

        orchestrator = TaskOrchestrator()
        task_id = orchestrator.start_task(TaskType.TRAINING, {"epochs": 50})

        assert task_id.startswith("training_")
        assert len(task_id) == len("training_") + 8  # 8 hex chars

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("multiprocessing.Process")
    def test_start_task_spawns_process(self, mock_process_cls, mock_write_status):
        """start_task spawn multiprocessing.Process và gọi start()."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_process_cls.return_value = mock_proc

        orchestrator = TaskOrchestrator()
        orchestrator.start_task(TaskType.BACKTEST, {"symbol": "FPT"})

        mock_process_cls.assert_called_once()
        mock_proc.start.assert_called_once()

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("multiprocessing.Process")
    def test_start_task_writes_initial_status(self, mock_process_cls, mock_write_status):
        """start_task ghi initial RUNNING status trước khi spawn."""
        mock_proc = MagicMock()
        mock_proc.pid = 100
        mock_process_cls.return_value = mock_proc

        orchestrator = TaskOrchestrator()
        task_id = orchestrator.start_task(TaskType.ANALYSIS, {"mode": "scan"})

        # Phải ghi status ít nhất 1 lần
        assert mock_write_status.call_count >= 1
        first_status = mock_write_status.call_args_list[0][0][0]
        assert first_status.task_id == task_id
        assert first_status.state == TaskState.RUNNING
        assert first_status.task_type == TaskType.ANALYSIS

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("multiprocessing.Process")
    def test_start_task_tracks_process(self, mock_process_cls, mock_write_status):
        """start_task lưu process vào _processes dict."""
        mock_proc = MagicMock()
        mock_proc.pid = 200
        mock_process_cls.return_value = mock_proc

        orchestrator = TaskOrchestrator()
        task_id = orchestrator.start_task(TaskType.TRAINING, {})

        assert task_id in orchestrator._processes
        assert orchestrator._processes[task_id] is mock_proc

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("multiprocessing.Process")
    def test_start_multiple_tasks_unique_ids(self, mock_process_cls, mock_write_status):
        """Nhiều lần start_task tạo task_id khác nhau."""
        mock_proc = MagicMock()
        mock_proc.pid = 300
        mock_process_cls.return_value = mock_proc

        orchestrator = TaskOrchestrator()
        id1 = orchestrator.start_task(TaskType.TRAINING, {})
        id2 = orchestrator.start_task(TaskType.TRAINING, {})
        id3 = orchestrator.start_task(TaskType.BACKTEST, {})

        assert id1 != id2
        assert id1 != id3
        assert id2 != id3


class TestStopTask:
    """Tests cho TaskOrchestrator.stop_task()."""

    @patch("orchestrator.task_orchestrator.read_status")
    @patch("orchestrator.task_orchestrator.write_status")
    def test_stop_task_nonexistent_returns_true(self, mock_write, mock_read):
        """stop_task trả về True khi task_id không trong _processes — vẫn update status file."""
        mock_read.return_value = None  # Status file không tồn tại
        orchestrator = TaskOrchestrator()
        result = orchestrator.stop_task("nonexistent_task")
        assert result is True

    @patch("orchestrator.task_orchestrator.read_status")
    @patch("orchestrator.task_orchestrator.write_status")
    @patch("multiprocessing.Process")
    def test_stop_task_terminates_process(self, mock_process_cls, mock_write, mock_read):
        """stop_task gọi terminate() trên process đang chạy."""
        mock_proc = MagicMock()
        mock_proc.pid = 500
        # is_alive trả True lần đầu (process đang chạy), False sau terminate
        mock_proc.is_alive.side_effect = [True, False]
        mock_process_cls.return_value = mock_proc

        mock_read.return_value = TaskStatus(
            task_id="training_12345678",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=50.0,
            message="Training in progress",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
        )

        orchestrator = TaskOrchestrator()
        orchestrator._processes["training_12345678"] = mock_proc

        result = orchestrator.stop_task("training_12345678")

        assert result is True
        mock_proc.terminate.assert_called_once()

    @patch("orchestrator.task_orchestrator.read_status")
    @patch("orchestrator.task_orchestrator.write_status")
    def test_stop_task_removes_from_tracking(self, mock_write, mock_read):
        """stop_task xóa process khỏi _processes dict."""
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False

        mock_read.return_value = TaskStatus(
            task_id="backtest_abcdef12",
            task_type=TaskType.BACKTEST,
            state=TaskState.RUNNING,
            progress_pct=30.0,
            message="Running",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
        )

        orchestrator = TaskOrchestrator()
        orchestrator._processes["backtest_abcdef12"] = mock_proc

        orchestrator.stop_task("backtest_abcdef12")

        assert "backtest_abcdef12" not in orchestrator._processes

    @patch("orchestrator.task_orchestrator.read_status")
    @patch("orchestrator.task_orchestrator.write_status")
    def test_stop_task_updates_status_to_completed(self, mock_write, mock_read):
        """stop_task cập nhật status thành COMPLETED."""
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False

        mock_read.return_value = TaskStatus(
            task_id="analysis_99887766",
            task_type=TaskType.ANALYSIS,
            state=TaskState.RUNNING,
            progress_pct=80.0,
            message="Analyzing",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
        )

        orchestrator = TaskOrchestrator()
        orchestrator._processes["analysis_99887766"] = mock_proc

        orchestrator.stop_task("analysis_99887766")

        # Kiểm tra write_status được gọi với state COMPLETED
        mock_write.assert_called()
        written_status = mock_write.call_args[0][0]
        assert written_status.state == TaskState.COMPLETED
        assert written_status.message == "Task stopped by user"

    @patch("orchestrator.task_orchestrator.read_status")
    @patch("orchestrator.task_orchestrator.write_status")
    def test_stop_task_force_kills_unresponsive(self, mock_write, mock_read):
        """stop_task force kill nếu process không dừng sau timeout."""
        mock_proc = MagicMock()
        # is_alive trả về True lần đầu (sau terminate), False sau kill
        mock_proc.is_alive.side_effect = [True, True, False]

        mock_read.return_value = TaskStatus(
            task_id="training_deadbeef",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=10.0,
            message="Stuck",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
        )

        orchestrator = TaskOrchestrator()
        orchestrator._processes["training_deadbeef"] = mock_proc

        result = orchestrator.stop_task("training_deadbeef")

        assert result is True
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()


class TestGetTaskStatus:
    """Tests cho TaskOrchestrator.get_task_status()."""

    @patch("orchestrator.task_orchestrator.read_status")
    def test_get_task_status_delegates_to_protocol(self, mock_read):
        """get_task_status delegate cho status_protocol.read_status()."""
        expected_status = TaskStatus(
            task_id="training_aabbccdd",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=45.0,
            message="Training epoch 5/100",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
        )
        mock_read.return_value = expected_status

        orchestrator = TaskOrchestrator()
        result = orchestrator.get_task_status("training_aabbccdd")

        mock_read.assert_called_once_with("training_aabbccdd")
        assert result is expected_status

    @patch("orchestrator.task_orchestrator.read_status")
    def test_get_task_status_returns_none_for_unknown(self, mock_read):
        """get_task_status trả về None nếu task không tồn tại."""
        mock_read.return_value = None

        orchestrator = TaskOrchestrator()
        result = orchestrator.get_task_status("unknown_task")

        assert result is None


class TestGetAllStatuses:
    """Tests cho TaskOrchestrator.get_all_statuses()."""

    @patch("orchestrator.task_orchestrator.read_all_statuses")
    def test_get_all_statuses_delegates_to_protocol(self, mock_read_all):
        """get_all_statuses delegate cho status_protocol.read_all_statuses()."""
        now = datetime.now()
        expected = {
            "training_11111111": TaskStatus(
                task_id="training_11111111",
                task_type=TaskType.TRAINING,
                state=TaskState.RUNNING,
                progress_pct=20.0,
                message="Running",
                heartbeat_ts=now,
                started_at=now,
            ),
            "backtest_22222222": TaskStatus(
                task_id="backtest_22222222",
                task_type=TaskType.BACKTEST,
                state=TaskState.COMPLETED,
                progress_pct=100.0,
                message="Done",
                heartbeat_ts=now,
                started_at=now,
            ),
        }
        mock_read_all.return_value = expected

        orchestrator = TaskOrchestrator()
        result = orchestrator.get_all_statuses()

        mock_read_all.assert_called_once()
        assert result == expected

    @patch("orchestrator.task_orchestrator.read_all_statuses")
    def test_get_all_statuses_empty(self, mock_read_all):
        """get_all_statuses trả về dict rỗng khi không có task nào."""
        mock_read_all.return_value = {}

        orchestrator = TaskOrchestrator()
        result = orchestrator.get_all_statuses()

        assert result == {}


class TestFailureIsolation:
    """Tests cho failure isolation - một task crash không ảnh hưởng task khác."""

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("multiprocessing.Process")
    def test_crash_does_not_affect_other_tasks(self, mock_process_cls, mock_write):
        """Khi một process crash, các process khác vẫn được tracked."""
        # Tạo 2 mock processes
        proc1 = MagicMock()
        proc1.pid = 1001
        proc1.is_alive.return_value = False  # Process 1 đã crash

        proc2 = MagicMock()
        proc2.pid = 1002
        proc2.is_alive.return_value = True  # Process 2 vẫn chạy

        mock_process_cls.side_effect = [proc1, proc2]

        orchestrator = TaskOrchestrator()
        id1 = orchestrator.start_task(TaskType.TRAINING, {})
        id2 = orchestrator.start_task(TaskType.BACKTEST, {})

        # Process 1 crash nhưng process 2 vẫn trong tracking
        assert id1 in orchestrator._processes
        assert id2 in orchestrator._processes
        assert orchestrator._processes[id2].is_alive()

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("multiprocessing.Process")
    def test_cleanup_removes_finished_only(self, mock_process_cls, mock_write):
        """cleanup_finished_processes chỉ xóa process đã kết thúc."""
        proc_alive = MagicMock()
        proc_alive.pid = 2001
        proc_alive.is_alive.return_value = True

        proc_dead = MagicMock()
        proc_dead.pid = 2002
        proc_dead.is_alive.return_value = False

        mock_process_cls.side_effect = [proc_alive, proc_dead]

        orchestrator = TaskOrchestrator()
        id_alive = orchestrator.start_task(TaskType.TRAINING, {})
        id_dead = orchestrator.start_task(TaskType.BACKTEST, {})

        orchestrator.cleanup_finished_processes()

        assert id_alive in orchestrator._processes
        assert id_dead not in orchestrator._processes


class TestConcurrentTaskManagement:
    """Tests cho concurrent task management - quản lý nhiều task đồng thời."""

    @patch("orchestrator.task_orchestrator.read_status")
    @patch("orchestrator.task_orchestrator.write_status")
    @patch("multiprocessing.Process")
    def test_start_and_stop_individual_tasks(self, mock_process_cls, mock_write, mock_read):
        """Start 3 tasks, stop 1, các task còn lại vẫn active."""
        procs = []
        for i in range(3):
            p = MagicMock()
            p.pid = 3000 + i
            p.is_alive.return_value = True
            procs.append(p)

        mock_process_cls.side_effect = procs

        orchestrator = TaskOrchestrator()
        id1 = orchestrator.start_task(TaskType.TRAINING, {"symbol": "VNM"})
        id2 = orchestrator.start_task(TaskType.BACKTEST, {"symbol": "FPT"})
        id3 = orchestrator.start_task(TaskType.ANALYSIS, {"mode": "scan"})

        # Tất cả 3 task đều được tracked
        assert len(orchestrator._processes) == 3

        # Stop task 2 - mock is_alive = False sau terminate
        procs[1].is_alive.return_value = False
        mock_read.return_value = TaskStatus(
            task_id=id2,
            task_type=TaskType.BACKTEST,
            state=TaskState.RUNNING,
            progress_pct=50.0,
            message="Running",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
        )

        result = orchestrator.stop_task(id2)

        assert result is True
        assert id2 not in orchestrator._processes
        assert id1 in orchestrator._processes
        assert id3 in orchestrator._processes
        assert len(orchestrator._processes) == 2

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("multiprocessing.Process")
    def test_multiple_task_types_coexist(self, mock_process_cls, mock_write):
        """Nhiều loại task khác nhau chạy đồng thời."""
        mock_proc = MagicMock()
        mock_proc.pid = 4000
        mock_proc.is_alive.return_value = True
        mock_process_cls.return_value = mock_proc

        orchestrator = TaskOrchestrator()
        id_train = orchestrator.start_task(TaskType.TRAINING, {})
        id_back = orchestrator.start_task(TaskType.BACKTEST, {})
        id_anal = orchestrator.start_task(TaskType.ANALYSIS, {})

        # Kiểm tra prefix đúng loại task
        assert id_train.startswith("training_")
        assert id_back.startswith("backtest_")
        assert id_anal.startswith("analysis_")

        # Tất cả đều được tracked
        assert id_train in orchestrator._processes
        assert id_back in orchestrator._processes
        assert id_anal in orchestrator._processes

    @patch("orchestrator.task_orchestrator.read_status")
    @patch("orchestrator.task_orchestrator.write_status")
    @patch("multiprocessing.Process")
    def test_stop_all_tasks_sequentially(self, mock_process_cls, mock_write, mock_read):
        """Stop tất cả tasks lần lượt, cuối cùng _processes rỗng."""
        procs = []
        for i in range(3):
            p = MagicMock()
            p.pid = 5000 + i
            p.is_alive.return_value = False
            procs.append(p)

        mock_process_cls.side_effect = procs

        orchestrator = TaskOrchestrator()
        ids = [
            orchestrator.start_task(TaskType.TRAINING, {}),
            orchestrator.start_task(TaskType.BACKTEST, {}),
            orchestrator.start_task(TaskType.ANALYSIS, {}),
        ]

        for task_id in ids:
            mock_read.return_value = TaskStatus(
                task_id=task_id,
                task_type=TaskType.TRAINING,
                state=TaskState.RUNNING,
                progress_pct=0.0,
                message="Running",
                heartbeat_ts=datetime.now(),
                started_at=datetime.now(),
            )
            orchestrator.stop_task(task_id)

        assert len(orchestrator._processes) == 0


class TestLifecycleEdgeCases:
    """Tests cho edge cases trong lifecycle transitions."""

    @patch("orchestrator.task_orchestrator.read_status")
    @patch("orchestrator.task_orchestrator.write_status")
    def test_stop_task_when_read_status_returns_none(self, mock_write, mock_read):
        """stop_task xử lý đúng khi read_status trả None (fallback status)."""
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        mock_read.return_value = None  # Status file không tồn tại

        orchestrator = TaskOrchestrator()
        orchestrator._processes["orphan_task_123"] = mock_proc

        result = orchestrator.stop_task("orphan_task_123")

        assert result is True
        assert "orphan_task_123" not in orchestrator._processes
        # Vẫn ghi status COMPLETED dù read trả None
        mock_write.assert_called()
        written_status = mock_write.call_args[0][0]
        assert written_status.state == TaskState.COMPLETED
        assert written_status.task_id == "orphan_task_123"

    @patch("orchestrator.task_orchestrator.read_status")
    @patch("orchestrator.task_orchestrator.write_status")
    def test_stop_already_dead_process(self, mock_write, mock_read):
        """stop_task xử lý process đã chết trước khi gọi stop."""
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False  # Process đã chết

        mock_read.return_value = TaskStatus(
            task_id="training_deadproc",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=75.0,
            message="Was running",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
        )

        orchestrator = TaskOrchestrator()
        orchestrator._processes["training_deadproc"] = mock_proc

        result = orchestrator.stop_task("training_deadproc")

        # Vẫn trả True và cleanup đúng
        assert result is True
        assert "training_deadproc" not in orchestrator._processes
        # Không gọi terminate vì process đã chết
        mock_proc.terminate.assert_not_called()

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("multiprocessing.Process")
    def test_start_task_with_empty_params(self, mock_process_cls, mock_write):
        """start_task chấp nhận empty params dict."""
        mock_proc = MagicMock()
        mock_proc.pid = 6000
        mock_process_cls.return_value = mock_proc

        orchestrator = TaskOrchestrator()
        task_id = orchestrator.start_task(TaskType.TRAINING, {})

        assert task_id is not None
        assert task_id in orchestrator._processes
        mock_proc.start.assert_called_once()


class TestFailureIsolationEdgeCases:
    """Tests bổ sung cho failure isolation scenarios."""

    @patch("orchestrator.task_orchestrator.read_status")
    @patch("orchestrator.task_orchestrator.write_status")
    @patch("multiprocessing.Process")
    def test_stop_crashed_task_does_not_affect_running_tasks(
        self, mock_process_cls, mock_write, mock_read
    ):
        """Stop một task đã crash không ảnh hưởng task đang chạy."""
        proc_running = MagicMock()
        proc_running.pid = 7001
        proc_running.is_alive.return_value = True

        proc_crashed = MagicMock()
        proc_crashed.pid = 7002
        proc_crashed.is_alive.return_value = False  # Đã crash

        mock_process_cls.side_effect = [proc_running, proc_crashed]

        orchestrator = TaskOrchestrator()
        id_running = orchestrator.start_task(TaskType.TRAINING, {"symbol": "HPG"})
        id_crashed = orchestrator.start_task(TaskType.BACKTEST, {"symbol": "FPT"})

        mock_read.return_value = TaskStatus(
            task_id=id_crashed,
            task_type=TaskType.BACKTEST,
            state=TaskState.RUNNING,
            progress_pct=30.0,
            message="Crashed",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
        )

        # Stop crashed task
        orchestrator.stop_task(id_crashed)

        # Running task vẫn nguyên vẹn
        assert id_running in orchestrator._processes
        assert orchestrator._processes[id_running].is_alive()

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("multiprocessing.Process")
    def test_cleanup_with_mix_of_alive_and_dead(self, mock_process_cls, mock_write):
        """cleanup_finished_processes xử lý đúng khi mix alive và dead."""
        procs = []
        # 2 alive, 3 dead
        for i in range(5):
            p = MagicMock()
            p.pid = 8000 + i
            p.is_alive.return_value = i < 2  # 0,1 alive; 2,3,4 dead
            procs.append(p)

        mock_process_cls.side_effect = procs

        orchestrator = TaskOrchestrator()
        ids = []
        for i in range(5):
            task_type = [TaskType.TRAINING, TaskType.BACKTEST, TaskType.ANALYSIS,
                         TaskType.TRAINING, TaskType.BACKTEST][i]
            ids.append(orchestrator.start_task(task_type, {}))

        assert len(orchestrator._processes) == 5

        orchestrator.cleanup_finished_processes()

        # Chỉ còn 2 processes alive
        assert len(orchestrator._processes) == 2
        assert ids[0] in orchestrator._processes
        assert ids[1] in orchestrator._processes
        for dead_id in ids[2:]:
            assert dead_id not in orchestrator._processes

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("multiprocessing.Process")
    def test_cleanup_empty_processes(self, mock_process_cls, mock_write):
        """cleanup_finished_processes không lỗi khi _processes rỗng."""
        orchestrator = TaskOrchestrator()
        # Không có process nào
        orchestrator.cleanup_finished_processes()
        assert len(orchestrator._processes) == 0
