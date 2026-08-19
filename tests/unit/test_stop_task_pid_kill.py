# -*- coding: utf-8 -*-
"""
Unit tests cho Task 3.3: Fix stop_task() to work without process reference (kill via PID).

Verify rằng khi _processes[task_id] không tồn tại, stop_task() vẫn có thể terminate
process thông qua PID đọc từ status file.

Test cases:
1. Stop via PID khi process không trong _processes
2. Stop khi PID invalid (process đã exit) — chỉ update status file
3. Stop khi không có PID trong status file (legacy status)
4. Existing behavior preserved (stop với process reference)
5. PermissionError khi kill — vẫn update status file
"""

import os
import signal
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from models.task_models import TaskState, TaskStatus, TaskType
from orchestrator.task_orchestrator import TaskOrchestrator


@pytest.fixture
def orchestrator():
    """Tạo TaskOrchestrator instance mới cho mỗi test."""
    return TaskOrchestrator()


@pytest.fixture
def running_status_with_pid():
    """Tạo TaskStatus RUNNING có PID trong details."""
    return TaskStatus(
        task_id="training_abc12345",
        task_type=TaskType.TRAINING,
        state=TaskState.RUNNING,
        progress_pct=50.0,
        message="Training in progress",
        heartbeat_ts=datetime.now(),
        started_at=datetime.now(),
        details={"pid": 99999, "phase": "phase_c"},
    )


@pytest.fixture
def running_status_without_pid():
    """Tạo TaskStatus RUNNING không có PID (legacy status)."""
    return TaskStatus(
        task_id="training_legacy01",
        task_type=TaskType.TRAINING,
        state=TaskState.RUNNING,
        progress_pct=30.0,
        message="Legacy task running",
        heartbeat_ts=datetime.now(),
        started_at=datetime.now(),
        details={"phase": "phase_c"},
    )


class TestStopTaskViaPID:
    """Tests cho stop_task() khi process không trong _processes — kill via PID."""

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("orchestrator.task_orchestrator.read_status")
    @patch("os.kill")
    @patch("time.sleep")
    def test_stop_via_pid_success(
        self, mock_sleep, mock_kill, mock_read, mock_write, orchestrator, running_status_with_pid
    ):
        """Stop task qua PID khi process không trong _processes — SIGTERM thành công."""
        mock_read.return_value = running_status_with_pid
        # os.kill(pid, SIGTERM) thành công, os.kill(pid, 0) raise ProcessLookupError (đã exit)
        mock_kill.side_effect = [None, ProcessLookupError("No such process")]

        result = orchestrator.stop_task("training_abc12345")

        assert result is True
        # Verify SIGTERM được gửi
        mock_kill.assert_any_call(99999, signal.SIGTERM)
        # Verify status file được cập nhật thành COMPLETED
        mock_write.assert_called()
        written_status = mock_write.call_args[0][0]
        assert written_status.state == TaskState.COMPLETED
        assert written_status.message == "Task stopped by user"

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("orchestrator.task_orchestrator.read_status")
    @patch("os.kill")
    @patch("time.sleep")
    def test_stop_via_pid_needs_force_kill(
        self, mock_sleep, mock_kill, mock_read, mock_write, orchestrator, running_status_with_pid
    ):
        """Stop task qua PID khi SIGTERM không đủ — cần force kill."""
        mock_read.return_value = running_status_with_pid

        if os.name == "nt":
            # Windows: SIGTERM → ok, check (os.kill 0) → ok (still alive), SIGTERM lại
            mock_kill.side_effect = [None, None, None]
        else:
            # Linux: SIGTERM → ok, check (os.kill 0) → ok (still alive), SIGKILL
            mock_kill.side_effect = [None, None, None]

        result = orchestrator.stop_task("training_abc12345")

        assert result is True
        # Verify SIGTERM được gọi đầu tiên
        first_call = mock_kill.call_args_list[0]
        assert first_call[0] == (99999, signal.SIGTERM)
        # Verify có thêm call (check alive hoặc force kill)
        assert mock_kill.call_count >= 2

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("orchestrator.task_orchestrator.read_status")
    @patch("os.kill")
    @patch("time.sleep")
    def test_stop_via_pid_process_already_exited(
        self, mock_sleep, mock_kill, mock_read, mock_write, orchestrator, running_status_with_pid
    ):
        """Stop task khi PID đã exit trước khi gửi signal — ProcessLookupError."""
        mock_read.return_value = running_status_with_pid
        # os.kill raise ProcessLookupError ngay lần đầu (process đã chết)
        mock_kill.side_effect = ProcessLookupError("No such process")

        result = orchestrator.stop_task("training_abc12345")

        assert result is True
        # Vẫn update status file thành COMPLETED
        mock_write.assert_called()
        written_status = mock_write.call_args[0][0]
        assert written_status.state == TaskState.COMPLETED

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("orchestrator.task_orchestrator.read_status")
    @patch("os.kill")
    @patch("time.sleep")
    def test_stop_via_pid_permission_denied(
        self, mock_sleep, mock_kill, mock_read, mock_write, orchestrator, running_status_with_pid
    ):
        """Stop task khi không có quyền kill — PermissionError, vẫn update status."""
        mock_read.return_value = running_status_with_pid
        mock_kill.side_effect = PermissionError("Operation not permitted")

        result = orchestrator.stop_task("training_abc12345")

        assert result is True
        # Vẫn update status file dù không kill được process
        mock_write.assert_called()
        written_status = mock_write.call_args[0][0]
        assert written_status.state == TaskState.COMPLETED


class TestStopTaskNoPIDInStatus:
    """Tests cho stop_task() khi status file không chứa PID (legacy)."""

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("orchestrator.task_orchestrator.read_status")
    @patch("os.kill")
    def test_stop_no_pid_in_status_file(
        self, mock_kill, mock_read, mock_write, orchestrator, running_status_without_pid
    ):
        """Stop task khi status file không có PID — chỉ update status, không kill."""
        mock_read.return_value = running_status_without_pid

        result = orchestrator.stop_task("training_legacy01")

        assert result is True
        # os.kill KHÔNG được gọi vì không có PID
        mock_kill.assert_not_called()
        # Vẫn update status file thành COMPLETED
        mock_write.assert_called()
        written_status = mock_write.call_args[0][0]
        assert written_status.state == TaskState.COMPLETED

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("orchestrator.task_orchestrator.read_status")
    @patch("os.kill")
    def test_stop_status_file_not_found(
        self, mock_kill, mock_read, mock_write, orchestrator
    ):
        """Stop task khi status file không tồn tại — tạo fallback status."""
        # read_status trả None cho cả 2 lần gọi (lần đầu check PID, lần sau update)
        mock_read.return_value = None

        result = orchestrator.stop_task("nonexistent_task99")

        assert result is True
        # os.kill KHÔNG được gọi
        mock_kill.assert_not_called()
        # Vẫn ghi fallback status
        mock_write.assert_called()
        written_status = mock_write.call_args[0][0]
        assert written_status.state == TaskState.COMPLETED
        assert written_status.task_id == "nonexistent_task99"

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("orchestrator.task_orchestrator.read_status")
    @patch("os.kill")
    def test_stop_empty_details_dict(
        self, mock_kill, mock_read, mock_write, orchestrator
    ):
        """Stop task khi details là dict rỗng — không có PID."""
        status = TaskStatus(
            task_id="training_emptydet",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=10.0,
            message="Running",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
            details={},
        )
        mock_read.return_value = status

        result = orchestrator.stop_task("training_emptydet")

        assert result is True
        mock_kill.assert_not_called()


class TestStopTaskExistingBehaviorPreserved:
    """Tests verify stop_task() với process reference vẫn hoạt động như cũ."""

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("orchestrator.task_orchestrator.read_status")
    def test_stop_with_process_reference_terminates(
        self, mock_read, mock_write, orchestrator
    ):
        """Stop task có process trong _processes — terminate process trực tiếp."""
        mock_proc = MagicMock()
        mock_proc.is_alive.side_effect = [True, False]  # Alive trước terminate, dead sau

        mock_read.return_value = TaskStatus(
            task_id="training_withproc",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=60.0,
            message="Training",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
            details={"pid": 12345},
        )

        orchestrator._processes["training_withproc"] = mock_proc
        result = orchestrator.stop_task("training_withproc")

        assert result is True
        mock_proc.terminate.assert_called_once()
        assert "training_withproc" not in orchestrator._processes
        # Status updated to COMPLETED
        mock_write.assert_called()
        written_status = mock_write.call_args[0][0]
        assert written_status.state == TaskState.COMPLETED

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("orchestrator.task_orchestrator.read_status")
    def test_stop_with_process_force_kill_if_unresponsive(
        self, mock_read, mock_write, orchestrator
    ):
        """Stop task — force kill nếu process không respond to terminate."""
        mock_proc = MagicMock()
        mock_proc.is_alive.side_effect = [True, True, False]  # Vẫn alive sau terminate

        mock_read.return_value = TaskStatus(
            task_id="training_stuck01",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=80.0,
            message="Stuck",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
        )

        orchestrator._processes["training_stuck01"] = mock_proc
        result = orchestrator.stop_task("training_stuck01")

        assert result is True
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()

    @patch("orchestrator.task_orchestrator.write_status")
    @patch("orchestrator.task_orchestrator.read_status")
    def test_stop_with_dead_process_reference(
        self, mock_read, mock_write, orchestrator
    ):
        """Stop task khi process đã chết nhưng vẫn trong _processes."""
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False  # Đã chết

        mock_read.return_value = TaskStatus(
            task_id="training_deadref",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=90.0,
            message="Was running",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
        )

        orchestrator._processes["training_deadref"] = mock_proc
        result = orchestrator.stop_task("training_deadref")

        assert result is True
        # Không gọi terminate vì process đã chết
        mock_proc.terminate.assert_not_called()
        # Vẫn xóa khỏi tracking
        assert "training_deadref" not in orchestrator._processes
