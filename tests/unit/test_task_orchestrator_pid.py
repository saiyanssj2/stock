# -*- coding: utf-8 -*-
"""
Unit test cho Task 3.2: Store PID in status file when starting task.

Verify rằng sau khi start_task(), PID được ghi vào status file details["pid"].
"""

import multiprocessing
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from models.task_models import TaskState, TaskStatus, TaskType
from orchestrator.status_protocol import read_status, write_status
from orchestrator.task_orchestrator import TaskOrchestrator


@pytest.fixture(autouse=True)
def clean_status_dir(tmp_path, monkeypatch):
    """Sử dụng tmp_path cho STATUS_DIR để không ảnh hưởng data thật."""
    monkeypatch.setattr("orchestrator.status_protocol.STATUS_DIR", str(tmp_path))
    monkeypatch.setattr("config.settings.STATUS_DIR", str(tmp_path))
    yield tmp_path


def _dummy_worker(task_type, task_id, params):
    """Worker giả - chỉ sleep ngắn rồi exit."""
    time.sleep(0.5)


class TestStartTaskStoresPID:
    """Verify start_task() ghi PID vào status file."""

    @patch("orchestrator.task_orchestrator.worker_entry", _dummy_worker)
    def test_pid_stored_in_status_file_after_start(self, clean_status_dir):
        """Sau start_task(), status file phải chứa details["pid"] là int > 0."""
        orchestrator = TaskOrchestrator()
        task_id = orchestrator.start_task(TaskType.TRAINING, {"phase": "phase_c"})

        # Đọc status file
        status = read_status(task_id)

        assert status is not None, "Status file phải tồn tại sau start_task()"
        assert "pid" in status.details, "Status file phải chứa 'pid' trong details"
        assert isinstance(status.details["pid"], int), "PID phải là int"
        assert status.details["pid"] > 0, "PID phải > 0"

        # Cleanup: stop process
        orchestrator.stop_task(task_id)

    @patch("orchestrator.task_orchestrator.worker_entry", _dummy_worker)
    def test_pid_matches_actual_process_pid(self, clean_status_dir):
        """PID trong status file phải match với process.pid thực tế."""
        orchestrator = TaskOrchestrator()
        task_id = orchestrator.start_task(TaskType.BACKTEST, {"mode": "auto"})

        # Lấy process từ _processes dict
        process = orchestrator._processes.get(task_id)
        assert process is not None, "Process phải có trong _processes dict"

        # Đọc PID từ status file
        status = read_status(task_id)
        assert status.details["pid"] == process.pid, (
            f"PID trong status file ({status.details['pid']}) "
            f"phải match process.pid ({process.pid})"
        )

        # Cleanup
        orchestrator.stop_task(task_id)

    @patch("orchestrator.task_orchestrator.worker_entry", _dummy_worker)
    def test_pid_available_immediately_after_start(self, clean_status_dir):
        """PID phải available ngay sau start_task() return, không cần chờ worker."""
        orchestrator = TaskOrchestrator()
        task_id = orchestrator.start_task(TaskType.TRAINING, {"epochs": 5})

        # Đọc ngay lập tức - không sleep
        status = read_status(task_id)
        assert status is not None
        assert "pid" in status.details
        assert status.details["pid"] > 0

        # Cleanup
        orchestrator.stop_task(task_id)

    @patch("orchestrator.task_orchestrator.worker_entry", _dummy_worker)
    def test_original_params_preserved_alongside_pid(self, clean_status_dir):
        """Params gốc trong details không bị mất khi thêm PID."""
        params = {"phase": "phase_c", "epochs": 10, "batch_size": 32}
        orchestrator = TaskOrchestrator()
        task_id = orchestrator.start_task(TaskType.TRAINING, params)

        status = read_status(task_id)

        # PID phải có
        assert "pid" in status.details

        # Params gốc phải preserved
        assert status.details["phase"] == "phase_c"
        assert status.details["epochs"] == 10
        assert status.details["batch_size"] == 32

        # Cleanup
        orchestrator.stop_task(task_id)
