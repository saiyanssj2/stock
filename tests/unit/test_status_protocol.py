# -*- coding: utf-8 -*-
"""
Unit tests cho orchestrator/status_protocol.py

Kiểm tra:
- write_status: atomic write TaskStatus ra JSON file
- read_status: đọc và deserialize từ JSON, xử lý file not found / corrupt
- read_all_statuses: đọc tất cả status files
- delete_status: xóa status file
- Serialization/deserialization datetime, Enum fields
"""

import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from models.task_models import TaskState, TaskStatus, TaskType
from orchestrator.status_protocol import (
    _dict_to_task_status,
    _task_status_to_dict,
    delete_status,
    read_all_statuses,
    read_status,
    write_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_status() -> TaskStatus:
    """Tạo một TaskStatus mẫu để test."""
    return TaskStatus(
        task_id="train_001",
        task_type=TaskType.TRAINING,
        state=TaskState.RUNNING,
        progress_pct=45.5,
        message="Training symbol FPT (3/10)",
        heartbeat_ts=datetime(2024, 6, 15, 10, 30, 0),
        started_at=datetime(2024, 6, 15, 9, 0, 0),
        error=None,
        details={"current_symbol": "FPT", "epoch": 5},
    )


@pytest.fixture
def status_with_error() -> TaskStatus:
    """Tạo TaskStatus có trạng thái lỗi."""
    return TaskStatus(
        task_id="backtest_002",
        task_type=TaskType.BACKTEST,
        state=TaskState.ERROR,
        progress_pct=70.0,
        message="Backtest failed at symbol HPG",
        heartbeat_ts=datetime(2024, 6, 15, 11, 0, 0),
        started_at=datetime(2024, 6, 15, 10, 0, 0),
        error="GPU OOM: insufficient memory",
        details={"failed_symbol": "HPG"},
    )


@pytest.fixture
def temp_status_dir(tmp_path, monkeypatch):
    """Patch STATUS_DIR để dùng thư mục tạm thời cho test."""
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    monkeypatch.setattr(
        "orchestrator.status_protocol.STATUS_DIR",
        str(status_dir),
    )
    return status_dir


# ---------------------------------------------------------------------------
# Test: _task_status_to_dict / _dict_to_task_status
# ---------------------------------------------------------------------------


class TestSerialization:
    """Test serialize/deserialize TaskStatus."""

    def test_serialize_basic(self, sample_status: TaskStatus):
        """Serialize TaskStatus thành dict với đúng format."""
        data = _task_status_to_dict(sample_status)

        assert data["task_id"] == "train_001"
        assert data["task_type"] == "training"
        assert data["state"] == "running"
        assert data["progress_pct"] == 45.5
        assert data["message"] == "Training symbol FPT (3/10)"
        assert data["heartbeat_ts"] == "2024-06-15T10:30:00"
        assert data["started_at"] == "2024-06-15T09:00:00"
        assert data["error"] is None
        assert data["details"] == {"current_symbol": "FPT", "epoch": 5}

    def test_serialize_with_error(self, status_with_error: TaskStatus):
        """Serialize TaskStatus có error field."""
        data = _task_status_to_dict(status_with_error)

        assert data["state"] == "error"
        assert data["error"] == "GPU OOM: insufficient memory"

    def test_deserialize_basic(self, sample_status: TaskStatus):
        """Deserialize dict thành TaskStatus round-trip."""
        data = _task_status_to_dict(sample_status)
        restored = _dict_to_task_status(data)

        assert restored.task_id == sample_status.task_id
        assert restored.task_type == sample_status.task_type
        assert restored.state == sample_status.state
        assert restored.progress_pct == sample_status.progress_pct
        assert restored.message == sample_status.message
        assert restored.heartbeat_ts == sample_status.heartbeat_ts
        assert restored.started_at == sample_status.started_at
        assert restored.error == sample_status.error
        assert restored.details == sample_status.details

    def test_deserialize_all_task_types(self):
        """Deserialize đúng tất cả TaskType enum values."""
        for task_type in TaskType:
            data = {
                "task_id": "test",
                "task_type": task_type.value,
                "state": "idle",
                "progress_pct": 0.0,
                "message": "",
                "heartbeat_ts": "2024-01-01T00:00:00",
                "started_at": "2024-01-01T00:00:00",
                "error": None,
                "details": {},
            }
            result = _dict_to_task_status(data)
            assert result.task_type == task_type

    def test_deserialize_all_task_states(self):
        """Deserialize đúng tất cả TaskState enum values."""
        for state in TaskState:
            data = {
                "task_id": "test",
                "task_type": "training",
                "state": state.value,
                "progress_pct": 0.0,
                "message": "",
                "heartbeat_ts": "2024-01-01T00:00:00",
                "started_at": "2024-01-01T00:00:00",
                "error": None,
                "details": {},
            }
            result = _dict_to_task_status(data)
            assert result.state == state


# ---------------------------------------------------------------------------
# Test: write_status
# ---------------------------------------------------------------------------


class TestWriteStatus:
    """Test write_status function."""

    def test_write_creates_json_file(self, temp_status_dir, sample_status):
        """write_status tạo file JSON đúng vị trí."""
        write_status(sample_status)

        expected_file = temp_status_dir / "train_001.json"
        assert expected_file.exists()

    def test_write_content_valid_json(self, temp_status_dir, sample_status):
        """File được ghi có nội dung JSON hợp lệ."""
        write_status(sample_status)

        file_path = temp_status_dir / "train_001.json"
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)

        assert data["task_id"] == "train_001"
        assert data["task_type"] == "training"

    def test_write_overwrites_existing(self, temp_status_dir, sample_status):
        """write_status ghi đè file cũ khi update status."""
        write_status(sample_status)

        # Cập nhật progress
        sample_status.progress_pct = 80.0
        sample_status.message = "Almost done"
        write_status(sample_status)

        file_path = temp_status_dir / "train_001.json"
        data = json.loads(file_path.read_text(encoding="utf-8"))
        assert data["progress_pct"] == 80.0
        assert data["message"] == "Almost done"

    def test_write_no_temp_file_left(self, temp_status_dir, sample_status):
        """Không còn temp file sau khi write thành công."""
        write_status(sample_status)

        tmp_files = list(temp_status_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_write_creates_status_dir_if_not_exists(self, tmp_path, monkeypatch):
        """write_status tự tạo thư mục status nếu chưa tồn tại."""
        new_dir = tmp_path / "new_status_dir"
        monkeypatch.setattr(
            "orchestrator.status_protocol.STATUS_DIR",
            str(new_dir),
        )

        status = TaskStatus(
            task_id="test_auto_dir",
            task_type=TaskType.ANALYSIS,
            state=TaskState.RUNNING,
            progress_pct=0.0,
            message="Starting",
            heartbeat_ts=datetime(2024, 1, 1, 0, 0, 0),
            started_at=datetime(2024, 1, 1, 0, 0, 0),
        )
        write_status(status)

        assert new_dir.exists()
        assert (new_dir / "test_auto_dir.json").exists()


# ---------------------------------------------------------------------------
# Test: read_status
# ---------------------------------------------------------------------------


class TestReadStatus:
    """Test read_status function."""

    def test_read_existing_status(self, temp_status_dir, sample_status):
        """Đọc status file đã ghi trước đó."""
        write_status(sample_status)
        result = read_status("train_001")

        assert result is not None
        assert result.task_id == "train_001"
        assert result.task_type == TaskType.TRAINING
        assert result.state == TaskState.RUNNING
        assert result.progress_pct == 45.5

    def test_read_nonexistent_returns_none(self, temp_status_dir):
        """Trả về None khi file không tồn tại."""
        result = read_status("nonexistent_task")
        assert result is None

    def test_read_corrupt_json_returns_none(self, temp_status_dir):
        """Trả về None khi file chứa JSON không hợp lệ."""
        corrupt_file = temp_status_dir / "corrupt_task.json"
        corrupt_file.write_text("{ invalid json content !!!", encoding="utf-8")

        result = read_status("corrupt_task")
        assert result is None

    def test_read_incomplete_json_returns_none(self, temp_status_dir):
        """Trả về None khi JSON thiếu field bắt buộc."""
        incomplete_file = temp_status_dir / "incomplete.json"
        incomplete_file.write_text(
            json.dumps({"task_id": "incomplete", "task_type": "training"}),
            encoding="utf-8",
        )

        result = read_status("incomplete")
        assert result is None

    def test_read_invalid_enum_value_returns_none(self, temp_status_dir):
        """Trả về None khi enum value không hợp lệ."""
        bad_enum_file = temp_status_dir / "bad_enum.json"
        bad_enum_file.write_text(
            json.dumps({
                "task_id": "bad_enum",
                "task_type": "invalid_type",
                "state": "running",
                "progress_pct": 0.0,
                "message": "",
                "heartbeat_ts": "2024-01-01T00:00:00",
                "started_at": "2024-01-01T00:00:00",
                "error": None,
                "details": {},
            }),
            encoding="utf-8",
        )

        result = read_status("bad_enum")
        assert result is None


# ---------------------------------------------------------------------------
# Test: read_all_statuses
# ---------------------------------------------------------------------------


class TestReadAllStatuses:
    """Test read_all_statuses function."""

    def test_read_all_empty_dir(self, temp_status_dir):
        """Trả về dict rỗng khi không có status file nào."""
        result = read_all_statuses()
        assert result == {}

    def test_read_all_multiple_statuses(self, temp_status_dir):
        """Đọc đúng tất cả status files hợp lệ."""
        statuses = [
            TaskStatus(
                task_id="train_001",
                task_type=TaskType.TRAINING,
                state=TaskState.RUNNING,
                progress_pct=50.0,
                message="Training",
                heartbeat_ts=datetime(2024, 1, 1, 10, 0, 0),
                started_at=datetime(2024, 1, 1, 9, 0, 0),
            ),
            TaskStatus(
                task_id="backtest_002",
                task_type=TaskType.BACKTEST,
                state=TaskState.COMPLETED,
                progress_pct=100.0,
                message="Done",
                heartbeat_ts=datetime(2024, 1, 1, 11, 0, 0),
                started_at=datetime(2024, 1, 1, 10, 0, 0),
            ),
        ]

        for s in statuses:
            write_status(s)

        result = read_all_statuses()
        assert len(result) == 2
        assert "train_001" in result
        assert "backtest_002" in result
        assert result["train_001"].state == TaskState.RUNNING
        assert result["backtest_002"].state == TaskState.COMPLETED

    def test_read_all_skips_corrupt_files(self, temp_status_dir):
        """Bỏ qua file corrupt, không crash."""
        # Ghi 1 file hợp lệ
        valid_status = TaskStatus(
            task_id="valid_task",
            task_type=TaskType.ANALYSIS,
            state=TaskState.RUNNING,
            progress_pct=25.0,
            message="Analyzing",
            heartbeat_ts=datetime(2024, 1, 1, 12, 0, 0),
            started_at=datetime(2024, 1, 1, 11, 0, 0),
        )
        write_status(valid_status)

        # Tạo 1 file corrupt
        corrupt_file = temp_status_dir / "corrupt.json"
        corrupt_file.write_text("not valid json!!!", encoding="utf-8")

        result = read_all_statuses()
        assert len(result) == 1
        assert "valid_task" in result

    def test_read_all_ignores_non_json_files(self, temp_status_dir):
        """Chỉ đọc file .json, bỏ qua file khác."""
        # Ghi status hợp lệ
        status = TaskStatus(
            task_id="task_x",
            task_type=TaskType.TRAINING,
            state=TaskState.IDLE,
            progress_pct=0.0,
            message="Idle",
            heartbeat_ts=datetime(2024, 1, 1, 0, 0, 0),
            started_at=datetime(2024, 1, 1, 0, 0, 0),
        )
        write_status(status)

        # Tạo file không phải .json
        (temp_status_dir / ".gitkeep").write_text("", encoding="utf-8")
        (temp_status_dir / "notes.txt").write_text("some notes", encoding="utf-8")

        result = read_all_statuses()
        assert len(result) == 1
        assert "task_x" in result


# ---------------------------------------------------------------------------
# Test: delete_status
# ---------------------------------------------------------------------------


class TestDeleteStatus:
    """Test delete_status function."""

    def test_delete_existing_file(self, temp_status_dir, sample_status):
        """Xóa thành công file status tồn tại."""
        write_status(sample_status)
        assert (temp_status_dir / "train_001.json").exists()

        result = delete_status("train_001")
        assert result is True
        assert not (temp_status_dir / "train_001.json").exists()

    def test_delete_nonexistent_returns_false(self, temp_status_dir):
        """Trả về False khi file không tồn tại."""
        result = delete_status("nonexistent_task")
        assert result is False

    def test_delete_then_read_returns_none(self, temp_status_dir, sample_status):
        """Sau khi xóa, read_status trả về None."""
        write_status(sample_status)
        delete_status("train_001")

        result = read_status("train_001")
        assert result is None


# ---------------------------------------------------------------------------
# Test: Round-trip (write → read)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Test write → read round-trip bảo toàn dữ liệu."""

    def test_roundtrip_preserves_all_fields(self, temp_status_dir, sample_status):
        """Write rồi read lại phải trả về data giống hệt."""
        write_status(sample_status)
        restored = read_status("train_001")

        assert restored is not None
        assert restored.task_id == sample_status.task_id
        assert restored.task_type == sample_status.task_type
        assert restored.state == sample_status.state
        assert restored.progress_pct == sample_status.progress_pct
        assert restored.message == sample_status.message
        assert restored.heartbeat_ts == sample_status.heartbeat_ts
        assert restored.started_at == sample_status.started_at
        assert restored.error == sample_status.error
        assert restored.details == sample_status.details

    def test_roundtrip_with_error_status(self, temp_status_dir, status_with_error):
        """Round-trip bảo toàn error field."""
        write_status(status_with_error)
        restored = read_status("backtest_002")

        assert restored is not None
        assert restored.error == "GPU OOM: insufficient memory"
        assert restored.state == TaskState.ERROR

    def test_roundtrip_empty_details(self, temp_status_dir):
        """Round-trip khi details là dict rỗng."""
        status = TaskStatus(
            task_id="minimal",
            task_type=TaskType.ANALYSIS,
            state=TaskState.COMPLETED,
            progress_pct=100.0,
            message="Completed",
            heartbeat_ts=datetime(2024, 6, 1, 12, 0, 0),
            started_at=datetime(2024, 6, 1, 11, 0, 0),
        )
        write_status(status)
        restored = read_status("minimal")

        assert restored is not None
        assert restored.details == {}
