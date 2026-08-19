# -*- coding: utf-8 -*-
"""
Unit tests cho engine/workers/training_checkpoint.py.

Kiểm tra:
- save_checkpoint: atomic write, serialize đầy đủ fields
- load_checkpoint: đọc đúng, trả None nếu missing/corrupt
- delete_checkpoint: xóa thành công, trả False nếu không tồn tại
- get_latest_checkpoint: tìm checkpoint mới nhất theo created_at
- resume_training_symbols: trả pending_symbols, skip completed

References: Req 1.8, 4.6, 4.9, 4.10
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from engine.workers.training_checkpoint import (
    DEFAULT_CHECKPOINT_DIR,
    delete_checkpoint,
    get_latest_checkpoint,
    load_checkpoint,
    resume_training_symbols,
    save_checkpoint,
)
from models.training_models import SessionCheckpoint, TrainingPhase


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Tạo thư mục tạm cho checkpoint tests."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return str(checkpoint_dir)


@pytest.fixture
def sample_checkpoint():
    """Tạo SessionCheckpoint mẫu để dùng trong tests."""
    return SessionCheckpoint(
        session_id="test-session-001",
        phase=TrainingPhase.PHASE_C,
        cycle_number=1,
        completed_symbols=["FPT", "VNM"],
        pending_symbols=["HPG", "MBB", "TCB"],
        current_symbol="HPG",
        current_epoch=5,
        total_epochs=10,
        model_path="data/engine/models/model_v1.pt",
        optimizer_state_path="data/engine/models/optimizer_v1.pt",
        created_at=datetime(2024, 6, 15, 10, 30, 0),
        metadata={"learning_rate": "0.001", "batch_size": "32"},
    )


class TestSaveCheckpoint:
    """Tests cho save_checkpoint function."""

    def test_save_creates_file(self, tmp_checkpoint_dir, sample_checkpoint):
        """save_checkpoint tạo file JSON đúng path."""
        result_path = save_checkpoint(sample_checkpoint, tmp_checkpoint_dir)

        assert os.path.exists(result_path)
        assert result_path.endswith("test-session-001.json")

    def test_save_returns_correct_path(self, tmp_checkpoint_dir, sample_checkpoint):
        """save_checkpoint trả về đường dẫn đến file đã lưu."""
        result_path = save_checkpoint(sample_checkpoint, tmp_checkpoint_dir)

        expected = str(Path(tmp_checkpoint_dir) / "test-session-001.json")
        assert result_path == expected

    def test_save_content_is_valid_json(self, tmp_checkpoint_dir, sample_checkpoint):
        """File được lưu phải là valid JSON."""
        result_path = save_checkpoint(sample_checkpoint, tmp_checkpoint_dir)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["session_id"] == "test-session-001"
        assert data["phase"] == "phase_c"
        assert data["cycle_number"] == 1

    def test_save_serializes_all_fields(self, tmp_checkpoint_dir, sample_checkpoint):
        """Tất cả fields phải được serialize đầy đủ."""
        result_path = save_checkpoint(sample_checkpoint, tmp_checkpoint_dir)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["completed_symbols"] == ["FPT", "VNM"]
        assert data["pending_symbols"] == ["HPG", "MBB", "TCB"]
        assert data["current_symbol"] == "HPG"
        assert data["current_epoch"] == 5
        assert data["total_epochs"] == 10
        assert data["model_path"] == "data/engine/models/model_v1.pt"
        assert data["optimizer_state_path"] == "data/engine/models/optimizer_v1.pt"
        assert data["created_at"] == "2024-06-15T10:30:00"
        assert data["metadata"] == {"learning_rate": "0.001", "batch_size": "32"}

    def test_save_serializes_datetime_as_iso(self, tmp_checkpoint_dir, sample_checkpoint):
        """datetime phải được serialize thành ISO format string."""
        result_path = save_checkpoint(sample_checkpoint, tmp_checkpoint_dir)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Kiểm tra parse ngược lại không lỗi
        parsed = datetime.fromisoformat(data["created_at"])
        assert parsed == datetime(2024, 6, 15, 10, 30, 0)

    def test_save_serializes_enum_as_value(self, tmp_checkpoint_dir, sample_checkpoint):
        """TrainingPhase enum phải được serialize thành string value."""
        result_path = save_checkpoint(sample_checkpoint, tmp_checkpoint_dir)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["phase"] == "phase_c"

    def test_save_creates_directory_if_not_exists(self, tmp_path, sample_checkpoint):
        """save_checkpoint tạo thư mục nếu chưa tồn tại."""
        new_dir = str(tmp_path / "new" / "nested" / "checkpoints")
        result_path = save_checkpoint(sample_checkpoint, new_dir)

        assert os.path.exists(result_path)

    def test_save_overwrites_existing_file(self, tmp_checkpoint_dir, sample_checkpoint):
        """save_checkpoint ghi đè file cũ nếu session_id trùng."""
        save_checkpoint(sample_checkpoint, tmp_checkpoint_dir)

        # Thay đổi checkpoint và lưu lại
        sample_checkpoint.current_epoch = 8
        save_checkpoint(sample_checkpoint, tmp_checkpoint_dir)

        loaded = load_checkpoint("test-session-001", tmp_checkpoint_dir)
        assert loaded is not None
        assert loaded.current_epoch == 8

    def test_save_no_temp_files_left(self, tmp_checkpoint_dir, sample_checkpoint):
        """Không có temp file nào bị bỏ lại sau khi save thành công."""
        save_checkpoint(sample_checkpoint, tmp_checkpoint_dir)

        files = os.listdir(tmp_checkpoint_dir)
        tmp_files = [f for f in files if f.endswith(".tmp")]
        assert len(tmp_files) == 0


class TestLoadCheckpoint:
    """Tests cho load_checkpoint function."""

    def test_load_returns_correct_checkpoint(self, tmp_checkpoint_dir, sample_checkpoint):
        """load_checkpoint trả về SessionCheckpoint đúng."""
        save_checkpoint(sample_checkpoint, tmp_checkpoint_dir)
        loaded = load_checkpoint("test-session-001", tmp_checkpoint_dir)

        assert loaded is not None
        assert loaded.session_id == "test-session-001"
        assert loaded.phase == TrainingPhase.PHASE_C
        assert loaded.cycle_number == 1
        assert loaded.completed_symbols == ["FPT", "VNM"]
        assert loaded.pending_symbols == ["HPG", "MBB", "TCB"]

    def test_load_preserves_all_fields(self, tmp_checkpoint_dir, sample_checkpoint):
        """Tất cả fields phải được preserve qua save/load round-trip."""
        save_checkpoint(sample_checkpoint, tmp_checkpoint_dir)
        loaded = load_checkpoint("test-session-001", tmp_checkpoint_dir)

        assert loaded is not None
        assert loaded.current_symbol == sample_checkpoint.current_symbol
        assert loaded.current_epoch == sample_checkpoint.current_epoch
        assert loaded.total_epochs == sample_checkpoint.total_epochs
        assert loaded.model_path == sample_checkpoint.model_path
        assert loaded.optimizer_state_path == sample_checkpoint.optimizer_state_path
        assert loaded.created_at == sample_checkpoint.created_at
        assert loaded.metadata == sample_checkpoint.metadata

    def test_load_returns_none_if_not_found(self, tmp_checkpoint_dir):
        """load_checkpoint trả về None nếu file không tồn tại."""
        result = load_checkpoint("nonexistent-session", tmp_checkpoint_dir)
        assert result is None

    def test_load_returns_none_if_corrupt_json(self, tmp_checkpoint_dir):
        """load_checkpoint trả về None nếu file JSON corrupt."""
        # Tạo file corrupt
        file_path = Path(tmp_checkpoint_dir) / "corrupt-session.json"
        file_path.write_text("{ invalid json content }", encoding="utf-8")

        result = load_checkpoint("corrupt-session", tmp_checkpoint_dir)
        assert result is None

    def test_load_returns_none_if_missing_required_fields(self, tmp_checkpoint_dir):
        """load_checkpoint trả về None nếu thiếu required fields."""
        # Tạo file thiếu session_id
        file_path = Path(tmp_checkpoint_dir) / "incomplete-session.json"
        data = {"phase": "phase_c", "cycle_number": 1}
        file_path.write_text(json.dumps(data), encoding="utf-8")

        result = load_checkpoint("incomplete-session", tmp_checkpoint_dir)
        assert result is None

    def test_load_returns_none_if_invalid_enum_value(self, tmp_checkpoint_dir):
        """load_checkpoint trả về None nếu enum value không hợp lệ."""
        file_path = Path(tmp_checkpoint_dir) / "bad-enum.json"
        data = {
            "session_id": "bad-enum",
            "phase": "invalid_phase_xyz",
            "cycle_number": 1,
            "completed_symbols": [],
            "pending_symbols": [],
            "created_at": "2024-01-01T00:00:00",
        }
        file_path.write_text(json.dumps(data), encoding="utf-8")

        result = load_checkpoint("bad-enum", tmp_checkpoint_dir)
        assert result is None

    def test_load_returns_none_if_dir_not_exists(self):
        """load_checkpoint trả về None nếu thư mục không tồn tại."""
        result = load_checkpoint("any-session", "/nonexistent/path/checkpoints")
        assert result is None

    def test_load_handles_phase_b(self, tmp_checkpoint_dir):
        """load_checkpoint xử lý đúng Phase B."""
        checkpoint = SessionCheckpoint(
            session_id="phase-b-session",
            phase=TrainingPhase.PHASE_B,
            cycle_number=3,
            completed_symbols=["FPT"],
            pending_symbols=["VNM"],
            created_at=datetime(2024, 7, 1, 12, 0, 0),
        )
        save_checkpoint(checkpoint, tmp_checkpoint_dir)
        loaded = load_checkpoint("phase-b-session", tmp_checkpoint_dir)

        assert loaded is not None
        assert loaded.phase == TrainingPhase.PHASE_B

    def test_load_handles_phase_a(self, tmp_checkpoint_dir):
        """load_checkpoint xử lý đúng Phase A."""
        checkpoint = SessionCheckpoint(
            session_id="phase-a-session",
            phase=TrainingPhase.PHASE_A,
            cycle_number=10,
            completed_symbols=["FPT", "VNM", "HPG"],
            pending_symbols=[],
            created_at=datetime(2024, 8, 1, 8, 0, 0),
        )
        save_checkpoint(checkpoint, tmp_checkpoint_dir)
        loaded = load_checkpoint("phase-a-session", tmp_checkpoint_dir)

        assert loaded is not None
        assert loaded.phase == TrainingPhase.PHASE_A


class TestDeleteCheckpoint:
    """Tests cho delete_checkpoint function."""

    def test_delete_removes_file(self, tmp_checkpoint_dir, sample_checkpoint):
        """delete_checkpoint xóa file thành công."""
        save_checkpoint(sample_checkpoint, tmp_checkpoint_dir)
        result = delete_checkpoint("test-session-001", tmp_checkpoint_dir)

        assert result is True
        file_path = Path(tmp_checkpoint_dir) / "test-session-001.json"
        assert not file_path.exists()

    def test_delete_returns_false_if_not_found(self, tmp_checkpoint_dir):
        """delete_checkpoint trả về False nếu file không tồn tại."""
        result = delete_checkpoint("nonexistent-session", tmp_checkpoint_dir)
        assert result is False

    def test_delete_returns_false_if_dir_not_exists(self):
        """delete_checkpoint trả về False nếu thư mục không tồn tại."""
        result = delete_checkpoint("any-session", "/nonexistent/path/checkpoints")
        assert result is False


class TestGetLatestCheckpoint:
    """Tests cho get_latest_checkpoint function."""

    def test_get_latest_returns_most_recent(self, tmp_checkpoint_dir):
        """get_latest_checkpoint trả về checkpoint có created_at mới nhất."""
        # Tạo 3 checkpoints với created_at khác nhau
        old = SessionCheckpoint(
            session_id="old-session",
            phase=TrainingPhase.PHASE_C,
            cycle_number=1,
            completed_symbols=[],
            pending_symbols=["FPT"],
            created_at=datetime(2024, 1, 1, 0, 0, 0),
        )
        middle = SessionCheckpoint(
            session_id="middle-session",
            phase=TrainingPhase.PHASE_B,
            cycle_number=2,
            completed_symbols=["FPT"],
            pending_symbols=["VNM"],
            created_at=datetime(2024, 6, 15, 12, 0, 0),
        )
        newest = SessionCheckpoint(
            session_id="newest-session",
            phase=TrainingPhase.PHASE_A,
            cycle_number=5,
            completed_symbols=["FPT", "VNM"],
            pending_symbols=["HPG"],
            created_at=datetime(2024, 12, 31, 23, 59, 59),
        )

        save_checkpoint(old, tmp_checkpoint_dir)
        save_checkpoint(middle, tmp_checkpoint_dir)
        save_checkpoint(newest, tmp_checkpoint_dir)

        latest = get_latest_checkpoint(tmp_checkpoint_dir)

        assert latest is not None
        assert latest.session_id == "newest-session"

    def test_get_latest_returns_none_if_empty_dir(self, tmp_checkpoint_dir):
        """get_latest_checkpoint trả về None nếu thư mục rỗng."""
        latest = get_latest_checkpoint(tmp_checkpoint_dir)
        assert latest is None

    def test_get_latest_returns_none_if_dir_not_exists(self):
        """get_latest_checkpoint trả về None nếu thư mục không tồn tại."""
        latest = get_latest_checkpoint("/nonexistent/path/checkpoints")
        assert latest is None

    def test_get_latest_skips_corrupt_files(self, tmp_checkpoint_dir):
        """get_latest_checkpoint bỏ qua file corrupt, trả về valid checkpoint."""
        valid = SessionCheckpoint(
            session_id="valid-session",
            phase=TrainingPhase.PHASE_C,
            cycle_number=1,
            completed_symbols=[],
            pending_symbols=["FPT"],
            created_at=datetime(2024, 3, 1, 0, 0, 0),
        )
        save_checkpoint(valid, tmp_checkpoint_dir)

        # Tạo file corrupt
        corrupt_path = Path(tmp_checkpoint_dir) / "corrupt.json"
        corrupt_path.write_text("not valid json", encoding="utf-8")

        latest = get_latest_checkpoint(tmp_checkpoint_dir)
        assert latest is not None
        assert latest.session_id == "valid-session"

    def test_get_latest_with_single_checkpoint(self, tmp_checkpoint_dir, sample_checkpoint):
        """get_latest_checkpoint hoạt động đúng khi chỉ có 1 file."""
        save_checkpoint(sample_checkpoint, tmp_checkpoint_dir)

        latest = get_latest_checkpoint(tmp_checkpoint_dir)
        assert latest is not None
        assert latest.session_id == "test-session-001"


class TestResumeTrainingSymbols:
    """Tests cho resume_training_symbols function."""

    def test_returns_pending_symbols(self, sample_checkpoint):
        """resume_training_symbols trả về pending_symbols."""
        result = resume_training_symbols(sample_checkpoint)
        assert result == ["HPG", "MBB", "TCB"]

    def test_does_not_include_completed_symbols(self, sample_checkpoint):
        """Kết quả không chứa bất kỳ completed symbol nào."""
        result = resume_training_symbols(sample_checkpoint)
        for completed in sample_checkpoint.completed_symbols:
            assert completed not in result

    def test_returns_empty_list_if_all_completed(self):
        """Trả về list rỗng nếu tất cả symbols đã hoàn thành."""
        checkpoint = SessionCheckpoint(
            session_id="done-session",
            phase=TrainingPhase.PHASE_C,
            cycle_number=1,
            completed_symbols=["FPT", "VNM", "HPG"],
            pending_symbols=[],
            created_at=datetime.now(),
        )
        result = resume_training_symbols(checkpoint)
        assert result == []

    def test_returns_all_if_none_completed(self):
        """Trả về toàn bộ pending nếu chưa có symbol nào completed."""
        checkpoint = SessionCheckpoint(
            session_id="fresh-session",
            phase=TrainingPhase.PHASE_C,
            cycle_number=1,
            completed_symbols=[],
            pending_symbols=["FPT", "VNM", "HPG", "MBB"],
            created_at=datetime.now(),
        )
        result = resume_training_symbols(checkpoint)
        assert result == ["FPT", "VNM", "HPG", "MBB"]

    def test_returns_copy_not_reference(self, sample_checkpoint):
        """Trả về bản copy, không phải reference trực tiếp đến pending_symbols."""
        result = resume_training_symbols(sample_checkpoint)
        result.append("EXTRA")
        # pending_symbols gốc không bị ảnh hưởng
        assert "EXTRA" not in sample_checkpoint.pending_symbols
