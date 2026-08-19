# -*- coding: utf-8 -*-
"""
Unit tests cho Training Trigger Monitor.

Test các scenario cụ thể:
- Req 1.2: auto_learner_cycle trigger event với cycle_number và elapsed_since_last_cycle
- Req 1.3: BackgroundTrainingManager caller context
- Req 1.6: I/O error → graceful degradation (log warning, không chặn training)
- Req 1.7: is_healthy() returns False khi I/O broken → block training

Requirements: 1.2, 1.3, 1.6, 1.7
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from engine.diagnostics.models import TriggerEvent, TriggerSource
from engine.diagnostics.trigger_monitor import TrainingTriggerMonitor


# === Fixtures ===


@pytest.fixture
def monitor(tmp_path):
    """Tạo TrainingTriggerMonitor với log_path trong tmp_path."""
    log_file = tmp_path / "trigger_log.jsonl"
    return TrainingTriggerMonitor(log_path=str(log_file))


@pytest.fixture
def auto_cycle_event():
    """Tạo TriggerEvent cho auto_learner_cycle (Req 1.2)."""
    return TriggerEvent(
        session_id="session-auto-001",
        timestamp="2024-06-15T10:30:00.000",
        trigger_source=TriggerSource.AUTO_LEARNER_CYCLE,
        caller_component="AutoLearner",
        metadata={
            "cycle_number": 5,
            "elapsed_since_last_cycle": 86400.5,
        },
    )


@pytest.fixture
def background_scheduled_event():
    """Tạo TriggerEvent cho background_scheduled (Req 1.3)."""
    return TriggerEvent(
        session_id="session-bg-002",
        timestamp="2024-06-15T14:00:00.000",
        trigger_source=TriggerSource.BACKGROUND_SCHEDULED,
        caller_component="BackgroundTrainingManager",
        metadata={
            "scheduled_time": "14:00",
            "scheduler_version": "1.2.0",
        },
    )


# === Test Req 1.2: auto_learner_cycle trigger event ===


class TestAutoLearnerCycleTrigger:
    """Test ghi nhận event auto_learner_cycle với metadata đầy đủ."""

    def test_log_auto_cycle_event_success(self, monitor, auto_cycle_event):
        """log_trigger() với auto_learner_cycle event trả về True."""
        result = monitor.log_trigger(auto_cycle_event)
        assert result is True

    def test_auto_cycle_event_persisted_with_metadata(
        self, monitor, auto_cycle_event, tmp_path
    ):
        """Event auto_learner_cycle được ghi vào JSONL với cycle_number và elapsed."""
        monitor.log_trigger(auto_cycle_event)

        # Đọc file JSONL và verify nội dung
        log_file = Path(monitor._log_path)
        assert log_file.exists()

        with open(log_file, "r", encoding="utf-8") as f:
            line = f.readline().strip()

        entry = json.loads(line)

        assert entry["session_id"] == "session-auto-001"
        assert entry["trigger_source"] == "auto_learner_cycle"
        assert entry["caller_component"] == "AutoLearner"
        assert entry["metadata"]["cycle_number"] == 5
        assert entry["metadata"]["elapsed_since_last_cycle"] == 86400.5

    def test_auto_cycle_event_timestamp_preserved(self, monitor, auto_cycle_event):
        """Timestamp ISO-8601 được ghi chính xác."""
        monitor.log_trigger(auto_cycle_event)

        log_file = Path(monitor._log_path)
        with open(log_file, "r", encoding="utf-8") as f:
            entry = json.loads(f.readline())

        assert entry["timestamp"] == "2024-06-15T10:30:00.000"

    def test_auto_cycle_multiple_events_append(self, monitor):
        """Nhiều auto_cycle events được append liên tiếp."""
        for i in range(3):
            event = TriggerEvent(
                session_id=f"session-auto-{i:03d}",
                timestamp=f"2024-06-15T10:{i:02d}:00.000",
                trigger_source=TriggerSource.AUTO_LEARNER_CYCLE,
                caller_component="AutoLearner",
                metadata={
                    "cycle_number": i + 1,
                    "elapsed_since_last_cycle": 3600.0 * (i + 1),
                },
            )
            monitor.log_trigger(event)

        log_file = Path(monitor._log_path)
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Verify mỗi entry có cycle_number tăng dần
        for i, line in enumerate(lines):
            entry = json.loads(line)
            assert entry["metadata"]["cycle_number"] == i + 1


# === Test Req 1.3: BackgroundTrainingManager caller context ===


class TestBackgroundScheduledCallerContext:
    """Test ghi nhận caller context cho BackgroundTrainingManager."""

    def test_log_background_event_success(
        self, monitor, background_scheduled_event
    ):
        """log_trigger() với background_scheduled event trả về True."""
        result = monitor.log_trigger(background_scheduled_event)
        assert result is True

    def test_background_event_caller_component_persisted(
        self, monitor, background_scheduled_event
    ):
        """caller_component = 'BackgroundTrainingManager' được ghi chính xác."""
        monitor.log_trigger(background_scheduled_event)

        log_file = Path(monitor._log_path)
        with open(log_file, "r", encoding="utf-8") as f:
            entry = json.loads(f.readline())

        assert entry["caller_component"] == "BackgroundTrainingManager"
        assert entry["trigger_source"] == "background_scheduled"

    def test_background_event_metadata_context(
        self, monitor, background_scheduled_event
    ):
        """Metadata chứa context bổ sung của caller."""
        monitor.log_trigger(background_scheduled_event)

        log_file = Path(monitor._log_path)
        with open(log_file, "r", encoding="utf-8") as f:
            entry = json.loads(f.readline())

        assert "scheduled_time" in entry["metadata"]
        assert entry["metadata"]["scheduled_time"] == "14:00"

    def test_different_caller_components_distinguished(self, monitor):
        """Phân biệt được các caller_component khác nhau."""
        events = [
            TriggerEvent(
                session_id="s1",
                timestamp="2024-06-15T10:00:00.000",
                trigger_source=TriggerSource.AUTO_LEARNER_CYCLE,
                caller_component="AutoLearner",
                metadata={"cycle_number": 1, "elapsed_since_last_cycle": 3600},
            ),
            TriggerEvent(
                session_id="s2",
                timestamp="2024-06-15T11:00:00.000",
                trigger_source=TriggerSource.BACKGROUND_SCHEDULED,
                caller_component="BackgroundTrainingManager",
                metadata={},
            ),
            TriggerEvent(
                session_id="s3",
                timestamp="2024-06-15T12:00:00.000",
                trigger_source=TriggerSource.USER_MANUAL,
                caller_component="TrainingPipeline",
                metadata={},
            ),
        ]

        for event in events:
            monitor.log_trigger(event)

        log_file = Path(monitor._log_path)
        with open(log_file, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f.readlines()]

        callers = [e["caller_component"] for e in entries]
        assert callers == ["AutoLearner", "BackgroundTrainingManager", "TrainingPipeline"]


# === Test Req 1.6: I/O error graceful degradation ===


class TestIOErrorGracefulDegradation:
    """Test I/O error → log warning, không chặn training."""

    def test_io_error_returns_false(self, tmp_path):
        """Khi không thể ghi file, log_trigger() trả về False."""
        log_file = tmp_path / "trigger_log.jsonl"
        monitor = TrainingTriggerMonitor(log_path=str(log_file))

        event = TriggerEvent(
            session_id="s-io-err",
            timestamp="2024-06-15T10:00:00.000",
            trigger_source=TriggerSource.USER_MANUAL,
            caller_component="TestCaller",
            metadata={},
        )

        # Mock open() để simulate I/O error (hoạt động trên cả Windows và Linux)
        with patch("builtins.open", side_effect=OSError("Permission denied")):
            result = monitor.log_trigger(event)
            assert result is False

    def test_io_error_logs_warning(self, tmp_path, caplog):
        """Khi I/O lỗi, warning được ghi vào application logger."""
        log_file = tmp_path / "trigger_log.jsonl"
        monitor = TrainingTriggerMonitor(log_path=str(log_file))

        event = TriggerEvent(
            session_id="s-io-err",
            timestamp="2024-06-15T10:00:00.000",
            trigger_source=TriggerSource.USER_MANUAL,
            caller_component="TestCaller",
            metadata={},
        )

        with caplog.at_level(logging.WARNING):
            with patch("builtins.open", side_effect=OSError("Disk full")):
                monitor.log_trigger(event)

        # Verify warning được log
        assert any(
            "I/O" in record.message or "lỗi" in record.message
            for record in caplog.records
        )

    def test_io_error_does_not_raise_exception(self, tmp_path):
        """I/O error không raise exception — training tiếp tục bình thường."""
        # Mock open() để raise OSError
        monitor = TrainingTriggerMonitor(
            log_path=str(tmp_path / "trigger_log.jsonl")
        )

        event = TriggerEvent(
            session_id="s-io-err",
            timestamp="2024-06-15T10:00:00.000",
            trigger_source=TriggerSource.USER_MANUAL,
            caller_component="TestCaller",
            metadata={},
        )

        with patch("builtins.open", side_effect=OSError("Disk full")):
            # Không raise exception
            result = monitor.log_trigger(event)
            assert result is False

    def test_io_error_sets_unhealthy_state(self, tmp_path):
        """Sau I/O error, _io_healthy chuyển thành False."""
        monitor = TrainingTriggerMonitor(
            log_path=str(tmp_path / "trigger_log.jsonl")
        )

        event = TriggerEvent(
            session_id="s-io-err",
            timestamp="2024-06-15T10:00:00.000",
            trigger_source=TriggerSource.USER_MANUAL,
            caller_component="TestCaller",
            metadata={},
        )

        with patch("builtins.open", side_effect=OSError("Disk full")):
            monitor.log_trigger(event)

        assert monitor._io_healthy is False


# === Test Req 1.7: Pre-training health check blocking ===


class TestPreTrainingHealthCheck:
    """Test is_healthy() để block training khi I/O lỗi."""

    def test_healthy_when_io_works(self, monitor):
        """is_healthy() trả về True khi I/O hoạt động bình thường."""
        assert monitor.is_healthy() is True

    def test_unhealthy_when_directory_not_writable(self, tmp_path):
        """is_healthy() trả về False khi directory không thể ghi."""
        log_file = tmp_path / "health_test" / "trigger_log.jsonl"
        monitor = TrainingTriggerMonitor(log_path=str(log_file))

        # Verify healthy trước khi mock
        assert monitor.is_healthy() is True

        # Mock probe write để simulate I/O error (cross-platform)
        with patch.object(Path, "write_text", side_effect=OSError("Read-only FS")):
            result = monitor.is_healthy()
            assert result is False

    def test_unhealthy_blocks_training_decision(self, tmp_path):
        """Khi is_healthy()=False, caller có thể dùng để block training."""
        monitor = TrainingTriggerMonitor(
            log_path=str(tmp_path / "trigger_log.jsonl")
        )

        # Simulate I/O error bằng cách mock probe write
        with patch.object(Path, "write_text", side_effect=OSError("No space")):
            healthy = monitor.is_healthy()

        # Pattern sử dụng: caller kiểm tra is_healthy() trước khi training
        assert healthy is False

        # Mô phỏng logic caller: nếu không healthy → không train
        should_train = healthy  # caller dùng kết quả này để quyết định
        assert should_train is False

    def test_healthy_after_io_recovers(self, monitor):
        """is_healthy() trả về True lại khi I/O phục hồi."""
        # Force unhealthy state
        monitor._io_healthy = False

        # Nhưng is_healthy() probe lại — nếu thư mục writable thì recover
        result = monitor.is_healthy()
        assert result is True

    def test_health_probe_does_not_leave_artifacts(self, monitor, tmp_path):
        """Health probe (.health_probe file) được cleanup sau khi check."""
        monitor.is_healthy()

        # Verify probe file không tồn tại sau check
        probe_path = Path(monitor._log_path).parent / ".health_probe"
        assert not probe_path.exists()
