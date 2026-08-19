"""
Unit tests cho engine/diagnostics/hooks.py — Trigger monitoring hooks.

Kiểm tra:
- Hook AutoLearner.run_cycle() log auto_learner_cycle trigger (Req 1.2)
- Hook BackgroundTrainingManager.start_training() log caller context (Req 1.3)
- Hook TrainingPipeline.train_full()/train_incremental() log user_manual (Req 1.1)
- Decorator pattern không thay đổi logic gốc
- Exception handling an toàn — hook lỗi không block method gốc
- setup_hooks() idempotent
- teardown_hooks() khôi phục methods gốc

Requirements: 1.1, 1.2, 1.3
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from engine.diagnostics.hooks import (
    _log_trigger_safe,
    get_monitor,
    hook_auto_learner_run_cycle,
    hook_background_training_start,
    hook_training_pipeline_train_full,
    hook_training_pipeline_train_incremental,
    set_monitor,
    setup_hooks,
    teardown_hooks,
)
from engine.diagnostics.models import TriggerEvent, TriggerSource
from engine.diagnostics.trigger_monitor import TrainingTriggerMonitor


@pytest.fixture
def mock_monitor(tmp_path):
    """Tạo monitor với temp path để isolate tests."""
    log_path = str(tmp_path / "test_trigger_log.jsonl")
    monitor = TrainingTriggerMonitor(log_path=log_path)
    set_monitor(monitor)
    yield monitor
    # Reset monitor sau test
    set_monitor(None)


@pytest.fixture(autouse=True)
def reset_hooks_state():
    """Reset hooks state trước mỗi test."""
    import engine.diagnostics.hooks as hooks_module

    hooks_module._hooks_installed = False
    yield
    hooks_module._hooks_installed = False


class TestLogTriggerSafe:
    """Test _log_trigger_safe — ghi log an toàn, không raise exception."""

    def test_logs_trigger_event_successfully(self, mock_monitor):
        """Ghi trigger event thành công với đầy đủ fields."""
        _log_trigger_safe(
            trigger_source=TriggerSource.USER_MANUAL,
            caller_component="TestComponent",
            metadata={"key": "value"},
        )

        # Verify event đã được ghi
        from datetime import datetime, timedelta

        now = datetime.now()
        events = mock_monitor.get_triggers_in_range(
            start_time=now - timedelta(seconds=5),
            end_time=now + timedelta(seconds=5),
        )
        assert len(events) == 1
        assert events[0].trigger_source == TriggerSource.USER_MANUAL
        assert events[0].caller_component == "TestComponent"
        assert events[0].metadata == {"key": "value"}
        # session_id là uuid4 hợp lệ
        uuid.UUID(events[0].session_id)
        # timestamp là ISO-8601 hợp lệ
        datetime.fromisoformat(events[0].timestamp)

    def test_does_not_raise_on_monitor_error(self):
        """Không raise exception khi monitor gặp lỗi."""
        # Set monitor thành None để force error
        set_monitor(None)

        # Patch get_monitor để raise
        with patch(
            "engine.diagnostics.hooks.get_monitor",
            side_effect=RuntimeError("test error"),
        ):
            # Không raise — chỉ log warning
            _log_trigger_safe(
                trigger_source=TriggerSource.AUTO_LEARNER_CYCLE,
                caller_component="Test",
            )

    def test_metadata_defaults_to_empty_dict(self, mock_monitor):
        """metadata = None → sử dụng dict rỗng."""
        _log_trigger_safe(
            trigger_source=TriggerSource.USER_MANUAL,
            caller_component="Test",
            metadata=None,
        )

        now = datetime.now()
        events = mock_monitor.get_triggers_in_range(
            start_time=now - timedelta(seconds=5),
            end_time=now + timedelta(seconds=5),
        )
        assert events[0].metadata == {}


class TestHookAutoLearnerRunCycle:
    """Test hook_auto_learner_run_cycle — log auto_learner_cycle trigger (Req 1.2)."""

    def test_logs_auto_learner_cycle_trigger(self, mock_monitor):
        """Hook ghi nhận event auto_learner_cycle với cycle_number."""
        # Giả lập AutoLearner instance
        mock_self = MagicMock()
        mock_self._last_cycle_time = None
        mock_self._cycle_count = 2

        # Method gốc
        original = MagicMock(return_value="cycle_result")

        # Wrap và gọi
        wrapped = hook_auto_learner_run_cycle(original)
        result = wrapped(mock_self, symbols=["VNM"])

        # Kiểm tra method gốc được gọi đúng
        original.assert_called_once_with(mock_self, symbols=["VNM"])
        assert result == "cycle_result"

        # Kiểm tra trigger đã được log
        now = datetime.now()
        events = mock_monitor.get_triggers_in_range(
            start_time=now - timedelta(seconds=5),
            end_time=now + timedelta(seconds=5),
        )
        assert len(events) == 1
        assert events[0].trigger_source == TriggerSource.AUTO_LEARNER_CYCLE
        assert events[0].caller_component == "AutoLearner"
        assert events[0].metadata["cycle_number"] == 3  # _cycle_count + 1

    def test_logs_elapsed_since_last_cycle(self, mock_monitor):
        """Hook ghi nhận elapsed_since_last_cycle khi có last_cycle_time."""
        mock_self = MagicMock()
        mock_self._last_cycle_time = datetime.now() - timedelta(hours=24)
        mock_self._cycle_count = 5

        original = MagicMock(return_value="result")
        wrapped = hook_auto_learner_run_cycle(original)
        wrapped(mock_self)

        now = datetime.now()
        events = mock_monitor.get_triggers_in_range(
            start_time=now - timedelta(seconds=5),
            end_time=now + timedelta(seconds=5),
        )
        assert len(events) == 1
        # elapsed ~ 86400 giây (24h)
        elapsed = events[0].metadata["elapsed_since_last_cycle"]
        assert 86390 < elapsed < 86410

    def test_original_method_called_even_on_hook_error(self):
        """Method gốc vẫn được gọi khi hook gặp lỗi."""
        # Patch _log_trigger_safe để raise
        with patch(
            "engine.diagnostics.hooks._log_trigger_safe",
            side_effect=RuntimeError("hook error"),
        ):
            mock_self = MagicMock()
            mock_self._last_cycle_time = None
            mock_self._cycle_count = 0

            original = MagicMock(return_value="ok")
            wrapped = hook_auto_learner_run_cycle(original)

            # _log_trigger_safe được gọi trong try...except bên trong wrapper
            # Nhưng wrapper gọi _log_trigger_safe trước original
            # Kiểm tra original vẫn được gọi
            # Lưu ý: _log_trigger_safe đã có try/except nội bộ
            # nên RuntimeError sẽ được bắt ở đó
            result = wrapped(mock_self)
            original.assert_called_once()


class TestHookBackgroundTrainingStart:
    """Test hook_background_training_start — log caller context (Req 1.3)."""

    def test_logs_background_scheduled_trigger(self, mock_monitor):
        """Hook ghi nhận trigger BACKGROUND_SCHEDULED với mode."""
        mock_self = MagicMock()

        original = MagicMock(return_value=True)
        wrapped = hook_background_training_start(original)
        result = wrapped(mock_self, symbol_data={"VNM": None}, mode="full")

        # Method gốc vẫn hoạt động
        original.assert_called_once_with(mock_self, symbol_data={"VNM": None}, mode="full")
        assert result is True

        # Verify trigger event
        now = datetime.now()
        events = mock_monitor.get_triggers_in_range(
            start_time=now - timedelta(seconds=5),
            end_time=now + timedelta(seconds=5),
        )
        assert len(events) == 1
        assert events[0].trigger_source == TriggerSource.BACKGROUND_SCHEDULED
        assert events[0].caller_component == "BackgroundTrainingManager"
        assert events[0].metadata["mode"] == "full"
        assert events[0].metadata["component"] == "BackgroundTrainingManager"

    def test_default_mode_incremental(self, mock_monitor):
        """Mode mặc định là incremental khi không truyền."""
        mock_self = MagicMock()
        original = MagicMock(return_value=False)
        wrapped = hook_background_training_start(original)
        wrapped(mock_self, symbol_data={})

        now = datetime.now()
        events = mock_monitor.get_triggers_in_range(
            start_time=now - timedelta(seconds=5),
            end_time=now + timedelta(seconds=5),
        )
        assert events[0].metadata["mode"] == "incremental"


class TestHookTrainingPipeline:
    """Test hooks cho TrainingPipeline.train_full/train_incremental (Req 1.1, 1.3)."""

    def test_train_full_logs_user_manual(self, mock_monitor):
        """train_full hook log USER_MANUAL trigger với mode=full."""
        mock_self = MagicMock()
        original = MagicMock(return_value="training_result")
        wrapped = hook_training_pipeline_train_full(original)
        result = wrapped(mock_self, symbol_data={"VNM": None})

        original.assert_called_once_with(mock_self, symbol_data={"VNM": None})
        assert result == "training_result"

        now = datetime.now()
        events = mock_monitor.get_triggers_in_range(
            start_time=now - timedelta(seconds=5),
            end_time=now + timedelta(seconds=5),
        )
        assert len(events) == 1
        assert events[0].trigger_source == TriggerSource.USER_MANUAL
        assert events[0].caller_component == "TrainingPipeline"
        assert events[0].metadata["mode"] == "full"

    def test_train_incremental_logs_user_manual(self, mock_monitor):
        """train_incremental hook log USER_MANUAL trigger với mode=incremental."""
        mock_self = MagicMock()
        original = MagicMock(return_value="incr_result")
        wrapped = hook_training_pipeline_train_incremental(original)
        result = wrapped(mock_self, new_data={"VNM": None})

        original.assert_called_once_with(mock_self, new_data={"VNM": None})
        assert result == "incr_result"

        now = datetime.now()
        events = mock_monitor.get_triggers_in_range(
            start_time=now - timedelta(seconds=5),
            end_time=now + timedelta(seconds=5),
        )
        assert len(events) == 1
        assert events[0].trigger_source == TriggerSource.USER_MANUAL
        assert events[0].metadata["mode"] == "incremental"


class TestSetupTeardownHooks:
    """Test setup_hooks() và teardown_hooks()."""

    def test_setup_hooks_is_idempotent(self, mock_monitor):
        """Gọi setup_hooks() nhiều lần chỉ cài đặt một lần."""
        import engine.diagnostics.hooks as hooks_module

        setup_hooks(monitor=mock_monitor)
        assert hooks_module._hooks_installed is True

        # Gọi lần 2 — không có side effect
        setup_hooks(monitor=mock_monitor)
        assert hooks_module._hooks_installed is True

        # Cleanup
        teardown_hooks()

    def test_teardown_restores_original_methods(self, mock_monitor):
        """teardown_hooks() khôi phục methods gốc."""
        from engine.auto_learner import AutoLearner
        from engine.training_pipeline import TrainingPipeline

        # Lưu references gốc
        original_run_cycle = AutoLearner.run_cycle
        original_train_full = TrainingPipeline.train_full

        setup_hooks(monitor=mock_monitor)

        # Sau setup, methods đã bị wrap
        assert hasattr(AutoLearner.run_cycle, "__wrapped__")
        assert hasattr(TrainingPipeline.train_full, "__wrapped__")

        teardown_hooks()

        # Sau teardown, methods được khôi phục
        assert AutoLearner.run_cycle == original_run_cycle
        assert TrainingPipeline.train_full == original_train_full

    def test_setup_with_custom_monitor(self, tmp_path):
        """setup_hooks() sử dụng monitor được cung cấp."""
        custom_monitor = TrainingTriggerMonitor(
            log_path=str(tmp_path / "custom.jsonl")
        )
        setup_hooks(monitor=custom_monitor)

        assert get_monitor() is custom_monitor

        teardown_hooks()


class TestHookExceptionSafety:
    """Test rằng hooks không bao giờ block original method execution."""

    def test_auto_learner_hook_exception_does_not_block(self):
        """Nếu monitor lỗi, run_cycle vẫn chạy bình thường."""
        # Set monitor lỗi
        broken_monitor = MagicMock()
        broken_monitor.log_trigger.side_effect = OSError("disk full")
        set_monitor(broken_monitor)

        mock_self = MagicMock()
        mock_self._last_cycle_time = None
        mock_self._cycle_count = 0

        original = MagicMock(return_value="success")
        wrapped = hook_auto_learner_run_cycle(original)
        result = wrapped(mock_self)

        # Original vẫn được gọi và trả kết quả
        assert result == "success"
        original.assert_called_once()

        # Cleanup
        set_monitor(None)
