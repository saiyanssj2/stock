"""
Training Trigger Monitoring Hooks — Decorator-based instrumentation cho engine modules.

Sử dụng monkey-patching/decorator pattern để hook vào các engine methods
mà KHÔNG thay đổi logic hiện có. Mỗi hook ghi log trigger event thông qua
TrainingTriggerMonitor.

Integration Points:
- AutoLearner.run_cycle() → log "auto_learner_cycle" trigger
- BackgroundTrainingManager.start_training() → log caller context
- TrainingPipeline.train_full() / train_incremental() → log "user_manual"

Requirements: 1.1, 1.2, 1.3
"""

import functools
import logging
import uuid
from datetime import datetime
from typing import Optional

from engine.diagnostics.models import TriggerEvent, TriggerSource
from engine.diagnostics.trigger_monitor import TrainingTriggerMonitor

logger = logging.getLogger(__name__)

# Singleton monitor instance — được khởi tạo khi setup_hooks() được gọi
_monitor: Optional[TrainingTriggerMonitor] = None

# Flag để tránh setup nhiều lần
_hooks_installed: bool = False


def get_monitor() -> TrainingTriggerMonitor:
    """Lấy monitor instance hiện tại, tạo mới nếu chưa có.

    Returns:
        TrainingTriggerMonitor singleton instance.
    """
    global _monitor
    if _monitor is None:
        _monitor = TrainingTriggerMonitor()
    return _monitor


def set_monitor(monitor: TrainingTriggerMonitor) -> None:
    """Thay thế monitor instance (dùng cho testing).

    Args:
        monitor: TrainingTriggerMonitor instance mới.
    """
    global _monitor
    _monitor = monitor


def _log_trigger_safe(
    trigger_source: TriggerSource,
    caller_component: str,
    metadata: Optional[dict] = None,
) -> None:
    """Ghi log trigger event an toàn — không raise exception nếu có lỗi.

    Tạo TriggerEvent với session_id (uuid4) và timestamp (ISO-8601),
    gọi monitor.log_trigger(). Nếu có exception → log warning, không block.

    Args:
        trigger_source: Nguồn trigger (enum value).
        caller_component: Tên component gọi.
        metadata: Thông tin bổ sung (optional).

    Requirements: 1.1 (session_id, timestamp, trigger_source)
    """
    try:
        monitor = get_monitor()
        event = TriggerEvent(
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            trigger_source=trigger_source,
            caller_component=caller_component,
            metadata=metadata or {},
        )
        monitor.log_trigger(event)
    except Exception as e:
        # Không bao giờ chặn logic gốc — chỉ cảnh báo
        logger.warning(f"Không thể ghi trigger log trong hook: {e}")


def hook_auto_learner_run_cycle(original_method):
    """Decorator wrap AutoLearner.run_cycle() — log auto_learner_cycle trigger.

    Ghi nhận event "auto_learner_cycle" với cycle_number và elapsed_since_last_cycle.

    Args:
        original_method: Method gốc AutoLearner.run_cycle.

    Returns:
        Wrapped method.

    Requirements: 1.1, 1.2
    """

    @functools.wraps(original_method)
    def wrapper(self, *args, **kwargs):
        try:
            # Tính elapsed_since_last_cycle (giây) trước khi chạy
            elapsed = None
            if self._last_cycle_time is not None:
                elapsed = (datetime.now() - self._last_cycle_time).total_seconds()

            metadata = {
                "cycle_number": self._cycle_count + 1,  # cycle_count sẽ tăng trong run_cycle
            }
            if elapsed is not None:
                metadata["elapsed_since_last_cycle"] = elapsed

            _log_trigger_safe(
                trigger_source=TriggerSource.AUTO_LEARNER_CYCLE,
                caller_component="AutoLearner",
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Hook auto_learner_run_cycle lỗi: {e}")

        # Gọi method gốc — không thay đổi logic
        return original_method(self, *args, **kwargs)

    return wrapper


def hook_background_training_start(original_method):
    """Decorator wrap BackgroundTrainingManager.start_training() — log caller context.

    Ghi nhận caller context bao gồm tên component và trigger_source
    để xác định component nào đã trigger training.

    Args:
        original_method: Method gốc BackgroundTrainingManager.start_training.

    Returns:
        Wrapped method.

    Requirements: 1.1, 1.3
    """

    @functools.wraps(original_method)
    def wrapper(self, *args, **kwargs):
        try:
            # Lấy mode từ arguments
            mode = kwargs.get("mode", "incremental")
            if len(args) >= 2:
                mode = args[1]

            metadata = {
                "mode": mode,
                "component": "BackgroundTrainingManager",
            }

            _log_trigger_safe(
                trigger_source=TriggerSource.BACKGROUND_SCHEDULED,
                caller_component="BackgroundTrainingManager",
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Hook background_training_start lỗi: {e}")

        # Gọi method gốc — không thay đổi logic
        return original_method(self, *args, **kwargs)

    return wrapper


def hook_training_pipeline_train_full(original_method):
    """Decorator wrap TrainingPipeline.train_full() — log user_manual trigger.

    Args:
        original_method: Method gốc TrainingPipeline.train_full.

    Returns:
        Wrapped method.

    Requirements: 1.1, 1.3
    """

    @functools.wraps(original_method)
    def wrapper(self, *args, **kwargs):
        try:
            metadata = {
                "mode": "full",
                "component": "TrainingPipeline",
            }

            _log_trigger_safe(
                trigger_source=TriggerSource.USER_MANUAL,
                caller_component="TrainingPipeline",
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Hook train_full lỗi: {e}")

        # Gọi method gốc — không thay đổi logic
        return original_method(self, *args, **kwargs)

    return wrapper


def hook_training_pipeline_train_incremental(original_method):
    """Decorator wrap TrainingPipeline.train_incremental() — log user_manual trigger.

    Args:
        original_method: Method gốc TrainingPipeline.train_incremental.

    Returns:
        Wrapped method.

    Requirements: 1.1, 1.3
    """

    @functools.wraps(original_method)
    def wrapper(self, *args, **kwargs):
        try:
            metadata = {
                "mode": "incremental",
                "component": "TrainingPipeline",
            }

            _log_trigger_safe(
                trigger_source=TriggerSource.USER_MANUAL,
                caller_component="TrainingPipeline",
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Hook train_incremental lỗi: {e}")

        # Gọi method gốc — không thay đổi logic
        return original_method(self, *args, **kwargs)

    return wrapper


def setup_hooks(monitor: Optional[TrainingTriggerMonitor] = None) -> None:
    """Cài đặt monitoring hooks vào engine modules bằng monkey-patching.

    Patch các method sau:
    - AutoLearner.run_cycle
    - BackgroundTrainingManager.start_training
    - TrainingPipeline.train_full
    - TrainingPipeline.train_incremental

    Hàm này idempotent — gọi nhiều lần chỉ cài đặt hooks một lần.

    Args:
        monitor: TrainingTriggerMonitor instance (optional, dùng default nếu None).
    """
    global _hooks_installed

    if _hooks_installed:
        logger.debug("Hooks đã được cài đặt trước đó, bỏ qua.")
        return

    # Thiết lập monitor nếu được cung cấp
    if monitor is not None:
        set_monitor(monitor)

    try:
        # Import engine modules
        from engine.auto_learner import AutoLearner
        from engine.background_training import BackgroundTrainingManager
        from engine.training_pipeline import TrainingPipeline

        # Patch AutoLearner.run_cycle
        AutoLearner.run_cycle = hook_auto_learner_run_cycle(AutoLearner.run_cycle)

        # Patch BackgroundTrainingManager.start_training
        BackgroundTrainingManager.start_training = hook_background_training_start(
            BackgroundTrainingManager.start_training
        )

        # Patch TrainingPipeline.train_full
        TrainingPipeline.train_full = hook_training_pipeline_train_full(
            TrainingPipeline.train_full
        )

        # Patch TrainingPipeline.train_incremental
        TrainingPipeline.train_incremental = hook_training_pipeline_train_incremental(
            TrainingPipeline.train_incremental
        )

        _hooks_installed = True
        logger.info("Diagnostics monitoring hooks đã được cài đặt thành công.")

    except ImportError as e:
        logger.warning(f"Không thể cài đặt hooks — module không tìm thấy: {e}")
    except Exception as e:
        logger.warning(f"Lỗi khi cài đặt monitoring hooks: {e}")


def teardown_hooks() -> None:
    """Gỡ bỏ monitoring hooks — khôi phục methods gốc.

    Hữu ích cho testing để tránh side effects giữa các test.
    """
    global _hooks_installed

    if not _hooks_installed:
        return

    try:
        from engine.auto_learner import AutoLearner
        from engine.background_training import BackgroundTrainingManager
        from engine.training_pipeline import TrainingPipeline

        # Khôi phục method gốc thông qua __wrapped__ (functools.wraps lưu lại)
        if hasattr(AutoLearner.run_cycle, "__wrapped__"):
            AutoLearner.run_cycle = AutoLearner.run_cycle.__wrapped__

        if hasattr(BackgroundTrainingManager.start_training, "__wrapped__"):
            BackgroundTrainingManager.start_training = (
                BackgroundTrainingManager.start_training.__wrapped__
            )

        if hasattr(TrainingPipeline.train_full, "__wrapped__"):
            TrainingPipeline.train_full = TrainingPipeline.train_full.__wrapped__

        if hasattr(TrainingPipeline.train_incremental, "__wrapped__"):
            TrainingPipeline.train_incremental = (
                TrainingPipeline.train_incremental.__wrapped__
            )

        _hooks_installed = False
        logger.info("Diagnostics monitoring hooks đã được gỡ bỏ.")

    except ImportError:
        _hooks_installed = False
    except Exception as e:
        logger.warning(f"Lỗi khi gỡ hooks: {e}")
        _hooks_installed = False
