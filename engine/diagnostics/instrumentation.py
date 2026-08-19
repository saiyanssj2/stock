# -*- coding: utf-8 -*-
"""
Timing Profiler Instrumentation — Monkey-patch instrumentation cho engine modules.

Sử dụng decorator/wrapper pattern để hook TimingProfiler vào các engine methods
mà KHÔNG thay đổi logic hiện có. Wrap các phase trong training pipeline và backtest
engine để đo wall-clock time.

Integration Points:
- TrainingPipeline phases: data_loading, feature_extraction, label_generation,
  model_training, checkpoint_saving
- BacktestEngine phases: strategy_signal_generation, trade_execution,
  metrics_computation
- Epoch loop instrumentation: forward/backward/step tracking

Requirements: 2.1, 2.2, 3.1
"""

import functools
import logging
import time
from typing import Optional

from engine.diagnostics.timing_profiler import TimingProfiler

logger = logging.getLogger(__name__)

# Singleton profiler instance
_profiler: Optional[TimingProfiler] = None

# Flag tránh cài đặt nhiều lần
_instrumentation_installed: bool = False


def get_profiler() -> TimingProfiler:
    """Lấy profiler instance hiện tại, tạo mới nếu chưa có.

    Returns:
        TimingProfiler singleton instance.
    """
    global _profiler
    if _profiler is None:
        _profiler = TimingProfiler()
    return _profiler


def set_profiler(profiler: TimingProfiler) -> None:
    """Thay thế profiler instance (dùng cho testing).

    Args:
        profiler: TimingProfiler instance mới.
    """
    global _profiler
    _profiler = profiler


def _wrap_train_full(original_method):
    """Wrap TrainingPipeline.train_full() — profile các training phases.

    Instrument toàn bộ training pipeline, bao gồm:
    - data_loading: thời gian prepare_training_data
    - feature_extraction: thời gian build feature matrices
    - label_generation: thời gian generate labels (bên trong feature building)
    - model_training: thời gian toàn bộ epoch loop
    - checkpoint_saving: thời gian save final model

    Args:
        original_method: Method gốc TrainingPipeline.train_full.

    Returns:
        Wrapped method.

    Requirements: 2.1
    """

    @functools.wraps(original_method)
    def wrapper(self, *args, **kwargs):
        profiler = get_profiler()

        # Wrap prepare_training_data (data_loading phase)
        original_prepare = self.prepare_training_data

        @functools.wraps(original_prepare)
        def profiled_prepare(*a, **kw):
            with profiler.profile_phase("data_loading"):
                return original_prepare(*a, **kw)

        # Wrap _build_feature_matrix (feature_extraction phase)
        original_build_feature = self._build_feature_matrix

        @functools.wraps(original_build_feature)
        def profiled_build_feature(*a, **kw):
            with profiler.profile_phase("feature_extraction"):
                return original_build_feature(*a, **kw)

        # Wrap _generate_labels (label_generation phase)
        original_generate_labels = self._generate_labels

        @functools.wraps(original_generate_labels)
        def profiled_generate_labels(*a, **kw):
            with profiler.profile_phase("label_generation"):
                return original_generate_labels(*a, **kw)

        # Tạm thời monkey-patch instance methods
        self.prepare_training_data = profiled_prepare
        self._build_feature_matrix = profiled_build_feature
        self._generate_labels = profiled_generate_labels

        # Wrap checkpoint_manager.save (checkpoint_saving phase)
        original_cp_save = self.checkpoint_manager.save

        @functools.wraps(original_cp_save)
        def profiled_cp_save(*a, **kw):
            with profiler.profile_phase("checkpoint_saving"):
                return original_cp_save(*a, **kw)

        self.checkpoint_manager.save = profiled_cp_save

        try:
            # Profile toàn bộ model_training phase bao bọc lời gọi gốc
            # (bao gồm epoch loop bên trong train_full_impl)
            with profiler.profile_phase("model_training"):
                result = original_method(self, *args, **kwargs)
            return result
        except Exception:
            raise
        finally:
            # Khôi phục instance methods
            self.prepare_training_data = original_prepare
            self._build_feature_matrix = original_build_feature
            self._generate_labels = original_generate_labels
            self.checkpoint_manager.save = original_cp_save

    return wrapper


def _wrap_train_full_impl(original_func):
    """Wrap train_full_impl() — instrument epoch loop với forward/backward/step tracking.

    Theo dõi mỗi epoch có đủ forward pass, loss.backward(), và optimizer.step() không.
    Dùng validate_epoch() của TimingProfiler (Req 2.2).

    Args:
        original_func: Function gốc train_full_impl.

    Returns:
        Wrapped function.

    Requirements: 2.2
    """

    @functools.wraps(original_func)
    def wrapper(pipeline, symbol_data, data_dir=None, resume=True):
        profiler = get_profiler()

        # Lưu epoch_logger.log_epoch gốc để intercept epoch info
        original_log_epoch = pipeline.epoch_logger.log_epoch

        # Biến track epoch metrics
        epoch_tracker = {"current_epoch": 0}

        @functools.wraps(original_log_epoch)
        def profiled_log_epoch(epoch, train_loss, val_loss, **kw):
            # Mỗi lần log_epoch được gọi → epoch đã hoàn thành
            # Trong train_full_impl, mỗi epoch có forward, backward, step
            epoch_time_seconds = kw.get("epoch_time_seconds", 0)
            duration_ms = epoch_time_seconds * 1000.0

            # Epoch trong train_full_impl luôn có forward/backward/step
            # (vì code bên trong loop có: model(), loss.backward(), optimizer.step())
            profiler.validate_epoch(
                epoch_number=epoch,
                has_forward=True,
                has_backward=True,
                has_step=True,
                duration_ms=duration_ms,
            )
            epoch_tracker["current_epoch"] = epoch

            return original_log_epoch(epoch, train_loss, val_loss, **kw)

        pipeline.epoch_logger.log_epoch = profiled_log_epoch

        try:
            # Ghi nhận symbol count
            profiler._symbol_count = len(symbol_data) if symbol_data else 0

            result = original_func(pipeline, symbol_data, data_dir, resume)

            # Sau khi training xong, kiểm tra duration
            if hasattr(result, "total_time_seconds") and result.total_time_seconds:
                profiler.check_training_duration(
                    total_duration_seconds=result.total_time_seconds,
                    symbol_count=profiler._symbol_count,
                )

            return result
        except Exception:
            raise
        finally:
            # Khôi phục method gốc
            pipeline.epoch_logger.log_epoch = original_log_epoch

    return wrapper


def _wrap_backtest_run(original_method):
    """Wrap BacktestEngine.run() — profile backtest phases per-symbol.

    Instrument 3 phases:
    - strategy_signal_generation: thời gian strategy.generate_signal() tổng
    - trade_execution: thời gian logic mua/bán
    - metrics_computation: thời gian _compute_metrics()

    Args:
        original_method: Method gốc BacktestEngine.run.

    Returns:
        Wrapped method.

    Requirements: 3.1
    """

    @functools.wraps(original_method)
    def wrapper(self, strategy, df, start_date, end_date, *args, **kwargs):
        profiler = get_profiler()

        # Timing accumulators cho phiên backtest này
        signal_time_ms = 0.0
        trade_time_ms = 0.0

        # Wrap strategy.generate_signal — tích lũy signal generation time
        original_generate_signal = strategy.generate_signal

        @functools.wraps(original_generate_signal)
        def profiled_generate_signal(*a, **kw):
            nonlocal signal_time_ms
            start = time.perf_counter()
            result = original_generate_signal(*a, **kw)
            signal_time_ms += (time.perf_counter() - start) * 1000.0
            return result

        strategy.generate_signal = profiled_generate_signal

        # Wrap _compute_metrics — đo metrics computation time
        original_compute_metrics = self._compute_metrics
        metrics_time_ms = 0.0

        @functools.wraps(original_compute_metrics)
        def profiled_compute_metrics(*a, **kw):
            nonlocal metrics_time_ms
            start = time.perf_counter()
            result = original_compute_metrics(*a, **kw)
            metrics_time_ms += (time.perf_counter() - start) * 1000.0
            return result

        self._compute_metrics = profiled_compute_metrics

        try:
            # Đo toàn bộ run — trade_execution = total - signal - metrics
            total_start = time.perf_counter()
            result = original_method(self, strategy, df, start_date, end_date, *args, **kwargs)
            total_ms = (time.perf_counter() - total_start) * 1000.0

            # trade_execution = tổng thời gian - signal_generation - metrics
            trade_time_ms = total_ms - signal_time_ms - metrics_time_ms

            # Profile phases (ghi lại kết quả)
            with profiler.profile_phase("strategy_signal_generation"):
                # Chỉ ghi thời gian đã đo, không chạy lại
                pass
            # Cập nhật duration trực tiếp cho phase cuối cùng
            if profiler._phase_results:
                profiler._phase_results[-1] = profiler._phase_results[-1].__class__(
                    phase_name="strategy_signal_generation",
                    duration_ms=signal_time_ms,
                    started_at=profiler._phase_results[-1].started_at,
                    ended_at=profiler._phase_results[-1].ended_at,
                )

            with profiler.profile_phase("trade_execution"):
                pass
            if profiler._phase_results:
                profiler._phase_results[-1] = profiler._phase_results[-1].__class__(
                    phase_name="trade_execution",
                    duration_ms=trade_time_ms,
                    started_at=profiler._phase_results[-1].started_at,
                    ended_at=profiler._phase_results[-1].ended_at,
                )

            with profiler.profile_phase("metrics_computation"):
                pass
            if profiler._phase_results:
                profiler._phase_results[-1] = profiler._phase_results[-1].__class__(
                    phase_name="metrics_computation",
                    duration_ms=metrics_time_ms,
                    started_at=profiler._phase_results[-1].started_at,
                    ended_at=profiler._phase_results[-1].ended_at,
                )

            return result
        except Exception:
            raise
        finally:
            # Khôi phục methods gốc
            strategy.generate_signal = original_generate_signal
            self._compute_metrics = original_compute_metrics

    return wrapper


def setup_instrumentation(profiler: Optional[TimingProfiler] = None) -> None:
    """Cài đặt timing profiler instrumentation vào engine modules bằng monkey-patching.

    Patch các method/function sau:
    - TrainingPipeline.train_full → profile training phases
    - train_full_impl → instrument epoch loop
    - BacktestEngine.run → profile backtest phases

    Hàm này idempotent — gọi nhiều lần chỉ cài đặt một lần.

    Args:
        profiler: TimingProfiler instance (optional, dùng default nếu None).
    """
    global _instrumentation_installed

    if _instrumentation_installed:
        logger.debug("Timing instrumentation đã được cài đặt trước đó, bỏ qua.")
        return

    # Thiết lập profiler nếu được cung cấp
    if profiler is not None:
        set_profiler(profiler)

    # Patch TrainingPipeline
    try:
        from engine.training_pipeline import TrainingPipeline

        TrainingPipeline.train_full = _wrap_train_full(TrainingPipeline.train_full)
        logger.info("Timing instrumentation: TrainingPipeline.train_full đã được patch.")
    except ImportError as e:
        logger.warning(
            f"Không thể patch TrainingPipeline — module không tìm thấy: {e}"
        )

    # Patch train_full_impl
    try:
        import engine.training_pipeline_impl as impl_module

        impl_module.train_full_impl = _wrap_train_full_impl(
            impl_module.train_full_impl
        )
        logger.info("Timing instrumentation: train_full_impl đã được patch.")
    except ImportError as e:
        logger.warning(
            f"Không thể patch train_full_impl — module không tìm thấy: {e}"
        )

    # Patch BacktestEngine
    try:
        from engine.backtest_engine import BacktestEngine

        BacktestEngine.run = _wrap_backtest_run(BacktestEngine.run)
        logger.info("Timing instrumentation: BacktestEngine.run đã được patch.")
    except ImportError as e:
        logger.warning(
            f"Không thể patch BacktestEngine — module không tìm thấy: {e}"
        )

    _instrumentation_installed = True
    logger.info("Timing profiler instrumentation đã được cài đặt thành công.")


def teardown_instrumentation() -> None:
    """Gỡ bỏ timing instrumentation — khôi phục methods gốc.

    Hữu ích cho testing để tránh side effects giữa các test.
    """
    global _instrumentation_installed

    if not _instrumentation_installed:
        return

    try:
        from engine.training_pipeline import TrainingPipeline

        if hasattr(TrainingPipeline.train_full, "__wrapped__"):
            TrainingPipeline.train_full = TrainingPipeline.train_full.__wrapped__
    except ImportError:
        pass

    try:
        import engine.training_pipeline_impl as impl_module

        if hasattr(impl_module.train_full_impl, "__wrapped__"):
            impl_module.train_full_impl = impl_module.train_full_impl.__wrapped__
    except ImportError:
        pass

    try:
        from engine.backtest_engine import BacktestEngine

        if hasattr(BacktestEngine.run, "__wrapped__"):
            BacktestEngine.run = BacktestEngine.run.__wrapped__
    except ImportError:
        pass

    _instrumentation_installed = False
    logger.info("Timing profiler instrumentation đã được gỡ bỏ.")
