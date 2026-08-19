# -*- coding: utf-8 -*-
"""
TaskOrchestrator - Quản lý lifecycle các worker process.

Spawn worker process qua multiprocessing.Process, track active processes,
graceful stop với timeout, và delegate status reading cho status_protocol.

References: Req 1.1, 1.2, 1.3, 1.5
"""

import logging
import multiprocessing
import os
import signal
import time
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from config.settings import HEARTBEAT_TIMEOUT_THRESHOLD
from models.task_models import TaskState, TaskStatus, TaskType
from orchestrator.status_protocol import (
    delete_status,
    read_all_statuses,
    read_status,
    write_status,
)

logger = logging.getLogger(__name__)

# Timeout mặc định khi chờ worker dừng gracefully (giây)
_STOP_TIMEOUT: float = 10.0


def worker_entry(task_type: TaskType, task_id: str, params: dict) -> None:
    """
    Entry point cho worker process.

    Đây là module-level function (bắt buộc cho multiprocessing).
    Dispatch sang TrainingEngineWorker hoặc BacktestEngineWorker tùy task_type.

    Parameters
    ----------
    task_type : TaskType
        Loại task (TRAINING, BACKTEST, ANALYSIS).
    task_id : str
        ID duy nhất của task.
    params : dict
        Tham số cấu hình cho task.
    """
    now = datetime.now()
    status = TaskStatus(
        task_id=task_id,
        task_type=task_type,
        state=TaskState.RUNNING,
        progress_pct=0.0,
        message=f"Worker started for {task_type.value}",
        heartbeat_ts=now,
        started_at=now,
        details=params,
    )
    write_status(status)

    try:
        if task_type == TaskType.TRAINING:
            _run_training_worker(task_id, params)
        elif task_type == TaskType.BACKTEST:
            _run_backtest_worker(task_id, params)
        else:
            # ANALYSIS hoặc task type khác - placeholder
            logger.info("Task type %s chưa có worker implementation", task_type.value)
    except Exception as e:
        # Ghi error status nếu worker crash
        error_status = TaskStatus(
            task_id=task_id,
            task_type=task_type,
            state=TaskState.ERROR,
            progress_pct=0.0,
            message=f"Worker crashed: {str(e)}",
            heartbeat_ts=datetime.now(),
            started_at=now,
            error=str(e),
            details=params,
        )
        write_status(error_status)
        logger.error("Worker %s crashed: %s", task_id, e, exc_info=True)


def _run_training_worker(task_id: str, params: dict) -> None:
    """
    Khởi tạo và chạy TrainingEngineWorker.

    Đọc danh sách symbols từ DataPipeline.get_tracked_symbols(),
    xác định phase từ params, và chạy training loop.

    Parameters
    ----------
    task_id : str
        ID duy nhất của training task.
    params : dict
        Cấu hình: phase, epochs, cycle_number, symbols (optional).
    """
    from engine.data_pipeline import DataPipeline
    from engine.workers.training_worker import TrainingEngineWorker
    from models.training_models import TrainingPhase

    # Xác định phase training
    phase_str = params.get("phase", "phase_c")
    try:
        phase = TrainingPhase(phase_str)
    except ValueError:
        phase = TrainingPhase.PHASE_C

    # Lấy danh sách symbols cần train
    symbols = params.get("symbols")
    if not symbols:
        pipeline = DataPipeline()
        symbols = pipeline.get_tracked_symbols()

    # Cấu hình training
    config = {
        "epochs": params.get("epochs", 10),
        "cycle_number": params.get("cycle_number", 1),
        "batch_size": params.get("batch_size", 32),
    }

    # Khởi tạo worker và chạy
    worker = TrainingEngineWorker(task_id=task_id)
    worker.run(symbols=symbols, phase=phase, config=config)


def _run_backtest_worker(task_id: str, params: dict) -> None:
    """
    Khởi tạo và chạy BacktestEngineWorker.

    Phân biệt mode manual vs auto từ params.

    Parameters
    ----------
    task_id : str
        ID duy nhất của backtest task.
    params : dict
        Cấu hình: mode (auto/manual), symbol, start_date, end_date, initial_capital.
    """
    from engine.workers.backtest_worker import BacktestEngineWorker
    from models.backtest_models import ManualBacktestParams

    worker = BacktestEngineWorker()
    mode = params.get("mode", "auto")

    if mode == "manual":
        # Chạy manual backtest
        from datetime import date as date_type

        backtest_params = ManualBacktestParams(
            symbol=params.get("symbol", "FPT"),
            start_date=date_type.fromisoformat(params.get("start_date", "2024-01-01")),
            end_date=date_type.fromisoformat(params.get("end_date", "2024-12-31")),
            initial_capital=float(params.get("initial_capital", 100_000_000)),
        )
        result = worker.run_manual(backtest_params)

        # Ghi kết quả vào status file
        result_status = TaskStatus(
            task_id=task_id,
            task_type=TaskType.BACKTEST,
            state=TaskState.COMPLETED,
            progress_pct=100.0,
            message=f"Manual backtest completed: {backtest_params.symbol}",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
            details={
                "mode": "manual",
                "symbol": backtest_params.symbol,
                "total_return": result.total_return if result else 0.0,
                "sharpe_ratio": result.sharpe_ratio if result else 0.0,
                "win_rate": result.win_rate if result else 0.0,
            },
        )
        write_status(result_status)
    else:
        # Chạy auto backtest
        from engine.data_pipeline import DataPipeline

        symbols = params.get("symbols")
        if not symbols:
            pipeline = DataPipeline()
            symbols = pipeline.get_tracked_symbols()

        # Callback ghi progress trung gian vào status file sau mỗi symbol
        def _progress_callback(completed: int, total: int, current_symbol: str) -> None:
            progress_pct = (completed / total) * 100.0 if total > 0 else 0.0
            progress_status = TaskStatus(
                task_id=task_id,
                task_type=TaskType.BACKTEST,
                state=TaskState.RUNNING,
                progress_pct=progress_pct,
                message=f"Backtest {completed}/{total}: {current_symbol}",
                heartbeat_ts=datetime.now(),
                started_at=datetime.now(),
                details={"mode": "auto", "current_symbol": current_symbol},
            )
            write_status(progress_status)

        result = worker.run_auto(symbols, progress_callback=_progress_callback)

        # Ghi kết quả vào status file
        result_status = TaskStatus(
            task_id=task_id,
            task_type=TaskType.BACKTEST,
            state=TaskState.COMPLETED,
            progress_pct=100.0,
            message=f"Auto backtest completed: {len(symbols)} symbols",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
            details={
                "mode": "auto",
                "cycle_number": params.get("cycle_number", 1),
                "symbols_tested": result.symbols_tested if result else [],
                "overall_sharpe": result.overall_sharpe if result else 0.0,
                "overall_win_rate": result.overall_win_rate if result else 0.0,
                "overall_return": result.overall_return if result else 0.0,
                "strategies_beaten": result.strategies_beaten if result else 0,
                "duration_seconds": result.duration_seconds if result else 0.0,
            },
        )
        write_status(result_status)


class TaskOrchestrator:
    """
    Quản lý lifecycle các worker process.

    Spawn worker process qua multiprocessing.Process, track active processes,
    hỗ trợ graceful stop, và delegate status reading cho status_protocol.
    """

    def __init__(self) -> None:
        """Khởi tạo TaskOrchestrator với dict tracking active processes."""
        self._processes: Dict[str, multiprocessing.Process] = {}

    def start_task(self, task_type: TaskType, params: dict) -> str:
        """
        Spawn worker process và trả về task_id.

        Tạo unique task_id, ghi initial RUNNING status,
        sau đó spawn Process chạy worker_entry.

        Parameters
        ----------
        task_type : TaskType
            Loại task cần chạy.
        params : dict
            Tham số cấu hình cho task.

        Returns
        -------
        str
            task_id duy nhất của task vừa spawn.
        """
        task_id = f"{task_type.value}_{uuid4().hex[:8]}"

        # Ghi initial RUNNING status trước khi spawn worker
        now = datetime.now()
        initial_status = TaskStatus(
            task_id=task_id,
            task_type=task_type,
            state=TaskState.RUNNING,
            progress_pct=0.0,
            message=f"Starting {task_type.value} task",
            heartbeat_ts=now,
            started_at=now,
            details=params,
        )
        write_status(initial_status)

        # Spawn worker process
        process = multiprocessing.Process(
            target=worker_entry,
            args=(task_type, task_id, params),
            name=f"worker-{task_id}",
            daemon=True,
        )
        process.start()
        self._processes[task_id] = process

        # Ghi PID vào status file để có thể kill process từ disk khi mất _processes reference
        initial_status.details = dict(initial_status.details or {})
        initial_status.details["pid"] = process.pid
        write_status(initial_status)

        logger.info("Task %s started (pid=%s)", task_id, process.pid)
        return task_id

    def stop_task(self, task_id: str) -> bool:
        """
        Gracefully stop task: gửi terminate → chờ timeout → force kill nếu cần.

        Cập nhật status file thành COMPLETED sau khi dừng.
        Xóa process khỏi tracking dict.

        Nếu process không có trong tracking (Streamlit session restart, process đã die),
        vẫn cập nhật status file để UI hiển thị đúng trạng thái.

        Parameters
        ----------
        task_id : str
            ID của task cần dừng.

        Returns
        -------
        bool
            True nếu dừng thành công (hoặc đã dừng rồi), False nếu không thể stop.
        """
        process = self._processes.get(task_id)

        if process is not None:
            # Có process trong tracking → terminate nó
            if process.is_alive():
                process.terminate()
                process.join(timeout=_STOP_TIMEOUT)

                # Force kill nếu vẫn còn sống sau timeout
                if process.is_alive():
                    logger.warning("Task %s not responding, force killing", task_id)
                    process.kill()
                    process.join(timeout=5.0)

            # Xóa khỏi tracking
            del self._processes[task_id]
        else:
            # Process không có trong tracking (session restart hoặc đã chết)
            # Thử kill process via PID từ status file
            logger.info(
                "Task %s not in active processes (session restart?), attempting PID-based kill",
                task_id,
            )
            current = read_status(task_id)
            pid = current.details.get("pid") if current and current.details else None

            if pid is not None and isinstance(pid, int):
                try:
                    # Gửi SIGTERM để terminate gracefully
                    os.kill(pid, signal.SIGTERM)
                    # Chờ ngắn để process kịp exit
                    time.sleep(1.0)
                    # Kiểm tra process còn sống không, nếu còn thì force kill
                    try:
                        os.kill(pid, 0)  # Không gửi signal, chỉ check process tồn tại
                        # Process vẫn còn sống → force kill
                        if os.name == "nt":
                            # Windows: SIGTERM đã gọi TerminateProcess, thử lại
                            os.kill(pid, signal.SIGTERM)
                        else:
                            os.kill(pid, signal.SIGKILL)
                        logger.warning("Task %s (pid=%s) force killed", task_id, pid)
                    except (ProcessLookupError, OSError):
                        # Process đã exit sau SIGTERM — thành công
                        pass
                    logger.info("Task %s (pid=%s) terminated via PID kill", task_id, pid)
                except ProcessLookupError:
                    # Process đã exit trước khi ta kill — chỉ cần update status file
                    logger.info("Task %s (pid=%s) already exited", task_id, pid)
                except PermissionError:
                    # Không có quyền kill process — log warning, vẫn update status file
                    logger.warning(
                        "Task %s (pid=%s) permission denied for kill, updating status only",
                        task_id,
                        pid,
                    )
            else:
                # Không có PID trong status file (legacy status) — chỉ update status file
                logger.info(
                    "Task %s has no PID in status file, updating status only", task_id
                )

        # Cập nhật status file thành COMPLETED
        current_status = read_status(task_id)
        if current_status is not None:
            current_status.state = TaskState.COMPLETED
            current_status.message = "Task stopped by user"
            current_status.heartbeat_ts = datetime.now()
            write_status(current_status)
        else:
            # Nếu không đọc được status cũ, tạo status mới
            now = datetime.now()
            stopped_status = TaskStatus(
                task_id=task_id,
                task_type=TaskType.TRAINING,  # Fallback
                state=TaskState.COMPLETED,
                progress_pct=0.0,
                message="Task stopped by user",
                heartbeat_ts=now,
                started_at=now,
            )
            write_status(stopped_status)

        logger.info("Task %s stopped successfully", task_id)
        return True

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Đọc status file của task.

        Delegate cho status_protocol.read_status().

        Parameters
        ----------
        task_id : str
            ID của task cần đọc status.

        Returns
        -------
        Optional[TaskStatus]
            TaskStatus nếu tìm thấy, None nếu không tồn tại.
        """
        return read_status(task_id)

    def get_all_statuses(self) -> Dict[str, TaskStatus]:
        """
        Trả về status của tất cả active tasks.

        Delegate cho status_protocol.read_all_statuses().

        Returns
        -------
        Dict[str, TaskStatus]
            Dictionary mapping task_id → TaskStatus.
        """
        return read_all_statuses()

    def get_active_tasks_from_disk(self) -> List[TaskStatus]:
        """
        Đọc tất cả status files từ disk và trả về danh sách tasks đang active.

        Method này xác định task active dựa trên:
        - state == RUNNING trong status file
        - heartbeat_ts còn fresh (chênh lệch < HEARTBEAT_TIMEOUT_THRESHOLD so với thời điểm hiện tại)

        KHÔNG phụ thuộc vào _processes dict — có thể gọi trên orchestrator instance mới
        (empty _processes) và vẫn phát hiện được tasks đang chạy từ disk.

        Returns
        -------
        List[TaskStatus]
            Danh sách TaskStatus của các tasks đang chạy với heartbeat còn fresh.
            Trả về list rỗng nếu không có task nào active.

        References
        ----------
        Requirements: 2.1, 2.2, 2.3
        """
        now = datetime.now()
        all_statuses = read_all_statuses()
        active_tasks: List[TaskStatus] = []

        for task_status in all_statuses.values():
            # Chỉ lấy tasks có state RUNNING
            if task_status.state != TaskState.RUNNING:
                continue

            # Kiểm tra heartbeat còn fresh (trong ngưỡng timeout)
            heartbeat_age_seconds = (now - task_status.heartbeat_ts).total_seconds()
            if heartbeat_age_seconds < HEARTBEAT_TIMEOUT_THRESHOLD:
                active_tasks.append(task_status)

        return active_tasks

    def cleanup_finished_processes(self) -> None:
        """
        Dọn dẹp các process đã kết thúc khỏi tracking dict.

        Gọi định kỳ để tránh memory leak từ zombie processes.
        """
        finished_ids = [
            task_id
            for task_id, proc in self._processes.items()
            if not proc.is_alive()
        ]
        for task_id in finished_ids:
            self._processes[task_id].join(timeout=1.0)
            del self._processes[task_id]
            logger.debug("Cleaned up finished process: %s", task_id)
