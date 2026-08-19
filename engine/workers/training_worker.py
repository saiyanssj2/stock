# -*- coding: utf-8 -*-
"""
Training Engine Worker - Worker process cho training ML.

Chạy trong subprocess riêng, iterate qua danh sách symbols,
train model cho từng symbol, và report progress qua status file.

Hỗ trợ 3 phases:
- Phase C: Supervised learning với simple labels từ price movement
- Phase B: Deep search validation với enhanced labels
- Phase A: Self-play, model cải thiện bằng cách đấu với version cũ

References: Req 4.1, 4.2, 4.3, 4.7, 4.8
"""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import HEARTBEAT_INTERVAL, POLL_INTERVAL
from models.task_models import TaskState, TaskStatus, TaskType
from models.training_models import (
    SymbolTrainingStatus,
    TrainingPhase,
    TrainingProgress,
)
from orchestrator.status_protocol import write_status


class TrainingEngineWorker:
    """
    Worker process cho training, chạy trong subprocess riêng.

    Iterate qua danh sách symbols, train model cho từng symbol,
    report progress vào status file mỗi POLL_INTERVAL giây.

    Attributes:
        task_id: ID duy nhất của training task
        phase: Phase training hiện tại (C, B, hoặc A)
        current_symbol: Symbol đang được train
        symbols_completed: Số symbol đã hoàn thành
        symbols_total: Tổng số symbol cần train
        per_symbol_status: Trạng thái chi tiết từng symbol
    """

    def __init__(self, task_id: Optional[str] = None) -> None:
        """
        Khởi tạo TrainingEngineWorker.

        Parameters
        ----------
        task_id : Optional[str]
            ID task. Nếu None, tự generate UUID.
        """
        self.task_id: str = task_id or str(uuid.uuid4())
        self.phase: TrainingPhase = TrainingPhase.PHASE_C
        self.current_symbol: str = ""
        self.symbols_completed: int = 0
        self.symbols_total: int = 0
        self.per_symbol_status: Dict[str, SymbolTrainingStatus] = {}
        self._last_progress_time: float = 0.0
        self._last_heartbeat_time: float = 0.0
        self._started_at: datetime = datetime.now()
        self._cycle_number: int = 1
        self._current_epoch: int = 0
        self._total_epochs: int = 0
        self._current_loss: float = 0.0
        self._running: bool = False

    def run(self, symbols: List[str], phase: TrainingPhase, config: dict) -> None:
        """
        Main training loop - iterate symbols, train từng symbol, report progress.

        Parameters
        ----------
        symbols : List[str]
            Danh sách mã cổ phiếu cần train.
        phase : TrainingPhase
            Phase training (PHASE_C, PHASE_B, PHASE_A).
        config : dict
            Cấu hình training (epochs, batch_size, etc.).
        """
        self.phase = phase
        self.symbols_total = len(symbols)
        self.symbols_completed = 0
        self._cycle_number = config.get("cycle_number", 1)
        self._running = True
        self._started_at = datetime.now()

        epochs_per_symbol = config.get("epochs", 10)

        # Khởi tạo per_symbol_status cho tất cả symbols
        for symbol in symbols:
            self.per_symbol_status[symbol] = SymbolTrainingStatus(
                symbol=symbol,
                status="pending",
                epochs_completed=0,
            )

        # Report trạng thái ban đầu
        self._report_initial_status()

        # Train từng symbol
        for symbol in symbols:
            if not self._running:
                break

            self.current_symbol = symbol
            self.per_symbol_status[symbol].status = "training"
            self._report_progress_if_needed(force=True)

            try:
                self.train_symbol(symbol, epochs_per_symbol)
                self.per_symbol_status[symbol].status = "completed"
                self.per_symbol_status[symbol].epochs_completed = epochs_per_symbol
                self.symbols_completed += 1
            except Exception as e:
                self.per_symbol_status[symbol].status = "failed"
                # Log error nhưng tiếp tục train symbol khác
                self.symbols_completed += 1

            self._report_progress_if_needed(force=True)

        # Report hoàn thành
        self._report_completed_status()
        self._running = False

    def train_symbol(self, symbol: str, epochs: int) -> None:
        """
        Train model cho một symbol (STUB - simulate training).

        Trong tương lai sẽ chứa logic ML thực tế.
        Hiện tại simulate training với sleep và progress update.

        Parameters
        ----------
        symbol : str
            Mã cổ phiếu cần train.
        epochs : int
            Số epoch cần train.
        """
        self._total_epochs = epochs
        symbol_start_time = time.time()

        for epoch in range(1, epochs + 1):
            if not self._running:
                break

            self._current_epoch = epoch
            # Simulate training - loss giảm dần theo epoch
            self._current_loss = self._simulate_loss(epoch, epochs)
            self.per_symbol_status[symbol].epochs_completed = epoch
            self.per_symbol_status[symbol].current_loss = self._current_loss

            # Simulate training time cho 1 epoch (stub)
            time.sleep(0.01)

            # Report progress theo POLL_INTERVAL
            self._report_progress_if_needed()

        # Ghi duration cho symbol
        duration = time.time() - symbol_start_time
        self.per_symbol_status[symbol].duration_seconds = duration

    def report_progress(self, progress: TrainingProgress) -> None:
        """
        Ghi progress vào status file thông qua status_protocol.

        Cập nhật heartbeat_ts với mỗi lần report.

        Parameters
        ----------
        progress : TrainingProgress
            Thông tin tiến độ training hiện tại.
        """
        progress_pct = (
            (self.symbols_completed / self.symbols_total) * 100.0
            if self.symbols_total > 0
            else 0.0
        )

        message = (
            f"Training {self.phase.value} - "
            f"{self.current_symbol} "
            f"[{self.symbols_completed}/{self.symbols_total}] "
            f"Epoch {progress.current_epoch}/{progress.total_epochs}"
        )

        # Serialize training progress details
        details = {
            "phase": progress.phase.value,
            "cycle_number": progress.cycle_number,
            "current_symbol": progress.current_symbol,
            "symbols_completed": progress.symbols_completed,
            "symbols_total": progress.symbols_total,
            "current_epoch": progress.current_epoch,
            "total_epochs": progress.total_epochs,
            "current_loss": progress.current_loss,
            "eta_seconds": progress.eta_seconds,
        }

        status = TaskStatus(
            task_id=self.task_id,
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=progress_pct,
            message=message,
            heartbeat_ts=datetime.now(),
            started_at=self._started_at,
            details=details,
        )

        write_status(status)
        self._last_heartbeat_time = time.time()

    def stop(self) -> None:
        """Dừng training loop gracefully."""
        self._running = False

    def _report_progress_if_needed(self, force: bool = False) -> None:
        """
        Report progress nếu đã đủ POLL_INTERVAL kể từ lần report trước.

        Parameters
        ----------
        force : bool
            Nếu True, report ngay bất kể thời gian.
        """
        now = time.time()
        elapsed = now - self._last_progress_time

        if force or elapsed >= POLL_INTERVAL:
            progress = self._build_progress()
            self.report_progress(progress)
            self._last_progress_time = now

    def _report_initial_status(self) -> None:
        """Ghi trạng thái ban đầu khi bắt đầu training."""
        status = TaskStatus(
            task_id=self.task_id,
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=0.0,
            message=f"Starting training {self.phase.value} - {self.symbols_total} symbols",
            heartbeat_ts=datetime.now(),
            started_at=self._started_at,
            details={
                "phase": self.phase.value,
                "symbols_total": self.symbols_total,
                "cycle_number": self._cycle_number,
            },
        )
        write_status(status)
        self._last_progress_time = time.time()
        self._last_heartbeat_time = time.time()

    def _report_completed_status(self) -> None:
        """Ghi trạng thái hoàn thành khi training xong."""
        status = TaskStatus(
            task_id=self.task_id,
            task_type=TaskType.TRAINING,
            state=TaskState.COMPLETED,
            progress_pct=100.0,
            message=f"Training {self.phase.value} completed - {self.symbols_completed}/{self.symbols_total} symbols",
            heartbeat_ts=datetime.now(),
            started_at=self._started_at,
            details={
                "phase": self.phase.value,
                "symbols_completed": self.symbols_completed,
                "symbols_total": self.symbols_total,
                "cycle_number": self._cycle_number,
            },
        )
        write_status(status)

    def _build_progress(self) -> TrainingProgress:
        """Tạo TrainingProgress object từ trạng thái hiện tại."""
        eta = self._estimate_eta()

        return TrainingProgress(
            phase=self.phase,
            cycle_number=self._cycle_number,
            current_symbol=self.current_symbol,
            symbols_completed=self.symbols_completed,
            symbols_total=self.symbols_total,
            current_epoch=self._current_epoch,
            total_epochs=self._total_epochs,
            current_loss=self._current_loss,
            eta_seconds=eta,
            per_symbol_status=dict(self.per_symbol_status),
        )

    def _estimate_eta(self) -> float:
        """
        Ước tính thời gian còn lại (giây) dựa trên tốc độ hiện tại.

        Returns
        -------
        float
            Thời gian ước tính còn lại (giây). 0.0 nếu chưa có đủ data.
        """
        if self.symbols_completed == 0:
            return 0.0

        elapsed = (datetime.now() - self._started_at).total_seconds()
        time_per_symbol = elapsed / self.symbols_completed
        remaining_symbols = self.symbols_total - self.symbols_completed
        return time_per_symbol * remaining_symbols

    def _simulate_loss(self, epoch: int, total_epochs: int) -> float:
        """
        Simulate loss giảm dần theo epoch (stub).

        Parameters
        ----------
        epoch : int
            Epoch hiện tại.
        total_epochs : int
            Tổng số epoch.

        Returns
        -------
        float
            Giá trị loss simulate.
        """
        # Loss bắt đầu ở 1.0 và giảm dần về gần 0
        base_loss = 1.0 - (epoch / total_epochs) * 0.8
        # Thêm chút noise theo phase
        if self.phase == TrainingPhase.PHASE_C:
            return base_loss * 1.0
        elif self.phase == TrainingPhase.PHASE_B:
            return base_loss * 0.8
        else:  # PHASE_A
            return base_loss * 0.6
