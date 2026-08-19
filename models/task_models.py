"""
Task models - Định nghĩa các model liên quan đến quản lý task trong TaskOrchestrator.

Bao gồm:
- TaskType: Loại task (TRAINING, BACKTEST, ANALYSIS)
- TaskState: Trạng thái task (IDLE, RUNNING, PAUSED, ERROR, COMPLETED)
- TaskStatus: Trạng thái đầy đủ của một task đang chạy
- ResourceAllocation: Phân bổ tài nguyên cho task

References: Req 1.4, 1.6, 1.7, 2.3
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class TaskType(Enum):
    """Loại task có thể chạy song song trong hệ thống."""

    TRAINING = "training"
    BACKTEST = "backtest"
    ANALYSIS = "analysis"


class TaskState(Enum):
    """Trạng thái lifecycle của một task."""

    IDLE = "idle"          # Task chưa bắt đầu hoặc đã reset
    RUNNING = "running"    # Task đang chạy bình thường
    PAUSED = "paused"      # Task tạm dừng (do resource hoặc user request)
    ERROR = "error"        # Task gặp lỗi và dừng
    COMPLETED = "completed"  # Task hoàn thành thành công


@dataclass
class TaskStatus:
    """
    Trạng thái đầy đủ của một task đang chạy.

    Được worker process ghi vào status file mỗi 10s,
    và UI polling đọc để hiển thị trên Dashboard.

    Attributes:
        task_id: ID duy nhất của task
        task_type: Loại task (TRAINING, BACKTEST, ANALYSIS)
        state: Trạng thái hiện tại
        progress_pct: Phần trăm hoàn thành (0.0 - 100.0)
        message: Thông điệp trạng thái human-readable
        heartbeat_ts: Timestamp lần cuối worker report còn sống
        started_at: Thời điểm task bắt đầu
        error: Thông báo lỗi nếu state == ERROR
        details: Dữ liệu bổ sung tùy loại task
    """

    task_id: str
    task_type: TaskType
    state: TaskState
    progress_pct: float
    message: str
    heartbeat_ts: datetime
    started_at: datetime
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """
    Phân bổ tài nguyên cho một task.

    TaskOrchestrator kiểm tra resource availability trước khi spawn worker.
    Nếu can_start == False, task sẽ không được bắt đầu.

    Attributes:
        gpu_memory_fraction: Phần GPU memory được phân bổ (tối đa 0.7)
        cpu_threads: Số CPU threads được phân bổ
        can_start: True nếu đủ resource để bắt đầu task
        reason: Lý do không thể start (khi can_start == False)
    """

    gpu_memory_fraction: float
    cpu_threads: int
    can_start: bool
    reason: Optional[str] = None
