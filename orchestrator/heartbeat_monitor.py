# -*- coding: utf-8 -*-
"""
Heartbeat Monitor - Phát hiện task bị frozen dựa trên heartbeat timestamp.

Worker process ghi heartbeat_ts mỗi 30s vào status file.
Main process kiểm tra frozen tasks mỗi poll cycle.
Task coi như frozen nếu heartbeat_ts > HEARTBEAT_TIMEOUT_THRESHOLD (60s).

References: Req 1.7
"""

from datetime import datetime
from typing import Dict, List

from config.settings import HEARTBEAT_TIMEOUT_THRESHOLD
from models.task_models import TaskState, TaskStatus
from orchestrator.status_protocol import read_all_statuses


def is_task_frozen(status: TaskStatus) -> bool:
    """
    Kiểm tra xem một task có bị frozen hay không.

    Chỉ check task đang ở trạng thái RUNNING.
    So sánh heartbeat_ts với thời điểm hiện tại, nếu vượt quá
    HEARTBEAT_TIMEOUT_THRESHOLD thì coi là frozen.

    Parameters
    ----------
    status : TaskStatus
        Trạng thái của task cần kiểm tra.

    Returns
    -------
    bool
        True nếu task bị frozen, False nếu không.
    """
    if status.state != TaskState.RUNNING:
        return False

    elapsed = (datetime.now() - status.heartbeat_ts).total_seconds()
    return elapsed > HEARTBEAT_TIMEOUT_THRESHOLD


def get_frozen_tasks(statuses: Dict[str, TaskStatus]) -> List[str]:
    """
    Lọc danh sách task_ids bị frozen từ dictionary statuses.

    Parameters
    ----------
    statuses : Dict[str, TaskStatus]
        Dictionary mapping task_id → TaskStatus.

    Returns
    -------
    List[str]
        Danh sách task_ids bị frozen (heartbeat_ts > 60s).
    """
    return [
        task_id
        for task_id, status in statuses.items()
        if is_task_frozen(status)
    ]


def check_heartbeats() -> List[str]:
    """
    Đọc tất cả status files, trả về danh sách task_ids bị frozen.

    Đây là entry point chính cho main process gọi mỗi poll cycle
    để phát hiện worker bị treo hoặc crash mà không kịp báo lỗi.

    Returns
    -------
    List[str]
        Danh sách task_ids có heartbeat_ts vượt quá 60s.
    """
    statuses = read_all_statuses()
    return get_frozen_tasks(statuses)
