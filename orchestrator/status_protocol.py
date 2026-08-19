# -*- coding: utf-8 -*-
"""
File-based status protocol cho giao tiếp giữa worker processes và main process.

Workers ghi trạng thái vào `data/engine/status/{task_id}.json` mỗi 10s.
Main process (UI) poll đọc status files mỗi 10s.
Atomic write dùng `os.replace()` (write to temp file → rename).

References: Req 1.6, 2.2
"""

import json
import os
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import STATUS_DIR
from models.task_models import TaskState, TaskStatus, TaskType


def _serialize_value(obj: Any) -> Any:
    """Chuyển đổi giá trị Python sang JSON-serializable."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _task_status_to_dict(status: TaskStatus) -> Dict[str, Any]:
    """Chuyển TaskStatus dataclass thành dictionary JSON-serializable."""
    return {
        "task_id": status.task_id,
        "task_type": status.task_type.value,
        "state": status.state.value,
        "progress_pct": status.progress_pct,
        "message": status.message,
        "heartbeat_ts": status.heartbeat_ts.isoformat(),
        "started_at": status.started_at.isoformat(),
        "error": status.error,
        "details": status.details,
    }


def _dict_to_task_status(data: Dict[str, Any]) -> TaskStatus:
    """Chuyển dictionary từ JSON thành TaskStatus dataclass."""
    return TaskStatus(
        task_id=data["task_id"],
        task_type=TaskType(data["task_type"]),
        state=TaskState(data["state"]),
        progress_pct=float(data["progress_pct"]),
        message=data["message"],
        heartbeat_ts=datetime.fromisoformat(data["heartbeat_ts"]),
        started_at=datetime.fromisoformat(data["started_at"]),
        error=data.get("error"),
        details=data.get("details", {}),
    )


def _get_status_dir() -> Path:
    """Trả về đường dẫn thư mục status, tạo nếu chưa tồn tại."""
    status_path = Path(STATUS_DIR)
    status_path.mkdir(parents=True, exist_ok=True)
    return status_path


def _get_status_file_path(task_id: str) -> Path:
    """Trả về đường dẫn file status cho task_id."""
    return _get_status_dir() / f"{task_id}.json"


def write_status(status: TaskStatus) -> None:
    """
    Ghi TaskStatus ra file JSON với atomic write.

    Sử dụng pattern: write to temp file → os.replace() để đảm bảo
    reader luôn đọc được file hoàn chỉnh (không bị corrupt giữa chừng).

    Trên Windows, os.replace() có thể fail với PermissionError khi target file
    đang bị đọc bởi process khác (Streamlit polling). Trong trường hợp đó,
    retry vài lần với delay ngắn, hoặc fallback sang ghi trực tiếp.

    Parameters
    ----------
    status : TaskStatus
        Trạng thái task cần ghi.
    """
    import time

    status_dir = _get_status_dir()
    target_path = status_dir / f"{status.task_id}.json"

    data = _task_status_to_dict(status)
    json_content = json.dumps(data, default=_serialize_value, ensure_ascii=False, indent=2)

    # Atomic write: ghi vào temp file cùng thư mục, sau đó rename
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f"{status.task_id}_",
        dir=str(status_dir),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json_content)

        # Retry os.replace() tối đa 5 lần — Windows có thể lock file tạm thời
        max_retries = 5
        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, str(target_path))
                return  # Thành công
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.05 * (attempt + 1))  # 50ms, 100ms, 150ms, 200ms
                else:
                    # Fallback: ghi trực tiếp vào target (không atomic nhưng ít nhất không crash)
                    try:
                        with open(str(target_path), "w", encoding="utf-8") as f:
                            f.write(json_content)
                    except PermissionError:
                        pass  # Bỏ qua nếu vẫn không ghi được — sẽ retry ở lần report tiếp
                    finally:
                        # Dọn temp file
                        if os.path.exists(tmp_path):
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                    return
    except Exception:
        # Dọn temp file nếu có lỗi khác
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def read_status(task_id: str) -> Optional[TaskStatus]:
    """
    Đọc và deserialize TaskStatus từ file JSON.

    Parameters
    ----------
    task_id : str
        ID của task cần đọc status.

    Returns
    -------
    Optional[TaskStatus]
        TaskStatus nếu đọc thành công, None nếu file không tồn tại hoặc corrupt.
    """
    file_path = _get_status_file_path(task_id)

    if not file_path.exists():
        return None

    try:
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)
        return _dict_to_task_status(data)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        # File corrupt hoặc format không hợp lệ → trả về None
        return None


def read_all_statuses() -> Dict[str, TaskStatus]:
    """
    Đọc tất cả status files trong thư mục status.

    Returns
    -------
    Dict[str, TaskStatus]
        Dictionary mapping task_id → TaskStatus cho tất cả file hợp lệ.
        File corrupt sẽ bị bỏ qua (không crash).
    """
    status_dir = _get_status_dir()
    result: Dict[str, TaskStatus] = {}

    for file_path in status_dir.glob("*.json"):
        task_id = file_path.stem
        status = read_status(task_id)
        if status is not None:
            result[status.task_id] = status

    return result


def delete_status(task_id: str) -> bool:
    """
    Xóa status file khi task hoàn thành.

    Parameters
    ----------
    task_id : str
        ID của task cần xóa status file.

    Returns
    -------
    bool
        True nếu xóa thành công, False nếu file không tồn tại.
    """
    file_path = _get_status_file_path(task_id)

    if not file_path.exists():
        return False

    try:
        file_path.unlink()
        return True
    except OSError:
        return False
