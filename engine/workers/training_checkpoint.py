# -*- coding: utf-8 -*-
"""
Training checkpoint module - Lưu và phục hồi trạng thái training session.

Cung cấp các hàm:
- save_checkpoint: Lưu SessionCheckpoint ra file JSON (atomic write)
- load_checkpoint: Đọc checkpoint từ file, trả về None nếu corrupt/missing
- delete_checkpoint: Xóa checkpoint file
- get_latest_checkpoint: Tìm checkpoint mới nhất theo created_at
- resume_training_symbols: Trả về danh sách symbols cần train (skip completed)

Atomic write dùng pattern: temp file + os.replace() (giống status_protocol).

References: Req 1.8, 4.6, 4.9, 4.10
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.training_models import SessionCheckpoint, TrainingPhase

logger = logging.getLogger(__name__)

# Thư mục mặc định lưu checkpoints
DEFAULT_CHECKPOINT_DIR = "data/engine/checkpoints"


def _serialize_checkpoint(checkpoint: SessionCheckpoint) -> Dict[str, Any]:
    """
    Chuyển SessionCheckpoint thành dictionary JSON-serializable.

    Parameters
    ----------
    checkpoint : SessionCheckpoint
        Checkpoint cần serialize.

    Returns
    -------
    Dict[str, Any]
        Dictionary chứa tất cả field, datetime → ISO format, enum → value.
    """
    return {
        "session_id": checkpoint.session_id,
        "phase": checkpoint.phase.value,
        "cycle_number": checkpoint.cycle_number,
        "completed_symbols": checkpoint.completed_symbols,
        "pending_symbols": checkpoint.pending_symbols,
        "current_symbol": checkpoint.current_symbol,
        "current_epoch": checkpoint.current_epoch,
        "total_epochs": checkpoint.total_epochs,
        "model_path": checkpoint.model_path,
        "optimizer_state_path": checkpoint.optimizer_state_path,
        "created_at": checkpoint.created_at.isoformat(),
        "metadata": checkpoint.metadata,
    }


def _deserialize_checkpoint(data: Dict[str, Any]) -> SessionCheckpoint:
    """
    Chuyển dictionary JSON thành SessionCheckpoint.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary đọc từ file JSON.

    Returns
    -------
    SessionCheckpoint
        Instance đã deserialize đầy đủ.

    Raises
    ------
    KeyError, ValueError
        Nếu data thiếu field bắt buộc hoặc giá trị không hợp lệ.
    """
    return SessionCheckpoint(
        session_id=data["session_id"],
        phase=TrainingPhase(data["phase"]),
        cycle_number=data["cycle_number"],
        completed_symbols=data["completed_symbols"],
        pending_symbols=data["pending_symbols"],
        current_symbol=data.get("current_symbol"),
        current_epoch=data.get("current_epoch", 0),
        total_epochs=data.get("total_epochs", 0),
        model_path=data.get("model_path"),
        optimizer_state_path=data.get("optimizer_state_path"),
        created_at=datetime.fromisoformat(data["created_at"]),
        metadata=data.get("metadata", {}),
    )


def save_checkpoint(
    checkpoint: SessionCheckpoint,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
) -> str:
    """
    Lưu SessionCheckpoint ra file JSON với atomic write.

    Sử dụng pattern: write to temp file → os.replace() để đảm bảo
    file luôn hoàn chỉnh, không bị corrupt nếu process crash giữa chừng.

    Parameters
    ----------
    checkpoint : SessionCheckpoint
        Checkpoint cần lưu.
    checkpoint_dir : str
        Thư mục chứa checkpoint files.

    Returns
    -------
    str
        Đường dẫn tuyệt đối đến file checkpoint đã lưu.
    """
    dir_path = Path(checkpoint_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    target_path = dir_path / f"{checkpoint.session_id}.json"

    data = _serialize_checkpoint(checkpoint)
    json_content = json.dumps(data, ensure_ascii=False, indent=2)

    # Atomic write: ghi vào temp file cùng thư mục, sau đó rename
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f"checkpoint_{checkpoint.session_id}_",
        dir=str(dir_path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json_content)
        os.replace(tmp_path, str(target_path))
        logger.debug(f"Checkpoint saved: {target_path}")
    except Exception:
        # Dọn temp file nếu rename thất bại
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return str(target_path)


def load_checkpoint(
    session_id: str,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
) -> Optional[SessionCheckpoint]:
    """
    Đọc và deserialize checkpoint từ file JSON.

    Parameters
    ----------
    session_id : str
        ID của session cần load checkpoint.
    checkpoint_dir : str
        Thư mục chứa checkpoint files.

    Returns
    -------
    Optional[SessionCheckpoint]
        SessionCheckpoint nếu đọc thành công, None nếu file không tồn tại hoặc corrupt.
    """
    file_path = Path(checkpoint_dir) / f"{session_id}.json"

    if not file_path.exists():
        return None

    try:
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)
        return _deserialize_checkpoint(data)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, OSError) as e:
        logger.warning(f"Checkpoint file corrupt hoặc không hợp lệ ({file_path}): {e}")
        return None


def delete_checkpoint(
    session_id: str,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
) -> bool:
    """
    Xóa checkpoint file.

    Parameters
    ----------
    session_id : str
        ID của session cần xóa checkpoint.
    checkpoint_dir : str
        Thư mục chứa checkpoint files.

    Returns
    -------
    bool
        True nếu xóa thành công, False nếu file không tồn tại hoặc lỗi.
    """
    file_path = Path(checkpoint_dir) / f"{session_id}.json"

    if not file_path.exists():
        return False

    try:
        file_path.unlink()
        logger.debug(f"Checkpoint deleted: {file_path}")
        return True
    except OSError as e:
        logger.warning(f"Không thể xóa checkpoint file ({file_path}): {e}")
        return False


def get_latest_checkpoint(
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
) -> Optional[SessionCheckpoint]:
    """
    Tìm checkpoint mới nhất theo created_at timestamp.

    Scan tất cả file .json trong checkpoint_dir, load và so sánh created_at.

    Parameters
    ----------
    checkpoint_dir : str
        Thư mục chứa checkpoint files.

    Returns
    -------
    Optional[SessionCheckpoint]
        Checkpoint có created_at mới nhất, None nếu không có checkpoint hợp lệ.
    """
    dir_path = Path(checkpoint_dir)

    if not dir_path.exists():
        return None

    latest: Optional[SessionCheckpoint] = None

    for file_path in dir_path.glob("*.json"):
        session_id = file_path.stem
        checkpoint = load_checkpoint(session_id, checkpoint_dir)
        if checkpoint is None:
            continue
        if latest is None or checkpoint.created_at > latest.created_at:
            latest = checkpoint

    return latest


def resume_training_symbols(checkpoint: SessionCheckpoint) -> List[str]:
    """
    Trả về danh sách symbols cần train khi resume từ checkpoint.

    Chỉ trả về pending_symbols, skip hoàn toàn completed_symbols.
    Đây là danh sách symbols mà TrainingEngine sẽ train.

    Parameters
    ----------
    checkpoint : SessionCheckpoint
        Checkpoint đã load.

    Returns
    -------
    List[str]
        Danh sách pending symbols cần train.
    """
    return list(checkpoint.pending_symbols)
