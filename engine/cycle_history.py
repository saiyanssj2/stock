"""
Cycle History - Quản lý lịch sử và tracking improvement cho Auto-Learner.

Cung cấp:
- persist_cycle(): Lưu CycleResult vào JSON (1 file/cycle)
- load_all_cycles(): Load tất cả cycle results, sorted theo cycle_number
- detect_improvement_trend(): Phát hiện trend cải thiện (3+ cycles tăng Sharpe)
- get_latest_cycle(): Lấy cycle gần nhất

Atomic write: dùng temp file + os.replace để tránh corrupt data.
References: Req 3.4, 3.5
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from models.data_models import CycleResult


# Thư mục mặc định lưu history
DEFAULT_HISTORY_DIR = "data/engine/history"


def _cycle_to_dict(result: CycleResult) -> dict:
    """Chuyển CycleResult thành dict để serialize JSON."""
    return {
        "cycle_number": result.cycle_number,
        "phase": result.phase,
        "sharpe_ratio": result.sharpe_ratio,
        "win_rate": result.win_rate,
        "total_return": result.total_return,
        "strategies_beaten": result.strategies_beaten,
        "validation_loss": result.validation_loss,
        "is_improving": result.is_improving,
        "timestamp": result.timestamp.isoformat(),
        "duration_seconds": result.duration_seconds,
        "notes": result.notes,
    }


def _dict_to_cycle(data: dict) -> CycleResult:
    """Chuyển dict từ JSON thành CycleResult."""
    return CycleResult(
        cycle_number=data["cycle_number"],
        phase=data["phase"],
        sharpe_ratio=data["sharpe_ratio"],
        win_rate=data["win_rate"],
        total_return=data["total_return"],
        strategies_beaten=data["strategies_beaten"],
        validation_loss=data["validation_loss"],
        is_improving=data["is_improving"],
        timestamp=datetime.fromisoformat(data["timestamp"]),
        duration_seconds=data.get("duration_seconds", 0.0),
        notes=data.get("notes", ""),
    )


def persist_cycle(
    result: CycleResult,
    history_dir: str = DEFAULT_HISTORY_DIR,
) -> str:
    """
    Lưu CycleResult vào file JSON. Mỗi cycle 1 file riêng.

    Sử dụng atomic write (temp file + os.replace) để đảm bảo
    file không bị corrupt nếu process crash giữa chừng.

    Args:
        result: CycleResult cần lưu
        history_dir: Thư mục lưu history files

    Returns:
        Đường dẫn file đã lưu

    Raises:
        OSError: Nếu không thể tạo thư mục hoặc ghi file
    """
    dir_path = Path(history_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    filename = f"cycle_{result.cycle_number}.json"
    target_path = dir_path / filename

    data = _cycle_to_dict(result)

    # Atomic write: ghi vào temp file rồi rename
    fd, tmp_path = tempfile.mkstemp(
        dir=str(dir_path),
        suffix=".tmp",
        prefix="cycle_",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(target_path))
    except Exception:
        # Cleanup temp file nếu có lỗi
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return str(target_path)


def load_all_cycles(
    history_dir: str = DEFAULT_HISTORY_DIR,
) -> List[CycleResult]:
    """
    Load tất cả cycle results từ thư mục history, sorted theo cycle_number.

    Bỏ qua file corrupt hoặc không đọc được (graceful handling).

    Args:
        history_dir: Thư mục chứa history files

    Returns:
        List[CycleResult] sorted ascending theo cycle_number
    """
    dir_path = Path(history_dir)

    if not dir_path.exists():
        return []

    cycles: List[CycleResult] = []

    for file_path in dir_path.glob("cycle_*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cycle = _dict_to_cycle(data)
            cycles.append(cycle)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
            # Bỏ qua file corrupt/invalid — không crash toàn bộ load
            continue

    # Sort theo cycle_number ascending
    cycles.sort(key=lambda c: c.cycle_number)
    return cycles


def detect_improvement_trend(
    cycles: List[CycleResult],
    min_consecutive: int = 3,
) -> bool:
    """
    Kiểm tra trend cải thiện: last min_consecutive cycles có Sharpe tăng strictly.

    Improvement = min_consecutive cycles cuối cùng có sharpe_ratio tăng liên tục.
    VD: min_consecutive=3, cycles[-3].sharpe < cycles[-2].sharpe < cycles[-1].sharpe → True

    Args:
        cycles: Danh sách CycleResult (đã sort theo cycle_number)
        min_consecutive: Số cycle liên tiếp cần tăng (mặc định 3)

    Returns:
        True nếu last min_consecutive cycles có Sharpe strictly increasing
    """
    if len(cycles) < min_consecutive:
        return False

    # Lấy min_consecutive cycles cuối cùng
    recent = cycles[-min_consecutive:]

    # Kiểm tra strictly increasing Sharpe ratio
    for i in range(1, len(recent)):
        if recent[i].sharpe_ratio <= recent[i - 1].sharpe_ratio:
            return False

    return True


def get_latest_cycle(
    history_dir: str = DEFAULT_HISTORY_DIR,
) -> Optional[CycleResult]:
    """
    Lấy cycle gần nhất (cycle_number lớn nhất).

    Args:
        history_dir: Thư mục chứa history files

    Returns:
        CycleResult mới nhất, hoặc None nếu chưa có cycle nào
    """
    cycles = load_all_cycles(history_dir)
    if not cycles:
        return None
    return cycles[-1]
