# -*- coding: utf-8 -*-
"""
File-based persistence cho user preferences.

Lưu và đọc preferences từ disk (data/preferences.json) để dữ liệu
không bị mất khi browser refresh, session reset, hoặc tab close.
Sử dụng atomic write (os.replace()) theo pattern giống status_protocol.

References: Req 2.4, 2.5
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Đường dẫn file preferences trên disk
PREFERENCES_FILE_PATH: str = "data/preferences.json"

# Các key bắt buộc trong preferences dict và kiểu dữ liệu expected
_REQUIRED_KEYS: Dict[str, type] = {
    "selected_symbols": list,
    "capital": (int, float),  # type: ignore[assignment]
    "auto_update_time": str,
    "cycle_interval": (int, float),  # type: ignore[assignment]
    "auto_update_enabled": bool,
}

# Các key optional (có thể có hoặc không)
_OPTIONAL_KEYS: Dict[str, type] = {
    "start_date": str,
    "end_date": str,
}


def _validate_preferences(prefs: Dict[str, Any]) -> bool:
    """
    Validate preferences dict trước khi serialize ra disk.

    Kiểm tra các key bắt buộc có tồn tại và đúng kiểu dữ liệu.

    Parameters
    ----------
    prefs : Dict[str, Any]
        Dictionary chứa preferences cần validate.

    Returns
    -------
    bool
        True nếu hợp lệ, False nếu không.
    """
    if not isinstance(prefs, dict):
        logger.error("Preferences phải là dict, nhận được: %s", type(prefs).__name__)
        return False

    for key, expected_type in _REQUIRED_KEYS.items():
        if key not in prefs:
            logger.error("Thiếu key bắt buộc trong preferences: %s", key)
            return False
        if not isinstance(prefs[key], expected_type):
            logger.error(
                "Key '%s' có kiểu sai: expected %s, got %s",
                key,
                expected_type,
                type(prefs[key]).__name__,
            )
            return False

    # Validate selected_symbols là list of strings
    if not all(isinstance(s, str) for s in prefs["selected_symbols"]):
        logger.error("selected_symbols phải chứa toàn string")
        return False

    # Validate capital > 0
    if prefs["capital"] <= 0:
        logger.error("capital phải > 0, nhận được: %s", prefs["capital"])
        return False

    return True


def save_preferences_to_file(prefs: Dict[str, Any]) -> None:
    """
    Ghi preferences ra file JSON trên disk dùng atomic write.

    Sử dụng pattern: write to temp file → os.replace() để đảm bảo
    reader luôn đọc được file hoàn chỉnh (không bị corrupt giữa chừng).

    Parameters
    ----------
    prefs : Dict[str, Any]
        Dictionary chứa preferences cần lưu. Phải có các key:
        - selected_symbols: List[str]
        - capital: float
        - auto_update_time: str
        - cycle_interval: float
        - auto_update_enabled: bool
        Optional:
        - start_date: str (ISO format)
        - end_date: str (ISO format)

    Raises
    ------
    ValueError
        Nếu prefs không hợp lệ (thiếu key, sai kiểu).
    IOError
        Nếu không thể ghi file ra disk.
    """
    if not _validate_preferences(prefs):
        raise ValueError("Preferences dict không hợp lệ — xem log để biết chi tiết.")

    target_path = Path(PREFERENCES_FILE_PATH)

    # Tạo thư mục cha nếu chưa tồn tại
    target_path.parent.mkdir(parents=True, exist_ok=True)

    json_content = json.dumps(prefs, ensure_ascii=False, indent=2)

    # Atomic write: ghi vào temp file cùng thư mục, sau đó rename
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix="preferences_",
        dir=str(target_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json_content)

        # os.replace() là atomic trên hầu hết filesystem
        # Trên Windows có thể fail nếu target file đang bị lock
        import time

        max_retries = 5
        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, str(target_path))
                logger.info("Đã lưu preferences ra: %s", target_path)
                return
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.05 * (attempt + 1))
                else:
                    # Fallback: ghi trực tiếp (không atomic nhưng không crash)
                    try:
                        with open(str(target_path), "w", encoding="utf-8") as f:
                            f.write(json_content)
                        logger.warning(
                            "Ghi preferences fallback (không atomic) do PermissionError."
                        )
                    except PermissionError:
                        logger.error("Không thể ghi preferences file: PermissionError")
                    finally:
                        if os.path.exists(tmp_path):
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                    return
    except Exception:
        # Dọn temp file nếu có lỗi
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def load_preferences_from_file() -> Optional[Dict[str, Any]]:
    """
    Đọc preferences từ file JSON trên disk.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary preferences nếu file tồn tại và hợp lệ.
        None nếu:
        - File không tồn tại (chưa từng save)
        - File chứa JSON không hợp lệ (corrupt)
        - File chứa dữ liệu không đúng format
    """
    target_path = Path(PREFERENCES_FILE_PATH)

    if not target_path.exists():
        logger.debug("File preferences không tồn tại: %s", target_path)
        return None

    try:
        content = target_path.read_text(encoding="utf-8")
        data = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning("File preferences bị corrupt hoặc invalid JSON: %s", e)
        return None
    except OSError as e:
        logger.warning("Không thể đọc file preferences: %s", e)
        return None

    if not isinstance(data, dict):
        logger.warning("File preferences không chứa dict, nhận được: %s", type(data).__name__)
        return None

    # Validate cơ bản — nếu thiếu key bắt buộc thì vẫn trả về None
    for key in _REQUIRED_KEYS:
        if key not in data:
            logger.warning("File preferences thiếu key bắt buộc: %s", key)
            return None

    logger.debug("Đã load preferences từ: %s", target_path)
    return data
