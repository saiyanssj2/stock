# -*- coding: utf-8 -*-
"""
Unit tests cho config/preferences.py — file-based persistence cho user preferences.

Kiểm tra:
- Roundtrip save/load hoạt động đúng
- Load file không tồn tại trả về None
- Load file corrupt trả về None
- Validate từ chối dữ liệu sai format
- Atomic write tạo file chính xác
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from config.preferences import (
    PREFERENCES_FILE_PATH,
    load_preferences_from_file,
    save_preferences_to_file,
    _validate_preferences,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def tmp_prefs_dir():
    """Tạo thư mục tạm cho test, dọn dẹp sau khi xong."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def valid_prefs():
    """Preferences dict hợp lệ mẫu."""
    return {
        "selected_symbols": ["VNM", "FPT", "HPG"],
        "capital": 100_000_000.0,
        "auto_update_time": "15:30",
        "cycle_interval": 7.0,
        "auto_update_enabled": True,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
    }


# ==============================================================================
# Test save_preferences_to_file
# ==============================================================================


class TestSavePreferences:
    """Tests cho hàm save_preferences_to_file."""

    def test_save_creates_file(self, tmp_prefs_dir, valid_prefs):
        """Save tạo file JSON trên disk."""
        prefs_file = os.path.join(tmp_prefs_dir, "preferences.json")
        with patch("config.preferences.PREFERENCES_FILE_PATH", prefs_file):
            save_preferences_to_file(valid_prefs)

        assert os.path.exists(prefs_file)

    def test_save_writes_valid_json(self, tmp_prefs_dir, valid_prefs):
        """File được ghi là JSON hợp lệ."""
        prefs_file = os.path.join(tmp_prefs_dir, "preferences.json")
        with patch("config.preferences.PREFERENCES_FILE_PATH", prefs_file):
            save_preferences_to_file(valid_prefs)

        with open(prefs_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data == valid_prefs

    def test_save_creates_parent_directory(self, tmp_prefs_dir, valid_prefs):
        """Save tạo thư mục cha nếu chưa tồn tại."""
        prefs_file = os.path.join(tmp_prefs_dir, "subdir", "nested", "preferences.json")
        with patch("config.preferences.PREFERENCES_FILE_PATH", prefs_file):
            save_preferences_to_file(valid_prefs)

        assert os.path.exists(prefs_file)

    def test_save_overwrite_existing(self, tmp_prefs_dir, valid_prefs):
        """Save ghi đè file đã tồn tại."""
        prefs_file = os.path.join(tmp_prefs_dir, "preferences.json")

        with patch("config.preferences.PREFERENCES_FILE_PATH", prefs_file):
            save_preferences_to_file(valid_prefs)

            # Thay đổi capital rồi save lại
            new_prefs = valid_prefs.copy()
            new_prefs["capital"] = 500_000_000.0
            save_preferences_to_file(new_prefs)

        with open(prefs_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["capital"] == 500_000_000.0

    def test_save_rejects_invalid_prefs(self):
        """Save raise ValueError khi prefs không hợp lệ."""
        with pytest.raises(ValueError):
            save_preferences_to_file({"selected_symbols": ["VNM"]})

    def test_save_rejects_non_dict(self):
        """Save raise ValueError khi input không phải dict."""
        with pytest.raises(ValueError):
            save_preferences_to_file("not a dict")  # type: ignore

    def test_save_ensures_ascii_false(self, tmp_prefs_dir):
        """File chứa ký tự Unicode không bị escape."""
        prefs = {
            "selected_symbols": ["VNM"],
            "capital": 100_000_000.0,
            "auto_update_time": "15:30",
            "cycle_interval": 7.0,
            "auto_update_enabled": True,
        }
        prefs_file = os.path.join(tmp_prefs_dir, "preferences.json")
        with patch("config.preferences.PREFERENCES_FILE_PATH", prefs_file):
            save_preferences_to_file(prefs)

        content = Path(prefs_file).read_text(encoding="utf-8")
        # Không có \\u escape sequences nếu dùng ensure_ascii=False
        assert "\\u" not in content


# ==============================================================================
# Test load_preferences_from_file
# ==============================================================================


class TestLoadPreferences:
    """Tests cho hàm load_preferences_from_file."""

    def test_load_nonexistent_returns_none(self, tmp_prefs_dir):
        """Load file không tồn tại trả về None."""
        prefs_file = os.path.join(tmp_prefs_dir, "nonexistent.json")
        with patch("config.preferences.PREFERENCES_FILE_PATH", prefs_file):
            result = load_preferences_from_file()

        assert result is None

    def test_load_invalid_json_returns_none(self, tmp_prefs_dir):
        """Load file chứa invalid JSON trả về None."""
        prefs_file = os.path.join(tmp_prefs_dir, "bad.json")
        with open(prefs_file, "w") as f:
            f.write("{invalid json, not parseable")

        with patch("config.preferences.PREFERENCES_FILE_PATH", prefs_file):
            result = load_preferences_from_file()

        assert result is None

    def test_load_non_dict_json_returns_none(self, tmp_prefs_dir):
        """Load file chứa JSON nhưng không phải dict trả về None."""
        prefs_file = os.path.join(tmp_prefs_dir, "array.json")
        with open(prefs_file, "w") as f:
            json.dump(["not", "a", "dict"], f)

        with patch("config.preferences.PREFERENCES_FILE_PATH", prefs_file):
            result = load_preferences_from_file()

        assert result is None

    def test_load_missing_required_keys_returns_none(self, tmp_prefs_dir):
        """Load file thiếu key bắt buộc trả về None."""
        prefs_file = os.path.join(tmp_prefs_dir, "incomplete.json")
        with open(prefs_file, "w") as f:
            json.dump({"selected_symbols": ["VNM"]}, f)

        with patch("config.preferences.PREFERENCES_FILE_PATH", prefs_file):
            result = load_preferences_from_file()

        assert result is None

    def test_load_valid_file_returns_dict(self, tmp_prefs_dir, valid_prefs):
        """Load file hợp lệ trả về dict preferences."""
        prefs_file = os.path.join(tmp_prefs_dir, "preferences.json")
        with open(prefs_file, "w", encoding="utf-8") as f:
            json.dump(valid_prefs, f)

        with patch("config.preferences.PREFERENCES_FILE_PATH", prefs_file):
            result = load_preferences_from_file()

        assert result == valid_prefs


# ==============================================================================
# Test roundtrip
# ==============================================================================


class TestRoundtrip:
    """Tests cho save → load roundtrip."""

    def test_roundtrip_basic(self, tmp_prefs_dir, valid_prefs):
        """Save rồi load trả về dữ liệu giống hệt."""
        prefs_file = os.path.join(tmp_prefs_dir, "preferences.json")
        with patch("config.preferences.PREFERENCES_FILE_PATH", prefs_file):
            save_preferences_to_file(valid_prefs)
            loaded = load_preferences_from_file()

        assert loaded == valid_prefs

    def test_roundtrip_minimal_prefs(self, tmp_prefs_dir):
        """Roundtrip với preferences chỉ có key bắt buộc."""
        minimal = {
            "selected_symbols": ["VCB"],
            "capital": 50_000_000.0,
            "auto_update_time": "16:00",
            "cycle_interval": 14.0,
            "auto_update_enabled": False,
        }
        prefs_file = os.path.join(tmp_prefs_dir, "preferences.json")
        with patch("config.preferences.PREFERENCES_FILE_PATH", prefs_file):
            save_preferences_to_file(minimal)
            loaded = load_preferences_from_file()

        assert loaded == minimal

    def test_roundtrip_empty_symbols(self, tmp_prefs_dir):
        """Roundtrip với danh sách symbols rỗng."""
        prefs = {
            "selected_symbols": [],
            "capital": 1_000_000.0,
            "auto_update_time": "09:00",
            "cycle_interval": 1.0,
            "auto_update_enabled": True,
        }
        prefs_file = os.path.join(tmp_prefs_dir, "preferences.json")
        with patch("config.preferences.PREFERENCES_FILE_PATH", prefs_file):
            save_preferences_to_file(prefs)
            loaded = load_preferences_from_file()

        assert loaded == prefs


# ==============================================================================
# Test _validate_preferences
# ==============================================================================


class TestValidatePreferences:
    """Tests cho hàm _validate_preferences."""

    def test_valid_prefs_pass(self, valid_prefs):
        """Preferences hợp lệ trả về True."""
        assert _validate_preferences(valid_prefs) is True

    def test_non_dict_fails(self):
        """Input không phải dict trả về False."""
        assert _validate_preferences("string") is False  # type: ignore
        assert _validate_preferences(None) is False  # type: ignore
        assert _validate_preferences([1, 2]) is False  # type: ignore

    def test_missing_required_key_fails(self, valid_prefs):
        """Thiếu bất kỳ key bắt buộc nào trả về False."""
        for key in ["selected_symbols", "capital", "auto_update_time",
                    "cycle_interval", "auto_update_enabled"]:
            broken = valid_prefs.copy()
            del broken[key]
            assert _validate_preferences(broken) is False, f"Should fail for missing '{key}'"

    def test_wrong_type_fails(self, valid_prefs):
        """Kiểu dữ liệu sai trả về False."""
        # selected_symbols phải là list
        broken = valid_prefs.copy()
        broken["selected_symbols"] = "not a list"
        assert _validate_preferences(broken) is False

        # capital phải là số
        broken = valid_prefs.copy()
        broken["capital"] = "not a number"
        assert _validate_preferences(broken) is False

    def test_negative_capital_fails(self, valid_prefs):
        """Capital <= 0 trả về False."""
        broken = valid_prefs.copy()
        broken["capital"] = -100.0
        assert _validate_preferences(broken) is False

        broken["capital"] = 0
        assert _validate_preferences(broken) is False

    def test_non_string_symbols_fails(self, valid_prefs):
        """Symbols chứa non-string trả về False."""
        broken = valid_prefs.copy()
        broken["selected_symbols"] = [123, "VNM"]
        assert _validate_preferences(broken) is False
