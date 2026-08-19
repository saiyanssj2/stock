"""
Unit tests cho engine/cycle_history.py

Kiểm tra:
- persist_cycle: lưu file JSON atomic, format đúng
- load_all_cycles: load + sort theo cycle_number, bỏ qua file corrupt
- detect_improvement_trend: phát hiện trend tăng Sharpe
- get_latest_cycle: lấy cycle mới nhất
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from engine.cycle_history import (
    _cycle_to_dict,
    _dict_to_cycle,
    detect_improvement_trend,
    get_latest_cycle,
    load_all_cycles,
    persist_cycle,
)
from models.data_models import CycleResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_history_dir(tmp_path):
    """Tạo thư mục tạm cho history files."""
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    return str(history_dir)


@pytest.fixture
def sample_cycle():
    """Tạo CycleResult mẫu."""
    return CycleResult(
        cycle_number=1,
        phase="PHASE_C",
        sharpe_ratio=1.5,
        win_rate=0.55,
        total_return=12.3,
        strategies_beaten=2,
        validation_loss=0.045,
        is_improving=True,
        timestamp=datetime(2024, 6, 15, 10, 30, 0),
        duration_seconds=3600.0,
        notes="first cycle",
    )


@pytest.fixture
def multiple_cycles():
    """Tạo danh sách nhiều CycleResult cho test trend."""
    base_time = datetime(2024, 6, 1)
    return [
        CycleResult(
            cycle_number=i,
            phase="PHASE_C",
            sharpe_ratio=sharpe,
            win_rate=0.5 + i * 0.01,
            total_return=5.0 + i,
            strategies_beaten=1,
            validation_loss=0.05 - i * 0.005,
            is_improving=i > 1,
            timestamp=base_time,
            duration_seconds=1800.0,
            notes="",
        )
        for i, sharpe in enumerate([0.8, 1.0, 1.2, 1.5, 1.9], start=1)
    ]


# ---------------------------------------------------------------------------
# Tests: _cycle_to_dict / _dict_to_cycle round-trip
# ---------------------------------------------------------------------------


class TestCycleSerializer:
    """Kiểm tra serialize/deserialize CycleResult."""

    def test_round_trip_preserves_all_fields(self, sample_cycle):
        """Dict -> CycleResult -> Dict giữ nguyên data."""
        data = _cycle_to_dict(sample_cycle)
        restored = _dict_to_cycle(data)

        assert restored.cycle_number == sample_cycle.cycle_number
        assert restored.phase == sample_cycle.phase
        assert restored.sharpe_ratio == sample_cycle.sharpe_ratio
        assert restored.win_rate == sample_cycle.win_rate
        assert restored.total_return == sample_cycle.total_return
        assert restored.strategies_beaten == sample_cycle.strategies_beaten
        assert restored.validation_loss == sample_cycle.validation_loss
        assert restored.is_improving == sample_cycle.is_improving
        assert restored.timestamp == sample_cycle.timestamp
        assert restored.duration_seconds == sample_cycle.duration_seconds
        assert restored.notes == sample_cycle.notes

    def test_to_dict_has_expected_keys(self, sample_cycle):
        """Dict output chứa đủ keys cần thiết."""
        data = _cycle_to_dict(sample_cycle)
        expected_keys = {
            "cycle_number", "phase", "sharpe_ratio", "win_rate",
            "total_return", "strategies_beaten", "validation_loss",
            "is_improving", "timestamp", "duration_seconds", "notes",
        }
        assert set(data.keys()) == expected_keys

    def test_timestamp_stored_as_iso_string(self, sample_cycle):
        """Timestamp được lưu dạng ISO format string."""
        data = _cycle_to_dict(sample_cycle)
        assert isinstance(data["timestamp"], str)
        # Phải parse lại được
        parsed = datetime.fromisoformat(data["timestamp"])
        assert parsed == sample_cycle.timestamp


# ---------------------------------------------------------------------------
# Tests: persist_cycle
# ---------------------------------------------------------------------------


class TestPersistCycle:
    """Kiểm tra lưu cycle vào file JSON."""

    def test_creates_file_with_correct_name(self, tmp_history_dir, sample_cycle):
        """File được tạo với tên cycle_{number}.json."""
        path = persist_cycle(sample_cycle, history_dir=tmp_history_dir)
        assert os.path.exists(path)
        assert Path(path).name == "cycle_1.json"

    def test_file_content_is_valid_json(self, tmp_history_dir, sample_cycle):
        """File chứa JSON hợp lệ."""
        path = persist_cycle(sample_cycle, history_dir=tmp_history_dir)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["cycle_number"] == 1
        assert data["sharpe_ratio"] == 1.5

    def test_creates_directory_if_not_exists(self, tmp_path, sample_cycle):
        """Tạo thư mục nếu chưa tồn tại."""
        new_dir = str(tmp_path / "new" / "nested" / "history")
        path = persist_cycle(sample_cycle, history_dir=new_dir)
        assert os.path.exists(path)

    def test_overwrites_existing_cycle_file(self, tmp_history_dir, sample_cycle):
        """Ghi đè nếu file đã tồn tại (cùng cycle_number)."""
        persist_cycle(sample_cycle, history_dir=tmp_history_dir)

        # Thay đổi sharpe và ghi lại
        updated = CycleResult(
            cycle_number=1,
            phase="PHASE_B",
            sharpe_ratio=2.0,
            win_rate=0.60,
            total_return=15.0,
            strategies_beaten=3,
            validation_loss=0.03,
            is_improving=True,
            timestamp=datetime(2024, 6, 16),
            duration_seconds=4000.0,
            notes="updated",
        )
        path = persist_cycle(updated, history_dir=tmp_history_dir)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["sharpe_ratio"] == 2.0
        assert data["phase"] == "PHASE_B"

    def test_no_temp_file_left_on_success(self, tmp_history_dir, sample_cycle):
        """Không còn file .tmp sau khi persist thành công."""
        persist_cycle(sample_cycle, history_dir=tmp_history_dir)
        tmp_files = list(Path(tmp_history_dir).glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_returns_path_string(self, tmp_history_dir, sample_cycle):
        """Trả về string path của file đã lưu."""
        result = persist_cycle(sample_cycle, history_dir=tmp_history_dir)
        assert isinstance(result, str)
        assert "cycle_1.json" in result


# ---------------------------------------------------------------------------
# Tests: load_all_cycles
# ---------------------------------------------------------------------------


class TestLoadAllCycles:
    """Kiểm tra load cycles từ history directory."""

    def test_empty_directory_returns_empty_list(self, tmp_history_dir):
        """Thư mục rỗng → trả về list rỗng."""
        cycles = load_all_cycles(history_dir=tmp_history_dir)
        assert cycles == []

    def test_nonexistent_directory_returns_empty_list(self, tmp_path):
        """Thư mục không tồn tại → trả về list rỗng."""
        cycles = load_all_cycles(history_dir=str(tmp_path / "nonexistent"))
        assert cycles == []

    def test_loads_persisted_cycles(self, tmp_history_dir, multiple_cycles):
        """Load được tất cả cycle đã persist."""
        for cycle in multiple_cycles:
            persist_cycle(cycle, history_dir=tmp_history_dir)

        loaded = load_all_cycles(history_dir=tmp_history_dir)
        assert len(loaded) == 5

    def test_sorted_by_cycle_number(self, tmp_history_dir, multiple_cycles):
        """Kết quả sorted ascending theo cycle_number."""
        # Persist theo thứ tự ngược
        for cycle in reversed(multiple_cycles):
            persist_cycle(cycle, history_dir=tmp_history_dir)

        loaded = load_all_cycles(history_dir=tmp_history_dir)
        numbers = [c.cycle_number for c in loaded]
        assert numbers == sorted(numbers)

    def test_skips_corrupt_json_files(self, tmp_history_dir, sample_cycle):
        """Bỏ qua file JSON bị corrupt, load file hợp lệ bình thường."""
        persist_cycle(sample_cycle, history_dir=tmp_history_dir)

        # Tạo file corrupt
        corrupt_path = Path(tmp_history_dir) / "cycle_99.json"
        corrupt_path.write_text("not valid json {{{", encoding="utf-8")

        loaded = load_all_cycles(history_dir=tmp_history_dir)
        assert len(loaded) == 1
        assert loaded[0].cycle_number == 1

    def test_skips_files_with_missing_keys(self, tmp_history_dir, sample_cycle):
        """Bỏ qua file thiếu required keys."""
        persist_cycle(sample_cycle, history_dir=tmp_history_dir)

        # Tạo file thiếu key
        incomplete_path = Path(tmp_history_dir) / "cycle_50.json"
        incomplete_data = {"cycle_number": 50, "phase": "PHASE_C"}
        incomplete_path.write_text(
            json.dumps(incomplete_data), encoding="utf-8"
        )

        loaded = load_all_cycles(history_dir=tmp_history_dir)
        assert len(loaded) == 1

    def test_ignores_non_cycle_files(self, tmp_history_dir, sample_cycle):
        """Chỉ đọc file có pattern cycle_*.json."""
        persist_cycle(sample_cycle, history_dir=tmp_history_dir)

        # Tạo file không phải cycle
        other_path = Path(tmp_history_dir) / "metadata.json"
        other_path.write_text("{}", encoding="utf-8")

        loaded = load_all_cycles(history_dir=tmp_history_dir)
        assert len(loaded) == 1

    def test_preserves_data_after_round_trip(self, tmp_history_dir, sample_cycle):
        """Data không bị mất sau persist → load."""
        persist_cycle(sample_cycle, history_dir=tmp_history_dir)
        loaded = load_all_cycles(history_dir=tmp_history_dir)

        assert len(loaded) == 1
        restored = loaded[0]
        assert restored.cycle_number == sample_cycle.cycle_number
        assert restored.phase == sample_cycle.phase
        assert restored.sharpe_ratio == sample_cycle.sharpe_ratio
        assert restored.win_rate == sample_cycle.win_rate
        assert restored.total_return == sample_cycle.total_return
        assert restored.strategies_beaten == sample_cycle.strategies_beaten
        assert restored.validation_loss == sample_cycle.validation_loss
        assert restored.is_improving == sample_cycle.is_improving
        assert restored.timestamp == sample_cycle.timestamp
        assert restored.duration_seconds == sample_cycle.duration_seconds
        assert restored.notes == sample_cycle.notes


# ---------------------------------------------------------------------------
# Tests: detect_improvement_trend
# ---------------------------------------------------------------------------


class TestDetectImprovementTrend:
    """Kiểm tra phát hiện trend cải thiện."""

    def test_improving_trend_returns_true(self, multiple_cycles):
        """5 cycles với Sharpe tăng liên tục → True (min_consecutive=3)."""
        result = detect_improvement_trend(multiple_cycles, min_consecutive=3)
        assert result is True

    def test_not_enough_cycles_returns_false(self):
        """Ít hơn min_consecutive cycles → False."""
        cycles = [
            CycleResult(
                cycle_number=1, phase="PHASE_C", sharpe_ratio=1.0,
                win_rate=0.5, total_return=5.0, strategies_beaten=1,
                validation_loss=0.05, is_improving=False,
                timestamp=datetime(2024, 1, 1),
            ),
            CycleResult(
                cycle_number=2, phase="PHASE_C", sharpe_ratio=1.2,
                win_rate=0.5, total_return=5.0, strategies_beaten=1,
                validation_loss=0.05, is_improving=True,
                timestamp=datetime(2024, 1, 2),
            ),
        ]
        result = detect_improvement_trend(cycles, min_consecutive=3)
        assert result is False

    def test_flat_sharpe_returns_false(self):
        """Sharpe bằng nhau (không strictly increasing) → False."""
        cycles = [
            CycleResult(
                cycle_number=i, phase="PHASE_C", sharpe_ratio=1.0,
                win_rate=0.5, total_return=5.0, strategies_beaten=1,
                validation_loss=0.05, is_improving=False,
                timestamp=datetime(2024, 1, i),
            )
            for i in range(1, 5)
        ]
        result = detect_improvement_trend(cycles, min_consecutive=3)
        assert result is False

    def test_decreasing_sharpe_returns_false(self):
        """Sharpe giảm → False."""
        cycles = [
            CycleResult(
                cycle_number=i, phase="PHASE_C", sharpe_ratio=sharpe,
                win_rate=0.5, total_return=5.0, strategies_beaten=1,
                validation_loss=0.05, is_improving=False,
                timestamp=datetime(2024, 1, i),
            )
            for i, sharpe in enumerate([2.0, 1.8, 1.5, 1.2], start=1)
        ]
        result = detect_improvement_trend(cycles, min_consecutive=3)
        assert result is False

    def test_only_last_n_cycles_matter(self):
        """Chỉ xét min_consecutive cycles cuối cùng."""
        # 2 cycles giảm + 3 cycles tăng → vẫn True
        cycles = [
            CycleResult(
                cycle_number=i, phase="PHASE_C", sharpe_ratio=sharpe,
                win_rate=0.5, total_return=5.0, strategies_beaten=1,
                validation_loss=0.05, is_improving=False,
                timestamp=datetime(2024, 1, i),
            )
            for i, sharpe in enumerate([2.0, 1.5, 0.8, 1.2, 1.6], start=1)
        ]
        result = detect_improvement_trend(cycles, min_consecutive=3)
        assert result is True

    def test_last_cycle_drops_returns_false(self):
        """Cycle cuối giảm → False dù trước đó tăng."""
        cycles = [
            CycleResult(
                cycle_number=i, phase="PHASE_C", sharpe_ratio=sharpe,
                win_rate=0.5, total_return=5.0, strategies_beaten=1,
                validation_loss=0.05, is_improving=False,
                timestamp=datetime(2024, 1, i),
            )
            for i, sharpe in enumerate([0.8, 1.0, 1.2, 1.1], start=1)
        ]
        result = detect_improvement_trend(cycles, min_consecutive=3)
        assert result is False

    def test_empty_list_returns_false(self):
        """List rỗng → False."""
        result = detect_improvement_trend([], min_consecutive=3)
        assert result is False

    def test_custom_min_consecutive(self):
        """Hỗ trợ custom min_consecutive."""
        cycles = [
            CycleResult(
                cycle_number=i, phase="PHASE_C", sharpe_ratio=float(i),
                win_rate=0.5, total_return=5.0, strategies_beaten=1,
                validation_loss=0.05, is_improving=True,
                timestamp=datetime(2024, 1, i),
            )
            for i in range(1, 6)
        ]
        # min_consecutive=5 → True (5 cycles tăng liên tục)
        assert detect_improvement_trend(cycles, min_consecutive=5) is True
        # min_consecutive=6 → False (chỉ có 5 cycles)
        assert detect_improvement_trend(cycles, min_consecutive=6) is False


# ---------------------------------------------------------------------------
# Tests: get_latest_cycle
# ---------------------------------------------------------------------------


class TestGetLatestCycle:
    """Kiểm tra lấy cycle gần nhất."""

    def test_returns_none_when_no_cycles(self, tmp_history_dir):
        """Không có cycle nào → None."""
        result = get_latest_cycle(history_dir=tmp_history_dir)
        assert result is None

    def test_returns_highest_cycle_number(self, tmp_history_dir, multiple_cycles):
        """Trả về cycle có cycle_number lớn nhất."""
        for cycle in multiple_cycles:
            persist_cycle(cycle, history_dir=tmp_history_dir)

        latest = get_latest_cycle(history_dir=tmp_history_dir)
        assert latest is not None
        assert latest.cycle_number == 5

    def test_returns_none_for_nonexistent_dir(self, tmp_path):
        """Thư mục không tồn tại → None."""
        result = get_latest_cycle(history_dir=str(tmp_path / "nope"))
        assert result is None
