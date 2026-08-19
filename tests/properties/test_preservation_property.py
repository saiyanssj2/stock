# -*- coding: utf-8 -*-
"""
Property-based tests cho Preservation — đảm bảo hành vi hiện tại KHÔNG bị thay đổi
khi implement bugfix cho process state visibility và settings persistence.

# Feature: streamlit-process-state-fix, Property 3: Preservation

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**

Observation-first methodology:
- Step 1: Quan sát hành vi hiện tại (unfixed code) cho non-buggy inputs
- Step 2: Viết property tests capturing observed behavior
- Step 3: Verify tests pass trên unfixed code (confirms baseline)

Properties:
1. write_status + read_status roundtrip bảo toàn dữ liệu (atomic write protocol preserved)
2. JSON serialization roundtrip cho preferences bảo toàn identity
3. Status files độc lập — đọc/ghi 1 task không ảnh hưởng task khác
4. stop_task() với valid process reference vẫn terminate + update status file
"""

import json
import multiprocessing
import os
import shutil
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict
from unittest.mock import patch

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from config.settings import STATUS_DIR
from models.task_models import TaskState, TaskStatus, TaskType


# ---------------------------------------------------------------------------
# Strategies — Sinh dữ liệu ngẫu nhiên hợp lệ cho property tests
# ---------------------------------------------------------------------------

# Loại task ngẫu nhiên
task_type_strategy = st.sampled_from(list(TaskType))

# Trạng thái task ngẫu nhiên
task_state_strategy = st.sampled_from(list(TaskState))

# Task ID: format matching thực tế (task_type_hex8)
task_id_strategy = st.from_regex(r"[a-z]+_[0-9a-f]{8}", fullmatch=True)

# Progress: 0.0 - 100.0
progress_strategy = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)

# Message: chuỗi text ngắn hợp lệ (ASCII an toàn cho JSON)
message_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Z"), max_codepoint=127),
    min_size=0,
    max_size=50,
)

# Datetime: ngày hợp lệ trong khoảng vài năm gần đây
datetime_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
)

# Details dict: key-value đơn giản (JSON-serializable)
details_strategy = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), max_codepoint=127),
        min_size=1,
        max_size=15,
    ),
    values=st.one_of(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "Z"), max_codepoint=127),
            min_size=0,
            max_size=30,
        ),
        st.integers(min_value=-1000, max_value=1000),
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        st.booleans(),
    ),
    min_size=0,
    max_size=5,
)


@st.composite
def task_status_strategy(draw):
    """Sinh TaskStatus ngẫu nhiên hợp lệ cho property tests."""
    task_id = draw(task_id_strategy)
    task_type = draw(task_type_strategy)
    state = draw(task_state_strategy)
    progress = draw(progress_strategy)
    message = draw(message_strategy)
    heartbeat_ts = draw(datetime_strategy)
    started_at = draw(datetime_strategy)
    # Error chỉ có khi state == ERROR
    error = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Z"), max_codepoint=127),
        min_size=1,
        max_size=30,
    )) if state == TaskState.ERROR else None
    details = draw(details_strategy)

    return TaskStatus(
        task_id=task_id,
        task_type=task_type,
        state=state,
        progress_pct=progress,
        message=message,
        heartbeat_ts=heartbeat_ts,
        started_at=started_at,
        error=error,
        details=details,
    )


# Strategies cho preferences
_SAMPLE_SYMBOLS = ["ACB", "BCM", "BID", "FPT", "GAS", "HPG", "VNM", "VCB", "MWG"]

selected_symbols_strategy = st.lists(
    st.sampled_from(_SAMPLE_SYMBOLS),
    min_size=0,
    max_size=len(_SAMPLE_SYMBOLS),
    unique=True,
)

date_strategy = st.dates(min_value=date(2010, 1, 1), max_value=date(2035, 12, 31))

capital_strategy = st.floats(
    min_value=1_000_000.0,
    max_value=100_000_000_000.0,
    allow_nan=False,
    allow_infinity=False,
)

auto_update_time_strategy = st.builds(
    lambda h, m: f"{h:02d}:{m:02d}",
    h=st.integers(min_value=0, max_value=23),
    m=st.integers(min_value=0, max_value=59),
)

cycle_interval_strategy = st.floats(
    min_value=1.0,
    max_value=168.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def date_range_strategy(draw):
    """Sinh cặp (start_date, end_date) hợp lệ với start < end."""
    start = draw(date_strategy)
    delta = draw(st.integers(min_value=1, max_value=3650))
    end = start + timedelta(days=delta)
    if end > date(2035, 12, 31):
        end = date(2035, 12, 31)
    if start >= end:
        start = end - timedelta(days=1)
    return start, end


@st.composite
def preferences_strategy(draw):
    """Sinh dict preferences hợp lệ cho property tests."""
    selected_symbols = draw(selected_symbols_strategy)
    start_date, end_date = draw(date_range_strategy())
    capital = draw(capital_strategy)
    auto_update_time = draw(auto_update_time_strategy)
    cycle_interval = draw(cycle_interval_strategy)
    auto_update_enabled = draw(st.booleans())

    return {
        "selected_symbols": selected_symbols,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "capital": capital,
        "auto_update_time": auto_update_time,
        "cycle_interval": cycle_interval,
        "auto_update_enabled": auto_update_enabled,
    }


# ---------------------------------------------------------------------------
# Helper: tạo/xóa temp directory riêng cho mỗi test case
# ---------------------------------------------------------------------------


def _make_temp_dir() -> Path:
    """Tạo temp directory mới cho mỗi test case."""
    return Path(tempfile.mkdtemp(prefix="pbt_preservation_"))


def _cleanup_temp_dir(path: Path) -> None:
    """Xóa temp directory sau test."""
    try:
        shutil.rmtree(str(path), ignore_errors=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Property 1: write_status + read_status roundtrip (atomic write protocol)
# ---------------------------------------------------------------------------


class TestAtomicWriteProtocolPreservation:
    """
    Property 1: write_status() + read_status() roundtrip bảo toàn dữ liệu.

    Observation: write_status(task_status) sử dụng atomic write protocol
    (temp file + os.replace()) để ghi JSON. read_status(task_id) đọc và
    parse file JSON trả về TaskStatus object giống hệt input.

    **Validates: Requirements 3.1**
    """

    @given(status=task_status_strategy())
    @settings(max_examples=50, deadline=None)
    def test_write_then_read_roundtrip_preserves_all_fields(self, status):
        """
        Property: Với bất kỳ TaskStatus hợp lệ nào, write_status() → read_status()
        phải trả về TaskStatus có cùng tất cả fields (roundtrip identity).

        **Validates: Requirements 3.1**
        """
        tmp_dir = _make_temp_dir()
        try:
            with patch("orchestrator.status_protocol._get_status_dir", return_value=tmp_dir):
                from orchestrator.status_protocol import write_status, read_status

                # Ghi status ra file
                write_status(status)

                # Đọc lại từ file
                loaded = read_status(status.task_id)

                # Assert roundtrip bảo toàn tất cả fields
                assert loaded is not None, f"read_status trả về None cho task_id={status.task_id}"
                assert loaded.task_id == status.task_id
                assert loaded.task_type == status.task_type
                assert loaded.state == status.state
                assert loaded.progress_pct == status.progress_pct
                assert loaded.message == status.message
                assert loaded.heartbeat_ts == status.heartbeat_ts
                assert loaded.started_at == status.started_at
                assert loaded.error == status.error
                assert loaded.details == status.details
        finally:
            _cleanup_temp_dir(tmp_dir)

    @given(status=task_status_strategy())
    @settings(max_examples=50, deadline=None)
    def test_write_status_produces_valid_json_with_required_schema(self, status):
        """
        Property: write_status() luôn tạo file JSON hợp lệ với đủ required fields.

        **Validates: Requirements 3.1**
        """
        tmp_dir = _make_temp_dir()
        try:
            with patch("orchestrator.status_protocol._get_status_dir", return_value=tmp_dir):
                from orchestrator.status_protocol import write_status

                write_status(status)

                # Verify file tồn tại và là valid JSON
                file_path = tmp_dir / f"{status.task_id}.json"
                assert file_path.exists(), f"Status file không được tạo: {file_path}"

                content = file_path.read_text(encoding="utf-8")
                data = json.loads(content)

                # Verify JSON schema có đủ required fields
                required_keys = {
                    "task_id", "task_type", "state", "progress_pct",
                    "message", "heartbeat_ts", "started_at",
                }
                assert required_keys.issubset(data.keys()), (
                    f"JSON thiếu keys. Expected: {required_keys}, Got: {set(data.keys())}"
                )

                # Verify giá trị type/state là string hợp lệ
                assert data["task_type"] in [t.value for t in TaskType]
                assert data["state"] in [s.value for s in TaskState]
        finally:
            _cleanup_temp_dir(tmp_dir)

    @given(status=task_status_strategy())
    @settings(max_examples=30, deadline=None)
    def test_write_status_overwrites_previous_completely(self, status):
        """
        Property: write_status() ghi đè file cũ hoàn toàn — đọc lại luôn
        trả về version mới nhất (không append, không merge).

        **Validates: Requirements 3.1**
        """
        tmp_dir = _make_temp_dir()
        try:
            with patch("orchestrator.status_protocol._get_status_dir", return_value=tmp_dir):
                from orchestrator.status_protocol import write_status, read_status

                # Ghi lần 1
                write_status(status)

                # Thay đổi state và message rồi ghi lần 2
                status.state = TaskState.COMPLETED
                status.message = "Done"
                write_status(status)

                # Đọc lại — phải là bản mới nhất
                loaded = read_status(status.task_id)
                assert loaded is not None
                assert loaded.state == TaskState.COMPLETED
                assert loaded.message == "Done"
        finally:
            _cleanup_temp_dir(tmp_dir)


# ---------------------------------------------------------------------------
# Property 2: Preferences JSON serialization roundtrip
# ---------------------------------------------------------------------------


class TestPreferencesSerializationPreservation:
    """
    Property 2: JSON serialization roundtrip cho preferences bảo toàn identity.

    Observation: Preferences dict (symbols list, dates, capital, toggles) khi
    serialize sang JSON string rồi deserialize lại phải trả về giá trị giống hệt.
    Atomic write pattern (temp file + os.replace) bảo toàn file integrity.

    **Validates: Requirements 3.5**
    """

    @given(prefs=preferences_strategy())
    @settings(max_examples=100, deadline=None)
    def test_preferences_json_roundtrip_identity(self, prefs):
        """
        Property: Bất kỳ tập preferences hợp lệ nào khi serialize → JSON string
        → deserialize phải trả về giá trị identical cho tất cả fields.

        **Validates: Requirements 3.5**
        """
        # Serialize → JSON string
        json_str = json.dumps(prefs, ensure_ascii=False)

        # Deserialize lại
        loaded = json.loads(json_str)

        # Assert roundtrip bảo toàn tất cả fields
        assert loaded["selected_symbols"] == prefs["selected_symbols"]
        assert loaded["start_date"] == prefs["start_date"]
        assert loaded["end_date"] == prefs["end_date"]
        assert loaded["capital"] == prefs["capital"]
        assert loaded["auto_update_time"] == prefs["auto_update_time"]
        assert loaded["cycle_interval"] == prefs["cycle_interval"]
        assert loaded["auto_update_enabled"] == prefs["auto_update_enabled"]

    @given(prefs=preferences_strategy())
    @settings(max_examples=50, deadline=None)
    def test_preferences_file_roundtrip_with_atomic_write(self, prefs):
        """
        Property: Preferences ghi ra file JSON dùng atomic write (temp → os.replace)
        → đọc lại từ file phải identical. Mô phỏng cùng protocol với write_status.

        **Validates: Requirements 3.5**
        """
        tmp_dir = _make_temp_dir()
        try:
            target_file = tmp_dir / "preferences.json"
            json_content = json.dumps(prefs, ensure_ascii=False, indent=2)

            # Atomic write: temp file → os.replace (cùng pattern status_protocol)
            tmp_file = tmp_dir / "preferences.tmp"
            tmp_file.write_text(json_content, encoding="utf-8")
            os.replace(str(tmp_file), str(target_file))

            # Đọc lại từ file
            loaded_content = target_file.read_text(encoding="utf-8")
            loaded = json.loads(loaded_content)

            # Assert roundtrip
            assert loaded == prefs
        finally:
            _cleanup_temp_dir(tmp_dir)

    @given(prefs=preferences_strategy())
    @settings(max_examples=30, deadline=None)
    def test_preferences_save_load_via_module(self, prefs):
        """
        Property: save_preferences_to_file() → load_preferences_from_file() roundtrip
        bảo toàn tất cả preferences data (dùng actual module functions).

        **Validates: Requirements 3.5**
        """
        # Cần ít nhất 1 symbol cho validation pass
        assume(len(prefs["selected_symbols"]) > 0)
        # Capital phải > 0
        assume(prefs["capital"] > 0)

        tmp_dir = _make_temp_dir()
        tmp_prefs_file = str(tmp_dir / "preferences.json")

        try:
            with patch("config.preferences.PREFERENCES_FILE_PATH", tmp_prefs_file):
                from config.preferences import save_preferences_to_file, load_preferences_from_file

                # Ghi preferences ra file
                save_preferences_to_file(prefs)

                # Đọc lại
                loaded = load_preferences_from_file()

                # Assert roundtrip
                assert loaded is not None, "load_preferences_from_file() trả về None"
                assert loaded["selected_symbols"] == prefs["selected_symbols"]
                assert loaded["capital"] == prefs["capital"]
                assert loaded["auto_update_time"] == prefs["auto_update_time"]
                assert loaded["cycle_interval"] == prefs["cycle_interval"]
                assert loaded["auto_update_enabled"] == prefs["auto_update_enabled"]
        finally:
            _cleanup_temp_dir(tmp_dir)


# ---------------------------------------------------------------------------
# Property 3: Status files độc lập (multi-task independence)
# ---------------------------------------------------------------------------


class TestMultiTaskIndependencePreservation:
    """
    Property 3: Status files độc lập — đọc/ghi task A không ảnh hưởng task B.

    Observation: Multiple tasks (training + backtest) tracked independently.
    Ghi status cho 1 task không ghi đè hoặc corrupt status file của task khác.

    **Validates: Requirements 3.7**
    """

    @given(
        status_a=task_status_strategy(),
        status_b=task_status_strategy(),
    )
    @settings(max_examples=50, deadline=None)
    def test_two_tasks_independent_write_read(self, status_a, status_b):
        """
        Property: Ghi status cho task A rồi ghi status cho task B
        → đọc lại task A phải trả về dữ liệu của A (không bị B ảnh hưởng).

        **Validates: Requirements 3.7**
        """
        # Đảm bảo 2 task có ID khác nhau
        assume(status_a.task_id != status_b.task_id)

        tmp_dir = _make_temp_dir()
        try:
            with patch("orchestrator.status_protocol._get_status_dir", return_value=tmp_dir):
                from orchestrator.status_protocol import write_status, read_status

                # Ghi task A
                write_status(status_a)

                # Ghi task B
                write_status(status_b)

                # Đọc lại task A — phải vẫn nguyên vẹn
                loaded_a = read_status(status_a.task_id)
                assert loaded_a is not None
                assert loaded_a.task_id == status_a.task_id
                assert loaded_a.task_type == status_a.task_type
                assert loaded_a.state == status_a.state
                assert loaded_a.progress_pct == status_a.progress_pct
                assert loaded_a.message == status_a.message
                assert loaded_a.heartbeat_ts == status_a.heartbeat_ts
                assert loaded_a.started_at == status_a.started_at

                # Đọc lại task B — phải đúng dữ liệu B
                loaded_b = read_status(status_b.task_id)
                assert loaded_b is not None
                assert loaded_b.task_id == status_b.task_id
                assert loaded_b.task_type == status_b.task_type
                assert loaded_b.state == status_b.state
        finally:
            _cleanup_temp_dir(tmp_dir)

    @given(
        data=st.data(),
        num_tasks=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_n_tasks_all_readable_via_read_all_statuses(self, data, num_tasks):
        """
        Property: Với N tasks ghi vào cùng thư mục status, read_all_statuses()
        phải trả về đúng N entries, mỗi entry giữ nguyên dữ liệu gốc.

        **Validates: Requirements 3.7**
        """
        tmp_dir = _make_temp_dir()
        try:
            with patch("orchestrator.status_protocol._get_status_dir", return_value=tmp_dir):
                from orchestrator.status_protocol import write_status, read_all_statuses

                # Sinh N TaskStatus với task_id unique
                statuses: Dict[str, TaskStatus] = {}
                used_ids = set()

                for _ in range(num_tasks):
                    status = data.draw(task_status_strategy())
                    while status.task_id in used_ids:
                        status = data.draw(task_status_strategy())
                    used_ids.add(status.task_id)
                    statuses[status.task_id] = status
                    write_status(status)

                # Đọc tất cả
                all_statuses = read_all_statuses()

                # Phải có đúng N entries
                assert len(all_statuses) == num_tasks, (
                    f"Expected {num_tasks} statuses, got {len(all_statuses)}"
                )

                # Mỗi entry phải đúng dữ liệu gốc
                for task_id, original in statuses.items():
                    assert task_id in all_statuses, (
                        f"Task {task_id} không có trong read_all_statuses()"
                    )
                    loaded = all_statuses[task_id]
                    assert loaded.task_type == original.task_type
                    assert loaded.state == original.state
                    assert loaded.progress_pct == original.progress_pct
        finally:
            _cleanup_temp_dir(tmp_dir)

    @given(
        status_a=task_status_strategy(),
        status_b=task_status_strategy(),
    )
    @settings(max_examples=30, deadline=None)
    def test_reading_one_task_does_not_modify_another(self, status_a, status_b):
        """
        Property: Đọc status của task A không side-effect lên file của task B.
        File content phải byte-identical trước và sau read.

        **Validates: Requirements 3.7**
        """
        assume(status_a.task_id != status_b.task_id)

        tmp_dir = _make_temp_dir()
        try:
            with patch("orchestrator.status_protocol._get_status_dir", return_value=tmp_dir):
                from orchestrator.status_protocol import write_status, read_status

                # Ghi cả 2 tasks
                write_status(status_a)
                write_status(status_b)

                # Snapshot nội dung file B trước khi đọc A
                file_b_path = tmp_dir / f"{status_b.task_id}.json"
                content_before = file_b_path.read_text(encoding="utf-8")

                # Đọc task A
                _ = read_status(status_a.task_id)

                # File B phải không thay đổi
                content_after = file_b_path.read_text(encoding="utf-8")
                assert content_before == content_after, (
                    "Đọc task A đã thay đổi file của task B!"
                )
        finally:
            _cleanup_temp_dir(tmp_dir)


# ---------------------------------------------------------------------------
# Property 4: stop_task() với valid process reference
# ---------------------------------------------------------------------------


def _dummy_worker_target(duration: float = 30.0):
    """Worker giả lập cho test — sleep rồi kết thúc."""
    time.sleep(duration)


class TestStopTaskWithProcessPreservation:
    """
    Property 4: stop_task() với valid process reference vẫn terminate process
    và update status file thành COMPLETED.

    Observation: Khi _processes[task_id] chứa process reference hợp lệ (alive),
    stop_task() gọi terminate() → join() → force kill nếu cần, rồi update
    status file thành COMPLETED + xóa khỏi _processes dict.

    **Validates: Requirements 3.2**
    """

    @given(
        task_type=task_type_strategy,
        progress=st.floats(min_value=0.0, max_value=99.0, allow_nan=False),
    )
    @settings(max_examples=5, deadline=None)
    def test_stop_task_with_process_terminates_and_updates_status(
        self, task_type, progress
    ):
        """
        Property: Khi _processes[task_id] chứa live process, stop_task() phải:
        1. Terminate process (not alive after stop)
        2. Xóa task_id khỏi _processes dict
        3. Update status file thành COMPLETED

        **Validates: Requirements 3.2**
        """
        from orchestrator.task_orchestrator import TaskOrchestrator
        from orchestrator.status_protocol import write_status, read_status

        tmp_dir = _make_temp_dir()
        try:
            with patch("orchestrator.status_protocol._get_status_dir", return_value=tmp_dir):
                task_id = f"{task_type.value}_test1234"

                # Tạo status file RUNNING (mô phỏng worker đã ghi)
                now = datetime.now()
                initial_status = TaskStatus(
                    task_id=task_id,
                    task_type=task_type,
                    state=TaskState.RUNNING,
                    progress_pct=progress,
                    message="Worker running",
                    heartbeat_ts=now,
                    started_at=now,
                )
                write_status(initial_status)

                # Tạo orchestrator và inject process reference
                orchestrator = TaskOrchestrator()
                process = multiprocessing.Process(
                    target=_dummy_worker_target,
                    args=(30.0,),
                    daemon=True,
                )
                process.start()
                orchestrator._processes[task_id] = process

                try:
                    # Stop task
                    result = orchestrator.stop_task(task_id)

                    # Assert: stop thành công
                    assert result is True

                    # Assert: process đã bị terminate
                    assert not process.is_alive(), "Process vẫn còn alive sau stop_task()"

                    # Assert: task_id bị xóa khỏi _processes
                    assert task_id not in orchestrator._processes

                    # Assert: status file updated thành COMPLETED
                    final_status = read_status(task_id)
                    assert final_status is not None
                    assert final_status.state == TaskState.COMPLETED
                    assert final_status.message == "Task stopped by user"
                finally:
                    # Cleanup: đảm bảo process bị kill nếu test fail
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5.0)
        finally:
            _cleanup_temp_dir(tmp_dir)

    @given(task_type=task_type_strategy)
    @settings(max_examples=10, deadline=None)
    def test_stop_task_without_process_still_updates_status_file(
        self, task_type
    ):
        """
        Property: Khi _processes[task_id] KHÔNG tồn tại (session restart),
        stop_task() vẫn update status file thành COMPLETED (không crash).

        Observation: Hành vi hiện tại đã đúng — stop_task() handle case
        process None bằng cách chỉ update status file.

        **Validates: Requirements 3.2**
        """
        from orchestrator.task_orchestrator import TaskOrchestrator
        from orchestrator.status_protocol import write_status, read_status

        tmp_dir = _make_temp_dir()
        try:
            with patch("orchestrator.status_protocol._get_status_dir", return_value=tmp_dir):
                task_id = f"{task_type.value}_noref123"

                # Tạo status file RUNNING (mô phỏng worker ghi từ trước)
                now = datetime.now()
                initial_status = TaskStatus(
                    task_id=task_id,
                    task_type=task_type,
                    state=TaskState.RUNNING,
                    progress_pct=50.0,
                    message="Worker running",
                    heartbeat_ts=now,
                    started_at=now,
                )
                write_status(initial_status)

                # Tạo orchestrator MỚI (empty _processes — simulating session restart)
                orchestrator = TaskOrchestrator()
                assert task_id not in orchestrator._processes

                # Stop task — phải không crash
                result = orchestrator.stop_task(task_id)

                # Assert: vẫn trả về True
                assert result is True

                # Assert: status file updated thành COMPLETED
                final_status = read_status(task_id)
                assert final_status is not None
                assert final_status.state == TaskState.COMPLETED
                assert final_status.message == "Task stopped by user"
        finally:
            _cleanup_temp_dir(tmp_dir)

    @given(task_type=task_type_strategy)
    @settings(max_examples=10, deadline=None)
    def test_stop_task_returns_true_even_when_status_file_missing(
        self, task_type
    ):
        """
        Property: stop_task() trả về True và tạo status COMPLETED ngay cả khi
        không tìm thấy status file cũ (edge case — tạo status mới).

        **Validates: Requirements 3.2**
        """
        from orchestrator.task_orchestrator import TaskOrchestrator
        from orchestrator.status_protocol import read_status

        tmp_dir = _make_temp_dir()
        try:
            with patch("orchestrator.status_protocol._get_status_dir", return_value=tmp_dir):
                task_id = f"{task_type.value}_ghost123"

                # KHÔNG tạo status file — mô phỏng case file bị xóa/corrupt
                orchestrator = TaskOrchestrator()

                # Stop task — phải không crash
                result = orchestrator.stop_task(task_id)

                # Assert: vẫn trả về True
                assert result is True

                # Assert: status file mới được tạo với state COMPLETED
                final_status = read_status(task_id)
                assert final_status is not None
                assert final_status.state == TaskState.COMPLETED
        finally:
            _cleanup_temp_dir(tmp_dir)
