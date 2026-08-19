# -*- coding: utf-8 -*-
"""
Property-based tests cho Preservation — đảm bảo hành vi hiện tại KHÔNG bị thay đổi
khi implement bugfix cho process state visibility và settings persistence.

# Feature: streamlit-process-state-fix, Property 3: Preservation

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**

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
# Strategies
# ---------------------------------------------------------------------------

# Loại task ngẫu nhiên
task_type_strategy = st.sampled_from(list(TaskType))

# Trạng thái task ngẫu nhiên
task_state_strategy = st.sampled_from(list(TaskState))

# Task ID: chuỗi alphanumeric hợp lệ (matching format thực tế)
task_id_strategy = st.from_regex(r"[a-z]+_[0-9a-f]{8}", fullmatch=True)

# Progress: 0.0 - 100.0
progress_strategy = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)

# Message: chuỗi text ngắn hợp lệ (chỉ dùng ASCII an toàn cho JSON)
message_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Z"), max_codepoint=127),
    min_size=0,
    max_size=50,
)

# Datetime strategy: ngày hợp lệ trong khoảng vài năm gần đây
datetime_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
)

# Details dict: key-value đơn giản (serializable)
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
    """Sinh TaskStatus ngẫu nhiên hợp lệ."""
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


# ---------------------------------------------------------------------------
# Helper: tạo temp directory riêng cho mỗi Hypothesis example
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
# Property 1: write_status + read_status roundtrip
# ---------------------------------------------------------------------------


class TestAtomicWriteProtocolPreservation:
    """
    Property 1: write_status() + read_status() roundtrip bảo toàn dữ liệu.

    Xác nhận atomic write protocol (temp file + os.replace) hoạt động đúng
    và không bị thay đổi bởi bugfix.

    **Validates: Requirements 3.1**
    """

    @given(status=task_status_strategy())
    @settings(max_examples=50, deadline=None)
    def test_write_then_read_roundtrip_preserves_data(self, status):
        """
        Property: Với bất kỳ TaskStatus hợp lệ nào, write_status() → read_status()
        phải trả về TaskStatus có cùng tất cả fields.

        # Feature: streamlit-process-state-fix, Property 3: Preservation
        **Validates: Requirements 3.1**
        """
        tmp_dir = _make_temp_dir()
        try:
            with patch("orchestrator.status_protocol._get_status_dir", return_value=tmp_dir):
                from orchestrator.status_protocol import write_status, read_status

                # Ghi status
                write_status(status)

                # Đọc lại
                loaded = read_status(status.task_id)

                # Assert roundtrip bảo toàn dữ liệu
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
    def test_write_status_produces_valid_json_file(self, status):
        """
        Property: write_status() luôn tạo ra file JSON hợp lệ có thể parse được.

        # Feature: streamlit-process-state-fix, Property 3: Preservation
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
                data = json.loads(content)  # Phải parse được, không raise exception

                # Verify JSON schema có đủ required fields
                required_keys = {
                    "task_id", "task_type", "state", "progress_pct",
                    "message", "heartbeat_ts", "started_at",
                }
                assert required_keys.issubset(data.keys()), (
                    f"JSON thiếu keys. Expected: {required_keys}, Got: {set(data.keys())}"
                )
        finally:
            _cleanup_temp_dir(tmp_dir)

    @given(status=task_status_strategy())
    @settings(max_examples=30, deadline=None)
    def test_write_status_overwrites_existing_file(self, status):
        """
        Property: write_status() ghi đè file cũ hoàn toàn (không append).

        # Feature: streamlit-process-state-fix, Property 3: Preservation
        **Validates: Requirements 3.1**
        """
        tmp_dir = _make_temp_dir()
        try:
            with patch("orchestrator.status_protocol._get_status_dir", return_value=tmp_dir):
                from orchestrator.status_protocol import write_status, read_status

                # Ghi lần 1
                write_status(status)

                # Thay đổi state rồi ghi lần 2
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
# Property 2: JSON serialization roundtrip cho preferences
# ---------------------------------------------------------------------------


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
    """Sinh cặp (start_date, end_date) hợp lệ."""
    start = draw(date_strategy)
    delta = draw(st.integers(min_value=1, max_value=3650))
    end = start + timedelta(days=delta)
    if end > date(2035, 12, 31):
        end = date(2035, 12, 31)
    if start >= end:
        start = end - timedelta(days=1)
    return start, end


class TestPreferencesSerializationPreservation:
    """
    Property 2: JSON serialization roundtrip cho preferences bảo toàn identity.

    Đảm bảo preferences dict khi serialize sang JSON rồi deserialize lại
    trả về dữ liệu giống hệt — chuẩn bị cho file-based persistence.

    **Validates: Requirements 3.5**
    """

    @given(
        selected_symbols=selected_symbols_strategy,
        date_range=date_range_strategy(),
        capital=capital_strategy,
        auto_update_time=auto_update_time_strategy,
        cycle_interval=cycle_interval_strategy,
        auto_update_enabled=st.booleans(),
    )
    @settings(max_examples=100, deadline=None)
    def test_preferences_json_roundtrip_identity(
        self,
        selected_symbols,
        date_range,
        capital,
        auto_update_time,
        cycle_interval,
        auto_update_enabled,
    ):
        """
        Property: Bất kỳ tập preferences hợp lệ nào khi serialize → JSON string
        → deserialize phải trả về giá trị identical.

        # Feature: streamlit-process-state-fix, Property 3: Preservation
        **Validates: Requirements 3.5**
        """
        start_date, end_date = date_range

        # Tạo preferences dict (format sẽ dùng cho file persistence)
        prefs = {
            "selected_symbols": selected_symbols,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "capital": capital,
            "auto_update_time": auto_update_time,
            "cycle_interval": cycle_interval,
            "auto_update_enabled": auto_update_enabled,
        }

        # Serialize → JSON string
        json_str = json.dumps(prefs, ensure_ascii=False)

        # Deserialize lại
        loaded = json.loads(json_str)

        # Assert roundtrip bảo toàn tất cả fields
        assert loaded["selected_symbols"] == selected_symbols
        assert loaded["start_date"] == start_date.isoformat()
        assert loaded["end_date"] == end_date.isoformat()
        assert loaded["capital"] == capital
        assert loaded["auto_update_time"] == auto_update_time
        assert loaded["cycle_interval"] == cycle_interval
        assert loaded["auto_update_enabled"] == auto_update_enabled

    @given(
        selected_symbols=selected_symbols_strategy,
        date_range=date_range_strategy(),
        capital=capital_strategy,
        auto_update_time=auto_update_time_strategy,
        cycle_interval=cycle_interval_strategy,
        auto_update_enabled=st.booleans(),
    )
    @settings(max_examples=50, deadline=None)
    def test_preferences_file_roundtrip_identity(
        self,
        selected_symbols,
        date_range,
        capital,
        auto_update_time,
        cycle_interval,
        auto_update_enabled,
    ):
        """
        Property: Preferences ghi ra file JSON → đọc lại từ file phải identical.
        Mô phỏng atomic write pattern (giống write_status).

        # Feature: streamlit-process-state-fix, Property 3: Preservation
        **Validates: Requirements 3.5**
        """
        start_date, end_date = date_range
        tmp_dir = _make_temp_dir()

        try:
            prefs = {
                "selected_symbols": selected_symbols,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "capital": capital,
                "auto_update_time": auto_update_time,
                "cycle_interval": cycle_interval,
                "auto_update_enabled": auto_update_enabled,
            }

            # Ghi ra file (mô phỏng atomic write pattern)
            target_file = tmp_dir / "preferences.json"
            json_content = json.dumps(prefs, ensure_ascii=False, indent=2)

            # Atomic write: temp file → os.replace
            tmp_file = tmp_dir / "preferences.tmp"
            tmp_file.write_text(json_content, encoding="utf-8")
            os.replace(str(tmp_file), str(target_file))

            # Đọc lại từ file
            loaded_content = target_file.read_text(encoding="utf-8")
            loaded = json.loads(loaded_content)

            # Assert roundtrip
            assert loaded["selected_symbols"] == selected_symbols
            assert loaded["start_date"] == start_date.isoformat()
            assert loaded["end_date"] == end_date.isoformat()
            assert loaded["capital"] == capital
            assert loaded["auto_update_time"] == auto_update_time
            assert loaded["cycle_interval"] == cycle_interval
            assert loaded["auto_update_enabled"] == auto_update_enabled
        finally:
            _cleanup_temp_dir(tmp_dir)


# ---------------------------------------------------------------------------
# Property 3: Status files độc lập
# ---------------------------------------------------------------------------


class TestMultiTaskIndependencePreservation:
    """
    Property 3: Status files độc lập — đọc/ghi task A không ảnh hưởng task B.

    Đảm bảo multi-task tracking (training + backtest song song) vẫn hoạt động.

    **Validates: Requirements 3.7**
    """

    @given(
        status_a=task_status_strategy(),
        status_b=task_status_strategy(),
    )
    @settings(max_examples=50, deadline=None)
    def test_two_tasks_independent_status_files(self, status_a, status_b):
        """
        Property: Ghi status cho task A rồi ghi status cho task B
        → đọc lại task A phải trả về dữ liệu của A (không bị B ghi đè).

        # Feature: streamlit-process-state-fix, Property 3: Preservation
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
        num_tasks=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_multiple_tasks_all_readable_independently(self, data, num_tasks):
        """
        Property: Với N tasks ghi vào cùng thư mục status, read_all_statuses()
        phải trả về đúng N entries, mỗi entry đúng dữ liệu gốc.

        # Feature: streamlit-process-state-fix, Property 3: Preservation
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
                    # Đảm bảo unique task_id
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

    Đảm bảo hành vi stop_task khi có process trong _processes dict KHÔNG thay đổi.

    **Validates: Requirements 3.2**
    """

    @given(
        task_type=task_type_strategy,
        progress=st.floats(min_value=0.0, max_value=99.0, allow_nan=False),
    )
    @settings(max_examples=5, deadline=None)
    def test_stop_task_with_process_reference_terminates_and_updates_status(
        self, task_type, progress
    ):
        """
        Property: Khi _processes[task_id] chứa process reference hợp lệ,
        stop_task() phải terminate process VÀ update status file thành COMPLETED.

        # Feature: streamlit-process-state-fix, Property 3: Preservation
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

                    # Assert: process đã bị terminate (không còn alive)
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
    def test_stop_task_without_process_reference_still_updates_status(
        self, task_type
    ):
        """
        Property: Khi _processes[task_id] KHÔNG tồn tại (session restart),
        stop_task() vẫn update status file thành COMPLETED (không crash).

        Đây là hành vi hiện tại đã đúng — preservation test đảm bảo không bị thay đổi.

        # Feature: streamlit-process-state-fix, Property 3: Preservation
        **Validates: Requirements 3.2**
        """
        from orchestrator.task_orchestrator import TaskOrchestrator
        from orchestrator.status_protocol import write_status, read_status

        tmp_dir = _make_temp_dir()
        try:
            with patch("orchestrator.status_protocol._get_status_dir", return_value=tmp_dir):
                task_id = f"{task_type.value}_noref123"

                # Tạo status file RUNNING
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

                # Tạo orchestrator MỚI (không có process reference)
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
