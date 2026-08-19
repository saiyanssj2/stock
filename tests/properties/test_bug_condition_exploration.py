# -*- coding: utf-8 -*-
"""
Bug Condition Exploration Property Test - Process State Lost When Session Resets.

# Feature: streamlit-process-state-fix, Property 1: Bug Condition

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6**

CRITICAL: Test này PHẢI FAIL trên code chưa fix — failure xác nhận bug tồn tại.
DO NOT fix code hoặc test khi nó fail.

Properties:
1. Process state: get_active_tasks_from_disk() SHALL return task as active
   khi status file có state=RUNNING và heartbeat fresh
2. Settings persistence: load_preferences_from_file() SHALL return saved preferences
   sau khi session reset (simulate browser close/refresh)

Bug Condition formal:
- isBugCondition(input) where processInMemory == False AND existsStatusFile(task_id, RUNNING)
  AND heartbeatFresh == True
- isBugCondition(input) where action == 'load_preferences' AND trigger IN
  ['browser_refresh', 'new_session'] AND preferencesFileNotExists()

Counterexamples expected:
- New orchestrator instance has _processes = {} → no way to detect running task
- multiprocessing.Process objects not serializable → lost on session boundary
- No preferences file exists → defaults returned after refresh
"""

import json
import os
import shutil
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from config.settings import HEARTBEAT_TIMEOUT_THRESHOLD
from config.vn_market_rules import VN30_SYMBOLS
from models.task_models import TaskState, TaskStatus, TaskType


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Loại task ngẫu nhiên
task_type_strategy = st.sampled_from([TaskType.TRAINING, TaskType.BACKTEST])

# Fresh heartbeat: elapsed < HEARTBEAT_TIMEOUT_THRESHOLD (60s)
fresh_heartbeat_elapsed_strategy = st.floats(
    min_value=0.0,
    max_value=HEARTBEAT_TIMEOUT_THRESHOLD - 1.0,
    allow_nan=False,
    allow_infinity=False,
)

# Progress ngẫu nhiên 0-100
progress_strategy = st.floats(min_value=0.0, max_value=99.9)

# Symbols selection strategy
_ALL_SYMBOLS = VN30_SYMBOLS + ["FOX", "GEE", "GEL", "GEX", "HSG", "HUT"]
selected_symbols_strategy = st.lists(
    st.sampled_from(_ALL_SYMBOLS),
    min_size=1,
    max_size=len(_ALL_SYMBOLS),
    unique=True,
)

# Capital strategy
capital_strategy = st.floats(
    min_value=1_000_000.0,
    max_value=100_000_000_000.0,
    allow_nan=False,
    allow_infinity=False,
)

# Auto-update time strategy (HH:MM format)
auto_update_time_strategy = st.builds(
    lambda h, m: f"{h:02d}:{m:02d}",
    h=st.integers(min_value=0, max_value=23),
    m=st.integers(min_value=0, max_value=59),
)

# Cycle interval strategy (giờ)
cycle_interval_strategy = st.floats(
    min_value=1.0,
    max_value=168.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def running_task_status_strategy(draw):
    """
    Sinh TaskStatus với state=RUNNING và heartbeat fresh (< 60s).
    Simulate trường hợp worker process đang chạy bình thường.
    """
    task_type = draw(task_type_strategy)
    task_id = f"{task_type.value}_{draw(st.text(alphabet='abcdef0123456789', min_size=8, max_size=8))}"
    progress = draw(progress_strategy)
    elapsed = draw(fresh_heartbeat_elapsed_strategy)

    now = datetime.now()
    heartbeat_ts = now - timedelta(seconds=elapsed)
    started_at = now - timedelta(minutes=draw(st.integers(min_value=1, max_value=60)))

    return TaskStatus(
        task_id=task_id,
        task_type=task_type,
        state=TaskState.RUNNING,
        progress_pct=progress,
        message=f"{task_type.value} in progress",
        heartbeat_ts=heartbeat_ts,
        started_at=started_at,
        details={"phase": "phase_c", "pid": 12345},
    )


@st.composite
def preferences_strategy(draw):
    """Sinh bộ preferences ngẫu nhiên hợp lệ."""
    start_date = draw(st.dates(min_value=date(2020, 1, 1), max_value=date(2025, 6, 1)))
    delta = draw(st.integers(min_value=30, max_value=365))
    end_date = start_date + timedelta(days=delta)
    if end_date > date(2035, 12, 31):
        end_date = date(2035, 12, 31)

    return {
        "selected_symbols": draw(selected_symbols_strategy),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "capital": draw(capital_strategy),
        "auto_update_time": draw(auto_update_time_strategy),
        "cycle_interval": draw(cycle_interval_strategy),
        "auto_update_enabled": draw(st.booleans()),
    }


# ---------------------------------------------------------------------------
# Property Tests - Bug Condition: Process State
# ---------------------------------------------------------------------------


class TestBugConditionProcessState:
    """
    Property 1: Bug Condition - Process State Lost When Session Resets.

    Test rằng hệ thống CÓ THỂ phát hiện running tasks từ disk
    khi _processes dict trống (new orchestrator instance / session reset).

    Trên code chưa fix: TaskOrchestrator KHÔNG có method get_active_tasks_from_disk()
    → test PHẢI FAIL (AttributeError hoặc assertion fail).
    """

    @given(task_status=running_task_status_strategy())
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_new_orchestrator_detects_running_task_from_disk(self, task_status: TaskStatus):
        """
        Property: Khi tạo orchestrator instance mới (simulate session reset),
        hệ thống SHALL phát hiện task đang chạy từ status files trên disk,
        BẤT KỂ _processes dict trống.

        Scenario: Orchestrator A start task → Orchestrator B (new session) → query active tasks
        Bug condition: processInMemory == False AND existsStatusFile(task_id, RUNNING)
            AND heartbeatFresh == True

        # Feature: streamlit-process-state-fix, Property 1: Bug Condition
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        # Tạo temp directory cho status files
        tmp_dir = tempfile.mkdtemp()
        try:
            status_dir = Path(tmp_dir) / "status"
            status_dir.mkdir(parents=True, exist_ok=True)

            # Ghi status file trực tiếp (simulate worker đang chạy ghi heartbeat)
            status_data = {
                "task_id": task_status.task_id,
                "task_type": task_status.task_type.value,
                "state": TaskState.RUNNING.value,
                "progress_pct": task_status.progress_pct,
                "message": task_status.message,
                "heartbeat_ts": task_status.heartbeat_ts.isoformat(),
                "started_at": task_status.started_at.isoformat(),
                "error": None,
                "details": task_status.details,
            }
            status_file = status_dir / f"{task_status.task_id}.json"
            status_file.write_text(
                json.dumps(status_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Tạo orchestrator MỚI (simulate session reset — _processes = {})
            from orchestrator.task_orchestrator import TaskOrchestrator

            orchestrator_b = TaskOrchestrator()
            assert orchestrator_b._processes == {}, "New orchestrator phải có _processes trống"

            # BUG CONDITION TEST: Gọi get_active_tasks_from_disk()
            # Method này KHÔNG tồn tại trên code chưa fix → AttributeError → test FAIL
            with patch("orchestrator.status_protocol.STATUS_DIR", str(status_dir)):
                active_tasks = orchestrator_b.get_active_tasks_from_disk()

            # Assert: task phải được phát hiện dù _processes trống
            assert len(active_tasks) > 0, (
                f"Bug condition confirmed: Orchestrator mới KHÔNG thể phát hiện running task "
                f"'{task_status.task_id}' từ disk khi _processes trống. "
                f"Expected at least 1 active task, got 0."
            )

            # Verify task_id đúng
            active_task_ids = [t.task_id for t in active_tasks]
            assert task_status.task_id in active_task_ids, (
                f"Task '{task_status.task_id}' không có trong active_tasks: {active_task_ids}"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @given(
        task_status_1=running_task_status_strategy(),
        task_status_2=running_task_status_strategy(),
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_multiple_running_tasks_detected_after_session_reset(
        self, task_status_1: TaskStatus, task_status_2: TaskStatus
    ):
        """
        Property: Sau session reset, ALL running tasks phải visible từ disk,
        không chỉ task cuối cùng được start.

        Scenario: Training + Backtest cùng chạy → session reset → cả 2 phải visible
        Bug condition: UI mất reference cả 2 tasks khi _processes = {}

        # Feature: streamlit-process-state-fix, Property 1: Bug Condition
        **Validates: Requirements 1.1, 1.2**
        """
        # Đảm bảo 2 task khác nhau
        assume(task_status_1.task_id != task_status_2.task_id)

        tmp_dir = tempfile.mkdtemp()
        try:
            status_dir = Path(tmp_dir) / "status"
            status_dir.mkdir(parents=True, exist_ok=True)

            # Ghi 2 status files (simulate 2 workers đang chạy)
            for ts in [task_status_1, task_status_2]:
                status_data = {
                    "task_id": ts.task_id,
                    "task_type": ts.task_type.value,
                    "state": TaskState.RUNNING.value,
                    "progress_pct": ts.progress_pct,
                    "message": ts.message,
                    "heartbeat_ts": ts.heartbeat_ts.isoformat(),
                    "started_at": ts.started_at.isoformat(),
                    "error": None,
                    "details": ts.details,
                }
                status_file = status_dir / f"{ts.task_id}.json"
                status_file.write_text(
                    json.dumps(status_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            # Session reset → new orchestrator
            from orchestrator.task_orchestrator import TaskOrchestrator

            orchestrator_new = TaskOrchestrator()

            with patch("orchestrator.status_protocol.STATUS_DIR", str(status_dir)):
                active_tasks = orchestrator_new.get_active_tasks_from_disk()

            # Phải detect cả 2 tasks
            assert len(active_tasks) >= 2, (
                f"Bug condition: Chỉ detect {len(active_tasks)} tasks thay vì 2. "
                f"_processes trống → mất reference đến cả 2 running tasks."
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Property Tests - Bug Condition: Settings Persistence
# ---------------------------------------------------------------------------


class TestBugConditionSettingsPersistence:
    """
    Property 2: Bug Condition - Settings Persistence Across Sessions.

    Test rằng preferences persist ra disk và có thể khôi phục sau session reset.

    Trên code chưa fix: KHÔNG có module config/preferences.py với hàm
    save_preferences_to_file() / load_preferences_from_file()
    → test PHẢI FAIL (ImportError).
    """

    @given(preferences=preferences_strategy())
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_preferences_persist_to_disk_and_survive_session_reset(
        self, preferences: Dict[str, Any]
    ):
        """
        Property: Sau khi save preferences, clear session (simulate browser close),
        load_preferences_from_file() SHALL return saved preferences chính xác.

        Scenario: Save preferences → clear session → load preferences → assert values preserved
        Bug condition: action == 'load_preferences' AND trigger == 'browser_refresh'
            AND preferencesFileNotExists()

        # Feature: streamlit-process-state-fix, Property 1: Bug Condition
        **Validates: Requirements 1.4, 1.5, 1.6**
        """
        # BUG CONDITION TEST: Import hàm persistence từ config/preferences.py
        # Module này KHÔNG tồn tại trên code chưa fix → ImportError → test FAIL
        from config.preferences import save_preferences_to_file, load_preferences_from_file

        tmp_dir = tempfile.mkdtemp()
        try:
            prefs_file = Path(tmp_dir) / "preferences.json"

            # Save preferences ra disk
            with patch("config.preferences.PREFERENCES_FILE_PATH", str(prefs_file)):
                save_preferences_to_file(preferences)

            # Simulate session reset: clear mọi in-memory state
            # (Trong thực tế, st.session_state bị xóa khi browser close/refresh)

            # Load preferences từ disk file
            with patch("config.preferences.PREFERENCES_FILE_PATH", str(prefs_file)):
                loaded = load_preferences_from_file()

            # Assert: loaded phải giống hệt saved preferences
            assert loaded is not None, (
                "Bug condition confirmed: load_preferences_from_file() trả về None "
                "sau session reset — không có disk persistence cho preferences."
            )

            assert loaded["selected_symbols"] == preferences["selected_symbols"], (
                f"selected_symbols mất sau session reset: "
                f"expected {preferences['selected_symbols']}, got {loaded.get('selected_symbols')}"
            )
            assert loaded["capital"] == preferences["capital"], (
                f"capital mất sau session reset: "
                f"expected {preferences['capital']}, got {loaded.get('capital')}"
            )
            assert loaded["auto_update_time"] == preferences["auto_update_time"], (
                f"auto_update_time mất: expected {preferences['auto_update_time']}, "
                f"got {loaded.get('auto_update_time')}"
            )
            assert loaded["cycle_interval"] == preferences["cycle_interval"], (
                f"cycle_interval mất: expected {preferences['cycle_interval']}, "
                f"got {loaded.get('cycle_interval')}"
            )
            assert loaded["auto_update_enabled"] == preferences["auto_update_enabled"], (
                f"auto_update_enabled mất: expected {preferences['auto_update_enabled']}, "
                f"got {loaded.get('auto_update_enabled')}"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @given(preferences=preferences_strategy())
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_init_session_state_hydrates_from_disk_file(self, preferences: Dict[str, Any]):
        """
        Property: Khi session mới bắt đầu (new_session trigger),
        _init_session_state() SHALL đọc preferences từ disk file
        thay vì dùng hardcoded defaults.

        Bug condition: trigger == 'new_session' AND widget defaults override saved values.

        # Feature: streamlit-process-state-fix, Property 1: Bug Condition
        **Validates: Requirements 1.5, 1.6**
        """
        from config.preferences import save_preferences_to_file, load_preferences_from_file

        tmp_dir = tempfile.mkdtemp()
        try:
            prefs_file = Path(tmp_dir) / "preferences.json"

            # Pre-condition: preferences đã được save trước đó
            with patch("config.preferences.PREFERENCES_FILE_PATH", str(prefs_file)):
                save_preferences_to_file(preferences)

            # Simulate new session: session_state trống hoàn toàn
            mock_session_state: Dict[str, Any] = {}

            # _init_session_state() phải hydrate từ disk file
            with patch("ui.pages.page_settings.st") as mock_st, \
                 patch("config.preferences.PREFERENCES_FILE_PATH", str(prefs_file)):
                mock_st.session_state = mock_session_state

                from ui.pages.page_settings import _init_session_state
                _init_session_state()

            # Assert: session_state phải có giá trị từ disk file, không phải defaults
            assert mock_session_state.get("pref_selected_symbols") == preferences["selected_symbols"], (
                f"Bug condition: _init_session_state() dùng hardcoded defaults thay vì disk. "
                f"Expected symbols: {preferences['selected_symbols']}, "
                f"got: {mock_session_state.get('pref_selected_symbols')}"
            )
            assert mock_session_state.get("pref_capital") == preferences["capital"], (
                f"Bug condition: capital reset về default thay vì load từ disk. "
                f"Expected: {preferences['capital']}, got: {mock_session_state.get('pref_capital')}"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
