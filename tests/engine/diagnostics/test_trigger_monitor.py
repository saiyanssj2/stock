# -*- coding: utf-8 -*-
"""
Property-based tests cho Training Trigger Monitor.

# Feature: train-backtest-verification, Property 1: Trigger log entry completeness
# Feature: train-backtest-verification, Property 3: Trigger summary accuracy

**Validates: Requirements 1.1, 1.5**

Property 1: For any trigger event with any valid TriggerSource, logging that event
SHALL produce a log entry containing a non-empty session_id, a valid ISO-8601
timestamp, the correct trigger_source, and the caller_component.

Property 3: For any set of trigger events and any time range [start, end],
the summary SHALL report the exact count of events per trigger source
that have timestamps within [start, end].
"""

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# Đảm bảo project root nằm trong sys.path
_project_root = str(Path(__file__).parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from engine.diagnostics.models import TriggerEvent, TriggerSource
from engine.diagnostics.trigger_monitor import TrainingTriggerMonitor


# ---------------------------------------------------------------------------
# Strategies — Sinh dữ liệu ngẫu nhiên cho property tests
# ---------------------------------------------------------------------------

# Nguồn trigger ngẫu nhiên
trigger_source_strategy = st.sampled_from(list(TriggerSource))

# Caller component ngẫu nhiên
caller_component_strategy = st.sampled_from([
    "AutoLearner",
    "BackgroundTrainingManager",
    "TrainingPipeline",
    "UserInterface",
    "DataUpdateHook",
])

# Timestamp trong khoảng hợp lệ (năm 2020-2030)
timestamp_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
)


@st.composite
def trigger_event_strategy(draw, timestamp=None):
    """Sinh TriggerEvent ngẫu nhiên hợp lệ.

    Args:
        timestamp: Nếu cung cấp, dùng timestamp này thay vì sinh ngẫu nhiên.
    """
    ts = timestamp if timestamp is not None else draw(timestamp_strategy)
    source = draw(trigger_source_strategy)
    caller = draw(caller_component_strategy)
    session_id = draw(st.uuids().map(str))

    return TriggerEvent(
        session_id=session_id,
        timestamp=ts.isoformat(),
        trigger_source=source,
        caller_component=caller,
        metadata={},
    )


@st.composite
def trigger_events_with_timestamps_strategy(draw):
    """Sinh danh sách TriggerEvents với timestamps ngẫu nhiên.

    Trả về tuple (events, timestamps_as_datetime) để dễ verify.
    """
    num_events = draw(st.integers(min_value=0, max_value=30))
    events = []
    timestamps = []

    for _ in range(num_events):
        ts = draw(timestamp_strategy)
        source = draw(trigger_source_strategy)
        caller = draw(caller_component_strategy)
        session_id = draw(st.uuids().map(str))

        event = TriggerEvent(
            session_id=session_id,
            timestamp=ts.isoformat(),
            trigger_source=source,
            caller_component=caller,
            metadata={},
        )
        events.append(event)
        timestamps.append(ts)

    return events, timestamps


@st.composite
def time_range_strategy(draw):
    """Sinh cặp (start_time, end_time) hợp lệ với start <= end."""
    start = draw(timestamp_strategy)
    # Offset từ 0 đến 365 ngày
    delta_hours = draw(st.integers(min_value=0, max_value=8760))
    end = start + timedelta(hours=delta_hours)
    # Giới hạn trong phạm vi hợp lệ
    if end > datetime(2030, 12, 31):
        end = datetime(2030, 12, 31)
    if start > end:
        start = end
    return start, end


# ---------------------------------------------------------------------------
# Helper: Tạo/xóa temp directory cho test isolation
# ---------------------------------------------------------------------------

def _make_temp_dir() -> Path:
    """Tạo temp directory mới cho mỗi test case."""
    return Path(tempfile.mkdtemp(prefix="pbt_trigger_summary_"))


def _cleanup_temp_dir(path: Path) -> None:
    """Xóa temp directory sau test."""
    try:
        shutil.rmtree(str(path), ignore_errors=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Property 1: Trigger log entry completeness
# ---------------------------------------------------------------------------


class TestTriggerLogEntryCompleteness:
    """
    Property 1: Trigger log entry completeness.

    For any trigger event with any valid TriggerSource, logging that event SHALL
    produce a log entry containing a non-empty session_id, a valid ISO-8601
    timestamp, the correct trigger_source, and the caller_component.

    Feature: train-backtest-verification, Property 1: Trigger log entry completeness

    **Validates: Requirements 1.1**
    """

    @given(
        source=trigger_source_strategy,
        caller=caller_component_strategy,
        session_id=st.uuids().map(str),
        ts=timestamp_strategy,
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_logged_entry_contains_all_required_fields(self, source, caller, session_id, ts):
        """
        Property: Bất kỳ TriggerEvent hợp lệ nào được log đều phải tạo ra entry
        trong storage chứa đầy đủ: session_id (non-empty), timestamp (ISO-8601 hợp lệ),
        trigger_source (khớp input), và caller_component (khớp input).

        **Validates: Requirements 1.1**
        """
        tmp_dir = _make_temp_dir()
        try:
            log_path = str(tmp_dir / "trigger_log.jsonl")
            monitor = TrainingTriggerMonitor(log_path=log_path)

            # Tạo event với dữ liệu sinh ngẫu nhiên
            event = TriggerEvent(
                session_id=session_id,
                timestamp=ts.isoformat(),
                trigger_source=source,
                caller_component=caller,
                metadata={},
            )

            # Ghi event vào storage
            result = monitor.log_trigger(event)
            assert result is True, "log_trigger phải trả về True khi ghi thành công"

            # Đọc lại entry từ storage (JSONL file)
            log_file = Path(log_path)
            assert log_file.exists(), "Log file phải tồn tại sau khi ghi"

            with open(log_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            assert len(lines) >= 1, "Log file phải có ít nhất 1 entry"

            # Parse entry cuối cùng (entry vừa ghi)
            entry = json.loads(lines[-1])

            # Verify session_id: non-empty string
            assert "session_id" in entry, "Entry thiếu trường session_id"
            assert isinstance(entry["session_id"], str), "session_id phải là string"
            assert len(entry["session_id"]) > 0, "session_id phải non-empty"
            assert entry["session_id"] == session_id, (
                f"session_id không khớp: expected '{session_id}', got '{entry['session_id']}'"
            )

            # Verify timestamp: valid ISO-8601
            assert "timestamp" in entry, "Entry thiếu trường timestamp"
            assert isinstance(entry["timestamp"], str), "timestamp phải là string"
            # Parse ISO-8601 — nếu không parse được sẽ raise ValueError
            parsed_ts = datetime.fromisoformat(entry["timestamp"])
            assert parsed_ts == ts, (
                f"timestamp không khớp: expected '{ts.isoformat()}', got '{entry['timestamp']}'"
            )

            # Verify trigger_source: khớp với input
            assert "trigger_source" in entry, "Entry thiếu trường trigger_source"
            assert entry["trigger_source"] == source.value, (
                f"trigger_source không khớp: expected '{source.value}', "
                f"got '{entry['trigger_source']}'"
            )

            # Verify caller_component: khớp với input
            assert "caller_component" in entry, "Entry thiếu trường caller_component"
            assert isinstance(entry["caller_component"], str), "caller_component phải là string"
            assert entry["caller_component"] == caller, (
                f"caller_component không khớp: expected '{caller}', "
                f"got '{entry['caller_component']}'"
            )

        finally:
            _cleanup_temp_dir(tmp_dir)

    @given(
        source=trigger_source_strategy,
        caller=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
            min_size=1,
            max_size=50,
        ),
        session_id=st.uuids().map(str),
        ts=timestamp_strategy,
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_logged_entry_preserves_arbitrary_caller_component(self, source, caller, session_id, ts):
        """
        Property: Với bất kỳ caller_component string nào (non-empty, arbitrary text),
        entry được log phải giữ nguyên giá trị caller_component.

        **Validates: Requirements 1.1**
        """
        tmp_dir = _make_temp_dir()
        try:
            log_path = str(tmp_dir / "trigger_log.jsonl")
            monitor = TrainingTriggerMonitor(log_path=log_path)

            event = TriggerEvent(
                session_id=session_id,
                timestamp=ts.isoformat(),
                trigger_source=source,
                caller_component=caller,
                metadata={},
            )

            result = monitor.log_trigger(event)
            assert result is True

            # Đọc lại entry
            with open(log_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            entry = json.loads(lines[-1])

            # Verify caller_component được bảo toàn nguyên vẹn
            assert entry["caller_component"] == caller, (
                f"caller_component bị thay đổi: expected '{caller}', "
                f"got '{entry['caller_component']}'"
            )

        finally:
            _cleanup_temp_dir(tmp_dir)

    @given(source=trigger_source_strategy)
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_all_trigger_sources_produce_valid_entries(self, source):
        """
        Property: Mọi giá trị TriggerSource hợp lệ đều phải tạo ra entry hợp lệ
        với trigger_source field chứa đúng string value tương ứng.

        **Validates: Requirements 1.1**
        """
        tmp_dir = _make_temp_dir()
        try:
            log_path = str(tmp_dir / "trigger_log.jsonl")
            monitor = TrainingTriggerMonitor(log_path=log_path)

            event = TriggerEvent(
                session_id="test-session-001",
                timestamp=datetime(2024, 6, 15, 10, 30, 0).isoformat(),
                trigger_source=source,
                caller_component="TestComponent",
                metadata={},
            )

            result = monitor.log_trigger(event)
            assert result is True

            # Đọc lại entry
            with open(log_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            entry = json.loads(lines[-1])

            # Verify trigger_source là một trong các giá trị hợp lệ
            valid_sources = [s.value for s in TriggerSource]
            assert entry["trigger_source"] in valid_sources, (
                f"trigger_source '{entry['trigger_source']}' không nằm trong "
                f"danh sách hợp lệ: {valid_sources}"
            )
            # Phải khớp chính xác với input
            assert entry["trigger_source"] == source.value

        finally:
            _cleanup_temp_dir(tmp_dir)


# ---------------------------------------------------------------------------
# Property 3: Trigger summary accuracy
# ---------------------------------------------------------------------------


class TestTriggerSummaryAccuracy:
    """
    Property 3: Trigger summary accuracy.

    For any set of trigger events and any time range [start, end], the summary
    SHALL report the exact count of events per trigger source that have
    timestamps within [start, end].

    **Validates: Requirements 1.5**
    """

    @given(
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_summary_counts_match_manual_counting(self, data):
        """
        Property: Với bất kỳ tập trigger events và khoảng thời gian [start, end],
        get_summary() phải trả về số lượng events chính xác cho mỗi trigger source
        bằng cách đếm thủ công các events có timestamp trong [start, end].

        **Validates: Requirements 1.5**
        """
        # Sinh danh sách events với timestamps ngẫu nhiên
        events_and_timestamps = data.draw(trigger_events_with_timestamps_strategy())
        events, timestamps = events_and_timestamps

        # Sinh time range ngẫu nhiên
        start_time, end_time = data.draw(time_range_strategy())

        tmp_dir = _make_temp_dir()
        try:
            log_path = str(tmp_dir / "trigger_log.jsonl")
            monitor = TrainingTriggerMonitor(log_path=log_path)

            # Ghi tất cả events vào monitor
            for event in events:
                monitor.log_trigger(event)

            # Lấy summary từ monitor
            summary = monitor.get_summary(start_time, end_time)

            # Đếm thủ công: đếm events theo source trong khoảng [start, end]
            expected_counts: Dict[str, int] = {source.value: 0 for source in TriggerSource}
            expected_total = 0

            for event, ts in zip(events, timestamps):
                if start_time <= ts <= end_time:
                    source_value = event.trigger_source.value
                    expected_counts[source_value] += 1
                    expected_total += 1

            # Assert: summary phải khớp chính xác với đếm thủ công
            assert summary["total_triggers"] == expected_total, (
                f"Total triggers sai: expected {expected_total}, got {summary['total_triggers']}"
            )

            for source in TriggerSource:
                assert summary["by_source"][source.value] == expected_counts[source.value], (
                    f"Count cho {source.value} sai: "
                    f"expected {expected_counts[source.value]}, "
                    f"got {summary['by_source'][source.value]}"
                )

        finally:
            _cleanup_temp_dir(tmp_dir)

    @given(
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_summary_excludes_events_outside_time_range(self, data):
        """
        Property: Events có timestamp ngoài [start, end] KHÔNG được tính vào summary.
        Tạo events chắc chắn ngoài range rồi verify count = 0 cho các events đó.

        **Validates: Requirements 1.5**
        """
        # Sinh time range cố định
        start_time, end_time = data.draw(time_range_strategy())
        assume(start_time < end_time)

        # Sinh events TRƯỚC start_time
        num_before = data.draw(st.integers(min_value=1, max_value=10))
        events_before = []
        for _ in range(num_before):
            # Timestamp trước start_time (ít nhất 1 giờ trước)
            offset_hours = data.draw(st.integers(min_value=1, max_value=8760))
            ts_before = start_time - timedelta(hours=offset_hours)
            if ts_before < datetime(2020, 1, 1):
                ts_before = datetime(2020, 1, 1)
            assume(ts_before < start_time)

            source = data.draw(trigger_source_strategy)
            event = TriggerEvent(
                session_id=data.draw(st.uuids().map(str)),
                timestamp=ts_before.isoformat(),
                trigger_source=source,
                caller_component="TestComponent",
                metadata={},
            )
            events_before.append(event)

        # Sinh events SAU end_time
        num_after = data.draw(st.integers(min_value=1, max_value=10))
        events_after = []
        for _ in range(num_after):
            # Timestamp sau end_time (ít nhất 1 giờ sau)
            offset_hours = data.draw(st.integers(min_value=1, max_value=8760))
            ts_after = end_time + timedelta(hours=offset_hours)
            if ts_after > datetime(2030, 12, 31):
                ts_after = datetime(2030, 12, 31)
            assume(ts_after > end_time)

            source = data.draw(trigger_source_strategy)
            event = TriggerEvent(
                session_id=data.draw(st.uuids().map(str)),
                timestamp=ts_after.isoformat(),
                trigger_source=source,
                caller_component="TestComponent",
                metadata={},
            )
            events_after.append(event)

        tmp_dir = _make_temp_dir()
        try:
            log_path = str(tmp_dir / "trigger_log.jsonl")
            monitor = TrainingTriggerMonitor(log_path=log_path)

            # Ghi tất cả events ngoài range
            for event in events_before + events_after:
                monitor.log_trigger(event)

            # Lấy summary
            summary = monitor.get_summary(start_time, end_time)

            # Assert: không event nào được đếm (tất cả ngoài range)
            assert summary["total_triggers"] == 0, (
                f"Expected 0 triggers in range, got {summary['total_triggers']}"
            )

            for source in TriggerSource:
                assert summary["by_source"][source.value] == 0, (
                    f"Expected 0 for {source.value}, got {summary['by_source'][source.value]}"
                )

        finally:
            _cleanup_temp_dir(tmp_dir)

    @given(
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_summary_time_range_metadata_correct(self, data):
        """
        Property: Summary trả về start_time và end_time khớp với input parameters.

        **Validates: Requirements 1.5**
        """
        start_time, end_time = data.draw(time_range_strategy())

        tmp_dir = _make_temp_dir()
        try:
            log_path = str(tmp_dir / "trigger_log.jsonl")
            monitor = TrainingTriggerMonitor(log_path=log_path)

            # Lấy summary (có thể không có events)
            summary = monitor.get_summary(start_time, end_time)

            # Assert: metadata thời gian phải chính xác
            assert summary["start_time"] == start_time.isoformat(), (
                f"start_time sai: expected {start_time.isoformat()}, got {summary['start_time']}"
            )
            assert summary["end_time"] == end_time.isoformat(), (
                f"end_time sai: expected {end_time.isoformat()}, got {summary['end_time']}"
            )

            # Assert: tất cả trigger sources phải có trong by_source
            for source in TriggerSource:
                assert source.value in summary["by_source"], (
                    f"Trigger source {source.value} không có trong summary by_source"
                )

        finally:
            _cleanup_temp_dir(tmp_dir)


# ---------------------------------------------------------------------------
# Property 2: Trigger storage FIFO invariant
# ---------------------------------------------------------------------------


class TestTriggerStorageFIFOInvariant:
    """
    Property 2: Trigger storage FIFO invariant.

    For any sequence of N trigger events logged to storage where N > 10,000,
    the storage SHALL contain exactly 10,000 entries AND the oldest entries
    (by timestamp) SHALL have been removed first (FIFO ordering preserved).

    Feature: train-backtest-verification, Property 2: Trigger storage FIFO invariant

    **Validates: Requirements 1.4**

    Sử dụng MAX_ENTRIES scaled-down (100) để test hiệu năng tốt hơn,
    nhưng property FIFO vẫn đúng cho mọi giá trị MAX_ENTRIES.
    """

    # Dùng giá trị nhỏ hơn cho test performance
    TEST_MAX_ENTRIES = 100

    @given(
        num_extra=st.integers(min_value=1, max_value=100),
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_fifo_exactly_max_entries_after_overflow(self, num_extra, data):
        """
        Property: Sau khi log N > MAX_ENTRIES events, storage chứa đúng MAX_ENTRIES entries.

        **Validates: Requirements 1.4**
        """
        max_entries = self.TEST_MAX_ENTRIES
        total_events = max_entries + num_extra

        # Tạo temp directory để test isolation
        tmp_dir = _make_temp_dir()
        try:
            log_path = str(tmp_dir / "trigger_log.jsonl")
            monitor = TrainingTriggerMonitor(log_path=log_path)
            monitor.MAX_ENTRIES = max_entries

            # Sinh và log events với timestamp tăng dần
            base_time = datetime(2024, 1, 1)
            for i in range(total_events):
                event_time = base_time + timedelta(seconds=i)
                source = data.draw(trigger_source_strategy)
                event = TriggerEvent(
                    session_id=f"session-{i:06d}",
                    timestamp=event_time.isoformat(),
                    trigger_source=source,
                    caller_component="TestComponent",
                    metadata={"index": i},
                )
                monitor.log_trigger(event)

            # Đọc file và đếm entries
            with open(log_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            # Property: storage chứa đúng MAX_ENTRIES entries
            assert len(lines) == max_entries, (
                f"Expected exactly {max_entries} entries after logging "
                f"{total_events} events, got {len(lines)}"
            )

        finally:
            _cleanup_temp_dir(tmp_dir)

    @given(
        num_extra=st.integers(min_value=1, max_value=100),
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_fifo_oldest_entries_removed_first(self, num_extra, data):
        """
        Property: Oldest entries (theo timestamp) bị xóa trước — chỉ giữ lại
        MAX_ENTRIES entries mới nhất. Entry còn lại phải tương ứng với
        entries cuối cùng được log (FIFO ordering).

        **Validates: Requirements 1.4**
        """
        max_entries = self.TEST_MAX_ENTRIES
        total_events = max_entries + num_extra

        tmp_dir = _make_temp_dir()
        try:
            log_path = str(tmp_dir / "trigger_log.jsonl")
            monitor = TrainingTriggerMonitor(log_path=log_path)
            monitor.MAX_ENTRIES = max_entries

            # Sinh events với timestamp tăng dần
            base_time = datetime(2024, 1, 1)
            for i in range(total_events):
                event_time = base_time + timedelta(seconds=i)
                source = data.draw(trigger_source_strategy)
                event = TriggerEvent(
                    session_id=f"session-{i:06d}",
                    timestamp=event_time.isoformat(),
                    trigger_source=source,
                    caller_component="TestComponent",
                    metadata={"index": i},
                )
                monitor.log_trigger(event)

            # Đọc file entries
            with open(log_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            stored_entries = [json.loads(line) for line in lines]

            # Property: entries còn lại phải là MAX_ENTRIES entries mới nhất
            # (tức là entries từ index (total_events - max_entries) đến total_events-1)
            expected_first_index = total_events - max_entries
            for idx, entry in enumerate(stored_entries):
                expected_index = expected_first_index + idx
                assert entry["metadata"]["index"] == expected_index, (
                    f"Entry tại vị trí {idx} có index {entry['metadata']['index']}, "
                    f"expected {expected_index}. Oldest entries phải bị xóa trước (FIFO)."
                )

        finally:
            _cleanup_temp_dir(tmp_dir)

    @given(
        num_extra=st.integers(min_value=1, max_value=100),
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_fifo_chronological_order_preserved(self, num_extra, data):
        """
        Property: Entries còn lại trong storage phải giữ thứ tự thời gian tăng dần
        (chronological order preserved sau rotation).

        **Validates: Requirements 1.4**
        """
        max_entries = self.TEST_MAX_ENTRIES
        total_events = max_entries + num_extra

        tmp_dir = _make_temp_dir()
        try:
            log_path = str(tmp_dir / "trigger_log.jsonl")
            monitor = TrainingTriggerMonitor(log_path=log_path)
            monitor.MAX_ENTRIES = max_entries

            # Sinh events với timestamp tăng dần
            base_time = datetime(2024, 1, 1)
            for i in range(total_events):
                event_time = base_time + timedelta(seconds=i)
                source = data.draw(trigger_source_strategy)
                event = TriggerEvent(
                    session_id=f"session-{i:06d}",
                    timestamp=event_time.isoformat(),
                    trigger_source=source,
                    caller_component="TestComponent",
                    metadata={"index": i},
                )
                monitor.log_trigger(event)

            # Đọc file entries
            with open(log_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            stored_entries = [json.loads(line) for line in lines]

            # Property: timestamps phải tăng dần (chronological order preserved)
            timestamps = [
                datetime.fromisoformat(entry["timestamp"])
                for entry in stored_entries
            ]
            for i in range(1, len(timestamps)):
                assert timestamps[i] > timestamps[i - 1], (
                    f"Chronological order bị vi phạm tại vị trí {i}: "
                    f"{timestamps[i-1].isoformat()} phải trước {timestamps[i].isoformat()}"
                )

        finally:
            _cleanup_temp_dir(tmp_dir)
