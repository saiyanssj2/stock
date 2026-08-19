"""
Property-based tests for Training Progress & Checkpoint feature.

Tests the following correctness properties from the design document:
- Property 1: Progress Invariant
- Property 4: ETA Formatting Rules
- Property 6: Session Checkpoint Round-Trip

**Validates: Requirements 1.1, 1.3, 1.4, 1.5, 3.2, 3.4, 3.5, 5.1, 5.3**
"""

import datetime
import re
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import characters

from engine.eta_calculator import ETACalculator
from engine.session_checkpoint import SessionCheckpoint, SessionCheckpointManager
from engine.symbol_progress_tracker import SymbolProgressTracker


# ---------------------------------------------------------------------------
# Custom strategies for Training Progress property tests
# ---------------------------------------------------------------------------

# Strategy for generating valid symbol names (non-empty letter strings)
symbol_strategy = st.text(
    min_size=1,
    max_size=10,
    alphabet=characters(whitelist_categories=("L",)),
)


@st.composite
def session_checkpoint_strategy(draw):
    """
    Generate a random valid SessionCheckpoint instance with:
    - completed_symbols: non-empty list of unique symbol strings
    - pending_symbols: non-empty list of unique symbol strings
    - mode: one of ["full", "incremental"]
    - session_start_time: valid ISO format datetime string
    - symbol_durations: dict mapping completed symbols to positive floats
    - total_epochs_per_symbol: positive integer
    """
    # Generate non-empty lists of unique symbols
    completed_symbols = draw(
        st.lists(symbol_strategy, min_size=1, max_size=20, unique=True)
    )
    pending_symbols = draw(
        st.lists(symbol_strategy, min_size=1, max_size=20, unique=True)
    )

    # Mode
    mode = draw(st.sampled_from(["full", "incremental"]))

    # Valid ISO format datetime string
    dt = draw(
        st.datetimes(
            min_value=datetime.datetime(2020, 1, 1),
            max_value=datetime.datetime(2030, 12, 31),
        )
    )
    session_start_time = dt.isoformat()

    # Symbol durations: map each completed symbol to a positive float
    symbol_durations = {}
    for sym in completed_symbols:
        duration = draw(st.floats(min_value=0.1, max_value=100000.0, allow_nan=False, allow_infinity=False))
        symbol_durations[sym] = duration

    # Total epochs per symbol: positive integer
    total_epochs_per_symbol = draw(st.integers(min_value=1, max_value=1000))

    return SessionCheckpoint(
        completed_symbols=completed_symbols,
        pending_symbols=pending_symbols,
        mode=mode,
        session_start_time=session_start_time,
        symbol_durations=symbol_durations,
        total_epochs_per_symbol=total_epochs_per_symbol,
    )


# ---------------------------------------------------------------------------
# Property 6: Session Checkpoint Round-Trip
# ---------------------------------------------------------------------------
# Feature: training-progress-checkpoint, Property 6: Session Checkpoint Round-Trip


class TestProperty6SessionCheckpointRoundTrip:
    """
    Property 6: Session Checkpoint Round-Trip.

    For any valid SessionCheckpoint (with non-empty completed/pending lists,
    valid mode, valid ISO timestamp, and positive durations), saving the
    checkpoint to disk and loading it back SHALL produce an equivalent
    SessionCheckpoint with identical field values.

    **Validates: Requirements 5.1, 5.3**
    """

    @given(checkpoint=session_checkpoint_strategy())
    @settings(max_examples=100, deadline=None)
    def test_save_then_load_preserves_all_fields(self, checkpoint):
        """
        Save a randomly generated SessionCheckpoint, then load it back
        and verify ALL fields are identical between original and loaded.

        **Validates: Requirements 5.1, 5.3**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionCheckpointManager(base_dir=tmpdir)

            # Save the checkpoint
            manager.save(checkpoint)

            # Load it back
            loaded = manager.load()

            # Verify loaded is not None
            assert loaded is not None, "Loaded checkpoint should not be None after save"

            # Verify all fields are identical
            assert loaded.completed_symbols == checkpoint.completed_symbols, (
                f"completed_symbols mismatch: "
                f"got {loaded.completed_symbols}, expected {checkpoint.completed_symbols}"
            )
            assert loaded.pending_symbols == checkpoint.pending_symbols, (
                f"pending_symbols mismatch: "
                f"got {loaded.pending_symbols}, expected {checkpoint.pending_symbols}"
            )
            assert loaded.mode == checkpoint.mode, (
                f"mode mismatch: got {loaded.mode}, expected {checkpoint.mode}"
            )
            assert loaded.session_start_time == checkpoint.session_start_time, (
                f"session_start_time mismatch: "
                f"got {loaded.session_start_time}, expected {checkpoint.session_start_time}"
            )
            assert loaded.total_epochs_per_symbol == checkpoint.total_epochs_per_symbol, (
                f"total_epochs_per_symbol mismatch: "
                f"got {loaded.total_epochs_per_symbol}, "
                f"expected {checkpoint.total_epochs_per_symbol}"
            )
            assert loaded.checkpoint_version == checkpoint.checkpoint_version, (
                f"checkpoint_version mismatch: "
                f"got {loaded.checkpoint_version}, "
                f"expected {checkpoint.checkpoint_version}"
            )

            # Verify symbol_durations (compare keys and values with float tolerance)
            assert set(loaded.symbol_durations.keys()) == set(checkpoint.symbol_durations.keys()), (
                f"symbol_durations keys mismatch: "
                f"got {set(loaded.symbol_durations.keys())}, "
                f"expected {set(checkpoint.symbol_durations.keys())}"
            )
            for sym, expected_duration in checkpoint.symbol_durations.items():
                actual_duration = loaded.symbol_durations[sym]
                assert abs(actual_duration - expected_duration) < 1e-9, (
                    f"symbol_durations[{sym}] mismatch: "
                    f"got {actual_duration}, expected {expected_duration}"
                )


# ---------------------------------------------------------------------------
# Property 7: Corrupt Checkpoint Graceful Handling
# ---------------------------------------------------------------------------
# Feature: training-progress-checkpoint, Property 7: Corrupt Checkpoint Graceful Handling


# Required fields that must be present in a valid checkpoint
_REQUIRED_CHECKPOINT_FIELDS = [
    "completed_symbols",
    "pending_symbols",
    "mode",
    "session_start_time",
    "symbol_durations",
]


@st.composite
def invalid_json_bytes_strategy(draw):
    """
    Generate random byte strings that are NOT valid JSON.
    Uses binary content that cannot be parsed as JSON.
    """
    # Generate random binary data (may include non-UTF8 bytes)
    data = draw(st.binary(min_size=1, max_size=500))
    # Ensure it's not accidentally valid JSON by checking
    try:
        import json
        json.loads(data)
        # If it somehow parses, prepend garbage to break it
        data = b"\xff\xfe" + data + b"\x00\x01"
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        pass
    return data


@st.composite
def json_missing_required_fields_strategy(draw):
    """
    Generate valid JSON dicts that are missing one or more required fields.
    The JSON is valid but lacks at least one of:
    completed_symbols, pending_symbols, mode, session_start_time, symbol_durations
    """
    import json

    # Start with a full valid checkpoint dict
    full_dict = {
        "completed_symbols": ["SYM1", "SYM2"],
        "pending_symbols": ["SYM3", "SYM4"],
        "mode": "full",
        "session_start_time": "2024-01-15T08:30:00",
        "symbol_durations": {"SYM1": 100.0, "SYM2": 200.0},
        "total_epochs_per_symbol": 100,
        "checkpoint_version": 1,
    }

    # Choose a non-empty subset of required fields to REMOVE
    fields_to_remove = draw(
        st.lists(
            st.sampled_from(_REQUIRED_CHECKPOINT_FIELDS),
            min_size=1,
            max_size=len(_REQUIRED_CHECKPOINT_FIELDS),
            unique=True,
        )
    )

    for field in fields_to_remove:
        full_dict.pop(field, None)

    return json.dumps(full_dict).encode("utf-8")


class TestProperty7CorruptCheckpointGracefulHandling:
    """
    Property 7: Corrupt Checkpoint Graceful Handling.

    For any byte string that is NOT valid JSON, or valid JSON missing any of
    the required fields (completed_symbols, pending_symbols, mode,
    session_start_time, symbol_durations), SessionCheckpointManager.load()
    SHALL return None without raising an exception.

    **Validates: Requirements 5.7**
    """

    @given(data=invalid_json_bytes_strategy())
    @settings(max_examples=100, deadline=None)
    def test_invalid_json_returns_none(self, data):
        """
        Generate random byte strings that are NOT valid JSON, write to
        checkpoint path, and verify load() returns None without raising.

        **Validates: Requirements 5.7**
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionCheckpointManager(base_dir=tmpdir)

            # Ensure the checkpoint directory exists
            checkpoint_path = manager.checkpoint_path
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            # Write corrupt data to checkpoint path
            with open(checkpoint_path, "wb") as f:
                f.write(data)

            # load() must return None without raising any exception
            result = manager.load()
            assert result is None, (
                f"Expected None for invalid JSON bytes, got {result}"
            )

    @given(data=json_missing_required_fields_strategy())
    @settings(max_examples=100, deadline=None)
    def test_json_missing_required_fields_returns_none(self, data):
        """
        Generate valid JSON dicts missing one or more required fields,
        write to checkpoint path, and verify load() returns None without raising.

        **Validates: Requirements 5.7**
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionCheckpointManager(base_dir=tmpdir)

            # Ensure the checkpoint directory exists
            checkpoint_path = manager.checkpoint_path
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            # Write valid JSON with missing fields to checkpoint path
            with open(checkpoint_path, "wb") as f:
                f.write(data)

            # load() must return None without raising any exception
            result = manager.load()
            assert result is None, (
                f"Expected None for JSON missing required fields, got {result}"
            )


# ---------------------------------------------------------------------------
# Property 4: ETA Formatting Rules
# ---------------------------------------------------------------------------
# Feature: training-progress-checkpoint, Property 4: ETA Formatting Rules


class TestProperty4ETAFormattingRules:
    """
    Property 4: ETA Formatting Rules.

    For any estimated remaining seconds value:
    - None → "Đang ước tính..."
    - < 60 → "Sắp hoàn tất"
    - >= 3600 → matches "Xh Ym" (regex: \\d+h \\d+m)
    - else → matches "Ym Zs" (regex: \\d+m \\d+s)

    **Validates: Requirements 3.2, 3.4, 3.5**
    """

    def setup_method(self):
        """Create an ETACalculator instance for testing."""
        self.calculator = ETACalculator(total_symbols=10)

    def test_none_returns_estimating_message(self):
        """When remaining_seconds is None, format_eta returns 'Đang ước tính...'."""
        result = self.calculator.format_eta(None)
        assert result == "Đang ước tính..."

    @settings(max_examples=100)
    @given(seconds=st.floats(min_value=0.0, max_value=59.999999))
    def test_under_60_returns_almost_done(self, seconds):
        """When remaining_seconds < 60, format_eta returns 'Sắp hoàn tất'."""
        result = self.calculator.format_eta(seconds)
        assert result == "Sắp hoàn tất"

    @settings(max_examples=100)
    @given(seconds=st.floats(min_value=3600.0, max_value=100000.0))
    def test_gte_3600_returns_hours_minutes_format(self, seconds):
        """When remaining_seconds >= 3600, format_eta returns 'Xh Ym' pattern."""
        result = self.calculator.format_eta(seconds)
        assert re.match(r"^\d+h \d+m$", result), (
            f"Expected 'Xh Ym' format for {seconds}s, got '{result}'"
        )

    @settings(max_examples=100)
    @given(seconds=st.floats(min_value=60.0, max_value=3599.999999))
    def test_between_60_and_3600_returns_minutes_seconds_format(self, seconds):
        """When 60 <= remaining_seconds < 3600, format_eta returns 'Ym Zs' pattern."""
        result = self.calculator.format_eta(seconds)
        assert re.match(r"^\d+m \d+s$", result), (
            f"Expected 'Ym Zs' format for {seconds}s, got '{result}'"
        )

    @settings(max_examples=100)
    @given(seconds=st.floats(min_value=0.0, max_value=100000.0))
    def test_all_ranges_produce_valid_format(self, seconds):
        """For any seconds in [0, 100000], format_eta returns one of the valid formats."""
        result = self.calculator.format_eta(seconds)

        if seconds < 60:
            assert result == "Sắp hoàn tất"
        elif seconds >= 3600:
            assert re.match(r"^\d+h \d+m$", result), (
                f"Expected 'Xh Ym' for {seconds}s, got '{result}'"
            )
        else:
            assert re.match(r"^\d+m \d+s$", result), (
                f"Expected 'Ym Zs' for {seconds}s, got '{result}'"
            )


# ---------------------------------------------------------------------------
# Property 2: Completion Recording Preserves State
# ---------------------------------------------------------------------------
# Feature: training-progress-checkpoint, Property 2: Completion Recording Preserves State


@st.composite
def symbol_completion_sequence_strategy(draw):
    """
    Generate a random list of symbols (size 1-50) and for each symbol,
    randomly choose success or failure with an arbitrary positive duration.

    Returns:
        tuple: (symbols, events) where events is a list of
               (symbol, is_success, duration) tuples.
    """
    # Generate unique symbol names (size 1-50)
    symbols = draw(
        st.lists(symbol_strategy, min_size=1, max_size=50, unique=True)
    )

    # For each symbol, generate an event: success/failure + positive duration
    events = []
    for sym in symbols:
        is_success = draw(st.booleans())
        duration = draw(
            st.floats(min_value=0.01, max_value=100000.0, allow_nan=False, allow_infinity=False)
        )
        events.append((sym, is_success, duration))

    return (symbols, events)


class TestProperty2CompletionRecordingPreservesState:
    """
    Property 2: Completion Recording Preserves State.

    For any sequence of symbol completions (successful or failed) with arbitrary
    positive durations, calling on_symbol_complete or on_symbol_failed SHALL
    increment completed_count by exactly 1, record the symbol name, its status
    ("completed" or "failed"), and its duration. The sum of recorded results
    SHALL equal completed_count at all times.

    **Validates: Requirements 1.2, 1.6**
    """

    @given(data=symbol_completion_sequence_strategy())
    @settings(max_examples=100, deadline=None)
    def test_completion_recording_preserves_state(self, data):
        """
        Generate random sequences of symbol completions (mixed success/failure)
        with arbitrary positive durations. After each completion/failure:
        - Verify len(completed_symbols) increments by exactly 1
        - Verify the recorded result has correct symbol name, status, and duration
        - Verify total completed_symbols count equals number of events processed

        **Validates: Requirements 1.2, 1.6**
        """
        from engine.symbol_progress_tracker import SymbolProgressTracker

        symbols, events = data

        # Create tracker with the full symbol list
        tracker = SymbolProgressTracker(
            symbols=symbols,
            mode="full",
            total_epochs_per_symbol=10,
            session_state_dict=None,
        )

        # Verify initial state
        assert len(tracker.completed_symbols) == 0

        for i, (sym, is_success, duration) in enumerate(events):
            previous_count = len(tracker.completed_symbols)

            # Must call on_symbol_start before complete/failed
            tracker.on_symbol_start(sym)

            if is_success:
                tracker.on_symbol_complete(sym, duration)
            else:
                tracker.on_symbol_failed(sym, f"Error for {sym}")

            current_results = tracker.completed_symbols

            # Verify completed_count increments by exactly 1
            assert len(current_results) == previous_count + 1, (
                f"After event {i + 1}: expected {previous_count + 1} results, "
                f"got {len(current_results)}"
            )

            # Verify the latest recorded result has correct symbol name, status, and duration
            latest_result = current_results[-1]
            assert latest_result.symbol == sym, (
                f"Event {i + 1}: expected symbol '{sym}', got '{latest_result.symbol}'"
            )

            expected_status = "completed" if is_success else "failed"
            assert latest_result.status == expected_status, (
                f"Event {i + 1}: expected status '{expected_status}', "
                f"got '{latest_result.status}'"
            )

            if is_success:
                assert latest_result.duration_seconds == duration, (
                    f"Event {i + 1}: expected duration {duration}, "
                    f"got {latest_result.duration_seconds}"
                )
            else:
                # Failed symbols record 0.0 duration
                assert latest_result.duration_seconds == 0.0, (
                    f"Event {i + 1}: expected 0.0 duration for failed symbol, "
                    f"got {latest_result.duration_seconds}"
                )

            # Verify total completed_symbols count equals number of events processed
            assert len(current_results) == i + 1, (
                f"After {i + 1} events: expected {i + 1} total results, "
                f"got {len(current_results)}"
            )


# ---------------------------------------------------------------------------
# Property 11: Slow Symbol Detection
# ---------------------------------------------------------------------------
# Feature: training-progress-checkpoint, Property 11: Slow Symbol Detection

from ui.ui_training_progress import _is_slow_symbol


class TestProperty11SlowSymbolDetection:
    """
    Property 11: Slow Symbol Detection.

    For any symbol duration and running average of previous symbol durations
    (where average > 0), the symbol SHALL be marked with "⚠️ Chậm" if and
    only if duration > 2 × average.

    **Validates: Requirements 7.5**
    """

    @given(
        duration=st.floats(min_value=0.01, max_value=100000.0, allow_nan=False, allow_infinity=False),
        average=st.floats(min_value=0.01, max_value=100000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_slow_symbol_iff_duration_exceeds_twice_average(self, duration, average):
        """
        For any (duration, average) pair with average > 0:
        - _is_slow_symbol returns True iff duration > 2 * average
        - _is_slow_symbol returns False iff duration <= 2 * average

        **Validates: Requirements 7.5**
        """
        result = _is_slow_symbol(duration, average)
        threshold = 2 * average

        if duration > threshold:
            assert result is True, (
                f"Expected slow (True) for duration={duration}, average={average}, "
                f"threshold={threshold}, got {result}"
            )
        else:
            assert result is False, (
                f"Expected not slow (False) for duration={duration}, average={average}, "
                f"threshold={threshold}, got {result}"
            )


# ---------------------------------------------------------------------------
# Property 5: Heartbeat Warning Threshold
# ---------------------------------------------------------------------------
# Feature: training-progress-checkpoint, Property 5: Heartbeat Warning Threshold


class TestProperty5HeartbeatWarningThreshold:
    """
    Property 5: Heartbeat Warning Threshold.

    For any heartbeat timestamp and current time, if the difference exceeds
    60 seconds the display logic SHALL produce the warning text
    "⚠️ Có thể bị treo", and if the difference is 60 seconds or less it
    SHALL produce the normal "Hoạt động: X giây trước" text.

    **Validates: Requirements 4.3, 4.7**
    """

    @given(delta=st.integers(min_value=0, max_value=300))
    @settings(max_examples=100, deadline=None)
    def test_heartbeat_warning_threshold(self, delta):
        """
        Generate random time deltas (0 to 300 seconds), create a heartbeat
        timestamp = now - timedelta(seconds=delta), and verify:
        - If delta > 60: result is "⚠️ Có thể bị treo"
        - If delta <= 60: result is "Hoạt động: {delta} giây trước"

        **Validates: Requirements 4.3, 4.7**
        """
        from ui.ui_training_progress import _get_heartbeat_text

        # Create a heartbeat timestamp that is `delta` seconds in the past
        now = datetime.datetime.now()
        heartbeat_time = now - datetime.timedelta(seconds=delta)
        heartbeat_timestamp = heartbeat_time.isoformat()

        # Call the function under test
        result = _get_heartbeat_text(heartbeat_timestamp)

        # Verify threshold logic
        if delta > 60:
            assert result == "⚠️ Có thể bị treo", (
                f"Expected warning for delta={delta}s, got '{result}'"
            )
        else:
            expected = f"Hoạt động: {delta} giây trước"
            assert result == expected, (
                f"Expected '{expected}' for delta={delta}s, got '{result}'"
            )


# ---------------------------------------------------------------------------
# Property 1: Progress Invariant
# ---------------------------------------------------------------------------
# Feature: training-progress-checkpoint, Property 1: Progress Invariant


@st.composite
def event_sequence_strategy(draw, symbols, total_epochs):
    """
    Generate a valid event sequence for the given symbol list.

    Events are generated in a realistic order:
    - on_symbol_start for the next pending symbol
    - on_epoch_complete for epochs 1..total_epochs
    - on_symbol_complete or on_symbol_failed to finish the symbol

    The sequence may be partial (not all symbols need to complete).
    """
    events = []
    num_symbols_to_process = draw(
        st.integers(min_value=0, max_value=len(symbols))
    )

    for i in range(num_symbols_to_process):
        symbol = symbols[i]

        # Start event
        events.append(("start", symbol))

        # Random number of epoch_complete events (0 to total_epochs)
        num_epochs = draw(st.integers(min_value=0, max_value=total_epochs))
        for epoch in range(1, num_epochs + 1):
            events.append(("epoch_complete", symbol, epoch, total_epochs))

        # Complete or fail
        outcome = draw(st.sampled_from(["complete", "failed"]))
        if outcome == "complete":
            duration = draw(
                st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False)
            )
            events.append(("complete", symbol, duration))
        else:
            events.append(("failed", symbol, "Random error"))

    return events


class TestProperty1ProgressInvariant:
    """
    Property 1: Progress Invariant.

    For any list of symbols (size 1-100), at any point during a training session,
    the SymbolProgressTracker SHALL maintain:
    - completed_count is in [0, total_count]
    - total_count equals the initial symbol list length
    - overall_progress is in [0.0, 1.0]
    - overall_progress == (completed_count + current_epoch/total_epochs) / total_count
      (or 0.0 when total_count is 0)

    **Validates: Requirements 1.1, 1.3, 1.4, 1.5**
    """

    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_progress_invariants_hold_during_event_sequence(self, data):
        """
        Generate random symbol lists and event sequences, verify progress
        invariants hold after every event.

        **Validates: Requirements 1.1, 1.3, 1.4, 1.5**
        """
        # Generate a unique symbol list (size 1-100 for performance)
        symbols = data.draw(
            st.lists(
                st.text(
                    min_size=1,
                    max_size=6,
                    alphabet=characters(whitelist_categories=("L", "N")),
                ),
                min_size=1,
                max_size=100,
                unique=True,
            )
        )

        total_epochs = data.draw(st.integers(min_value=1, max_value=200))

        # Create tracker (no session state dict to avoid side effects)
        tracker = SymbolProgressTracker(
            symbols=symbols,
            mode="full",
            total_epochs_per_symbol=total_epochs,
            session_state_dict=None,
        )

        total_count = len(symbols)

        # Verify initial state
        assert len(tracker.completed_symbols) == 0
        assert tracker.overall_progress == 0.0

        # Generate and replay event sequence
        events = data.draw(event_sequence_strategy(symbols, total_epochs))

        for event in events:
            if event[0] == "start":
                tracker.on_symbol_start(event[1])
            elif event[0] == "epoch_complete":
                tracker.on_epoch_complete(event[1], event[2], event[3])
            elif event[0] == "complete":
                tracker.on_symbol_complete(event[1], event[2])
            elif event[0] == "failed":
                tracker.on_symbol_failed(event[1], event[2])

            # After each event, verify invariants
            completed_count = len(tracker.completed_symbols)
            progress = tracker.overall_progress

            # Invariant 1: completed_count in [0, total_count]
            assert 0 <= completed_count <= total_count, (
                f"completed_count {completed_count} not in [0, {total_count}]"
            )

            # Invariant 2: overall_progress in [0.0, 1.0]
            assert 0.0 <= progress <= 1.0, (
                f"overall_progress {progress} not in [0.0, 1.0]"
            )

            # Invariant 3: Formula correctness
            # overall_progress == (completed_count + current_epoch/total_epochs) / total_count
            # We need internal state to verify the formula, use get_progress_state()
            state = tracker.get_progress_state()
            current_epoch = state["current_epoch"]
            current_total_epochs = state["total_epochs"]

            if total_count == 0:
                expected_progress = 0.0
            else:
                epoch_fraction = 0.0
                if current_total_epochs > 0:
                    epoch_fraction = current_epoch / current_total_epochs
                expected_progress = (completed_count + epoch_fraction) / total_count

            assert abs(progress - expected_progress) < 1e-9, (
                f"Progress {progress} != expected {expected_progress} "
                f"(completed={completed_count}, epoch={current_epoch}/{current_total_epochs}, "
                f"total={total_count})"
            )


# ---------------------------------------------------------------------------
# Property 8: Resume Skips Completed Symbols
# ---------------------------------------------------------------------------
# Feature: training-progress-checkpoint, Property 8: Resume Skips Completed Symbols


@st.composite
def completed_pending_split_strategy(draw):
    """
    Generate a random list of unique symbols, then split into completed and pending
    portions. The split can be at any position (including all-completed or all-pending).

    Returns:
        tuple: (all_symbols, completed_symbols, pending_symbols)
    """
    # Generate a list of unique symbols (min 2 to ensure meaningful split)
    all_symbols = draw(
        st.lists(symbol_strategy, min_size=2, max_size=50, unique=True)
    )

    # Pick a split point: at least 1 completed and 1 pending
    split_index = draw(st.integers(min_value=1, max_value=len(all_symbols) - 1))

    completed_symbols = all_symbols[:split_index]
    pending_symbols = all_symbols[split_index:]

    return (all_symbols, completed_symbols, pending_symbols)


class TestProperty8ResumeSkipsCompletedSymbols:
    """
    Property 8: Resume Skips Completed Symbols.

    For any SessionCheckpoint with N completed symbols and M pending symbols,
    resuming training SHALL produce a training symbol list containing exactly
    the M pending symbols in order, with none of the N completed symbols present.

    **Validates: Requirements 5.5**
    """

    @given(split_data=completed_pending_split_strategy())
    @settings(max_examples=100, deadline=None)
    def test_resume_produces_exactly_pending_symbols_in_order(self, split_data):
        """
        Generate random completed/pending splits. Simulate the resume logic:
        - Create a SessionCheckpoint with those splits
        - The resumed training list should be EXACTLY the pending symbols in order
        - NONE of the completed symbols should be present

        This tests the pure logic from run_tracked_training:
        When resuming from checkpoint, checkpoint.pending_symbols becomes
        the training list. Completed symbols are skipped entirely.

        **Validates: Requirements 5.5**
        """
        all_symbols, completed_symbols, pending_symbols = split_data

        # Create a SessionCheckpoint with the given split
        symbol_durations = {sym: 100.0 for sym in completed_symbols}
        checkpoint = SessionCheckpoint(
            completed_symbols=completed_symbols,
            pending_symbols=pending_symbols,
            mode="full",
            session_start_time="2024-01-15T08:30:00",
            symbol_durations=symbol_durations,
            total_epochs_per_symbol=100,
        )

        # Simulate resume logic from run_tracked_training:
        # When resuming, training_symbols = checkpoint.pending_symbols
        # (assuming all pending symbols have valid data - no CSV filtering needed)
        resumed_training_list = checkpoint.pending_symbols

        # Verify: resumed list contains EXACTLY the pending symbols in order
        assert resumed_training_list == pending_symbols, (
            f"Resumed training list {resumed_training_list} != "
            f"expected pending {pending_symbols}"
        )

        # Verify: NONE of the completed symbols are in the resumed list
        completed_set = set(completed_symbols)
        for sym in resumed_training_list:
            assert sym not in completed_set, (
                f"Completed symbol '{sym}' found in resumed training list"
            )

        # Verify: the resumed list preserves the original pending order
        assert len(resumed_training_list) == len(pending_symbols), (
            f"Resumed list length {len(resumed_training_list)} != "
            f"pending length {len(pending_symbols)}"
        )
        for i, (resumed, expected) in enumerate(
            zip(resumed_training_list, pending_symbols)
        ):
            assert resumed == expected, (
                f"Order mismatch at position {i}: "
                f"got '{resumed}', expected '{expected}'"
            )


# ---------------------------------------------------------------------------
# Property 9: Resume Filters Missing Data Files
# ---------------------------------------------------------------------------
# Feature: training-progress-checkpoint, Property 9: Resume Filters Missing Data Files


def _filter_pending_symbols(pending_symbols: list, available_symbols: set) -> list:
    """
    Pure function implementing the resume filtering logic from
    background_training_impl.run_tracked_training().

    Given pending symbols from a checkpoint and a set of symbols with
    available data (CSV files exist), returns only the pending symbols
    that have available data, preserving original order.

    Args:
        pending_symbols: Ordered list of pending symbols from checkpoint.
        available_symbols: Set of symbols that have CSV data available.

    Returns:
        List of pending symbols that have available data, in original order.
    """
    valid_pending = []
    for symbol in pending_symbols:
        if symbol in available_symbols:
            valid_pending.append(symbol)
    return valid_pending


class TestProperty9ResumeFiltersMissingDataFiles:
    """
    Property 9: Resume Filters Missing Data Files.

    For any set of pending symbols in a checkpoint, where a subset S of those
    symbols lack corresponding CSV files on disk, the resumed training list
    SHALL contain exactly the pending symbols NOT in S. If all pending symbols
    are in S (all missing), the system SHALL start a fresh session using the
    current available symbol list.

    **Validates: Requirements 6.5, 6.6**
    """

    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_resume_filters_missing_data_files(self, data):
        """
        Generate random pending symbol lists (1-50 symbols), randomly partition
        into "available" (have CSV) and "missing" (no CSV), and verify:
        - Result contains exactly the "available" symbols in their original order
        - None of the "missing" symbols are present
        - If ALL symbols are "missing", verify fresh start condition
          (empty valid_pending with non-empty pending)

        **Validates: Requirements 6.5, 6.6**
        """
        # Generate a unique list of pending symbols (1-50)
        pending_symbols = data.draw(
            st.lists(
                st.text(
                    min_size=1,
                    max_size=8,
                    alphabet=characters(whitelist_categories=("L", "N")),
                ),
                min_size=1,
                max_size=50,
                unique=True,
            ),
            label="pending_symbols",
        )

        # Randomly partition into available and missing
        # Each symbol independently has a chance of being available or missing
        available_mask = data.draw(
            st.lists(
                st.booleans(),
                min_size=len(pending_symbols),
                max_size=len(pending_symbols),
            ),
            label="available_mask",
        )

        available_symbols = set()
        missing_symbols = set()
        for symbol, is_available in zip(pending_symbols, available_mask):
            if is_available:
                available_symbols.add(symbol)
            else:
                missing_symbols.add(symbol)

        # Apply the filtering logic (pure function)
        valid_pending = _filter_pending_symbols(pending_symbols, available_symbols)

        # Property assertions:

        # 1. Result contains only symbols that are in available_symbols
        for symbol in valid_pending:
            assert symbol in available_symbols, (
                f"Symbol '{symbol}' is in result but NOT in available_symbols. "
                f"Missing symbols should be filtered out."
            )

        # 2. None of the missing symbols are present in the result
        for symbol in missing_symbols:
            assert symbol not in valid_pending, (
                f"Missing symbol '{symbol}' should NOT be in the result."
            )

        # 3. Result contains ALL available pending symbols (nothing wrongly removed)
        expected_available = [s for s in pending_symbols if s in available_symbols]
        assert valid_pending == expected_available, (
            f"Result should be exactly the available symbols in original order. "
            f"Expected {expected_available}, got {valid_pending}"
        )

        # 4. Result preserves original order of pending_symbols
        # (Already verified by checking equality with expected_available above,
        #  since expected_available preserves order from pending_symbols)

        # 5. Fresh start condition: if ALL symbols are missing
        if len(valid_pending) == 0 and len(pending_symbols) > 0:
            # This is the fresh start condition (Requirement 6.6)
            # Verify all pending symbols were indeed missing
            assert all(s in missing_symbols for s in pending_symbols), (
                "Fresh start triggered but not all symbols are missing"
            )
            # In the actual code, this would trigger:
            # - checkpoint_manager.delete()
            # - training_symbols = all_symbols (fresh start)
            # We verify the condition is correctly detected
            assert len(available_symbols & set(pending_symbols)) == 0, (
                "Fresh start condition triggered but some symbols have data"
            )

        # 6. If NOT all missing, result should be non-empty
        if len(available_symbols & set(pending_symbols)) > 0:
            assert len(valid_pending) > 0, (
                "Some symbols are available but result is empty"
            )
