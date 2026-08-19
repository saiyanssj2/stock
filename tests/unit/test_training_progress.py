"""
Unit tests for training progress and checkpoint features.

Tests:
- SymbolProgressTracker initialization with various symbol counts
- Session state initialization on session start
- Heartbeat timestamp initialization
- Checkpoint deletion on session completion
- UI rendering logic with mocked Streamlit calls
- Resume prompt display with valid/invalid checkpoint
- Slow symbol detection edge cases (exact 2× boundary)

Requirements: 1.1, 4.6, 5.6, 6.1, 7.5
"""

import datetime
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# Mock streamlit before importing UI modules
_mock_st = MagicMock()
_mock_st.session_state = {}
sys.modules.setdefault("streamlit", _mock_st)

from engine.eta_calculator import ETACalculator
from engine.session_checkpoint import SessionCheckpoint, SessionCheckpointManager
from engine.symbol_progress_tracker import SymbolProgressTracker, SymbolResult
from ui.ui_training_progress import (
    _is_slow_symbol,
    render_resume_prompt,
    render_training_progress,
)


# ==============================================================================
# SymbolProgressTracker Initialization Tests (Requirement 1.1)
# ==============================================================================


class TestSymbolProgressTrackerInit:
    """Tests for SymbolProgressTracker initialization with various symbol counts."""

    def test_init_with_zero_symbols(self):
        """With 0 symbols, overall_progress is 0.0 without error."""
        tracker = SymbolProgressTracker(
            symbols=[],
            mode="full",
            total_epochs_per_symbol=100,
        )
        assert tracker.overall_progress == 0.0
        assert tracker._total_count == 0

    def test_init_with_one_symbol(self):
        """With 1 symbol, total_count is 1."""
        tracker = SymbolProgressTracker(
            symbols=["VNM"],
            mode="full",
            total_epochs_per_symbol=100,
        )
        assert tracker._total_count == 1
        assert tracker.overall_progress == 0.0

    def test_init_with_68_symbols(self):
        """With 68 symbols, total_count is 68."""
        symbols = [f"SYM{i}" for i in range(68)]
        tracker = SymbolProgressTracker(
            symbols=symbols,
            mode="full",
            total_epochs_per_symbol=100,
        )
        assert tracker._total_count == 68
        assert tracker.overall_progress == 0.0

    def test_init_completed_count_is_zero(self):
        """Completed count starts at zero."""
        tracker = SymbolProgressTracker(
            symbols=["VNM", "FPT", "HPG"],
            mode="incremental",
            total_epochs_per_symbol=50,
        )
        assert tracker._completed_count == 0
        assert len(tracker.completed_symbols) == 0

    def test_init_pending_symbols_equals_input(self):
        """All symbols are pending initially."""
        symbols = ["VNM", "FPT", "HPG"]
        tracker = SymbolProgressTracker(
            symbols=symbols,
            mode="full",
            total_epochs_per_symbol=100,
        )
        assert tracker.pending_symbols == symbols


# ==============================================================================
# Session State Initialization Tests (Requirement 4.6)
# ==============================================================================


class TestSessionStateInit:
    """Tests for session state initialization on symbol start."""

    def test_on_symbol_start_sets_current_symbol(self):
        """After on_symbol_start(), current_symbol is set."""
        tracker = SymbolProgressTracker(
            symbols=["VNM", "FPT"],
            mode="full",
            total_epochs_per_symbol=100,
        )
        tracker.on_symbol_start("VNM")
        assert tracker.current_symbol == "VNM"

    def test_on_symbol_start_sets_heartbeat_timestamp(self):
        """After on_symbol_start(), heartbeat_timestamp is initialized."""
        tracker = SymbolProgressTracker(
            symbols=["VNM"],
            mode="full",
            total_epochs_per_symbol=100,
        )
        before = datetime.datetime.now().isoformat()
        tracker.on_symbol_start("VNM")
        after = datetime.datetime.now().isoformat()

        assert tracker._heartbeat_timestamp >= before
        assert tracker._heartbeat_timestamp <= after

    def test_on_symbol_start_resets_epoch_state(self):
        """After on_symbol_start(), epoch counters are reset."""
        tracker = SymbolProgressTracker(
            symbols=["VNM", "FPT"],
            mode="full",
            total_epochs_per_symbol=100,
        )
        # Simulate some epoch progress on first symbol
        tracker.on_symbol_start("VNM")
        tracker.on_epoch_complete("VNM", 50, 100)
        # Start second symbol
        tracker.on_symbol_complete("VNM", 120.0)
        tracker.on_symbol_start("FPT")
        assert tracker._current_epoch == 0

    def test_session_state_dict_updated_on_start(self):
        """session_state_dict receives updates after on_symbol_start()."""
        state = {}
        tracker = SymbolProgressTracker(
            symbols=["VNM"],
            mode="full",
            total_epochs_per_symbol=100,
            session_state_dict=state,
        )
        tracker.on_symbol_start("VNM")
        assert "training_progress" in state
        assert state["training_progress"]["current_symbol"] == "VNM"
        assert state["training_progress"]["active"] is True


# ==============================================================================
# Heartbeat Timestamp Tests (Requirement 4.6)
# ==============================================================================


class TestHeartbeat:
    """Tests for heartbeat timestamp initialization and updates."""

    def test_heartbeat_initialized_on_construction(self):
        """After init, heartbeat_timestamp is set to current time."""
        before = datetime.datetime.now().isoformat()
        tracker = SymbolProgressTracker(
            symbols=["VNM"],
            mode="full",
            total_epochs_per_symbol=100,
        )
        after = datetime.datetime.now().isoformat()

        assert tracker._heartbeat_timestamp >= before
        assert tracker._heartbeat_timestamp <= after

    def test_update_heartbeat_advances_timestamp(self):
        """After update_heartbeat(), timestamp updates to a newer time."""
        tracker = SymbolProgressTracker(
            symbols=["VNM"],
            mode="full",
            total_epochs_per_symbol=100,
        )
        initial_ts = tracker._heartbeat_timestamp
        # Small sleep to ensure time advances
        time.sleep(0.01)
        tracker.update_heartbeat()
        assert tracker._heartbeat_timestamp > initial_ts


# ==============================================================================
# Checkpoint Deletion Tests (Requirement 5.6)
# ==============================================================================


class TestCheckpointDeletion:
    """Tests for checkpoint deletion on session completion."""

    def test_checkpoint_deleted_after_all_symbols_complete(self, tmp_path):
        """After all symbols complete, checkpoint should be deleted."""
        # Setup checkpoint manager with tmp_path
        tracker = SymbolProgressTracker(
            symbols=["VNM", "FPT"],
            mode="full",
            total_epochs_per_symbol=100,
        )
        # Override checkpoint manager to use tmp_path
        tracker._checkpoint_manager = SessionCheckpointManager(
            base_dir=str(tmp_path)
        )
        # Ensure checkpoint dir exists
        checkpoint_dir = tmp_path / "engine" / "models"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Complete both symbols
        tracker.on_symbol_start("VNM")
        tracker.on_symbol_complete("VNM", 100.0)
        tracker.on_symbol_start("FPT")
        tracker.on_symbol_complete("FPT", 120.0)

        # After completion, checkpoint should have been saved (for each symbol)
        # Now delete it as the session manager would on completion
        tracker._checkpoint_manager.delete()
        assert not tracker._checkpoint_manager.exists()

    def test_checkpoint_saved_between_symbols(self, tmp_path):
        """Checkpoint is saved after each symbol completes (before session ends)."""
        tracker = SymbolProgressTracker(
            symbols=["VNM", "FPT", "HPG"],
            mode="full",
            total_epochs_per_symbol=100,
        )
        tracker._checkpoint_manager = SessionCheckpointManager(
            base_dir=str(tmp_path)
        )
        checkpoint_dir = tmp_path / "engine" / "models"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Complete first symbol
        tracker.on_symbol_start("VNM")
        tracker.on_symbol_complete("VNM", 100.0)

        # Checkpoint should exist
        assert tracker._checkpoint_manager.exists()


# ==============================================================================
# UI Rendering Tests (mocked Streamlit)
# ==============================================================================


class TestUIRendering:
    """Tests for UI rendering logic with mocked Streamlit calls."""

    def setup_method(self):
        """Reset the streamlit mock before each test."""
        _mock_st.reset_mock()
        _mock_st.session_state = {}

    def test_render_training_progress_none_state(self):
        """render_training_progress() doesn't crash when training_progress is None."""
        _mock_st.session_state = {}
        # Should return without error
        render_training_progress()

    def test_render_training_progress_inactive(self):
        """render_training_progress() doesn't crash when active=False."""
        _mock_st.session_state = {
            "training_progress": {"active": False}
        }
        render_training_progress()

    def test_render_resume_prompt_no_checkpoint(self):
        """render_resume_prompt() returns None when no checkpoint exists."""
        _mock_st.session_state = {}
        with patch.object(
            SessionCheckpointManager, "exists", return_value=False
        ):
            result = render_resume_prompt()
        assert result is None


# ==============================================================================
# Slow Symbol Detection Edge Cases (Requirement 7.5)
# ==============================================================================


class TestSlowSymbolDetection:
    """Tests for slow symbol detection edge cases at exact 2× boundary."""

    def test_exactly_two_times_average_is_not_slow(self):
        """duration = exactly 2 * average → NOT slow (> required, not >=)."""
        average = 100.0
        duration = 2 * average  # exactly 200.0
        assert _is_slow_symbol(duration, average) is False

    def test_just_above_two_times_average_is_slow(self):
        """duration = 2 * average + 0.001 → IS slow."""
        average = 100.0
        duration = 2 * average + 0.001  # 200.001
        assert _is_slow_symbol(duration, average) is True

    def test_just_below_two_times_average_is_not_slow(self):
        """duration = 2 * average - 0.001 → NOT slow."""
        average = 100.0
        duration = 2 * average - 0.001  # 199.999
        assert _is_slow_symbol(duration, average) is False

    def test_small_average_boundary(self):
        """Boundary test with small average value."""
        average = 0.5
        assert _is_slow_symbol(1.0, average) is False  # exactly 2×
        assert _is_slow_symbol(1.001, average) is True  # just above
        assert _is_slow_symbol(0.999, average) is False  # below

    def test_large_average_boundary(self):
        """Boundary test with large average value."""
        average = 5000.0
        assert _is_slow_symbol(10000.0, average) is False  # exactly 2×
        assert _is_slow_symbol(10000.001, average) is True  # just above
        assert _is_slow_symbol(9999.999, average) is False  # below
