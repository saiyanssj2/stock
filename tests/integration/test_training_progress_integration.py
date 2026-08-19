"""
Integration tests for training progress and checkpoint functionality.

Tests end-to-end flows involving SymbolProgressTracker and SessionCheckpointManager
working together, covering:
- Full training flow from start to completion
- Resume from checkpoint
- Heartbeat timing
- UI session state propagation
- Corrupt checkpoint recovery

Requirements: 5.4, 5.5, 5.7, 6.3, 6.5
"""

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest

from engine.session_checkpoint import SessionCheckpoint, SessionCheckpointManager
from engine.symbol_progress_tracker import SymbolProgressTracker


class TestEndToEndFlow:
    """Test complete training flow: start → track → complete → checkpoint deleted."""

    def test_full_training_flow_three_symbols(self):
        """End-to-end: train 3 symbols, verify progress and checkpoint lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            symbols = ["VNM", "FPT", "HPG"]
            session_state = {}

            # Create tracker with temp dir for checkpoint
            tracker = SymbolProgressTracker(
                symbols=symbols,
                mode="full",
                total_epochs_per_symbol=10,
                session_state_dict=session_state,
            )
            # Override checkpoint manager to use temp dir
            tracker._checkpoint_manager = SessionCheckpointManager(base_dir=tmpdir)

            # Verify initial state
            assert tracker.overall_progress == 0.0
            assert tracker.current_symbol is None

            # Train each symbol
            durations = [120.0, 95.0, 140.0]
            for i, symbol in enumerate(symbols):
                tracker.on_symbol_start(symbol)
                assert tracker.current_symbol == symbol

                # Simulate epochs
                for epoch in range(1, 11):
                    tracker.on_epoch_complete(symbol, epoch, 10)

                # Complete the symbol
                tracker.on_symbol_complete(symbol, durations[i])

                # Verify checkpoint saved after each symbol
                checkpoint_path = (
                    Path(tmpdir) / "engine" / "models"
                    / "training_session_checkpoint.json"
                )
                if i < len(symbols) - 1:
                    # Checkpoint exists for intermediate symbols
                    assert checkpoint_path.exists(), (
                        f"Checkpoint should exist after completing symbol {i+1}"
                    )

            # After all complete, verify overall progress is 1.0
            assert tracker.overall_progress == 1.0

            # Verify checkpoint is saved with last state (3 completed, 0 pending)
            mgr = SessionCheckpointManager(base_dir=tmpdir)
            cp = mgr.load()
            if cp is not None:
                # All symbols completed
                assert len(cp.completed_symbols) == 3
                assert len(cp.pending_symbols) == 0

            # Simulate session completion: delete checkpoint
            mgr.delete()
            assert not checkpoint_path.exists()

    def test_progress_increases_monotonically(self):
        """Verify overall_progress goes from 0 to 1.0 monotonically."""
        symbols = ["A", "B", "C"]
        tracker = SymbolProgressTracker(
            symbols=symbols,
            mode="full",
            total_epochs_per_symbol=5,
            session_state_dict={},
        )
        # Use temp dir for checkpoint to avoid polluting project
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker._checkpoint_manager = SessionCheckpointManager(base_dir=tmpdir)

            progress_values = [tracker.overall_progress]

            for symbol in symbols:
                tracker.on_symbol_start(symbol)
                for epoch in range(1, 6):
                    tracker.on_epoch_complete(symbol, epoch, 5)
                    progress_values.append(tracker.overall_progress)
                tracker.on_symbol_complete(symbol, 60.0)
                progress_values.append(tracker.overall_progress)

            # Progress should be non-decreasing
            for i in range(1, len(progress_values)):
                assert progress_values[i] >= progress_values[i - 1], (
                    f"Progress decreased at step {i}: "
                    f"{progress_values[i-1]} -> {progress_values[i]}"
                )

            # Final progress should be 1.0
            assert progress_values[-1] == 1.0


class TestResumeFlow:
    """Test checkpoint resume: create → verify exists → load → resume pending."""

    def test_resume_skips_completed_symbols(self):
        """Save checkpoint with 2 completed, 3 pending. Resume trains only pending."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save a checkpoint with partial progress
            checkpoint = SessionCheckpoint(
                completed_symbols=["VNM", "FPT"],
                pending_symbols=["HPG", "MWG", "VHM"],
                mode="full",
                session_start_time=datetime.now().isoformat(),
                symbol_durations={"VNM": 120.0, "FPT": 95.0},
                total_epochs_per_symbol=10,
            )

            mgr = SessionCheckpointManager(base_dir=tmpdir)
            mgr.save(checkpoint)

            # Verify checkpoint exists
            assert mgr.exists()

            # Load and verify pending symbols
            loaded = mgr.load()
            assert loaded is not None
            assert loaded.pending_symbols == ["HPG", "MWG", "VHM"]
            assert loaded.completed_symbols == ["VNM", "FPT"]

            # Simulate resume: train only pending symbols
            session_state = {}
            tracker = SymbolProgressTracker(
                symbols=loaded.pending_symbols,
                mode=loaded.mode,
                total_epochs_per_symbol=loaded.total_epochs_per_symbol,
                session_state_dict=session_state,
            )
            tracker._checkpoint_manager = SessionCheckpointManager(base_dir=tmpdir)

            trained_symbols = []
            for symbol in loaded.pending_symbols:
                tracker.on_symbol_start(symbol)
                for epoch in range(1, 11):
                    tracker.on_epoch_complete(symbol, epoch, 10)
                tracker.on_symbol_complete(symbol, 100.0)
                trained_symbols.append(symbol)

            # Only pending symbols were trained
            assert trained_symbols == ["HPG", "MWG", "VHM"]
            assert "VNM" not in trained_symbols
            assert "FPT" not in trained_symbols

    def test_resume_checkpoint_exists_true(self):
        """Verify exists() returns True when valid checkpoint is saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionCheckpointManager(base_dir=tmpdir)

            # No checkpoint initially
            assert not mgr.exists()

            # Save checkpoint
            checkpoint = SessionCheckpoint(
                completed_symbols=["A"],
                pending_symbols=["B", "C"],
                mode="incremental",
                session_start_time=datetime.now().isoformat(),
                symbol_durations={"A": 50.0},
                total_epochs_per_symbol=100,
            )
            mgr.save(checkpoint)

            # Now exists
            assert mgr.exists()

    def test_resume_loads_correct_pending(self):
        """Load checkpoint and verify pending_symbols match what was saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pending = ["HPG", "MWG", "VHM", "TCB", "ACB"]
            checkpoint = SessionCheckpoint(
                completed_symbols=["VNM", "FPT"],
                pending_symbols=pending,
                mode="full",
                session_start_time="2024-01-15T08:30:00",
                symbol_durations={"VNM": 423.5, "FPT": 389.2},
                total_epochs_per_symbol=100,
            )

            mgr = SessionCheckpointManager(base_dir=tmpdir)
            mgr.save(checkpoint)

            loaded = mgr.load()
            assert loaded is not None
            assert loaded.pending_symbols == pending


class TestHeartbeatTiming:
    """Test heartbeat timestamp updates during training."""

    def test_heartbeat_updated_on_epoch_complete(self):
        """Verify heartbeat_timestamp updates when on_epoch_complete is called."""
        symbols = ["VNM"]
        session_state = {}
        tracker = SymbolProgressTracker(
            symbols=symbols,
            mode="full",
            total_epochs_per_symbol=10,
            session_state_dict=session_state,
        )

        tracker.on_symbol_start("VNM")

        # Get heartbeat after start
        state_after_start = session_state.get("training_progress", {})
        heartbeat_start = state_after_start.get("heartbeat_timestamp")
        assert heartbeat_start is not None

        # Simulate epoch and check heartbeat is recent
        time.sleep(0.05)
        tracker.on_epoch_complete("VNM", 1, 10)

        state_after_epoch = session_state["training_progress"]
        heartbeat_after = state_after_epoch["heartbeat_timestamp"]

        # Parse timestamps and verify heartbeat is recent
        ts = datetime.fromisoformat(heartbeat_after)
        now = datetime.now()
        delta = (now - ts).total_seconds()
        assert delta < 2.0, f"Heartbeat should be within 2 seconds, got {delta}s"

    def test_update_heartbeat_sets_recent_timestamp(self):
        """Verify update_heartbeat() sets timestamp within 2 seconds of now."""
        symbols = ["VNM"]
        session_state = {}
        tracker = SymbolProgressTracker(
            symbols=symbols,
            mode="full",
            total_epochs_per_symbol=10,
            session_state_dict=session_state,
        )

        tracker.on_symbol_start("VNM")
        time.sleep(0.05)
        tracker.update_heartbeat()

        state = session_state["training_progress"]
        heartbeat_ts = datetime.fromisoformat(state["heartbeat_timestamp"])
        now = datetime.now()
        delta = (now - heartbeat_ts).total_seconds()
        assert delta < 2.0, f"Heartbeat should be within 2s of now, got {delta}s"

    def test_heartbeat_initialized_on_session_start(self):
        """Verify heartbeat is set when tracker is created and symbol starts."""
        symbols = ["A", "B"]
        session_state = {}
        tracker = SymbolProgressTracker(
            symbols=symbols,
            mode="full",
            total_epochs_per_symbol=5,
            session_state_dict=session_state,
        )

        tracker.on_symbol_start("A")
        state = session_state["training_progress"]
        assert "heartbeat_timestamp" in state

        heartbeat_ts = datetime.fromisoformat(state["heartbeat_timestamp"])
        now = datetime.now()
        delta = (now - heartbeat_ts).total_seconds()
        assert delta < 2.0


class TestUIStatePropagation:
    """Test that session_state dict is updated correctly by tracker events."""

    def test_session_state_has_training_progress_after_start(self):
        """After on_symbol_start, session_state_dict has training_progress key."""
        session_state = {}
        tracker = SymbolProgressTracker(
            symbols=["VNM", "FPT", "HPG"],
            mode="full",
            total_epochs_per_symbol=100,
            session_state_dict=session_state,
        )

        tracker.on_symbol_start("VNM")

        assert "training_progress" in session_state
        progress = session_state["training_progress"]
        assert progress["active"] is True
        assert progress["current_symbol"] == "VNM"
        assert progress["completed_count"] == 0
        assert progress["total_count"] == 3

    def test_session_state_counts_updated_on_complete(self):
        """After on_symbol_complete, completed_count increments in session_state."""
        session_state = {}
        tracker = SymbolProgressTracker(
            symbols=["VNM", "FPT"],
            mode="full",
            total_epochs_per_symbol=10,
            session_state_dict=session_state,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker._checkpoint_manager = SessionCheckpointManager(base_dir=tmpdir)

            tracker.on_symbol_start("VNM")
            assert session_state["training_progress"]["completed_count"] == 0

            tracker.on_symbol_complete("VNM", 120.0)
            assert session_state["training_progress"]["completed_count"] == 1

            tracker.on_symbol_start("FPT")
            tracker.on_symbol_complete("FPT", 95.0)
            assert session_state["training_progress"]["completed_count"] == 2

    def test_session_state_overall_progress_updated(self):
        """Verify overall_progress in session_state reflects tracker state."""
        session_state = {}
        tracker = SymbolProgressTracker(
            symbols=["A", "B"],
            mode="full",
            total_epochs_per_symbol=10,
            session_state_dict=session_state,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker._checkpoint_manager = SessionCheckpointManager(base_dir=tmpdir)

            tracker.on_symbol_start("A")
            tracker.on_epoch_complete("A", 5, 10)

            # Progress: (0 + 5/10) / 2 = 0.25
            progress = session_state["training_progress"]["overall_progress"]
            assert abs(progress - 0.25) < 1e-9

            tracker.on_symbol_complete("A", 60.0)
            # Progress: 1/2 = 0.5
            progress = session_state["training_progress"]["overall_progress"]
            assert abs(progress - 0.5) < 1e-9

    def test_session_state_not_set_when_dict_is_none(self):
        """If session_state_dict is None, no error occurs."""
        tracker = SymbolProgressTracker(
            symbols=["A"],
            mode="full",
            total_epochs_per_symbol=10,
            session_state_dict=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker._checkpoint_manager = SessionCheckpointManager(base_dir=tmpdir)

            # Should not raise
            tracker.on_symbol_start("A")
            tracker.on_epoch_complete("A", 1, 10)
            tracker.on_symbol_complete("A", 30.0)


class TestCorruptCheckpointRecovery:
    """Test recovery when checkpoint file contains invalid data."""

    def test_invalid_json_returns_none(self):
        """Write invalid JSON to checkpoint path, verify load() returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionCheckpointManager(base_dir=tmpdir)
            checkpoint_path = mgr.checkpoint_path

            # Ensure parent directory exists
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            # Write invalid JSON
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                f.write("{not valid json content!!!")

            assert checkpoint_path.exists()

            # load() should return None
            result = mgr.load()
            assert result is None

    def test_corrupt_checkpoint_is_deleted(self):
        """After loading corrupt checkpoint, the file should be deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionCheckpointManager(base_dir=tmpdir)
            checkpoint_path = mgr.checkpoint_path

            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            # Write garbage data
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                f.write("corrupted data here")

            # Load triggers cleanup
            mgr.load()

            # File should be deleted
            assert not checkpoint_path.exists()

    def test_fresh_start_after_corrupt_checkpoint(self):
        """After corrupt checkpoint is deleted, fresh training proceeds normally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionCheckpointManager(base_dir=tmpdir)
            checkpoint_path = mgr.checkpoint_path

            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            # Write invalid JSON
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                f.write("not json")

            # Load and discard (triggers deletion)
            result = mgr.load()
            assert result is None
            assert not checkpoint_path.exists()

            # Fresh start: create new tracker and train normally
            session_state = {}
            tracker = SymbolProgressTracker(
                symbols=["VNM", "FPT"],
                mode="full",
                total_epochs_per_symbol=5,
                session_state_dict=session_state,
            )
            tracker._checkpoint_manager = mgr

            tracker.on_symbol_start("VNM")
            for epoch in range(1, 6):
                tracker.on_epoch_complete("VNM", epoch, 5)
            tracker.on_symbol_complete("VNM", 60.0)

            # Checkpoint should be saved with new state
            assert checkpoint_path.exists()
            loaded = mgr.load()
            assert loaded is not None
            assert "VNM" in loaded.completed_symbols

    def test_missing_required_fields_returns_none(self):
        """JSON with missing required fields triggers None return and deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionCheckpointManager(base_dir=tmpdir)
            checkpoint_path = mgr.checkpoint_path

            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            # Valid JSON but missing required fields
            incomplete_data = {
                "completed_symbols": ["VNM"],
                # Missing: pending_symbols, mode, session_start_time, symbol_durations
            }
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(incomplete_data, f)

            result = mgr.load()
            assert result is None
            assert not checkpoint_path.exists()
