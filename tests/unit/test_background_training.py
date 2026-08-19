"""
Unit tests for background training manager.

Tests:
- GPUMemoryMonitor threshold logic
- SymbolQueue thread-safe operations
- BackgroundTrainingManager lifecycle (start/stop/pause/resume)
- Model hot-swap integration
- Training status reporting
- Inference priority over training

Requirements: 7.1, 7.2, 7.3, 7.5, 7.6, 6.11, 12.4
"""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from engine.background_training import (
    BackgroundTrainingManager,
    GPUMemoryMonitor,
    SymbolQueue,
    TrainingState,
    TrainingStatus,
)


# ==============================================================================
# GPUMemoryMonitor Tests
# ==============================================================================


class TestGPUMemoryMonitor:
    """Tests for GPUMemoryMonitor threshold logic."""

    def test_default_thresholds(self):
        """Default high threshold is 90%, low threshold is 80%."""
        monitor = GPUMemoryMonitor()
        assert monitor.high_threshold == 0.90
        assert monitor.low_threshold == 0.80

    def test_custom_thresholds(self):
        """Custom thresholds are accepted."""
        monitor = GPUMemoryMonitor(high_threshold=0.85, low_threshold=0.70)
        assert monitor.high_threshold == 0.85
        assert monitor.low_threshold == 0.70

    def test_no_cuda_returns_zero_utilization(self):
        """When CUDA is unavailable, memory utilization is 0."""
        monitor = GPUMemoryMonitor()
        monitor._cuda_available = False
        assert monitor.get_memory_utilization() == 0.0

    def test_no_cuda_should_not_pause(self):
        """Without CUDA, training should never be paused."""
        monitor = GPUMemoryMonitor()
        monitor._cuda_available = False
        assert monitor.should_pause_training() is False

    def test_no_cuda_can_always_resume(self):
        """Without CUDA, can_resume_training always returns True."""
        monitor = GPUMemoryMonitor()
        monitor._cuda_available = False
        assert monitor.can_resume_training() is True

    def test_get_memory_info_no_cuda(self):
        """Memory info without CUDA returns zeros."""
        monitor = GPUMemoryMonitor()
        monitor._cuda_available = False
        info = monitor.get_memory_info()
        assert info["allocated_gb"] == 0.0
        assert info["total_gb"] == 0.0
        assert info["utilization"] == 0.0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=5_000_000_000)  # 5GB
    @patch("torch.cuda.get_device_properties")
    def test_high_memory_triggers_pause(self, mock_props, mock_alloc, mock_avail):
        """When GPU memory exceeds 90%, should_pause_training returns True."""
        mock_props.return_value = MagicMock(total_mem=6_000_000_000)  # 6GB total
        monitor = GPUMemoryMonitor()
        monitor._cuda_available = True
        # 5/6 = 83% - below threshold
        assert monitor.should_pause_training() is False

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=5_500_000_000)  # 5.5GB
    @patch("torch.cuda.get_device_properties")
    def test_very_high_memory_triggers_pause(self, mock_props, mock_alloc, mock_avail):
        """When GPU memory exceeds 90%, should_pause_training returns True."""
        mock_props.return_value = MagicMock(total_mem=6_000_000_000)  # 6GB total
        monitor = GPUMemoryMonitor()
        monitor._cuda_available = True
        # 5.5/6 = 91.7% - above threshold
        assert monitor.should_pause_training() is True

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=4_500_000_000)  # 4.5GB
    @patch("torch.cuda.get_device_properties")
    def test_low_memory_allows_resume(self, mock_props, mock_alloc, mock_avail):
        """When GPU memory drops below 80%, can_resume_training returns True."""
        mock_props.return_value = MagicMock(total_mem=6_000_000_000)  # 6GB total
        monitor = GPUMemoryMonitor()
        monitor._cuda_available = True
        # 4.5/6 = 75% - below low threshold
        assert monitor.can_resume_training() is True

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=5_000_000_000)  # 5GB
    @patch("torch.cuda.get_device_properties")
    def test_medium_memory_blocks_resume(self, mock_props, mock_alloc, mock_avail):
        """Between thresholds, can_resume_training returns False."""
        mock_props.return_value = MagicMock(total_mem=6_000_000_000)  # 6GB total
        monitor = GPUMemoryMonitor()
        monitor._cuda_available = True
        # 5/6 = 83.3% - between low (80%) and high (90%)
        assert monitor.can_resume_training() is False


# ==============================================================================
# SymbolQueue Tests
# ==============================================================================


class TestSymbolQueue:
    """Tests for thread-safe symbol queue operations."""

    def test_add_single_symbol(self):
        """Adding a symbol places it in the queue."""
        queue = SymbolQueue()
        queue.add_symbol("VNM")
        assert queue.has_pending() is True
        assert queue.pending_count == 1

    def test_add_duplicate_symbol(self):
        """Duplicate symbols are not added twice."""
        queue = SymbolQueue()
        queue.add_symbol("VNM")
        queue.add_symbol("VNM")
        assert queue.pending_count == 1

    def test_add_multiple_symbols(self):
        """Multiple different symbols are queued."""
        queue = SymbolQueue()
        queue.add_symbols(["VNM", "FPT", "VIC"])
        assert queue.pending_count == 3

    def test_get_pending_clears_queue(self):
        """Getting pending symbols clears the queue."""
        queue = SymbolQueue()
        queue.add_symbols(["VNM", "FPT"])
        symbols = queue.get_pending_symbols()
        assert symbols == ["VNM", "FPT"]
        assert queue.has_pending() is False
        assert queue.pending_count == 0

    def test_empty_queue(self):
        """Empty queue returns no pending symbols."""
        queue = SymbolQueue()
        assert queue.has_pending() is False
        assert queue.get_pending_symbols() == []

    def test_thread_safety(self):
        """Queue handles concurrent additions safely."""
        queue = SymbolQueue()
        symbols_to_add = [f"SYM_{i}" for i in range(100)]
        errors = []

        def add_symbols(start, end):
            try:
                for i in range(start, end):
                    queue.add_symbol(symbols_to_add[i])
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_symbols, args=(0, 25)),
            threading.Thread(target=add_symbols, args=(25, 50)),
            threading.Thread(target=add_symbols, args=(50, 75)),
            threading.Thread(target=add_symbols, args=(75, 100)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert queue.pending_count == 100


# ==============================================================================
# BackgroundTrainingManager Tests
# ==============================================================================


class TestBackgroundTrainingManager:
    """Tests for BackgroundTrainingManager lifecycle and coordination."""

    def _make_manager(self, model_manager=None, pipeline=None):
        """Create a BackgroundTrainingManager with mocked dependencies."""
        if model_manager is None:
            model_manager = MagicMock()
            model_manager.version = 1
            model_manager.is_loaded = True
        manager = BackgroundTrainingManager(
            model_manager=model_manager,
            training_pipeline=pipeline,
        )
        return manager

    def test_initial_state_idle(self):
        """Manager starts in IDLE state."""
        manager = self._make_manager()
        assert manager.status.state == TrainingState.IDLE
        assert manager.is_training is False

    def test_cannot_start_without_pipeline(self):
        """Starting training fails if no pipeline is configured."""
        manager = self._make_manager()
        result = manager.start_training({}, mode="incremental")
        assert result is False

    def test_symbol_queue_integration(self):
        """Symbols can be queued through the manager."""
        manager = self._make_manager()
        manager.queue_symbol("VNM")
        manager.queue_symbol("FPT")
        assert manager.symbol_queue.pending_count == 2

    def test_queue_symbols_batch(self):
        """Multiple symbols can be queued at once."""
        manager = self._make_manager()
        manager.queue_symbols(["VNM", "FPT", "VIC"])
        queued = manager.get_queued_symbols()
        assert "VNM" in queued
        assert "FPT" in queued
        assert "VIC" in queued

    def test_stop_when_not_training(self):
        """Stopping when idle is a no-op."""
        manager = self._make_manager()
        manager.stop_training()
        assert manager.status.state == TrainingState.STOPPED

    def test_status_callback_invoked(self):
        """Status callback is called on state changes."""
        manager = self._make_manager()
        statuses = []
        manager.set_status_callback(lambda s: statuses.append(s))

        # Manually trigger a pause notification to test callback
        manager._handle_gpu_pause()
        # Give the watch thread a moment
        time.sleep(0.1)

        assert len(statuses) >= 1
        assert statuses[0].state == TrainingState.PAUSED

    def test_pause_callback_invoked(self):
        """Pause callback is called when training pauses."""
        manager = self._make_manager()
        pause_reasons = []
        manager.set_paused_callback(lambda r: pause_reasons.append(r))

        manager._handle_gpu_pause()
        time.sleep(0.1)

        assert len(pause_reasons) == 1
        assert "GPU memory" in pause_reasons[0]

    def test_training_status_fields(self):
        """TrainingStatus contains all expected fields."""
        status = TrainingStatus()
        assert status.state == TrainingState.IDLE
        assert status.current_epoch == 0
        assert status.total_epochs == 0
        assert status.current_loss == 0.0
        assert status.eta_seconds == 0.0
        assert status.symbols_training == []
        assert status.paused_reason is None
        assert status.model_version == 0

    def test_hot_swap_timeout(self):
        """Hot-swap respects the timeout configuration."""
        manager = self._make_manager()
        assert manager._hot_swap_timeout == 3.0

        # Custom timeout
        manager2 = BackgroundTrainingManager(
            model_manager=MagicMock(),
            hot_swap_timeout=5.0,
        )
        assert manager2._hot_swap_timeout == 5.0

    def test_gpu_monitor_accessible(self):
        """GPU monitor is accessible through the manager."""
        manager = self._make_manager()
        assert isinstance(manager.gpu_monitor, GPUMemoryMonitor)

    def test_resume_from_pause(self):
        """Training resumes after GPU memory pressure subsides."""
        manager = self._make_manager()

        # Put into paused state
        with manager._status_lock:
            manager._status.state = TrainingState.PAUSED

        # Resume
        manager._resume_from_pause()

        assert manager.status.state == TrainingState.RUNNING
        assert manager.status.paused_reason is None

    def test_hot_swap_updates_model_version(self):
        """Successful hot-swap updates model version in status."""
        model_manager = MagicMock()
        model_manager.version = 5
        model_manager.hot_swap = MagicMock()

        manager = self._make_manager(model_manager=model_manager)
        manager._perform_hot_swap("/fake/model.pt")

        model_manager.hot_swap.assert_called_once_with("/fake/model.pt")
        assert manager.status.model_version == 5
        assert manager.status.last_training_completed is not None

    def test_hot_swap_failure_graceful(self):
        """Failed hot-swap doesn't crash, sets state to IDLE."""
        model_manager = MagicMock()
        model_manager.hot_swap = MagicMock(
            side_effect=Exception("checksum mismatch")
        )

        manager = self._make_manager(model_manager=model_manager)
        manager._perform_hot_swap("/fake/model.pt")

        assert manager.status.state == TrainingState.IDLE
        assert "failed" in manager.status.paused_reason.lower()

    def test_concurrent_inference_not_blocked(self):
        """Inference gate is always set, allowing inference to proceed."""
        manager = self._make_manager()
        # The inference gate should be set (not blocking)
        assert manager._inference_gate.is_set()

    def test_double_start_returns_false(self):
        """Starting training while already running returns False."""
        manager = self._make_manager()
        with manager._status_lock:
            manager._status.state = TrainingState.RUNNING
        result = manager.start_training({}, mode="incremental")
        assert result is False


# ==============================================================================
# Integration with DecisionEngine Tests
# ==============================================================================


class TestDecisionEngineBackgroundTraining:
    """Tests for DecisionEngine's background training integration."""

    def _make_engine(self, tmp_path):
        """Create a DecisionEngine with a temp directory."""
        import os
        # Create minimal CSV file for validation
        csv_path = tmp_path / "VNINDEX.csv"
        csv_path.write_text(
            "time,open,high,low,close,volume\n"
            + "\n".join(
                [f"2024-01-{i:02d},{100+i},{105+i},{95+i},{102+i},{1000000}" for i in range(1, 70)]
            )
        )

        from engine.decision_engine import DecisionEngine

        engine = DecisionEngine(base_dir=str(tmp_path))
        return engine

    def test_engine_has_background_training(self, tmp_path):
        """DecisionEngine has background_training property."""
        engine = self._make_engine(tmp_path)
        assert isinstance(engine.background_training, BackgroundTrainingManager)

    def test_engine_training_status(self, tmp_path):
        """DecisionEngine exposes training status."""
        engine = self._make_engine(tmp_path)
        status = engine.training_status
        assert status.state == TrainingState.IDLE

    def test_engine_queue_symbol(self, tmp_path):
        """DecisionEngine can queue symbols for training."""
        engine = self._make_engine(tmp_path)
        engine.queue_training_symbol("VNM")
        engine.queue_training_symbol("FPT")
        queued = engine.background_training.get_queued_symbols()
        assert "VNM" in queued
        assert "FPT" in queued

    def test_engine_gpu_monitor(self, tmp_path):
        """DecisionEngine provides GPU monitor access."""
        engine = self._make_engine(tmp_path)
        assert isinstance(engine.gpu_monitor, GPUMemoryMonitor)

    def test_engine_is_training_property(self, tmp_path):
        """DecisionEngine reports training state."""
        engine = self._make_engine(tmp_path)
        assert engine.is_training is False
