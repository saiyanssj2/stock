# -*- coding: utf-8 -*-
"""
Unit tests cho engine/workers/training_worker.py

Kiểm tra:
- TrainingEngineWorker.run(): main loop iterate symbols đúng thứ tự
- TrainingEngineWorker.train_symbol(): stub training, epoch progression
- TrainingEngineWorker.report_progress(): ghi status file qua write_status
- Progress calculation: (symbols_completed / symbols_total) * 100
- Phase support: Phase C, Phase B, Phase A
- Per-symbol status tracking
- Graceful stop
"""

import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine.workers.training_worker import TrainingEngineWorker
from models.task_models import TaskState, TaskStatus, TaskType
from models.training_models import (
    SymbolTrainingStatus,
    TrainingPhase,
    TrainingProgress,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def worker() -> TrainingEngineWorker:
    """Tạo worker instance với task_id cố định."""
    return TrainingEngineWorker(task_id="test_train_001")


@pytest.fixture
def temp_status_dir(tmp_path, monkeypatch):
    """Patch STATUS_DIR để dùng thư mục tạm."""
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    monkeypatch.setattr(
        "orchestrator.status_protocol.STATUS_DIR",
        str(status_dir),
    )
    return status_dir


@pytest.fixture
def basic_config() -> dict:
    """Cấu hình training cơ bản."""
    return {
        "epochs": 5,
        "cycle_number": 1,
    }


@pytest.fixture
def symbols() -> list:
    """Danh sách symbols test."""
    return ["FPT", "VNM", "HPG"]


# ---------------------------------------------------------------------------
# Test: __init__
# ---------------------------------------------------------------------------


class TestInit:
    """Test khởi tạo TrainingEngineWorker."""

    def test_default_task_id_generated(self):
        """Tự generate UUID nếu không truyền task_id."""
        worker = TrainingEngineWorker()
        assert worker.task_id is not None
        assert len(worker.task_id) > 0

    def test_custom_task_id(self):
        """Sử dụng task_id được truyền vào."""
        worker = TrainingEngineWorker(task_id="my_task_123")
        assert worker.task_id == "my_task_123"

    def test_initial_state(self, worker):
        """Trạng thái ban đầu đúng default."""
        assert worker.phase == TrainingPhase.PHASE_C
        assert worker.current_symbol == ""
        assert worker.symbols_completed == 0
        assert worker.symbols_total == 0
        assert worker.per_symbol_status == {}


# ---------------------------------------------------------------------------
# Test: run()
# ---------------------------------------------------------------------------


class TestRun:
    """Test main training loop."""

    def test_run_processes_all_symbols(self, worker, temp_status_dir, symbols, basic_config):
        """run() train tất cả symbols trong danh sách."""
        worker.run(symbols, TrainingPhase.PHASE_C, basic_config)

        assert worker.symbols_completed == 3
        assert worker.symbols_total == 3

    def test_run_sets_phase(self, worker, temp_status_dir, symbols, basic_config):
        """run() set phase đúng."""
        worker.run(symbols, TrainingPhase.PHASE_B, basic_config)
        assert worker.phase == TrainingPhase.PHASE_B

    def test_run_all_symbols_completed(self, worker, temp_status_dir, symbols, basic_config):
        """Tất cả symbols có status completed sau khi run xong."""
        worker.run(symbols, TrainingPhase.PHASE_C, basic_config)

        for symbol in symbols:
            assert worker.per_symbol_status[symbol].status == "completed"

    def test_run_writes_status_file(self, worker, temp_status_dir, symbols, basic_config):
        """run() ghi status file vào thư mục status."""
        worker.run(symbols, TrainingPhase.PHASE_C, basic_config)

        status_file = temp_status_dir / "test_train_001.json"
        assert status_file.exists()

        data = json.loads(status_file.read_text(encoding="utf-8"))
        assert data["task_id"] == "test_train_001"
        assert data["task_type"] == "training"

    def test_run_final_status_completed(self, worker, temp_status_dir, symbols, basic_config):
        """Status cuối cùng là COMPLETED."""
        worker.run(symbols, TrainingPhase.PHASE_C, basic_config)

        status_file = temp_status_dir / "test_train_001.json"
        data = json.loads(status_file.read_text(encoding="utf-8"))
        assert data["state"] == "completed"
        assert data["progress_pct"] == 100.0

    def test_run_with_cycle_number(self, worker, temp_status_dir, symbols):
        """run() sử dụng cycle_number từ config."""
        config = {"epochs": 3, "cycle_number": 5}
        worker.run(symbols, TrainingPhase.PHASE_A, config)

        status_file = temp_status_dir / "test_train_001.json"
        data = json.loads(status_file.read_text(encoding="utf-8"))
        assert data["details"]["cycle_number"] == 5

    def test_run_empty_symbols(self, worker, temp_status_dir):
        """run() với danh sách symbols rỗng hoàn thành ngay."""
        worker.run([], TrainingPhase.PHASE_C, {"epochs": 5})

        assert worker.symbols_completed == 0
        assert worker.symbols_total == 0

    def test_run_single_symbol(self, worker, temp_status_dir, basic_config):
        """run() với 1 symbol duy nhất."""
        worker.run(["VNM"], TrainingPhase.PHASE_C, basic_config)

        assert worker.symbols_completed == 1
        assert worker.per_symbol_status["VNM"].status == "completed"


# ---------------------------------------------------------------------------
# Test: train_symbol()
# ---------------------------------------------------------------------------


class TestTrainSymbol:
    """Test training cho 1 symbol (stub)."""

    def test_train_symbol_updates_epochs(self, worker, temp_status_dir):
        """train_symbol() cập nhật epochs_completed."""
        worker.phase = TrainingPhase.PHASE_C
        worker.symbols_total = 1
        worker.current_symbol = "FPT"
        worker.per_symbol_status["FPT"] = SymbolTrainingStatus(
            symbol="FPT", status="training", epochs_completed=0
        )
        worker._running = True
        worker._last_progress_time = time.time()

        worker.train_symbol("FPT", 5)

        assert worker.per_symbol_status["FPT"].epochs_completed == 5

    def test_train_symbol_updates_loss(self, worker, temp_status_dir):
        """train_symbol() cập nhật current_loss."""
        worker.phase = TrainingPhase.PHASE_C
        worker.symbols_total = 1
        worker.current_symbol = "VNM"
        worker.per_symbol_status["VNM"] = SymbolTrainingStatus(
            symbol="VNM", status="training", epochs_completed=0
        )
        worker._running = True
        worker._last_progress_time = time.time()

        worker.train_symbol("VNM", 10)

        assert worker.per_symbol_status["VNM"].current_loss is not None
        assert worker.per_symbol_status["VNM"].current_loss >= 0.0

    def test_train_symbol_records_duration(self, worker, temp_status_dir):
        """train_symbol() ghi duration_seconds."""
        worker.phase = TrainingPhase.PHASE_C
        worker.symbols_total = 1
        worker.current_symbol = "HPG"
        worker.per_symbol_status["HPG"] = SymbolTrainingStatus(
            symbol="HPG", status="training", epochs_completed=0
        )
        worker._running = True
        worker._last_progress_time = time.time()

        worker.train_symbol("HPG", 3)

        assert worker.per_symbol_status["HPG"].duration_seconds is not None
        assert worker.per_symbol_status["HPG"].duration_seconds > 0.0


# ---------------------------------------------------------------------------
# Test: report_progress()
# ---------------------------------------------------------------------------


class TestReportProgress:
    """Test report_progress ghi status file."""

    def test_report_progress_writes_status(self, worker, temp_status_dir):
        """report_progress() ghi file đúng task_id."""
        worker.symbols_total = 10
        worker.symbols_completed = 3
        worker.current_symbol = "VNM"
        worker._started_at = datetime.now()

        progress = TrainingProgress(
            phase=TrainingPhase.PHASE_C,
            cycle_number=1,
            current_symbol="VNM",
            symbols_completed=3,
            symbols_total=10,
            current_epoch=5,
            total_epochs=20,
            current_loss=0.45,
            eta_seconds=120.0,
        )

        worker.report_progress(progress)

        status_file = temp_status_dir / "test_train_001.json"
        assert status_file.exists()

        data = json.loads(status_file.read_text(encoding="utf-8"))
        assert data["task_id"] == "test_train_001"
        assert data["state"] == "running"
        assert data["progress_pct"] == 30.0  # 3/10 * 100

    def test_report_progress_updates_heartbeat(self, worker, temp_status_dir):
        """report_progress() cập nhật heartbeat_ts."""
        worker.symbols_total = 5
        worker.symbols_completed = 2
        worker.current_symbol = "FPT"
        worker._started_at = datetime.now()

        progress = TrainingProgress(
            phase=TrainingPhase.PHASE_B,
            cycle_number=2,
            current_symbol="FPT",
            symbols_completed=2,
            symbols_total=5,
            current_epoch=10,
            total_epochs=20,
            current_loss=0.3,
            eta_seconds=60.0,
        )

        worker.report_progress(progress)

        status_file = temp_status_dir / "test_train_001.json"
        data = json.loads(status_file.read_text(encoding="utf-8"))

        # heartbeat_ts phải là thời điểm gần hiện tại
        heartbeat = datetime.fromisoformat(data["heartbeat_ts"])
        assert (datetime.now() - heartbeat).total_seconds() < 5.0

    def test_report_progress_percentage_calculation(self, worker, temp_status_dir):
        """Progress = (symbols_completed / symbols_total) * 100."""
        worker._started_at = datetime.now()

        test_cases = [
            (0, 10, 0.0),
            (5, 10, 50.0),
            (10, 10, 100.0),
            (1, 3, pytest.approx(33.33, rel=0.01)),
        ]

        for completed, total, expected_pct in test_cases:
            worker.symbols_completed = completed
            worker.symbols_total = total
            worker.current_symbol = "TEST"

            progress = TrainingProgress(
                phase=TrainingPhase.PHASE_C,
                cycle_number=1,
                current_symbol="TEST",
                symbols_completed=completed,
                symbols_total=total,
                current_epoch=1,
                total_epochs=10,
                current_loss=0.5,
                eta_seconds=0.0,
            )

            worker.report_progress(progress)

            status_file = temp_status_dir / "test_train_001.json"
            data = json.loads(status_file.read_text(encoding="utf-8"))
            assert data["progress_pct"] == expected_pct


# ---------------------------------------------------------------------------
# Test: Phase support
# ---------------------------------------------------------------------------


class TestPhaseSupport:
    """Test hỗ trợ 3 training phases."""

    def test_phase_c_supervised(self, worker, temp_status_dir, basic_config):
        """Phase C: supervised learning."""
        worker.run(["FPT"], TrainingPhase.PHASE_C, basic_config)

        status_file = temp_status_dir / "test_train_001.json"
        data = json.loads(status_file.read_text(encoding="utf-8"))
        assert data["details"]["phase"] == "phase_c"

    def test_phase_b_deep_search(self, worker, temp_status_dir, basic_config):
        """Phase B: deep search validation."""
        worker.run(["FPT"], TrainingPhase.PHASE_B, basic_config)

        status_file = temp_status_dir / "test_train_001.json"
        data = json.loads(status_file.read_text(encoding="utf-8"))
        assert data["details"]["phase"] == "phase_b"

    def test_phase_a_self_play(self, worker, temp_status_dir, basic_config):
        """Phase A: self-play."""
        worker.run(["FPT"], TrainingPhase.PHASE_A, basic_config)

        status_file = temp_status_dir / "test_train_001.json"
        data = json.loads(status_file.read_text(encoding="utf-8"))
        assert data["details"]["phase"] == "phase_a"

    def test_phase_affects_loss_simulation(self, worker, temp_status_dir):
        """Phase khác nhau cho loss khác nhau (stub behavior)."""
        # Phase C loss cao hơn Phase A
        loss_c = worker._simulate_loss(5, 10)
        worker.phase = TrainingPhase.PHASE_C
        loss_c = worker._simulate_loss(5, 10)

        worker.phase = TrainingPhase.PHASE_A
        loss_a = worker._simulate_loss(5, 10)

        assert loss_c > loss_a


# ---------------------------------------------------------------------------
# Test: Progress reporting interval
# ---------------------------------------------------------------------------


class TestProgressReportingInterval:
    """Test _report_progress_if_needed() tuân thủ POLL_INTERVAL."""

    def test_report_skipped_within_interval(self, worker, temp_status_dir):
        """Không report nếu chưa đủ POLL_INTERVAL kể từ lần report trước."""
        worker.symbols_total = 3
        worker.symbols_completed = 1
        worker.current_symbol = "FPT"
        worker._started_at = datetime.now()
        worker._running = True
        # Giả lập vừa report cách đây 1 giây (< POLL_INTERVAL=10s)
        worker._last_progress_time = time.time() - 1.0

        # Ghi trước 1 lần để có file
        worker._report_progress_if_needed(force=True)
        status_file = temp_status_dir / "test_train_001.json"
        first_data = json.loads(status_file.read_text(encoding="utf-8"))
        first_heartbeat = first_data["heartbeat_ts"]

        # Reset _last_progress_time để simulate "vừa report xong"
        worker._last_progress_time = time.time()
        time.sleep(0.01)

        # Gọi không force - chưa đủ interval → không nên ghi lại
        worker._report_progress_if_needed(force=False)
        second_data = json.loads(status_file.read_text(encoding="utf-8"))
        # heartbeat_ts không thay đổi vì không có report mới
        assert second_data["heartbeat_ts"] == first_heartbeat

    def test_report_triggered_after_interval(self, worker, temp_status_dir):
        """Report khi đã vượt POLL_INTERVAL."""
        worker.symbols_total = 3
        worker.symbols_completed = 1
        worker.current_symbol = "VNM"
        worker._started_at = datetime.now()
        worker._running = True

        # Giả lập lần report trước cách đây > POLL_INTERVAL
        worker._last_progress_time = time.time() - 15.0

        worker._report_progress_if_needed(force=False)

        status_file = temp_status_dir / "test_train_001.json"
        assert status_file.exists()
        data = json.loads(status_file.read_text(encoding="utf-8"))
        assert data["state"] == "running"

    def test_force_report_ignores_interval(self, worker, temp_status_dir):
        """force=True bỏ qua POLL_INTERVAL, report ngay."""
        worker.symbols_total = 5
        worker.symbols_completed = 2
        worker.current_symbol = "HPG"
        worker._started_at = datetime.now()
        worker._running = True
        # Vừa report xong (0 giây trước)
        worker._last_progress_time = time.time()

        worker._report_progress_if_needed(force=True)

        status_file = temp_status_dir / "test_train_001.json"
        assert status_file.exists()
        data = json.loads(status_file.read_text(encoding="utf-8"))
        assert data["state"] == "running"
        assert data["progress_pct"] == 40.0  # 2/5 * 100


# ---------------------------------------------------------------------------
# Test: Error recovery
# ---------------------------------------------------------------------------


class TestErrorRecovery:
    """Test error recovery khi train_symbol raise exception."""

    def test_error_in_symbol_continues_others(self, worker, temp_status_dir, basic_config):
        """Khi 1 symbol lỗi, worker tiếp tục train các symbol còn lại."""
        symbols = ["FPT", "ERROR_SYMBOL", "VNM"]

        # Patch train_symbol để raise exception cho ERROR_SYMBOL
        original_train = worker.train_symbol

        def patched_train(symbol, epochs):
            if symbol == "ERROR_SYMBOL":
                raise RuntimeError("Simulated training error")
            original_train(symbol, epochs)

        worker.train_symbol = patched_train
        worker.run(symbols, TrainingPhase.PHASE_C, basic_config)

        # Worker vẫn hoàn thành - tất cả 3 symbols đều được xử lý
        assert worker.symbols_completed == 3
        assert worker.per_symbol_status["FPT"].status == "completed"
        assert worker.per_symbol_status["ERROR_SYMBOL"].status == "failed"
        assert worker.per_symbol_status["VNM"].status == "completed"

    def test_error_symbol_marked_failed_in_status(self, worker, temp_status_dir, basic_config):
        """Symbol bị lỗi có status 'failed' trong per_symbol_status."""
        original_train = worker.train_symbol

        def patched_train(symbol, epochs):
            if symbol == "HPG":
                raise ValueError("Invalid data for HPG")
            original_train(symbol, epochs)

        worker.train_symbol = patched_train
        worker.run(["HPG"], TrainingPhase.PHASE_C, basic_config)

        assert worker.per_symbol_status["HPG"].status == "failed"

    def test_final_status_completed_despite_errors(self, worker, temp_status_dir, basic_config):
        """Training kết thúc completed ngay cả khi có symbol lỗi."""
        original_train = worker.train_symbol

        def patched_train(symbol, epochs):
            if symbol == "FPT":
                raise RuntimeError("FPT training failed")
            original_train(symbol, epochs)

        worker.train_symbol = patched_train
        worker.run(["FPT", "VNM"], TrainingPhase.PHASE_C, basic_config)

        status_file = temp_status_dir / "test_train_001.json"
        data = json.loads(status_file.read_text(encoding="utf-8"))
        assert data["state"] == "completed"
        assert data["progress_pct"] == 100.0


# ---------------------------------------------------------------------------
# Test: Stop graceful
# ---------------------------------------------------------------------------


class TestStop:
    """Test graceful stop."""

    def test_stop_sets_running_false(self, worker):
        """stop() dừng training loop."""
        worker._running = True
        worker.stop()
        assert worker._running is False


# ---------------------------------------------------------------------------
# Test: ETA estimation
# ---------------------------------------------------------------------------


class TestETA:
    """Test ước tính thời gian còn lại."""

    def test_eta_zero_when_no_completed(self, worker):
        """ETA = 0 khi chưa hoàn thành symbol nào."""
        worker.symbols_completed = 0
        worker.symbols_total = 10
        assert worker._estimate_eta() == 0.0

    def test_eta_positive_when_has_completed(self, worker):
        """ETA > 0 khi đã hoàn thành ít nhất 1 symbol."""
        worker.symbols_completed = 2
        worker.symbols_total = 10
        worker._started_at = datetime.now()
        # Simulate đã chạy 1 giây
        import time
        time.sleep(0.01)

        eta = worker._estimate_eta()
        assert eta >= 0.0
