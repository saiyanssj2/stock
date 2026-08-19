# -*- coding: utf-8 -*-
"""
Integration test: Training flow end-to-end.

Kiểm tra luồng training hoàn chỉnh:
- start → progress → checkpoint → resume

Sử dụng mock cho GPU operations và file I/O thực tế với tmp_path.

Requirements: 1.1 (parallel execution), 1.8 (checkpoint resume)
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from engine.workers.training_checkpoint import (
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from engine.workers.training_worker import TrainingEngineWorker
from models.task_models import TaskState, TaskStatus, TaskType
from models.training_models import SessionCheckpoint, TrainingPhase
from orchestrator.status_protocol import read_status, write_status


class TestTrainingFlowEndToEnd:
    """Test toàn bộ flow training: start → progress → checkpoint → resume."""

    def test_start_training_reports_progress(self, tmp_path: Path) -> None:
        """Start training → worker report progress qua status file."""
        # Thiết lập status dir tạm
        status_dir = tmp_path / "status"
        status_dir.mkdir()

        with patch("orchestrator.status_protocol.STATUS_DIR", str(status_dir)):
            with patch("engine.workers.training_worker.write_status") as mock_write:
                # Cấu hình worker
                worker = TrainingEngineWorker(task_id="train_test_001")
                symbols = ["FPT", "VNM", "HPG"]
                config = {"epochs": 3, "cycle_number": 1}

                # Chạy training
                worker.run(symbols=symbols, phase=TrainingPhase.PHASE_C, config=config)

                # Worker đã report progress nhiều lần
                assert mock_write.call_count > 0

                # Lần report cuối phải là COMPLETED
                last_call = mock_write.call_args_list[-1]
                last_status: TaskStatus = last_call[0][0]
                assert last_status.state == TaskState.COMPLETED
                assert last_status.progress_pct == 100.0

    def test_training_progress_increases_monotonically(self, tmp_path: Path) -> None:
        """Progress percentage tăng monotonic trong suốt training."""
        progress_values: list = []

        def capture_status(status: TaskStatus) -> None:
            """Capture mỗi lần write_status để kiểm tra progress."""
            progress_values.append(status.progress_pct)

        with patch(
            "engine.workers.training_worker.write_status", side_effect=capture_status
        ):
            worker = TrainingEngineWorker(task_id="train_mono_001")
            symbols = ["A", "B", "C"]
            config = {"epochs": 2, "cycle_number": 1}

            worker.run(symbols=symbols, phase=TrainingPhase.PHASE_C, config=config)

        # Progress phải non-decreasing
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1], (
                f"Progress giảm tại step {i}: "
                f"{progress_values[i - 1]} → {progress_values[i]}"
            )

        # Cuối cùng phải đạt 100%
        assert progress_values[-1] == 100.0

    def test_checkpoint_save_during_training(self, tmp_path: Path) -> None:
        """Checkpoint được save đúng format, có thể load lại."""
        checkpoint_dir = str(tmp_path / "checkpoints")

        # Tạo checkpoint giả lập như worker sẽ save
        checkpoint = SessionCheckpoint(
            session_id="session_001",
            phase=TrainingPhase.PHASE_C,
            cycle_number=1,
            completed_symbols=["FPT", "VNM"],
            pending_symbols=["HPG", "MWG"],
            current_symbol="HPG",
            current_epoch=5,
            total_epochs=10,
            model_path="models/phase_c_latest.pt",
            optimizer_state_path="models/optimizer_state.pt",
            created_at=datetime.now(),
            metadata={"batch_size": "32", "learning_rate": "0.001"},
        )

        # Save checkpoint
        saved_path = save_checkpoint(checkpoint, checkpoint_dir=checkpoint_dir)
        assert os.path.exists(saved_path)

        # Load checkpoint và verify round-trip
        loaded = load_checkpoint("session_001", checkpoint_dir=checkpoint_dir)
        assert loaded is not None
        assert loaded.session_id == checkpoint.session_id
        assert loaded.phase == TrainingPhase.PHASE_C
        assert loaded.completed_symbols == ["FPT", "VNM"]
        assert loaded.pending_symbols == ["HPG", "MWG"]
        assert loaded.current_epoch == 5
        assert loaded.total_epochs == 10

    def test_resume_from_checkpoint_skips_completed(self, tmp_path: Path) -> None:
        """Resume training từ checkpoint chỉ train pending symbols."""
        checkpoint_dir = str(tmp_path / "checkpoints")

        # Tạo checkpoint với 2 completed, 2 pending
        checkpoint = SessionCheckpoint(
            session_id="resume_session",
            phase=TrainingPhase.PHASE_C,
            cycle_number=2,
            completed_symbols=["FPT", "VNM"],
            pending_symbols=["HPG", "MWG"],
            current_symbol=None,
            current_epoch=0,
            total_epochs=5,
            created_at=datetime.now(),
        )

        save_checkpoint(checkpoint, checkpoint_dir=checkpoint_dir)

        # Load checkpoint
        loaded = load_checkpoint("resume_session", checkpoint_dir=checkpoint_dir)
        assert loaded is not None

        # Dùng pending symbols từ checkpoint cho worker
        trained_symbols: list = []

        def capture_status(status: TaskStatus) -> None:
            """Track symbols đã train qua progress messages."""
            if "completed" in status.message.lower():
                return
            details = status.details
            if details and "current_symbol" in details:
                sym = details["current_symbol"]
                if sym and sym not in trained_symbols:
                    trained_symbols.append(sym)

        with patch(
            "engine.workers.training_worker.write_status", side_effect=capture_status
        ):
            worker = TrainingEngineWorker(task_id="resume_worker")
            # Chỉ train pending symbols
            worker.run(
                symbols=loaded.pending_symbols,
                phase=loaded.phase,
                config={"epochs": 3, "cycle_number": loaded.cycle_number},
            )

        # Worker đã train pending symbols
        assert worker.symbols_completed == 2
        assert worker.symbols_total == 2

        # Không train completed symbols
        for sym in loaded.completed_symbols:
            assert sym not in worker.per_symbol_status or (
                worker.per_symbol_status.get(sym) is None
            )

    def test_full_training_lifecycle(self, tmp_path: Path) -> None:
        """
        Lifecycle hoàn chỉnh: start → progress → stop → checkpoint → resume.

        Mô phỏng scenario:
        1. Bắt đầu training với 4 symbols
        2. Training xong 2 symbols
        3. Crash/stop → save checkpoint
        4. Resume → chỉ train 2 symbols còn lại
        """
        checkpoint_dir = str(tmp_path / "checkpoints")
        all_symbols = ["FPT", "VNM", "HPG", "MWG"]

        # ---- Phase 1: Training bắt đầu và train xong 2 symbols ----
        statuses_phase1: list = []

        def capture_phase1(status: TaskStatus) -> None:
            statuses_phase1.append(status)

        with patch(
            "engine.workers.training_worker.write_status",
            side_effect=capture_phase1,
        ):
            worker1 = TrainingEngineWorker(task_id="lifecycle_001")
            # Chỉ train 2 symbols đầu (giả lập train partial trước khi crash)
            worker1.run(
                symbols=all_symbols[:2],
                phase=TrainingPhase.PHASE_C,
                config={"epochs": 3, "cycle_number": 1},
            )

        # Verify 2 symbols hoàn thành
        assert worker1.symbols_completed == 2

        # ---- Phase 2: Save checkpoint sau khi train partial ----
        checkpoint = SessionCheckpoint(
            session_id="lifecycle_001",
            phase=TrainingPhase.PHASE_C,
            cycle_number=1,
            completed_symbols=all_symbols[:2],
            pending_symbols=all_symbols[2:],
            current_epoch=0,
            total_epochs=3,
            created_at=datetime.now(),
        )
        save_checkpoint(checkpoint, checkpoint_dir=checkpoint_dir)

        # ---- Phase 3: Resume từ checkpoint ----
        loaded = load_checkpoint("lifecycle_001", checkpoint_dir=checkpoint_dir)
        assert loaded is not None
        assert loaded.pending_symbols == ["HPG", "MWG"]

        statuses_phase2: list = []

        def capture_phase2(status: TaskStatus) -> None:
            statuses_phase2.append(status)

        with patch(
            "engine.workers.training_worker.write_status",
            side_effect=capture_phase2,
        ):
            worker2 = TrainingEngineWorker(task_id="lifecycle_001_resumed")
            worker2.run(
                symbols=loaded.pending_symbols,
                phase=loaded.phase,
                config={"epochs": 3, "cycle_number": loaded.cycle_number},
            )

        # Verify worker2 chỉ train pending
        assert worker2.symbols_completed == 2
        assert worker2.symbols_total == 2

        # Verify toàn bộ lifecycle: 2 + 2 = 4 symbols trained
        total_trained = worker1.symbols_completed + worker2.symbols_completed
        assert total_trained == len(all_symbols)

        # ---- Phase 4: Cleanup checkpoint sau khi hoàn tất ----
        deleted = delete_checkpoint("lifecycle_001", checkpoint_dir=checkpoint_dir)
        assert deleted is True
        assert load_checkpoint("lifecycle_001", checkpoint_dir=checkpoint_dir) is None

    def test_training_per_symbol_status_tracking(self, tmp_path: Path) -> None:
        """Verify per-symbol status tracking: pending → training → completed."""
        with patch("engine.workers.training_worker.write_status"):
            worker = TrainingEngineWorker(task_id="status_track_001")
            symbols = ["A", "B", "C"]
            config = {"epochs": 2, "cycle_number": 1}

            worker.run(symbols=symbols, phase=TrainingPhase.PHASE_B, config=config)

        # Tất cả symbols phải có status = completed
        for sym in symbols:
            assert sym in worker.per_symbol_status
            assert worker.per_symbol_status[sym].status == "completed"
            assert worker.per_symbol_status[sym].epochs_completed == 2

    def test_training_phase_information_in_status(self, tmp_path: Path) -> None:
        """Status report bao gồm thông tin phase (C/B/A)."""
        reported_phases: list = []

        def capture_status(status: TaskStatus) -> None:
            if status.details and "phase" in status.details:
                reported_phases.append(status.details["phase"])

        with patch(
            "engine.workers.training_worker.write_status",
            side_effect=capture_status,
        ):
            worker = TrainingEngineWorker(task_id="phase_info_001")
            worker.run(
                symbols=["FPT"],
                phase=TrainingPhase.PHASE_B,
                config={"epochs": 2, "cycle_number": 3},
            )

        # Tất cả reports phải chứa đúng phase
        assert len(reported_phases) > 0
        for phase in reported_phases:
            assert phase == "phase_b"
