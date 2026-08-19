# -*- coding: utf-8 -*-
"""
Integration test: Auto-Learning Cycle end-to-end.

Kiểm tra luồng auto-learning hoàn chỉnh:
- backtest → analyze mistakes → generate hard examples → phase transition → persist cycle

Sử dụng BacktestEngineWorker với data thực (CSV files) hoặc mock data,
AutoLearnerWiring để wire các component thật lại với nhau.

Requirements: 3.1 (auto-learning cycle)
"""

import os
from datetime import date, datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from engine.auto_learner_wiring import AutoLearnerWiring
from engine.cycle_history import load_all_cycles, persist_cycle
from engine.mistake_analyzer import (
    generate_hard_examples,
    identify_incorrect_predictions,
)
from engine.workers.backtest_worker import BacktestEngineWorker
from engine.workers.phase_transition import check_phase_transition
from models.backtest_models import AutoBacktestResult, BacktestResult, Trade
from models.data_models import AutoLearnerConfig, CycleResult
from models.training_models import TrainingPhase


def _make_trades_with_losses() -> List[Trade]:
    """Tạo danh sách trades test với cả winning và losing."""
    return [
        Trade(
            symbol="FPT",
            buy_date=date(2024, 1, 5),
            sell_date=date(2024, 1, 15),
            buy_price=80.0,
            sell_price=85.0,
            shares=100,
            pnl=500_000.0,
            pnl_pct=0.0625,
            holding_days=10,
        ),
        Trade(
            symbol="VNM",
            buy_date=date(2024, 1, 8),
            sell_date=date(2024, 1, 18),
            buy_price=70.0,
            sell_price=65.0,
            shares=200,
            pnl=-1_000_000.0,
            pnl_pct=-0.0714,
            holding_days=10,
        ),
        Trade(
            symbol="HPG",
            buy_date=date(2024, 1, 10),
            sell_date=date(2024, 1, 20),
            buy_price=25.0,
            sell_price=23.0,
            shares=400,
            pnl=-800_000.0,
            pnl_pct=-0.08,
            holding_days=10,
        ),
        Trade(
            symbol="MWG",
            buy_date=date(2024, 1, 12),
            sell_date=date(2024, 1, 22),
            buy_price=50.0,
            sell_price=55.0,
            shares=200,
            pnl=1_000_000.0,
            pnl_pct=0.10,
            holding_days=10,
        ),
    ]


class TestAutoLearningCycleEndToEnd:
    """Test luồng auto-learning: backtest → analyze → retrain."""

    def test_full_cycle_flow_with_mock_backtest(self, tmp_path: Path) -> None:
        """
        Full cycle: backtest → mistake analysis → hard examples → persist.

        Dùng mock cho BacktestEngineWorker.run_auto() để trả về trades có cả
        winning và losing, verify phần analyze và persist hoạt động đúng.
        """
        history_dir = str(tmp_path / "history")
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Mock backtest result với trades hỗn hợp
        mock_trades = _make_trades_with_losses()
        mock_auto_result = AutoBacktestResult(
            cycle_number=1,
            symbols_tested=["FPT", "VNM", "HPG", "MWG"],
            overall_sharpe=1.2,
            overall_win_rate=50.0,
            overall_return=5.0,
            strategies_beaten=2,
            benchmark_results=[0.8, 1.0, 1.5, 0.5],
            trades=mock_trades,
            duration_seconds=10.0,
        )

        # Tạo wiring với mock backtest worker
        mock_worker = MagicMock(spec=BacktestEngineWorker)
        mock_worker.run_auto.return_value = mock_auto_result

        wiring = AutoLearnerWiring(
            config=AutoLearnerConfig(cycle_interval_hours=24.0),
            backtest_worker=mock_worker,
            history_dir=history_dir,
            data_dir=data_dir,
        )

        # Chạy full cycle
        result = wiring.run_full_cycle(symbols=["FPT", "VNM", "HPG", "MWG"])

        # Verify cycle result
        assert result.cycle_number == 1
        assert result.phase == TrainingPhase.PHASE_C.value
        assert result.sharpe_ratio == 1.2
        assert result.win_rate == 50.0
        assert result.duration_seconds >= 0.0

        # Verify cycle được persist vào history
        cycles = load_all_cycles(history_dir)
        assert len(cycles) == 1
        assert cycles[0].cycle_number == 1
        assert cycles[0].sharpe_ratio == 1.2

    def test_mistake_analysis_identifies_losing_trades(self) -> None:
        """MistakeAnalyzer nhận dạng đúng trades có pnl < 0."""
        trades = _make_trades_with_losses()

        # Phân tích mistakes
        incorrect = identify_incorrect_predictions(trades)

        # 2 trades lỗ: VNM (pnl=-1M) và HPG (pnl=-800k)
        assert len(incorrect) == 2
        symbols_with_loss = {t.symbol for t in incorrect}
        assert "VNM" in symbols_with_loss
        assert "HPG" in symbols_with_loss

        # Trades winning không bị đánh dấu
        assert "FPT" not in symbols_with_loss
        assert "MWG" not in symbols_with_loss

    def test_hard_examples_generated_from_mistakes(self) -> None:
        """Hard examples được tạo đúng từ incorrect trades."""
        trades = _make_trades_with_losses()
        incorrect = identify_incorrect_predictions(trades)
        hard_examples = generate_hard_examples(incorrect)

        # Mỗi incorrect trade tạo 1 hard example
        assert len(hard_examples) == 2

        # Verify cấu trúc hard example
        for ex in hard_examples:
            assert "symbol" in ex
            assert "buy_date" in ex
            assert "sell_date" in ex
            assert "pnl" in ex
            assert "outcome" in ex
            assert ex["outcome"] == "loss"
            assert ex["pnl"] < 0

    def test_multiple_cycles_track_improvement(self, tmp_path: Path) -> None:
        """Nhiều cycles liên tiếp → detect improvement trend."""
        history_dir = str(tmp_path / "history")
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Tạo wiring
        config = AutoLearnerConfig(
            cycle_interval_hours=24.0,
            min_improvement_cycles=3,
        )

        # Chạy 4 cycles với Sharpe ratio tăng dần
        sharpe_values = [0.5, 0.8, 1.1, 1.5]

        for i, sharpe in enumerate(sharpe_values):
            mock_trades = [
                Trade(
                    symbol="FPT",
                    buy_date=date(2024, 1, 5),
                    sell_date=date(2024, 1, 15),
                    buy_price=80.0,
                    sell_price=85.0 + i * 2,
                    shares=100,
                    pnl=500_000.0 + i * 100_000.0,
                    pnl_pct=0.05 + i * 0.01,
                    holding_days=10,
                ),
            ]

            mock_result = AutoBacktestResult(
                cycle_number=i + 1,
                symbols_tested=["FPT"],
                overall_sharpe=sharpe,
                overall_win_rate=60.0 + i * 5,
                overall_return=5.0 + i * 2,
                strategies_beaten=min(i + 1, 4),
                benchmark_results=[0.3, 0.6, 0.9, 1.2],
                trades=mock_trades,
                duration_seconds=5.0,
            )

            mock_worker = MagicMock(spec=BacktestEngineWorker)
            mock_worker.run_auto.return_value = mock_result

            wiring = AutoLearnerWiring(
                config=config,
                backtest_worker=mock_worker,
                history_dir=history_dir,
                data_dir=data_dir,
            )

            wiring.run_full_cycle(symbols=["FPT"])

        # Verify improvement trend detected (3+ cycles increasing Sharpe)
        wiring_check = AutoLearnerWiring(
            config=config,
            history_dir=history_dir,
            data_dir=data_dir,
        )
        assert wiring_check.get_improvement_trend() is True

    def test_phase_transition_triggered_after_criteria_met(
        self, tmp_path: Path
    ) -> None:
        """Phase transition C→B xảy ra khi criteria được đáp ứng."""
        history_dir = str(tmp_path / "history")
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = AutoLearnerConfig(
            cycle_interval_hours=24.0,
            phase_c_sharpe_threshold=2,
        )

        # Tạo 3 cycles liên tiếp với strategies_beaten >= 2 (điều kiện Phase C→B)
        for i in range(3):
            mock_result = AutoBacktestResult(
                cycle_number=i + 1,
                symbols_tested=["FPT"],
                overall_sharpe=1.5 + i * 0.1,
                overall_win_rate=60.0,
                overall_return=8.0,
                strategies_beaten=3,  # Beat 3/4 strategies
                benchmark_results=[0.5, 1.0, 1.2, 2.0],
                trades=[],
                duration_seconds=5.0,
            )

            mock_worker = MagicMock(spec=BacktestEngineWorker)
            mock_worker.run_auto.return_value = mock_result

            wiring = AutoLearnerWiring(
                config=config,
                backtest_worker=mock_worker,
                history_dir=history_dir,
                data_dir=data_dir,
            )

            result = wiring.run_full_cycle(symbols=["FPT"])

        # Sau 3 cycles đáp ứng criteria → phase chuyển sang PHASE_B
        assert wiring.get_current_phase() == TrainingPhase.PHASE_B

    def test_manual_backtest_incorporated_into_cycle(self, tmp_path: Path) -> None:
        """Manual backtest results được merge vào auto-learning cycle."""
        history_dir = str(tmp_path / "history")
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Tạo manual backtest result
        manual_result = BacktestResult(
            symbol="TCB",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=100_000_000.0,
            final_capital=105_000_000.0,
            total_return=5.0,
            sharpe_ratio=1.3,
            win_rate=55.0,
            max_drawdown=8.0,
            total_trades=3,
            trades=[
                Trade(
                    symbol="TCB",
                    buy_date=date(2024, 1, 10),
                    sell_date=date(2024, 1, 25),
                    buy_price=30.0,
                    sell_price=28.0,
                    shares=300,
                    pnl=-600_000.0,
                    pnl_pct=-0.0667,
                    holding_days=15,
                ),
            ],
        )

        # Mock auto backtest (không có trades)
        mock_auto_result = AutoBacktestResult(
            cycle_number=1,
            symbols_tested=["FPT"],
            overall_sharpe=1.0,
            overall_win_rate=50.0,
            overall_return=3.0,
            strategies_beaten=1,
            benchmark_results=[0.5, 0.8, 1.2, 1.5],
            trades=[],
            duration_seconds=5.0,
        )

        mock_worker = MagicMock(spec=BacktestEngineWorker)
        mock_worker.run_auto.return_value = mock_auto_result

        wiring = AutoLearnerWiring(
            config=AutoLearnerConfig(),
            backtest_worker=mock_worker,
            history_dir=history_dir,
            data_dir=data_dir,
        )

        # Incorporate manual backtest
        wiring.incorporate_manual_backtest(manual_result)
        assert len(wiring.manual_backtest_queue) == 1

        # Chạy cycle → manual trades sẽ được merge vào analysis
        result = wiring.run_full_cycle(symbols=["FPT"])

        # Queue đã được clear sau cycle
        assert len(wiring.manual_backtest_queue) == 0

        # Cycle result persisted
        cycles = load_all_cycles(history_dir)
        assert len(cycles) == 1

    def test_cycle_history_persistence_across_instances(self, tmp_path: Path) -> None:
        """Cycle history persist trên disk, có thể đọc lại từ instance mới."""
        history_dir = str(tmp_path / "history")

        # Persist 3 cycles manually
        for i in range(1, 4):
            cycle = CycleResult(
                cycle_number=i,
                phase="phase_c",
                sharpe_ratio=0.5 * i,
                win_rate=50.0 + i * 5,
                total_return=3.0 + i * 2,
                strategies_beaten=i,
                validation_loss=0.5 - i * 0.1,
                is_improving=True,
                timestamp=datetime.now(),
                duration_seconds=10.0,
                notes=f"Cycle {i}",
            )
            persist_cycle(cycle, history_dir=history_dir)

        # Load từ disk bằng instance mới
        loaded_cycles = load_all_cycles(history_dir)

        assert len(loaded_cycles) == 3
        assert loaded_cycles[0].cycle_number == 1
        assert loaded_cycles[1].cycle_number == 2
        assert loaded_cycles[2].cycle_number == 3
        assert loaded_cycles[2].sharpe_ratio == 1.5

    def test_wiring_restores_state_from_history(self, tmp_path: Path) -> None:
        """AutoLearnerWiring restore cycle_count và phase từ history trên disk."""
        history_dir = str(tmp_path / "history")
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Persist cycle #5 ở phase_b
        cycle = CycleResult(
            cycle_number=5,
            phase="phase_b",
            sharpe_ratio=2.0,
            win_rate=65.0,
            total_return=12.0,
            strategies_beaten=3,
            validation_loss=0.1,
            is_improving=True,
            timestamp=datetime.now(),
            duration_seconds=30.0,
        )
        persist_cycle(cycle, history_dir=history_dir)

        # Tạo wiring mới → phải restore state từ disk
        wiring = AutoLearnerWiring(
            config=AutoLearnerConfig(),
            history_dir=history_dir,
            data_dir=data_dir,
        )

        assert wiring.get_cycle_count() == 5
        assert wiring.get_current_phase() == TrainingPhase.PHASE_B

    def test_is_cycle_due_respects_interval(self, tmp_path: Path) -> None:
        """is_cycle_due() trả về True khi chưa có cycle nào chạy."""
        history_dir = str(tmp_path / "history")
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        wiring = AutoLearnerWiring(
            config=AutoLearnerConfig(cycle_interval_hours=24.0),
            history_dir=history_dir,
            data_dir=data_dir,
        )

        # Chưa chạy cycle nào → due
        assert wiring.is_cycle_due() is True
