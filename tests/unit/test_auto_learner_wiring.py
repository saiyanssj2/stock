# -*- coding: utf-8 -*-
"""
Unit tests cho engine/auto_learner_wiring.py.

Kiểm tra AutoLearnerWiring:
- run_full_cycle(): chạy full cycle kết nối backtest → mistakes → hard examples → retrain
- incorporate_manual_backtest(): manual backtest feed back vào cycle
- Phase transition logic triggered sau mỗi cycle
- Persistent cycle history
- State restoration

References: Req 3.1, 3.6, 7.3
"""

import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from engine.auto_learner_wiring import AutoLearnerWiring
from engine.workers.backtest_worker import BacktestEngineWorker
from models.backtest_models import (
    AutoBacktestResult,
    BacktestResult,
    Trade,
)
from models.data_models import AutoLearnerConfig, CycleResult
from models.training_models import TrainingPhase


# --- Helpers ---


def make_trade(
    symbol: str = "FPT",
    pnl: float = 100.0,
    buy_price: float = 50.0,
    sell_price: float = 51.0,
) -> Trade:
    """Tạo Trade cho test."""
    return Trade(
        symbol=symbol,
        buy_date=date(2024, 1, 1),
        sell_date=date(2024, 1, 5),
        buy_price=buy_price,
        sell_price=sell_price,
        shares=100,
        pnl=pnl,
        pnl_pct=(pnl / (buy_price * 100)) if buy_price > 0 else 0.0,
        holding_days=4,
    )


def make_auto_backtest_result(
    trades: List[Trade] = None,
    sharpe: float = 1.0,
    win_rate: float = 60.0,
    strategies_beaten: int = 2,
) -> AutoBacktestResult:
    """Tạo AutoBacktestResult giả lập cho test."""
    if trades is None:
        trades = [make_trade(pnl=100.0), make_trade(pnl=-50.0)]
    return AutoBacktestResult(
        cycle_number=1,
        symbols_tested=["FPT", "VNM"],
        overall_sharpe=sharpe,
        overall_win_rate=win_rate,
        overall_return=5.0,
        strategies_beaten=strategies_beaten,
        benchmark_results=[0.5, 0.8, -0.2, 0.3],
        trades=trades,
        duration_seconds=2.0,
    )


def make_backtest_result(trades: List[Trade] = None) -> BacktestResult:
    """Tạo BacktestResult cho test."""
    if trades is None:
        trades = [make_trade(pnl=200.0), make_trade(pnl=-100.0)]
    return BacktestResult(
        symbol="FPT",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
        initial_capital=100_000_000.0,
        final_capital=105_000_000.0,
        total_return=5.0,
        sharpe_ratio=1.2,
        win_rate=60.0,
        max_drawdown=-5.0,
        total_trades=len(trades),
        trades=trades,
        equity_curve=[100_000_000.0, 105_000_000.0],
    )


# --- Tests cho run_full_cycle ---


class TestRunFullCycle:
    """Tests cho run_full_cycle() - end-to-end wiring."""

    def test_returns_cycle_result(self, tmp_path):
        """run_full_cycle() trả về CycleResult."""
        mock_worker = MagicMock(spec=BacktestEngineWorker)
        mock_worker.run_auto.return_value = make_auto_backtest_result()

        wiring = AutoLearnerWiring(
            backtest_worker=mock_worker,
            history_dir=str(tmp_path / "history"),
        )
        result = wiring.run_full_cycle(symbols=["FPT", "VNM"])

        assert isinstance(result, CycleResult)

    def test_calls_backtest_worker_run_auto(self, tmp_path):
        """run_full_cycle() phải gọi BacktestEngineWorker.run_auto()."""
        mock_worker = MagicMock(spec=BacktestEngineWorker)
        mock_worker.run_auto.return_value = make_auto_backtest_result()

        wiring = AutoLearnerWiring(
            backtest_worker=mock_worker,
            history_dir=str(tmp_path / "history"),
        )
        wiring.run_full_cycle(symbols=["FPT", "VNM"])

        mock_worker.run_auto.assert_called_once_with(
            symbols=["FPT", "VNM"], cycle_number=1
        )

    def test_cycle_result_has_backtest_metrics(self, tmp_path):
        """CycleResult chứa metrics từ auto backtest."""
        mock_worker = MagicMock(spec=BacktestEngineWorker)
        mock_worker.run_auto.return_value = make_auto_backtest_result(
            sharpe=1.5, win_rate=65.0, strategies_beaten=3
        )

        wiring = AutoLearnerWiring(
            backtest_worker=mock_worker,
            history_dir=str(tmp_path / "history"),
        )
        result = wiring.run_full_cycle(symbols=["FPT"])

        assert result.sharpe_ratio == 1.5
        assert result.win_rate == 65.0
        assert result.strategies_beaten == 3

    def test_increments_cycle_count(self, tmp_path):
        """Mỗi lần run_full_cycle() tăng cycle_count."""
        mock_worker = MagicMock(spec=BacktestEngineWorker)
        mock_worker.run_auto.return_value = make_auto_backtest_result()

        wiring = AutoLearnerWiring(
            backtest_worker=mock_worker,
            history_dir=str(tmp_path / "history"),
        )

        wiring.run_full_cycle(symbols=["FPT"])
        assert wiring.get_cycle_count() == 1

        wiring.run_full_cycle(symbols=["FPT"])
        assert wiring.get_cycle_count() == 2

    def test_mistakes_analyzed_from_backtest_trades(self, tmp_path):
        """Mistakes được phân tích từ trades của auto backtest."""
        losing_trade = make_trade(pnl=-200.0)
        winning_trade = make_trade(pnl=300.0)

        mock_worker = MagicMock(spec=BacktestEngineWorker)
        mock_worker.run_auto.return_value = make_auto_backtest_result(
            trades=[losing_trade, winning_trade]
        )

        wiring = AutoLearnerWiring(
            backtest_worker=mock_worker,
            history_dir=str(tmp_path / "history"),
        )
        result = wiring.run_full_cycle(symbols=["FPT"])

        # 1 losing trade → 1 hard example
        assert "Hard examples: 1" in result.notes

    def test_persists_cycle_to_history(self, tmp_path):
        """Cycle result được lưu vào persistent history."""
        history_dir = str(tmp_path / "history")

        mock_worker = MagicMock(spec=BacktestEngineWorker)
        mock_worker.run_auto.return_value = make_auto_backtest_result()

        wiring = AutoLearnerWiring(
            backtest_worker=mock_worker,
            history_dir=history_dir,
        )
        wiring.run_full_cycle(symbols=["FPT"])

        # Kiểm tra file history tồn tại
        history_path = Path(history_dir)
        assert history_path.exists()
        json_files = list(history_path.glob("cycle_*.json"))
        assert len(json_files) == 1


# --- Tests cho incorporate_manual_backtest ---


class TestIncorporateManualBacktest:
    """Tests cho incorporate_manual_backtest() trong wiring."""

    def test_manual_backtest_feeds_into_next_cycle(self, tmp_path):
        """Manual backtest trades được merge vào auto backtest trong cycle."""
        manual_losing_trade = make_trade(symbol="VNM", pnl=-500.0)
        manual_result = make_backtest_result(trades=[manual_losing_trade])

        # Auto backtest có 1 winning trade
        auto_trades = [make_trade(pnl=200.0)]
        mock_worker = MagicMock(spec=BacktestEngineWorker)
        mock_worker.run_auto.return_value = make_auto_backtest_result(
            trades=auto_trades
        )

        wiring = AutoLearnerWiring(
            backtest_worker=mock_worker,
            history_dir=str(tmp_path / "history"),
        )

        # Incorporate manual backtest
        wiring.incorporate_manual_backtest(manual_result)

        # Run cycle → manual losing trade phải được detect
        result = wiring.run_full_cycle(symbols=["FPT"])

        # 1 losing trade from manual → 1 hard example
        assert "Hard examples: 1" in result.notes

    def test_manual_queue_cleared_after_cycle(self, tmp_path):
        """Manual backtest queue được xóa sau khi cycle hoàn thành."""
        mock_worker = MagicMock(spec=BacktestEngineWorker)
        mock_worker.run_auto.return_value = make_auto_backtest_result()

        wiring = AutoLearnerWiring(
            backtest_worker=mock_worker,
            history_dir=str(tmp_path / "history"),
        )

        manual_result = make_backtest_result()
        wiring.incorporate_manual_backtest(manual_result)
        assert len(wiring.manual_backtest_queue) == 1

        wiring.run_full_cycle(symbols=["FPT"])
        assert len(wiring.manual_backtest_queue) == 0


# --- Tests cho phase transition ---


class TestPhaseTransition:
    """Tests cho phase transition logic trong wiring."""

    def test_starts_at_phase_c(self, tmp_path):
        """Mặc định bắt đầu ở Phase C."""
        mock_worker = MagicMock(spec=BacktestEngineWorker)

        wiring = AutoLearnerWiring(
            backtest_worker=mock_worker,
            history_dir=str(tmp_path / "history"),
        )
        assert wiring.get_current_phase() == TrainingPhase.PHASE_C

    def test_phase_transition_triggered_after_criteria_met(self, tmp_path):
        """Phase C→B transition khi beat >= 2 strategies × 3 cycles liên tiếp."""
        history_dir = str(tmp_path / "history")

        mock_worker = MagicMock(spec=BacktestEngineWorker)
        # Mỗi cycle beat 3 strategies → đủ criteria cho Phase C→B
        mock_worker.run_auto.return_value = make_auto_backtest_result(
            strategies_beaten=3
        )

        wiring = AutoLearnerWiring(
            backtest_worker=mock_worker,
            history_dir=history_dir,
        )

        # Chạy 3 cycles liên tiếp beat >= 2 strategies
        for _ in range(3):
            wiring.run_full_cycle(symbols=["FPT"])

        # Sau 3 cycles với strategies_beaten=3 → phải transition sang Phase B
        assert wiring.get_current_phase() == TrainingPhase.PHASE_B

    def test_no_transition_without_meeting_criteria(self, tmp_path):
        """Không chuyển phase nếu chưa đủ criteria."""
        mock_worker = MagicMock(spec=BacktestEngineWorker)
        # Beat 1 strategy — không đủ threshold (cần >= 2)
        mock_worker.run_auto.return_value = make_auto_backtest_result(
            strategies_beaten=1
        )

        wiring = AutoLearnerWiring(
            backtest_worker=mock_worker,
            history_dir=str(tmp_path / "history"),
        )

        for _ in range(3):
            wiring.run_full_cycle(symbols=["FPT"])

        # Vẫn ở Phase C vì chưa đủ criteria
        assert wiring.get_current_phase() == TrainingPhase.PHASE_C


# --- Tests cho state restoration ---


class TestStateRestoration:
    """Tests cho restore state từ persistent history."""

    def test_restores_cycle_count_from_history(self, tmp_path):
        """Cycle count được restore từ history files."""
        history_dir = str(tmp_path / "history")

        # Chạy 2 cycles đầu
        mock_worker = MagicMock(spec=BacktestEngineWorker)
        mock_worker.run_auto.return_value = make_auto_backtest_result()

        wiring1 = AutoLearnerWiring(
            backtest_worker=mock_worker,
            history_dir=history_dir,
        )
        wiring1.run_full_cycle(symbols=["FPT"])
        wiring1.run_full_cycle(symbols=["FPT"])

        # Tạo instance mới → phải restore cycle_count = 2
        wiring2 = AutoLearnerWiring(
            backtest_worker=mock_worker,
            history_dir=history_dir,
        )
        assert wiring2.get_cycle_count() == 2

    def test_is_cycle_due_respects_interval(self, tmp_path):
        """is_cycle_due() hoạt động đúng với interval config."""
        mock_worker = MagicMock(spec=BacktestEngineWorker)
        mock_worker.run_auto.return_value = make_auto_backtest_result()

        wiring = AutoLearnerWiring(
            config=AutoLearnerConfig(cycle_interval_hours=24.0),
            backtest_worker=mock_worker,
            history_dir=str(tmp_path / "history"),
        )

        # Chưa chạy cycle nào → due
        assert wiring.is_cycle_due() is True

        # Chạy 1 cycle
        wiring.run_full_cycle(symbols=["FPT"])

        # Vừa chạy xong → not due
        assert wiring.is_cycle_due() is False
