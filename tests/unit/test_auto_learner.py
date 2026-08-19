# -*- coding: utf-8 -*-
"""
Unit tests cho engine/auto_learner.py.

Kiểm tra AutoLearner:
- schedule_cycle(): lên lịch cycle tiếp theo
- run_cycle(): chạy 1 auto-learning cycle
- analyze_mistakes(): phân tích trades sai (PnL < 0)
- generate_hard_examples(): tạo hard examples từ mistakes
- incorporate_manual_backtest(): thêm manual backtest signal
- is_cycle_due(): kiểm tra đã đến lúc chạy cycle chưa
- get_cycle_interval_hours(): trả về interval config

References: Req 3.1, 3.2, 3.3, 7.3
"""

from datetime import date, datetime, timedelta

import pytest

from engine.auto_learner import AutoLearner
from models.backtest_models import BacktestResult, Trade
from models.data_models import AutoLearnerConfig, CycleResult
from models.training_models import TrainingPhase


# --- Helpers ---


def make_trade(
    symbol: str = "FPT",
    pnl: float = 100.0,
    buy_price: float = 50000.0,
    sell_price: float = 51000.0,
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
        pnl_pct=(pnl / (buy_price * 100)) * 100 if buy_price > 0 else 0.0,
        holding_days=4,
    )


def make_backtest_result(trades: list = None) -> BacktestResult:
    """Tạo BacktestResult cho test."""
    if trades is None:
        trades = [make_trade(pnl=100.0), make_trade(pnl=-50.0)]
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
        equity_curve=[100_000_000.0, 102_000_000.0, 105_000_000.0],
    )


# --- Tests cho __init__ ---


class TestAutoLearnerInit:
    """Tests cho khởi tạo AutoLearner."""

    def test_default_config(self):
        """Khởi tạo với config mặc định."""
        learner = AutoLearner()
        assert learner.config.cycle_interval_hours == 24.0
        assert learner._last_cycle_time is None
        assert learner._next_cycle_time is None
        assert learner._manual_backtest_results == []
        assert learner._cycle_history == []
        assert learner._current_phase == TrainingPhase.PHASE_C
        assert learner._cycle_count == 0

    def test_custom_config(self):
        """Khởi tạo với config tùy chỉnh."""
        config = AutoLearnerConfig(cycle_interval_hours=12.0)
        learner = AutoLearner(config=config)
        assert learner.config.cycle_interval_hours == 12.0


# --- Tests cho schedule_cycle ---


class TestScheduleCycle:
    """Tests cho schedule_cycle()."""

    def test_schedule_sets_next_cycle_time(self):
        """schedule_cycle() phải set _next_cycle_time."""
        learner = AutoLearner()
        assert learner._next_cycle_time is None

        learner.schedule_cycle()
        assert learner._next_cycle_time is not None

    def test_schedule_uses_config_interval(self):
        """next_cycle_time phải cách now khoảng cycle_interval_hours."""
        config = AutoLearnerConfig(cycle_interval_hours=6.0)
        learner = AutoLearner(config=config)

        before = datetime.now()
        learner.schedule_cycle()
        after = datetime.now()

        # next_cycle_time phải nằm trong khoảng [before + 6h, after + 6h]
        expected_min = before + timedelta(hours=6.0)
        expected_max = after + timedelta(hours=6.0)

        assert expected_min <= learner._next_cycle_time <= expected_max


# --- Tests cho analyze_mistakes ---


class TestAnalyzeMistakes:
    """Tests cho analyze_mistakes()."""

    def test_empty_trades_returns_empty(self):
        """Không có trades → không có mistakes."""
        learner = AutoLearner()
        result = learner.analyze_mistakes([])
        assert result == []

    def test_all_profitable_returns_empty(self):
        """Tất cả trades đều lời → không có mistakes."""
        learner = AutoLearner()
        trades = [
            make_trade(pnl=100.0),
            make_trade(pnl=50.0),
            make_trade(pnl=200.0),
        ]
        result = learner.analyze_mistakes(trades)
        assert result == []

    def test_identifies_negative_pnl_trades(self):
        """Trades có PnL < 0 được nhận diện là mistakes."""
        learner = AutoLearner()
        losing_trade = make_trade(pnl=-100.0)
        winning_trade = make_trade(pnl=200.0)
        trades = [winning_trade, losing_trade]

        result = learner.analyze_mistakes(trades)
        assert len(result) == 1
        assert result[0] is losing_trade

    def test_all_losing_returns_all(self):
        """Tất cả trades đều lỗ → trả về tất cả."""
        learner = AutoLearner()
        trades = [
            make_trade(pnl=-50.0),
            make_trade(pnl=-100.0),
            make_trade(pnl=-25.0),
        ]
        result = learner.analyze_mistakes(trades)
        assert len(result) == 3

    def test_zero_pnl_not_counted_as_mistake(self):
        """PnL = 0 không phải mistake (hòa vốn)."""
        learner = AutoLearner()
        trades = [make_trade(pnl=0.0)]
        result = learner.analyze_mistakes(trades)
        assert result == []

    def test_none_pnl_ignored(self):
        """Trades chưa đóng (pnl=None) bị bỏ qua."""
        learner = AutoLearner()
        open_trade = Trade(
            symbol="FPT",
            buy_date=date(2024, 1, 1),
            sell_date=None,
            buy_price=50000.0,
            sell_price=None,
            shares=100,
            pnl=None,
            holding_days=3,
        )
        result = learner.analyze_mistakes([open_trade])
        assert result == []


# --- Tests cho generate_hard_examples ---


class TestGenerateHardExamples:
    """Tests cho generate_hard_examples()."""

    def test_empty_input_returns_empty(self):
        """Không có trades sai → không tạo examples."""
        learner = AutoLearner()
        result = learner.generate_hard_examples([])
        assert result == []

    def test_generates_example_per_trade(self):
        """Mỗi trade sai tạo ra 1 hard example."""
        learner = AutoLearner()
        trades = [make_trade(pnl=-100.0), make_trade(pnl=-50.0)]
        result = learner.generate_hard_examples(trades)
        assert len(result) == 2

    def test_example_contains_required_fields(self):
        """Hard example chứa đầy đủ thông tin cần thiết."""
        learner = AutoLearner()
        trade = make_trade(symbol="VNM", pnl=-200.0, buy_price=80000.0)
        result = learner.generate_hard_examples([trade])

        assert len(result) == 1
        example = result[0]
        assert example["symbol"] == "VNM"
        assert example["buy_price"] == 80000.0
        assert example["pnl"] == -200.0
        assert example["outcome"] == "loss"

    def test_example_has_buy_date(self):
        """Hard example lưu buy_date dạng string."""
        learner = AutoLearner()
        trade = make_trade(pnl=-50.0)
        result = learner.generate_hard_examples([trade])
        assert result[0]["buy_date"] == "2024-01-01"


# --- Tests cho incorporate_manual_backtest ---


class TestIncorporateManualBacktest:
    """Tests cho incorporate_manual_backtest()."""

    def test_adds_result_to_list(self):
        """Kết quả backtest manual được thêm vào danh sách."""
        learner = AutoLearner()
        result = make_backtest_result()

        learner.incorporate_manual_backtest(result)
        assert len(learner._manual_backtest_results) == 1
        assert learner._manual_backtest_results[0] is result

    def test_multiple_results_accumulated(self):
        """Có thể incorporate nhiều backtest results."""
        learner = AutoLearner()
        result1 = make_backtest_result()
        result2 = make_backtest_result()

        learner.incorporate_manual_backtest(result1)
        learner.incorporate_manual_backtest(result2)
        assert len(learner._manual_backtest_results) == 2

    def test_manual_trades_used_in_run_cycle(self):
        """Trades từ manual backtest được dùng trong run_cycle()."""
        learner = AutoLearner()
        losing_trade = make_trade(pnl=-500.0)
        result = make_backtest_result(trades=[losing_trade])

        learner.incorporate_manual_backtest(result)
        cycle_result = learner.run_cycle()

        # Cycle phải nhận diện losing trade từ manual backtest
        assert "Hard examples generated: 1" in cycle_result.notes


# --- Tests cho get_cycle_interval_hours ---


class TestGetCycleIntervalHours:
    """Tests cho get_cycle_interval_hours()."""

    def test_returns_default_interval(self):
        """Mặc định trả về 24h."""
        learner = AutoLearner()
        assert learner.get_cycle_interval_hours() == 24.0

    def test_returns_custom_interval(self):
        """Trả về interval từ config tùy chỉnh."""
        config = AutoLearnerConfig(cycle_interval_hours=8.0)
        learner = AutoLearner(config=config)
        assert learner.get_cycle_interval_hours() == 8.0


# --- Tests cho is_cycle_due ---


class TestIsCycleDue:
    """Tests cho is_cycle_due()."""

    def test_no_previous_cycle_is_due(self):
        """Chưa chạy cycle nào → luôn due."""
        learner = AutoLearner()
        assert learner.is_cycle_due() is True

    def test_just_ran_is_not_due(self):
        """Vừa chạy xong → chưa due."""
        learner = AutoLearner()
        learner._last_cycle_time = datetime.now()
        assert learner.is_cycle_due() is False

    def test_after_interval_is_due(self):
        """Qua hết interval → due."""
        config = AutoLearnerConfig(cycle_interval_hours=1.0)
        learner = AutoLearner(config=config)
        # Giả lập last_cycle_time là 2 giờ trước
        learner._last_cycle_time = datetime.now() - timedelta(hours=2.0)
        assert learner.is_cycle_due() is True

    def test_before_interval_not_due(self):
        """Chưa hết interval → not due."""
        config = AutoLearnerConfig(cycle_interval_hours=24.0)
        learner = AutoLearner(config=config)
        # Giả lập last_cycle_time là 1 giờ trước
        learner._last_cycle_time = datetime.now() - timedelta(hours=1.0)
        assert learner.is_cycle_due() is False


# --- Tests cho run_cycle ---


class TestRunCycle:
    """Tests cho run_cycle()."""

    def test_returns_cycle_result(self):
        """run_cycle() trả về CycleResult."""
        learner = AutoLearner()
        result = learner.run_cycle()
        assert isinstance(result, CycleResult)

    def test_increments_cycle_count(self):
        """Mỗi lần run_cycle() tăng cycle_count."""
        learner = AutoLearner()
        learner.run_cycle()
        assert learner._cycle_count == 1
        learner.run_cycle()
        assert learner._cycle_count == 2

    def test_cycle_result_has_correct_number(self):
        """CycleResult.cycle_number khớp với thứ tự cycle."""
        learner = AutoLearner()
        result1 = learner.run_cycle()
        result2 = learner.run_cycle()
        assert result1.cycle_number == 1
        assert result2.cycle_number == 2

    def test_cycle_result_has_phase(self):
        """CycleResult.phase khớp với phase hiện tại."""
        learner = AutoLearner()
        result = learner.run_cycle()
        assert result.phase == TrainingPhase.PHASE_C.value

    def test_cycle_result_has_duration(self):
        """CycleResult ghi nhận duration > 0."""
        learner = AutoLearner()
        result = learner.run_cycle()
        assert result.duration_seconds >= 0.0

    def test_updates_last_cycle_time(self):
        """run_cycle() cập nhật _last_cycle_time."""
        learner = AutoLearner()
        assert learner._last_cycle_time is None

        before = datetime.now()
        learner.run_cycle()
        after = datetime.now()

        assert learner._last_cycle_time is not None
        assert before <= learner._last_cycle_time <= after

    def test_appends_to_cycle_history(self):
        """Mỗi cycle được thêm vào history."""
        learner = AutoLearner()
        learner.run_cycle()
        learner.run_cycle()
        assert len(learner._cycle_history) == 2

    def test_cycle_with_manual_backtest_incorporates_trades(self):
        """Cycle sử dụng trades từ manual backtest đã incorporate."""
        learner = AutoLearner()
        trades = [make_trade(pnl=-100.0), make_trade(pnl=200.0)]
        result = make_backtest_result(trades=trades)
        learner.incorporate_manual_backtest(result)

        cycle_result = learner.run_cycle()
        # 1 trade có PnL < 0 → 1 hard example
        assert "Hard examples generated: 1" in cycle_result.notes
