"""
Unit tests bổ sung cho BacktestEngineWorker - Edge cases.

Bổ sung cho test_backtest_worker.py, tập trung:
- VN rules enforcement edge cases (T+2.5, ±7%, lot 100)
- Manual backtest với dữ liệu mẫu chi tiết
- Benchmark comparison logic

Requirements: 7.1, 7.2, 7.4, 7.5
"""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from config.vn_market_rules import LOT_SIZE, PRICE_LIMIT_PCT, SETTLEMENT_DAYS
from engine.workers.backtest_worker import BacktestEngineWorker
from engine.workers.vn_rules import (
    compute_earliest_sell_date,
    enforce_vn_rules,
    is_within_settlement_period,
    validate_lot_size,
    validate_price_limit,
)
from models.backtest_models import (
    AutoBacktestResult,
    BacktestResult,
    ManualBacktestParams,
    Trade,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def trending_up_csv(tmp_path) -> Path:
    """Tạo CSV với trend tăng rõ ràng → sẽ có BUY signal."""
    dates = pd.bdate_range(start="2024-01-02", periods=60)
    # Giá tăng đều từ 50 lên 70 → EMA ngắn sẽ cắt lên EMA dài
    close = np.linspace(50.0, 70.0, 60)
    # Thêm chút noise nhỏ
    np.random.seed(123)
    close = close + np.random.randn(60) * 0.1

    df = pd.DataFrame({
        "time": dates.strftime("%Y-%m-%d"),
        "open": close - 0.1,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": np.random.randint(100000, 500000, 60),
    })
    csv_path = tmp_path / "UPTREND.csv"
    df.to_csv(csv_path, index=False)
    return tmp_path


@pytest.fixture
def volatile_csv(tmp_path) -> Path:
    """Tạo CSV với giá volatile lớn → có thể vi phạm ±7%."""
    dates = pd.bdate_range(start="2024-01-02", periods=40)
    # Giá dao động mạnh: có ngày thay đổi > 7%
    close = [50.0]
    for i in range(1, 40):
        # Cứ 5 ngày lại có 1 ngày nhảy lớn (>7%)
        if i % 5 == 0:
            close.append(close[-1] * 1.08)  # +8% vi phạm biên độ
        else:
            close.append(close[-1] * (1 + np.random.uniform(-0.02, 0.02)))
    close = np.array(close)

    df = pd.DataFrame({
        "time": dates.strftime("%Y-%m-%d"),
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.randint(100000, 500000, 40),
    })
    csv_path = tmp_path / "VOLATILE.csv"
    df.to_csv(csv_path, index=False)
    return tmp_path


@pytest.fixture
def short_csv(tmp_path) -> Path:
    """Tạo CSV với ít ngày giao dịch (< LONG_MA_PERIOD) → chỉ HOLD."""
    dates = pd.bdate_range(start="2024-01-02", periods=10)
    close = np.linspace(50.0, 52.0, 10)

    df = pd.DataFrame({
        "time": dates.strftime("%Y-%m-%d"),
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": np.random.randint(100000, 500000, 10),
    })
    csv_path = tmp_path / "SHORT.csv"
    df.to_csv(csv_path, index=False)
    return tmp_path


@pytest.fixture
def multi_symbol_csv(tmp_path) -> Path:
    """Tạo nhiều file CSV cho test run_auto."""
    dates = pd.bdate_range(start="2024-01-02", periods=60)
    np.random.seed(42)

    for symbol in ["AAA", "BBB", "CCC"]:
        close = 50.0 + np.cumsum(np.random.randn(60) * 0.3)
        close = np.maximum(close, 10.0)
        df = pd.DataFrame({
            "time": dates.strftime("%Y-%m-%d"),
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.random.randint(100000, 500000, 60),
        })
        csv_path = tmp_path / f"{symbol}.csv"
        df.to_csv(csv_path, index=False)
    return tmp_path


# ==============================================================================
# Test VN Rules Enforcement Edge Cases (Req 7.5)
# ==============================================================================


class TestVNRulesSettlementPeriod:
    """Test T+2.5 settlement period enforcement trong backtest."""

    def test_sell_within_settlement_blocked(self, trending_up_csv):
        """Sell signal trong 3 ngày đầu phải bị skip (Req 5.5)."""
        worker = BacktestEngineWorker(data_dir=trending_up_csv)
        params = ManualBacktestParams(
            symbol="UPTREND",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=100_000_000.0,
            enforce_vn_rules=True,
        )
        result = worker.run_manual(params)
        # Kiểm tra mọi trade đều giữ >= 3 ngày giao dịch
        for trade in result.trades:
            if trade.buy_date and trade.sell_date:
                # holding_days (calendar days) phải >= 3
                # vì 3 trading days >= 3 calendar days tối thiểu
                assert trade.holding_days >= 3

    def test_is_within_settlement_true_for_early_sell(self):
        """Ngày bán trước earliest_sell_date phải trả về True."""
        buy_date = date(2024, 1, 2)  # Thứ Ba
        sell_date = date(2024, 1, 3)  # Thứ Tư (chỉ 1 trading day)
        assert is_within_settlement_period(buy_date, sell_date) is True

    def test_is_within_settlement_false_after_settlement(self):
        """Ngày bán sau earliest_sell_date phải trả về False."""
        buy_date = date(2024, 1, 2)  # Thứ Ba
        # Earliest sell = T+3 trading days = 2024-01-05 (Thứ Sáu)
        sell_date = date(2024, 1, 8)  # Thứ Hai tuần sau
        assert is_within_settlement_period(buy_date, sell_date) is False

    def test_earliest_sell_date_skips_weekends(self):
        """Earliest sell date phải bỏ qua weekends."""
        buy_date = date(2024, 1, 4)  # Thứ Năm
        # +3 trading days: Fri(5), Mon(8), Tue(9)
        earliest = compute_earliest_sell_date(buy_date)
        assert earliest == date(2024, 1, 9)

    def test_enforce_vn_rules_adjusts_sell_date_if_too_early(self):
        """enforce_vn_rules phải điều chỉnh sell_date nếu trong settlement."""
        trade = Trade(
            symbol="TEST",
            buy_date=date(2024, 1, 2),
            sell_date=date(2024, 1, 3),  # Quá sớm
            buy_price=50.0,
            sell_price=52.0,
            shares=100,
            pnl=200.0,
            pnl_pct=0.04,
            holding_days=1,
        )
        adjusted = enforce_vn_rules(trade)
        earliest = compute_earliest_sell_date(date(2024, 1, 2))
        assert adjusted.sell_date >= earliest

    def test_no_trades_when_vn_rules_enabled_short_data(self, short_csv):
        """Data quá ngắn kết hợp VN rules → không đủ signals → 0 trades."""
        worker = BacktestEngineWorker(data_dir=short_csv)
        params = ManualBacktestParams(
            symbol="SHORT",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 15),
            initial_capital=100_000_000.0,
            enforce_vn_rules=True,
        )
        result = worker.run_manual(params)
        # Với 10 ngày data (< LONG_MA_PERIOD=20), tất cả signals sẽ là HOLD
        assert result.total_trades == 0


class TestVNRulesPriceLimit:
    """Test ±7% daily price limit enforcement (Req 7.5)."""

    def test_price_within_limit_accepted(self):
        """Giá thay đổi 5% phải hợp lệ (< 7%)."""
        assert validate_price_limit(52.5, 50.0) is True  # +5%

    def test_price_at_limit_boundary_accepted(self):
        """Giá thay đổi đúng 7% phải hợp lệ (<=7%)."""
        assert validate_price_limit(53.5, 50.0) is True  # +7%

    def test_price_exceeds_limit_rejected(self):
        """Giá thay đổi > 7% phải bị reject."""
        assert validate_price_limit(54.0, 50.0) is False  # +8%

    def test_price_floor_limit_accepted(self):
        """Giá giảm 7% (floor) phải hợp lệ."""
        assert validate_price_limit(46.5, 50.0) is True  # -7%

    def test_price_below_floor_rejected(self):
        """Giá giảm > 7% (dưới floor) phải bị reject."""
        assert validate_price_limit(46.0, 50.0) is False  # -8%

    def test_zero_reference_price_rejected(self):
        """Giá tham chiếu = 0 phải trả về False."""
        assert validate_price_limit(50.0, 0.0) is False

    def test_negative_price_rejected(self):
        """Giá âm phải trả về False."""
        assert validate_price_limit(-5.0, 50.0) is False

    def test_check_price_limit_in_worker(self, trending_up_csv):
        """Worker._check_price_limit phải cùng logic với validate_price_limit."""
        worker = BacktestEngineWorker(data_dir=trending_up_csv)
        # Trong biên độ
        assert worker._check_price_limit(53.0, 50.0) is True
        # Vượt biên độ
        assert worker._check_price_limit(54.5, 50.0) is False


class TestVNRulesLotSize:
    """Test lot size = bội 100 enforcement (Req 7.5)."""

    def test_valid_lot_sizes(self):
        """Bội số 100 phải hợp lệ."""
        assert validate_lot_size(100) is True
        assert validate_lot_size(500) is True
        assert validate_lot_size(1000) is True

    def test_invalid_lot_sizes(self):
        """Không phải bội 100 phải không hợp lệ."""
        assert validate_lot_size(50) is False
        assert validate_lot_size(150) is False
        assert validate_lot_size(99) is False

    def test_zero_or_negative_lot_invalid(self):
        """Lot size 0 hoặc âm phải không hợp lệ."""
        assert validate_lot_size(0) is False
        assert validate_lot_size(-100) is False

    def test_enforce_vn_rules_rounds_down_shares(self):
        """enforce_vn_rules phải làm tròn xuống bội 100."""
        trade = Trade(
            symbol="TEST",
            buy_date=date(2024, 1, 2),
            sell_date=date(2024, 1, 10),
            buy_price=50.0,
            sell_price=52.0,
            shares=350,  # Không phải bội 100
            pnl=700.0,
            pnl_pct=0.04,
            holding_days=8,
        )
        adjusted = enforce_vn_rules(trade)
        assert adjusted.shares == 300
        assert adjusted.shares % LOT_SIZE == 0

    def test_enforce_vn_rules_minimum_one_lot(self):
        """enforce_vn_rules với shares < 100 phải đặt minimum = 100."""
        trade = Trade(
            symbol="TEST",
            buy_date=date(2024, 1, 2),
            sell_date=date(2024, 1, 10),
            buy_price=50.0,
            sell_price=52.0,
            shares=50,  # < LOT_SIZE
            pnl=100.0,
            pnl_pct=0.04,
            holding_days=8,
        )
        adjusted = enforce_vn_rules(trade)
        assert adjusted.shares == LOT_SIZE


# ==============================================================================
# Test Manual Backtest với sample data (Req 7.1, 7.2)
# ==============================================================================


class TestManualBacktestDetailed:
    """Test chi tiết manual backtest results."""

    def test_equity_curve_length_matches_data_period(self, trending_up_csv):
        """Equity curve phải có length = số ngày giao dịch trong period (Req 7.2)."""
        worker = BacktestEngineWorker(data_dir=trending_up_csv)
        params = ManualBacktestParams(
            symbol="UPTREND",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=100_000_000.0,
        )
        result = worker.run_manual(params)
        assert len(result.equity_curve) > 0
        # Equity curve phải <= 60 (số ngày data)
        assert len(result.equity_curve) <= 60

    def test_equity_curve_starts_near_initial_capital(self, trending_up_csv):
        """Điểm đầu equity curve phải gần initial_capital (Req 7.2)."""
        worker = BacktestEngineWorker(data_dir=trending_up_csv)
        params = ManualBacktestParams(
            symbol="UPTREND",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=100_000_000.0,
        )
        result = worker.run_manual(params)
        # Trước khi mua, equity = capital → điểm đầu = initial_capital
        assert result.equity_curve[0] == 100_000_000.0

    def test_final_capital_matches_equity_curve_end(self, trending_up_csv):
        """final_capital phải bằng giá trị cuối equity curve (Req 7.2)."""
        worker = BacktestEngineWorker(data_dir=trending_up_csv)
        params = ManualBacktestParams(
            symbol="UPTREND",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=100_000_000.0,
        )
        result = worker.run_manual(params)
        if result.equity_curve:
            assert result.final_capital == result.equity_curve[-1]

    def test_metrics_all_present_in_result(self, trending_up_csv):
        """Result phải chứa tất cả metrics: win_rate, sharpe, drawdown, return (Req 7.2)."""
        worker = BacktestEngineWorker(data_dir=trending_up_csv)
        params = ManualBacktestParams(
            symbol="UPTREND",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=100_000_000.0,
        )
        result = worker.run_manual(params)
        assert hasattr(result, "win_rate")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "total_return")
        # Kiểm tra kiểu dữ liệu
        assert isinstance(result.win_rate, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.total_return, float)

    def test_win_rate_in_valid_range(self, trending_up_csv):
        """Win rate phải trong khoảng [0, 100]% (Req 7.2)."""
        worker = BacktestEngineWorker(data_dir=trending_up_csv)
        params = ManualBacktestParams(
            symbol="UPTREND",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=100_000_000.0,
        )
        result = worker.run_manual(params)
        assert 0.0 <= result.win_rate <= 100.0

    def test_max_drawdown_non_negative(self, trending_up_csv):
        """Max drawdown phải >= 0 (Req 7.2)."""
        worker = BacktestEngineWorker(data_dir=trending_up_csv)
        params = ManualBacktestParams(
            symbol="UPTREND",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=100_000_000.0,
        )
        result = worker.run_manual(params)
        assert result.max_drawdown >= 0.0

    def test_trades_have_all_required_fields(self, trending_up_csv):
        """Mỗi trade phải có đủ fields: symbol, dates, prices, pnl (Req 7.2)."""
        worker = BacktestEngineWorker(data_dir=trending_up_csv)
        params = ManualBacktestParams(
            symbol="UPTREND",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=100_000_000.0,
        )
        result = worker.run_manual(params)
        for trade in result.trades:
            assert trade.symbol == "UPTREND"
            assert trade.buy_date is not None
            assert trade.sell_date is not None
            assert trade.buy_price > 0
            assert trade.sell_price > 0
            assert trade.shares > 0
            assert trade.pnl is not None

    def test_backtest_without_vn_rules(self, trending_up_csv):
        """Backtest với enforce_vn_rules=False cho phép sell sớm hơn (Req 7.1)."""
        worker = BacktestEngineWorker(data_dir=trending_up_csv)
        params_no_rules = ManualBacktestParams(
            symbol="UPTREND",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=100_000_000.0,
            enforce_vn_rules=False,
        )
        params_with_rules = ManualBacktestParams(
            symbol="UPTREND",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=100_000_000.0,
            enforce_vn_rules=True,
        )
        result_no_rules = worker.run_manual(params_no_rules)
        result_with_rules = worker.run_manual(params_with_rules)
        # Có thể có nhiều trades hơn khi không enforce rules
        # (vì không bị block bởi settlement)
        assert result_no_rules.total_trades >= result_with_rules.total_trades


# ==============================================================================
# Test Benchmark Comparison (Req 7.4)
# ==============================================================================


class TestBenchmarkComparison:
    """Test comparison với benchmark strategies (Req 7.4)."""

    def test_benchmark_generates_exactly_4_results(self, trending_up_csv):
        """Phải có đúng 4 benchmark strategies: Wyckoff, Technical, Momentum, Mean Reversion."""
        worker = BacktestEngineWorker(data_dir=trending_up_csv)
        benchmarks = worker._generate_benchmark_results(1.5)
        assert len(benchmarks) == 4

    def test_benchmark_results_are_floats(self, trending_up_csv):
        """Tất cả benchmark results phải là float."""
        worker = BacktestEngineWorker(data_dir=trending_up_csv)
        benchmarks = worker._generate_benchmark_results(1.0)
        for b in benchmarks:
            assert isinstance(b, float)

    def test_strategies_beaten_count_correct(self, multi_symbol_csv):
        """strategies_beaten phải đúng = số benchmarks mà AI thắng (Req 7.4)."""
        worker = BacktestEngineWorker(data_dir=multi_symbol_csv)
        result = worker.run_auto(symbols=["AAA", "BBB", "CCC"], cycle_number=1)
        # Đếm thủ công: bao nhiêu benchmarks mà overall_sharpe > benchmark
        expected_beaten = sum(
            1 for b in result.benchmark_results
            if result.overall_sharpe > b
        )
        assert result.strategies_beaten == expected_beaten

    def test_auto_backtest_aggregates_metrics_correctly(self, multi_symbol_csv):
        """Auto backtest phải aggregate metrics từ nhiều symbols (Req 7.4)."""
        worker = BacktestEngineWorker(data_dir=multi_symbol_csv)
        result = worker.run_auto(symbols=["AAA", "BBB", "CCC"], cycle_number=1)
        assert len(result.symbols_tested) == 3
        # Metrics phải là giá trị hữu hạn
        assert np.isfinite(result.overall_sharpe)
        assert np.isfinite(result.overall_win_rate)
        assert np.isfinite(result.overall_return)

    def test_auto_backtest_skips_invalid_symbols(self, multi_symbol_csv):
        """Auto backtest phải bỏ qua symbols không có data (Req 7.4)."""
        worker = BacktestEngineWorker(data_dir=multi_symbol_csv)
        result = worker.run_auto(
            symbols=["AAA", "INVALID1", "BBB", "INVALID2"], cycle_number=1
        )
        assert "AAA" in result.symbols_tested
        assert "BBB" in result.symbols_tested
        assert "INVALID1" not in result.symbols_tested
        assert "INVALID2" not in result.symbols_tested
        assert len(result.symbols_tested) == 2
