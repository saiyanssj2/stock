"""
Unit tests cho BacktestEngineWorker.

Kiểm tra:
- run_manual(): Load data, generate signals, execute trades, tính metrics
- run_auto(): Iterate symbols, aggregate metrics, benchmark comparison
- _load_data(): Xử lý file không tồn tại, file thiếu columns
- _generate_signals(): MA crossover logic
- _calculate_metrics(): total_return, sharpe, win_rate, max_drawdown
- VN rules enforcement trong trade execution
"""

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.workers.backtest_worker import BacktestEngineWorker
from models.backtest_models import (
    AutoBacktestResult,
    BacktestResult,
    ManualBacktestParams,
)


@pytest.fixture
def sample_csv_data() -> pd.DataFrame:
    """Tạo DataFrame OHLCV mẫu cho test (60 ngày giao dịch)."""
    dates = pd.bdate_range(start="2024-01-02", periods=60)
    np.random.seed(42)

    # Sinh giá close ngẫu nhiên quanh 50 (đơn vị 1000 VND)
    close = 50.0 + np.cumsum(np.random.randn(60) * 0.5)
    close = np.maximum(close, 10.0)  # Đảm bảo giá > 0

    df = pd.DataFrame({
        "time": dates.strftime("%Y-%m-%d"),
        "open": close - np.random.uniform(0, 0.3, 60),
        "high": close + np.random.uniform(0, 0.5, 60),
        "low": close - np.random.uniform(0, 0.5, 60),
        "close": close,
        "volume": np.random.randint(100000, 500000, 60),
    })
    return df


@pytest.fixture
def data_dir_with_csv(sample_csv_data, tmp_path) -> Path:
    """Tạo thư mục data tạm với file CSV."""
    csv_path = tmp_path / "TEST.csv"
    sample_csv_data.to_csv(csv_path, index=False)
    return tmp_path


@pytest.fixture
def worker(data_dir_with_csv) -> BacktestEngineWorker:
    """Tạo BacktestEngineWorker với data dir tạm."""
    return BacktestEngineWorker(data_dir=data_dir_with_csv)


class TestLoadData:
    """Test _load_data method."""

    def test_load_existing_file(self, worker):
        """Load file CSV tồn tại phải trả về DataFrame hợp lệ."""
        df = worker._load_data("TEST")
        assert df is not None
        assert not df.empty
        assert "time" in df.columns
        assert "close" in df.columns

    def test_load_nonexistent_file(self, worker):
        """Load file không tồn tại phải trả về None."""
        df = worker._load_data("NONEXISTENT")
        assert df is None

    def test_load_file_missing_columns(self, tmp_path):
        """Load file thiếu columns bắt buộc phải trả về None."""
        csv_path = tmp_path / "BAD.csv"
        pd.DataFrame({"time": ["2024-01-01"], "price": [50]}).to_csv(
            csv_path, index=False
        )
        w = BacktestEngineWorker(data_dir=tmp_path)
        df = w._load_data("BAD")
        assert df is None


class TestGenerateSignals:
    """Test _generate_signals method."""

    def test_signals_length_matches_data(self, worker, sample_csv_data):
        """Signals phải cùng chiều dài với data."""
        signals = worker._generate_signals(sample_csv_data)
        assert len(signals) == len(sample_csv_data)

    def test_signals_valid_values(self, worker, sample_csv_data):
        """Tất cả signals phải là BUY, SELL, hoặc HOLD."""
        signals = worker._generate_signals(sample_csv_data)
        valid = {"BUY", "SELL", "HOLD"}
        assert all(s in valid for s in signals)

    def test_first_signals_are_hold(self, worker, sample_csv_data):
        """Signals đầu tiên (chưa đủ MA) phải là HOLD."""
        signals = worker._generate_signals(sample_csv_data)
        # Ít nhất LONG_MA_PERIOD (20) signals đầu phải là HOLD
        from engine.workers.backtest_worker import LONG_MA_PERIOD
        for i in range(LONG_MA_PERIOD):
            assert signals[i] == "HOLD"


class TestRunManual:
    """Test run_manual method."""

    def test_returns_backtest_result(self, worker):
        """run_manual phải trả về BacktestResult."""
        params = ManualBacktestParams(
            symbol="TEST",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=100_000_000.0,
        )
        result = worker.run_manual(params)
        assert isinstance(result, BacktestResult)

    def test_result_has_correct_symbol(self, worker):
        """Kết quả phải chứa đúng symbol."""
        params = ManualBacktestParams(
            symbol="TEST",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=100_000_000.0,
        )
        result = worker.run_manual(params)
        assert result.symbol == "TEST"

    def test_result_has_equity_curve(self, worker):
        """Kết quả phải có equity curve không rỗng."""
        params = ManualBacktestParams(
            symbol="TEST",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=100_000_000.0,
        )
        result = worker.run_manual(params)
        assert len(result.equity_curve) > 0

    def test_empty_result_for_missing_symbol(self, worker):
        """Symbol không tồn tại phải trả về empty result."""
        params = ManualBacktestParams(
            symbol="MISSING",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=100_000_000.0,
        )
        result = worker.run_manual(params)
        assert result.total_trades == 0
        assert result.total_return == 0.0

    def test_empty_result_for_invalid_date_range(self, worker):
        """Date range ngoài data phải trả về empty result."""
        params = ManualBacktestParams(
            symbol="TEST",
            start_date=date(2030, 1, 1),
            end_date=date(2030, 12, 31),
            initial_capital=100_000_000.0,
        )
        result = worker.run_manual(params)
        assert result.total_trades == 0

    def test_initial_capital_preserved_in_result(self, worker):
        """initial_capital phải đúng trong result."""
        capital = 50_000_000.0
        params = ManualBacktestParams(
            symbol="TEST",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=capital,
        )
        result = worker.run_manual(params)
        assert result.initial_capital == capital

    def test_trades_have_valid_lot_size(self, worker):
        """Tất cả trades phải có shares là bội số 100 (enforce VN rules)."""
        params = ManualBacktestParams(
            symbol="TEST",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            initial_capital=100_000_000.0,
            enforce_vn_rules=True,
        )
        result = worker.run_manual(params)
        for trade in result.trades:
            assert trade.shares % 100 == 0


class TestRunAuto:
    """Test run_auto method."""

    def test_returns_auto_backtest_result(self, worker):
        """run_auto phải trả về AutoBacktestResult."""
        result = worker.run_auto(symbols=["TEST"], cycle_number=1)
        assert isinstance(result, AutoBacktestResult)

    def test_cycle_number_preserved(self, worker):
        """cycle_number phải đúng trong result."""
        result = worker.run_auto(symbols=["TEST"], cycle_number=5)
        assert result.cycle_number == 5

    def test_tested_symbols_only_valid(self, worker):
        """Chỉ symbols có data mới xuất hiện trong symbols_tested."""
        result = worker.run_auto(
            symbols=["TEST", "NONEXISTENT"], cycle_number=1
        )
        assert "TEST" in result.symbols_tested
        assert "NONEXISTENT" not in result.symbols_tested

    def test_benchmark_results_has_4_strategies(self, worker):
        """Phải có đúng 4 benchmark strategies (Req 7.4)."""
        result = worker.run_auto(symbols=["TEST"], cycle_number=1)
        assert len(result.benchmark_results) == 4

    def test_strategies_beaten_within_range(self, worker):
        """strategies_beaten phải từ 0 đến 4."""
        result = worker.run_auto(symbols=["TEST"], cycle_number=1)
        assert 0 <= result.strategies_beaten <= 4

    def test_duration_positive(self, worker):
        """duration_seconds phải > 0."""
        result = worker.run_auto(symbols=["TEST"], cycle_number=1)
        assert result.duration_seconds > 0

    def test_empty_symbols_list(self, worker):
        """Symbols rỗng phải trả về result với metrics = 0."""
        result = worker.run_auto(symbols=[], cycle_number=1)
        assert result.overall_return == 0.0
        assert result.overall_sharpe == 0.0
        assert result.symbols_tested == []


class TestCalculateMetrics:
    """Test _calculate_metrics method."""

    def test_zero_metrics_for_empty_trades(self, worker):
        """Không có trades phải trả về metrics = 0."""
        metrics = worker._calculate_metrics([], 100_000_000.0, [100_000_000.0])
        assert metrics["win_rate"] == 0.0

    def test_positive_return(self, worker):
        """Equity tăng phải cho total_return > 0."""
        equity = [100_000_000.0, 105_000_000.0, 110_000_000.0]
        metrics = worker._calculate_metrics([], 100_000_000.0, equity)
        assert metrics["total_return"] > 0

    def test_max_drawdown_non_negative(self, worker):
        """Max drawdown luôn >= 0."""
        equity = [100.0, 90.0, 95.0, 85.0, 100.0]
        metrics = worker._calculate_metrics([], 100.0, equity)
        assert metrics["max_drawdown"] >= 0


class TestComputeMaxDrawdown:
    """Test _compute_max_drawdown riêng."""

    def test_no_drawdown_for_increasing_equity(self, worker):
        """Equity tăng liên tục phải có drawdown = 0."""
        equity = [100.0, 110.0, 120.0, 130.0]
        assert worker._compute_max_drawdown(equity) == 0.0

    def test_correct_drawdown_calculation(self, worker):
        """Tính drawdown chính xác từ peak đến trough."""
        # Peak = 100, trough = 80 → drawdown = 20%
        equity = [100.0, 80.0, 90.0]
        dd = worker._compute_max_drawdown(equity)
        assert abs(dd - 20.0) < 0.01

    def test_single_point_equity(self, worker):
        """Equity 1 điểm phải có drawdown = 0."""
        assert worker._compute_max_drawdown([100.0]) == 0.0


class TestComputeSharpeRatio:
    """Test _compute_sharpe_ratio riêng."""

    def test_zero_sharpe_for_flat_equity(self, worker):
        """Equity không đổi phải có Sharpe = 0 (std = 0)."""
        equity = [100.0, 100.0, 100.0, 100.0]
        assert worker._compute_sharpe_ratio(equity) == 0.0

    def test_single_point_returns_zero(self, worker):
        """Equity 1 điểm phải trả về 0."""
        assert worker._compute_sharpe_ratio([100.0]) == 0.0

    def test_positive_sharpe_for_increasing_equity(self, worker):
        """Equity tăng đều phải cho Sharpe > 0."""
        equity = [100.0 + i * 1.0 for i in range(50)]
        sharpe = worker._compute_sharpe_ratio(equity)
        assert sharpe > 0
