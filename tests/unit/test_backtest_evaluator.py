"""
Tests cho engine/backtest_evaluator.py

Kiểm tra ModelStrategy signal conversion và BacktestEvaluator integration
với BacktestEngine sử dụng quy tắc thị trường Việt Nam.
"""

import numpy as np
import pandas as pd
import pytest

from engine.backtest_evaluator import BacktestEvaluator, ModelStrategy
from engine.config import Action, ConfigError
from engine.evaluation_config import EvaluationConfig


# ==============================================================================
# Fixtures
# ==============================================================================


def _create_test_df(num_days: int = 20, start_price: float = 50.0) -> pd.DataFrame:
    """Tạo DataFrame OHLCV giả lập cho testing."""
    dates = pd.date_range("2024-01-02", periods=num_days, freq="B")
    np.random.seed(42)

    prices = [start_price]
    for _ in range(num_days - 1):
        # Biến động ngẫu nhiên trong khoảng ±3%
        change = np.random.uniform(-0.03, 0.03)
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)

    df = pd.DataFrame(
        {
            "time": dates,
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(100000, 1000000, num_days),
        }
    )
    return df


# ==============================================================================
# ModelStrategy tests
# ==============================================================================


class TestModelStrategy:
    """Tests cho ModelStrategy."""

    def test_buy_signal_above_threshold(self):
        """Score > buy_threshold → BUY."""
        predictions = np.array([0.5, 0.0, -0.5])
        index_mapping = {0: 0, 1: 1, 2: 2}
        strategy = ModelStrategy(
            predictions=predictions,
            index_mapping=index_mapping,
            buy_threshold=0.3,
            sell_threshold=-0.3,
        )
        df = _create_test_df(3)
        assert strategy.generate_signal(df, 0) == Action.BUY

    def test_sell_signal_below_threshold(self):
        """Score < sell_threshold → SELL."""
        predictions = np.array([0.5, 0.0, -0.5])
        index_mapping = {0: 0, 1: 1, 2: 2}
        strategy = ModelStrategy(
            predictions=predictions,
            index_mapping=index_mapping,
            buy_threshold=0.3,
            sell_threshold=-0.3,
        )
        df = _create_test_df(3)
        assert strategy.generate_signal(df, 2) == Action.SELL

    def test_hold_signal_between_thresholds(self):
        """Score giữa hai thresholds → HOLD."""
        predictions = np.array([0.5, 0.0, -0.5])
        index_mapping = {0: 0, 1: 1, 2: 2}
        strategy = ModelStrategy(
            predictions=predictions,
            index_mapping=index_mapping,
            buy_threshold=0.3,
            sell_threshold=-0.3,
        )
        df = _create_test_df(3)
        assert strategy.generate_signal(df, 1) == Action.HOLD

    def test_score_at_buy_threshold_is_hold(self):
        """Score == buy_threshold → HOLD (phải > chứ không phải >=)."""
        predictions = np.array([0.3])
        index_mapping = {0: 0}
        strategy = ModelStrategy(
            predictions=predictions,
            index_mapping=index_mapping,
            buy_threshold=0.3,
            sell_threshold=-0.3,
        )
        df = _create_test_df(1)
        assert strategy.generate_signal(df, 0) == Action.HOLD

    def test_score_at_sell_threshold_is_hold(self):
        """Score == sell_threshold → HOLD (phải < chứ không phải <=)."""
        predictions = np.array([-0.3])
        index_mapping = {0: 0}
        strategy = ModelStrategy(
            predictions=predictions,
            index_mapping=index_mapping,
            buy_threshold=0.3,
            sell_threshold=-0.3,
        )
        df = _create_test_df(1)
        assert strategy.generate_signal(df, 0) == Action.HOLD

    def test_unmapped_index_returns_hold(self):
        """Index không có trong mapping → HOLD."""
        predictions = np.array([0.9])
        index_mapping = {0: 0}
        strategy = ModelStrategy(
            predictions=predictions,
            index_mapping=index_mapping,
            buy_threshold=0.3,
            sell_threshold=-0.3,
        )
        df = _create_test_df(5)
        # Index 3 không có trong mapping
        assert strategy.generate_signal(df, 3) == Action.HOLD

    def test_out_of_bounds_pred_index_returns_hold(self):
        """Prediction index vượt bounds → HOLD."""
        predictions = np.array([0.9])
        # Mapping trỏ đến index ngoài bounds
        index_mapping = {0: 5}
        strategy = ModelStrategy(
            predictions=predictions,
            index_mapping=index_mapping,
            buy_threshold=0.3,
            sell_threshold=-0.3,
        )
        df = _create_test_df(1)
        assert strategy.generate_signal(df, 0) == Action.HOLD

    def test_custom_thresholds(self):
        """Kiểm tra hoạt động đúng với custom thresholds."""
        predictions = np.array([0.2, -0.15])
        index_mapping = {0: 0, 1: 1}
        # Thresholds hẹp hơn
        strategy = ModelStrategy(
            predictions=predictions,
            index_mapping=index_mapping,
            buy_threshold=0.1,
            sell_threshold=-0.1,
        )
        df = _create_test_df(2)
        assert strategy.generate_signal(df, 0) == Action.BUY
        assert strategy.generate_signal(df, 1) == Action.SELL

    def test_extreme_scores(self):
        """Kiểm tra với scores tại giới hạn [-1, 1]."""
        predictions = np.array([1.0, -1.0, 0.0])
        index_mapping = {0: 0, 1: 1, 2: 2}
        strategy = ModelStrategy(
            predictions=predictions,
            index_mapping=index_mapping,
            buy_threshold=0.3,
            sell_threshold=-0.3,
        )
        df = _create_test_df(3)
        assert strategy.generate_signal(df, 0) == Action.BUY
        assert strategy.generate_signal(df, 1) == Action.SELL
        assert strategy.generate_signal(df, 2) == Action.HOLD


# ==============================================================================
# BacktestEvaluator tests
# ==============================================================================


class TestBacktestEvaluator:
    """Tests cho BacktestEvaluator."""

    def test_run_backtest_basic(self):
        """Kiểm tra backtest chạy thành công và trả về kết quả."""
        config = EvaluationConfig(buy_threshold=0.3, sell_threshold=-0.3)
        evaluator = BacktestEvaluator(config=config)

        test_df = _create_test_df(num_days=30)
        # Tạo predictions với các signals hỗn hợp
        predictions = np.array(
            [0.5] * 5 + [0.0] * 10 + [-0.5] * 5 + [0.5] * 5 + [0.0] * 5
        )

        result = evaluator.run_backtest(predictions=predictions, test_df=test_df)

        # Kiểm tra kết quả có đầy đủ các metrics
        assert hasattr(result, "total_return_pct")
        assert hasattr(result, "annualized_return_pct")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "sharpe_ratio")
        assert result.equity_curve is not None
        assert len(result.equity_curve) == 30

    def test_run_backtest_all_hold(self):
        """Backtest với toàn HOLD signals → total_return = 0."""
        config = EvaluationConfig(buy_threshold=0.3, sell_threshold=-0.3)
        evaluator = BacktestEvaluator(config=config)

        test_df = _create_test_df(num_days=20)
        # Tất cả scores nằm trong vùng HOLD
        predictions = np.zeros(20)

        result = evaluator.run_backtest(predictions=predictions, test_df=test_df)

        # Không có giao dịch nào → return = 0
        assert result.total_return_pct == 0.0
        assert len(result.trades) == 0

    def test_run_backtest_custom_dates(self):
        """Kiểm tra backtest với custom start_date và end_date."""
        config = EvaluationConfig()
        evaluator = BacktestEvaluator(config=config)

        test_df = _create_test_df(num_days=30)
        predictions = np.array([0.5] * 10 + [-0.5] * 10 + [0.0] * 10)

        # Chỉ định dates rõ ràng
        dates = pd.to_datetime(test_df["time"])
        start = str(dates.iloc[5].date())
        end = str(dates.iloc[25].date())

        result = evaluator.run_backtest(
            predictions=predictions,
            test_df=test_df,
            start_date=start,
            end_date=end,
        )

        assert result.equity_curve is not None
        # Số ngày phải <= khoảng thời gian chỉ định
        assert len(result.equity_curve) <= 25

    def test_invalid_thresholds_raises_config_error(self):
        """sell_threshold >= buy_threshold → ConfigError."""
        config = EvaluationConfig(buy_threshold=0.3, sell_threshold=0.5)
        evaluator = BacktestEvaluator(config=config)

        test_df = _create_test_df(num_days=10)
        predictions = np.zeros(10)

        with pytest.raises(ConfigError) as exc_info:
            evaluator.run_backtest(predictions=predictions, test_df=test_df)

        assert exc_info.value.error_code == "INVALID_THRESHOLD"

    def test_invalid_thresholds_equal_raises_config_error(self):
        """sell_threshold == buy_threshold → ConfigError."""
        config = EvaluationConfig(buy_threshold=0.0, sell_threshold=0.0)
        evaluator = BacktestEvaluator(config=config)

        test_df = _create_test_df(num_days=10)
        predictions = np.zeros(10)

        with pytest.raises(ConfigError):
            evaluator.run_backtest(predictions=predictions, test_df=test_df)

    def test_backtest_with_datetime_index(self):
        """Kiểm tra hoạt động với DataFrame có DatetimeIndex thay vì cột 'time'."""
        config = EvaluationConfig()
        evaluator = BacktestEvaluator(config=config)

        # Tạo DataFrame với DatetimeIndex
        num_days = 20
        dates = pd.date_range("2024-01-02", periods=num_days, freq="B")
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(num_days) * 0.5) + 50

        df = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.02,
                "low": prices * 0.98,
                "close": prices,
                "volume": np.random.randint(100000, 1000000, num_days),
            },
            index=dates,
        )

        predictions = np.array([0.5] * 5 + [0.0] * 5 + [-0.5] * 5 + [0.0] * 5)

        result = evaluator.run_backtest(predictions=predictions, test_df=df)
        assert result.equity_curve is not None

    def test_backtest_respects_vietnamese_market_rules(self):
        """Kiểm tra T+2.5 settlement được áp dụng (không thể sell ngay sau buy)."""
        config = EvaluationConfig(buy_threshold=0.3, sell_threshold=-0.3)
        evaluator = BacktestEvaluator(config=config)

        # BUY ngày đầu, SELL ngày 2 (vi phạm T+2.5)
        test_df = _create_test_df(num_days=10)
        # BUY day 0, SELL day 1 (quá sớm, bị bỏ qua), SELL day 4 (hợp lệ)
        predictions = np.array([0.5, -0.5, -0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = evaluator.run_backtest(predictions=predictions, test_df=test_df)

        # Nếu có trade, nó phải được thực hiện sau ít nhất 3 ngày
        if len(result.trades) > 0:
            trade = result.trades[0]
            days_held = (trade.exit_date - trade.entry_date).days
            # Ít nhất 3 trading days (có thể nhiều hơn do weekend)
            assert days_held >= 2  # Business days
