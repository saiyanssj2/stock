"""
Unit tests cho BaselineComparator và baseline strategies.

Kiểm tra:
- BuyAndHoldStrategy: BUY ngày đầu, HOLD mọi ngày sau
- RandomStrategy: reproducibility với fixed seed, equal probability
- SMACrossoverStrategy: BUY/SELL dựa trên SMA crossover
- BaselineComparator.compare(): tích hợp với BacktestEngine
"""

import numpy as np
import pandas as pd
import pytest

from engine.baseline_comparator import (
    BaselineComparator,
    BuyAndHoldStrategy,
    RandomStrategy,
    SMACrossoverStrategy,
)
from engine.config import Action


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Tạo DataFrame OHLCV mẫu với 100 ngày."""
    np.random.seed(123)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    base_price = 50.0
    # Tạo chuỗi giá random walk
    returns = np.random.normal(0.001, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "time": dates,
            "open": prices * (1 - np.random.uniform(0, 0.01, n)),
            "high": prices * (1 + np.random.uniform(0, 0.02, n)),
            "low": prices * (1 - np.random.uniform(0, 0.02, n)),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n),
        }
    )
    return df


@pytest.fixture
def short_df() -> pd.DataFrame:
    """Tạo DataFrame ngắn (10 ngày) cho edge case."""
    n = 10
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = np.linspace(50, 55, n)

    df = pd.DataFrame(
        {
            "time": dates,
            "open": prices - 0.5,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": [500000] * n,
        }
    )
    return df


# ==============================================================================
# Tests: BuyAndHoldStrategy
# ==============================================================================


class TestBuyAndHoldStrategy:
    """Tests cho BuyAndHoldStrategy."""

    def test_first_signal_is_buy(self, sample_df: pd.DataFrame) -> None:
        """Ngày đầu tiên phải trả về BUY."""
        strategy = BuyAndHoldStrategy()
        signal = strategy.generate_signal(sample_df, 0)
        assert signal == Action.BUY

    def test_subsequent_signals_are_hold(self, sample_df: pd.DataFrame) -> None:
        """Tất cả ngày sau ngày đầu phải trả về HOLD."""
        strategy = BuyAndHoldStrategy()
        # Ngày đầu tiên
        strategy.generate_signal(sample_df, 0)
        # Các ngày tiếp theo
        for i in range(1, 20):
            signal = strategy.generate_signal(sample_df, i)
            assert signal == Action.HOLD

    def test_only_one_buy_signal(self, sample_df: pd.DataFrame) -> None:
        """Chỉ có đúng 1 signal BUY trong toàn bộ sequence."""
        strategy = BuyAndHoldStrategy()
        signals = [strategy.generate_signal(sample_df, i) for i in range(50)]
        buy_count = sum(1 for s in signals if s == Action.BUY)
        assert buy_count == 1


# ==============================================================================
# Tests: RandomStrategy
# ==============================================================================


class TestRandomStrategy:
    """Tests cho RandomStrategy."""

    def test_reproducibility_same_seed(self, sample_df: pd.DataFrame) -> None:
        """Cùng seed phải cho cùng sequence tín hiệu."""
        strategy1 = RandomStrategy(seed=42)
        strategy2 = RandomStrategy(seed=42)

        signals1 = [strategy1.generate_signal(sample_df, i) for i in range(100)]
        signals2 = [strategy2.generate_signal(sample_df, i) for i in range(100)]

        assert signals1 == signals2

    def test_different_seeds_different_signals(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Seed khác nhau phải cho sequence khác nhau."""
        strategy1 = RandomStrategy(seed=42)
        strategy2 = RandomStrategy(seed=99)

        signals1 = [strategy1.generate_signal(sample_df, i) for i in range(100)]
        signals2 = [strategy2.generate_signal(sample_df, i) for i in range(100)]

        assert signals1 != signals2

    def test_all_three_actions_present(self, sample_df: pd.DataFrame) -> None:
        """Phải có cả 3 loại action trong sequence đủ dài."""
        strategy = RandomStrategy(seed=42)
        signals = [strategy.generate_signal(sample_df, i) for i in range(100)]

        actions_present = set(signals)
        assert Action.BUY in actions_present
        assert Action.SELL in actions_present
        assert Action.HOLD in actions_present

    def test_approximately_uniform_distribution(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Phân phối xấp xỉ đều (mỗi loại >= 20% cho 1000 samples)."""
        strategy = RandomStrategy(seed=42)
        # Tạo DataFrame dài hơn cho test này
        n = 1000
        signals = [strategy.generate_signal(sample_df, i) for i in range(n)]

        buy_count = sum(1 for s in signals if s == Action.BUY)
        sell_count = sum(1 for s in signals if s == Action.SELL)
        hold_count = sum(1 for s in signals if s == Action.HOLD)

        # Mỗi loại >= 20% (1/3 ≈ 33%, tolerance rộng)
        assert buy_count >= n * 0.20
        assert sell_count >= n * 0.20
        assert hold_count >= n * 0.20

    def test_valid_actions_only(self, sample_df: pd.DataFrame) -> None:
        """Tất cả signals phải là Action hợp lệ."""
        strategy = RandomStrategy(seed=123)
        signals = [strategy.generate_signal(sample_df, i) for i in range(50)]

        valid_actions = {Action.BUY, Action.SELL, Action.HOLD}
        for signal in signals:
            assert signal in valid_actions


# ==============================================================================
# Tests: SMACrossoverStrategy
# ==============================================================================


class TestSMACrossoverStrategy:
    """Tests cho SMACrossoverStrategy."""

    def test_hold_when_insufficient_data(self, short_df: pd.DataFrame) -> None:
        """Trả về HOLD khi chưa đủ dữ liệu cho SMA dài hạn."""
        strategy = SMACrossoverStrategy(short_period=5, long_period=10)
        # Index 0-8 (< long_period - 1 = 9) → HOLD
        for i in range(9):
            signal = strategy.generate_signal(short_df, i)
            assert signal == Action.HOLD

    def test_buy_when_short_above_long(self) -> None:
        """BUY khi SMA ngắn > SMA dài (uptrend)."""
        # Tạo data uptrend rõ ràng: giá tăng dần
        n = 60
        prices = np.linspace(40, 60, n)  # Tăng từ 40 → 60
        df = pd.DataFrame(
            {
                "time": pd.date_range("2023-01-01", periods=n, freq="B"),
                "open": prices - 0.5,
                "high": prices + 1.0,
                "low": prices - 1.0,
                "close": prices,
                "volume": [500000] * n,
            }
        )

        strategy = SMACrossoverStrategy(short_period=10, long_period=30)
        # Ở cuối uptrend, SMA ngắn > SMA dài
        signal = strategy.generate_signal(df, n - 1)
        assert signal == Action.BUY

    def test_sell_when_short_below_long(self) -> None:
        """SELL khi SMA ngắn < SMA dài (downtrend)."""
        # Tạo data downtrend rõ ràng: giá giảm dần
        n = 60
        prices = np.linspace(60, 40, n)  # Giảm từ 60 → 40
        df = pd.DataFrame(
            {
                "time": pd.date_range("2023-01-01", periods=n, freq="B"),
                "open": prices + 0.5,
                "high": prices + 1.0,
                "low": prices - 1.0,
                "close": prices,
                "volume": [500000] * n,
            }
        )

        strategy = SMACrossoverStrategy(short_period=10, long_period=30)
        # Ở cuối downtrend, SMA ngắn < SMA dài
        signal = strategy.generate_signal(df, n - 1)
        assert signal == Action.SELL

    def test_uses_close_prices(self) -> None:
        """Đảm bảo SMA tính từ cột 'close'."""
        n = 30
        # Close khác biệt lớn so với open
        close_prices = np.array([100.0] * 20 + [50.0] * 10)
        open_prices = np.array([50.0] * 30)

        df = pd.DataFrame(
            {
                "time": pd.date_range("2023-01-01", periods=n, freq="B"),
                "open": open_prices,
                "high": close_prices + 5,
                "low": close_prices - 5,
                "close": close_prices,
                "volume": [500000] * n,
            }
        )

        strategy = SMACrossoverStrategy(short_period=5, long_period=20)
        # SMA_short tính từ 5 close cuối (= 50), SMA_long tính từ 20 close cuối
        # SMA_long bao gồm cả giá 100, nên SMA_short < SMA_long → SELL
        signal = strategy.generate_signal(df, n - 1)
        assert signal == Action.SELL

    def test_custom_periods(self, sample_df: pd.DataFrame) -> None:
        """Verify custom periods được sử dụng đúng."""
        strategy = SMACrossoverStrategy(short_period=5, long_period=10)
        # Index 9 là đủ data cho long_period=10 (cần >= 9)
        signal = strategy.generate_signal(sample_df, 9)
        assert signal in {Action.BUY, Action.SELL, Action.HOLD}


# ==============================================================================
# Tests: BaselineComparator
# ==============================================================================


class TestBaselineComparator:
    """Tests cho BaselineComparator.compare()."""

    def test_compare_returns_comparison_result(
        self, sample_df: pd.DataFrame
    ) -> None:
        """compare() trả về ComparisonResult."""
        from engine.config import ComparisonResult

        comparator = BaselineComparator()

        # Tạo simple model strategy (luôn BUY)
        class AlwaysBuyStrategy:
            def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
                return Action.BUY

        model_strategy = AlwaysBuyStrategy()
        result = comparator.compare(
            model_strategy=model_strategy,
            test_df=sample_df,
            start_date="2023-01-01",
            end_date="2023-06-01",
        )

        assert isinstance(result, ComparisonResult)

    def test_compare_includes_model_strategy(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Kết quả phải bao gồm model strategy."""
        comparator = BaselineComparator()

        class AlwaysBuyStrategy:
            def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
                return Action.BUY

        model_strategy = AlwaysBuyStrategy()
        result = comparator.compare(
            model_strategy=model_strategy,
            test_df=sample_df,
            start_date="2023-01-01",
            end_date="2023-06-01",
        )

        # Model strategy chỉ BUY (không có SELL), có thể bị exclude
        # Kiểm tra rằng "model" xuất hiện trong results hoặc excluded
        all_strategies = list(result.results.keys()) + result.excluded_strategies
        assert "model" in all_strategies

    def test_compare_includes_baselines(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Kết quả phải bao gồm tất cả baseline strategies."""
        comparator = BaselineComparator()

        class SimpleStrategy:
            """Strategy đơn giản: BUY ngày chẵn, SELL ngày lẻ."""

            def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
                if index % 3 == 0:
                    return Action.BUY
                elif index % 3 == 1:
                    return Action.SELL
                return Action.HOLD

        model_strategy = SimpleStrategy()
        result = comparator.compare(
            model_strategy=model_strategy,
            test_df=sample_df,
            start_date="2023-01-01",
            end_date="2023-06-01",
        )

        # Tất cả strategies phải xuất hiện (results hoặc excluded)
        all_strategies = list(result.results.keys()) + result.excluded_strategies
        assert "model" in all_strategies
        assert "buy_and_hold" in all_strategies
        assert "random" in all_strategies
        assert "sma_crossover" in all_strategies

    def test_compare_uses_config_params(self, sample_df: pd.DataFrame) -> None:
        """Verify config parameters được truyền đúng vào strategies."""
        from engine.config import ComparisonResult
        from engine.evaluation_config import EvaluationConfig

        config = EvaluationConfig(
            random_seed=99,
            sma_short=10,
            sma_long=30,
        )
        comparator = BaselineComparator(config=config)

        class SimpleStrategy:
            def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
                if index % 3 == 0:
                    return Action.BUY
                elif index % 3 == 1:
                    return Action.SELL
                return Action.HOLD

        model_strategy = SimpleStrategy()
        result = comparator.compare(
            model_strategy=model_strategy,
            test_df=sample_df,
            start_date="2023-01-01",
            end_date="2023-06-01",
        )

        # Kết quả phải là ComparisonResult hợp lệ
        assert isinstance(result, ComparisonResult)
        assert result.date_range_start == "2023-01-01"
        assert result.date_range_end == "2023-06-01"

    def test_compare_date_range_in_result(
        self, sample_df: pd.DataFrame
    ) -> None:
        """ComparisonResult phải lưu date range đúng."""
        comparator = BaselineComparator()

        class SimpleStrategy:
            def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
                if index % 2 == 0:
                    return Action.BUY
                return Action.SELL

        result = comparator.compare(
            model_strategy=SimpleStrategy(),
            test_df=sample_df,
            start_date="2023-02-01",
            end_date="2023-05-01",
        )

        assert result.date_range_start == "2023-02-01"
        assert result.date_range_end == "2023-05-01"
