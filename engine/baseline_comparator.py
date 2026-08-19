"""
BaselineComparator và baseline strategies cho module đánh giá sau huấn luyện.

So sánh hiệu suất model với 3 chiến lược cơ sở:
- BuyAndHoldStrategy: Mua ngày đầu, giữ đến cuối
- RandomStrategy: Random BUY/SELL/HOLD với equal probability (fixed seed)
- SMACrossoverStrategy: BUY khi SMA ngắn > SMA dài, SELL khi ngược lại

Sử dụng BacktestEngine.compare_strategies() để đảm bảo so sánh công bằng
với cùng initial capital, date range, và market rules.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from engine.backtest_engine import BacktestEngine, Strategy
from engine.config import Action, ComparisonResult
from engine.evaluation_config import EvaluationConfig

logger = logging.getLogger(__name__)


# ==============================================================================
# Baseline Strategies
# ==============================================================================


class BuyAndHoldStrategy(Strategy):
    """
    Chiến lược mua và giữ.

    BUY ngày đầu tiên, HOLD tất cả các ngày sau đó.
    Đây là baseline đơn giản nhất để đánh giá xem model có vượt trội
    hơn việc đơn giản mua và giữ hay không.
    """

    def __init__(self) -> None:
        self._first_signal_sent = False

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        """
        BUY ngày đầu tiên, HOLD mọi ngày sau.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame OHLCV (không sử dụng trực tiếp).
        index : int
            Positional index trong DataFrame.

        Returns
        -------
        Action
            BUY cho ngày đầu tiên, HOLD cho tất cả ngày sau.
        """
        if not self._first_signal_sent:
            self._first_signal_sent = True
            return Action.BUY
        return Action.HOLD


class RandomStrategy(Strategy):
    """
    Chiến lược random với fixed seed cho reproducibility.

    Sinh tín hiệu BUY/SELL/HOLD với xác suất bằng nhau (1/3 mỗi loại)
    sử dụng numpy random generator với seed cố định.

    Parameters
    ----------
    seed : int
        Seed cho random number generator. Mặc định: 42.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        """
        Random BUY/SELL/HOLD với equal probability (1/3 mỗi loại).

        Mapping: 0 → BUY, 1 → SELL, 2 → HOLD.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame OHLCV (không sử dụng trực tiếp).
        index : int
            Positional index trong DataFrame.

        Returns
        -------
        Action
            Random action: BUY, SELL, hoặc HOLD.
        """
        value = self.rng.integers(0, 3)
        if value == 0:
            return Action.BUY
        elif value == 1:
            return Action.SELL
        else:
            return Action.HOLD


class SMACrossoverStrategy(Strategy):
    """
    Chiến lược SMA Crossover.

    BUY khi SMA ngắn hạn cắt lên trên SMA dài hạn (golden cross),
    SELL khi SMA ngắn hạn cắt xuống dưới SMA dài hạn (death cross),
    HOLD khi chưa đủ dữ liệu hoặc hai đường bằng nhau.

    Parameters
    ----------
    short_period : int
        Chu kỳ SMA ngắn hạn. Mặc định: 20.
    long_period : int
        Chu kỳ SMA dài hạn. Mặc định: 50.
    """

    def __init__(self, short_period: int = 20, long_period: int = 50) -> None:
        self.short_period = short_period
        self.long_period = long_period

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        """
        BUY khi SMA_short > SMA_long, SELL khi SMA_short < SMA_long.

        Tính SMA từ close prices trong df từ đầu đến vị trí hiện tại (index).
        Nếu chưa đủ dữ liệu cho SMA dài hạn, trả về HOLD.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame OHLCV với cột 'close'.
        index : int
            Positional index trong DataFrame (0-based).

        Returns
        -------
        Action
            BUY, SELL, hoặc HOLD dựa trên SMA crossover.
        """
        # Cần ít nhất long_period điểm dữ liệu để tính SMA dài hạn
        if index < self.long_period - 1:
            return Action.HOLD

        # Lấy close prices từ đầu đến vị trí hiện tại (inclusive)
        close_prices = df["close"].iloc[: index + 1].values

        # Tính SMA ngắn hạn và dài hạn
        sma_short = float(np.mean(close_prices[-self.short_period:]))
        sma_long = float(np.mean(close_prices[-self.long_period:]))

        # So sánh SMA crossover
        if sma_short > sma_long:
            return Action.BUY
        elif sma_short < sma_long:
            return Action.SELL
        else:
            return Action.HOLD


# ==============================================================================
# BaselineComparator
# ==============================================================================


class BaselineComparator:
    """
    So sánh hiệu suất model với baseline strategies.

    Sử dụng BacktestEngine.compare_strategies() để chạy backtest đồng thời
    model strategy cùng 3 baseline strategies trên cùng test period,
    đảm bảo so sánh công bằng.

    Parameters
    ----------
    config : EvaluationConfig, optional
        Cấu hình đánh giá. Sử dụng giá trị mặc định nếu không cung cấp.
    engine : BacktestEngine, optional
        BacktestEngine instance. Tạo mới nếu không cung cấp.
    """

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        engine: Optional[BacktestEngine] = None,
    ) -> None:
        self.config = config or EvaluationConfig()
        self.engine = engine or BacktestEngine()

    def compare(
        self,
        model_strategy: Strategy,
        test_df: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> ComparisonResult:
        """
        So sánh model vs baselines sử dụng BacktestEngine.compare_strategies().

        Tạo 3 baseline strategies (Buy-and-Hold, Random, SMA Crossover) và
        gọi compare_strategies() với model strategy + 3 baselines để đảm bảo
        so sánh công bằng với cùng initial capital, date range, và market rules.

        Parameters
        ----------
        model_strategy : Strategy
            Strategy của model (ModelStrategy từ BacktestEvaluator).
        test_df : pd.DataFrame
            DataFrame OHLCV cho test period.
        start_date : str
            Ngày bắt đầu test period (format: 'YYYY-MM-DD').
        end_date : str
            Ngày kết thúc test period (format: 'YYYY-MM-DD').

        Returns
        -------
        ComparisonResult
            Kết quả so sánh chứa BacktestResult cho mỗi strategy.
        """
        # Tạo 3 baseline strategies
        buy_and_hold = BuyAndHoldStrategy()
        random_strategy = RandomStrategy(seed=self.config.random_seed)
        sma_crossover = SMACrossoverStrategy(
            short_period=self.config.sma_short,
            long_period=self.config.sma_long,
        )

        # Chuẩn bị dict strategies cho compare_strategies()
        strategies = {
            "model": model_strategy,
            "buy_and_hold": buy_and_hold,
            "random": random_strategy,
            "sma_crossover": sma_crossover,
        }

        logger.info(
            f"So sánh model với 3 baseline strategies "
            f"từ {start_date} đến {end_date}"
        )

        # Gọi BacktestEngine.compare_strategies()
        result = self.engine.compare_strategies(
            strategies=strategies,
            df=test_df,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(
            f"So sánh hoàn tất: {len(result.results)} strategies có kết quả, "
            f"{len(result.excluded_strategies)} strategies bị loại"
        )

        return result
