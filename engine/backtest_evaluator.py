"""
BacktestEvaluator và ModelStrategy cho module đánh giá sau huấn luyện.

BacktestEvaluator chạy backtest mô hình trên tập test data sử dụng BacktestEngine
với quy tắc thị trường Việt Nam. ModelStrategy chuyển đổi Position_Score thành
tín hiệu giao dịch BUY/HOLD/SELL dựa trên configurable thresholds.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from engine.backtest_engine import BacktestEngine
from engine.config import Action, BacktestResult, ConfigError
from engine.evaluation_config import EvaluationConfig

logger = logging.getLogger(__name__)


# ==============================================================================
# ModelStrategy - Chuyển Position_Score → Trading Signal
# ==============================================================================


class ModelStrategy:
    """
    Strategy wrapper cho model predictions, tuân thủ Strategy Protocol.

    Chuyển đổi Position_Score thành tín hiệu giao dịch:
    - score > buy_threshold → BUY
    - score < sell_threshold → SELL
    - otherwise → HOLD

    Parameters
    ----------
    predictions : np.ndarray
        Array Position_Score cho mỗi ngày trong test period.
    index_mapping : Dict[int, int]
        Mapping từ positional index trong DataFrame gốc → index trong predictions array.
    buy_threshold : float
        Ngưỡng trên để sinh tín hiệu BUY (mặc định: 0.3).
    sell_threshold : float
        Ngưỡng dưới để sinh tín hiệu SELL (mặc định: -0.3).
    """

    def __init__(
        self,
        predictions: np.ndarray,
        index_mapping: Dict[int, int],
        buy_threshold: float = 0.3,
        sell_threshold: float = -0.3,
    ):
        self.predictions = predictions
        self.index_mapping = index_mapping
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        """
        Chuyển Position_Score tại index thành trading signal.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame OHLCV gốc (không sử dụng trực tiếp, cần cho Strategy Protocol).
        index : int
            Positional index trong DataFrame gốc.

        Returns
        -------
        Action
            BUY nếu score > buy_threshold,
            SELL nếu score < sell_threshold,
            HOLD trong các trường hợp còn lại.
        """
        # Lấy index trong predictions array từ mapping
        pred_idx = self.index_mapping.get(index)

        # Nếu không tìm thấy mapping, trả về HOLD
        if pred_idx is None:
            return Action.HOLD

        # Kiểm tra bounds của predictions array
        if pred_idx < 0 or pred_idx >= len(self.predictions):
            return Action.HOLD

        score = float(self.predictions[pred_idx])

        # Chuyển đổi score → signal dựa trên thresholds
        if score > self.buy_threshold:
            return Action.BUY
        elif score < self.sell_threshold:
            return Action.SELL
        else:
            return Action.HOLD


# ==============================================================================
# BacktestEvaluator - Chạy backtest mô hình trên test data
# ==============================================================================


class BacktestEvaluator:
    """
    Thực hiện backtest mô hình trên tập test data.

    Sử dụng BacktestEngine với quy tắc thị trường Việt Nam:
    - T+2.5 settlement
    - ±7% price limit
    - 100-share lot size
    - 20% max position

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
    ):
        self.config = config or EvaluationConfig()
        self.engine = engine or BacktestEngine()

    def run_backtest(
        self,
        predictions: np.ndarray,
        test_df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """
        Chạy backtest sử dụng BacktestEngine với model predictions.

        Tạo ModelStrategy từ predictions array, xây dựng index mapping
        giữa DataFrame và predictions, rồi chạy qua BacktestEngine.run()
        với Vietnamese market rules.

        Parameters
        ----------
        predictions : np.ndarray
            Array Position_Score cho mỗi ngày trong test period.
            Độ dài phải khớp với số hàng trong test_df.
        test_df : pd.DataFrame
            DataFrame OHLCV cho test period. Phải chứa cột 'time' hoặc DatetimeIndex,
            cùng các cột 'open', 'high', 'low', 'close', 'volume'.
        start_date : str, optional
            Ngày bắt đầu test period. Nếu None, lấy từ test_df.
        end_date : str, optional
            Ngày kết thúc test period. Nếu None, lấy từ test_df.

        Returns
        -------
        BacktestResult
            Kết quả backtest với đầy đủ metrics: total return, annualized return,
            win rate, max drawdown, Sharpe ratio.

        Raises
        ------
        ConfigError
            Nếu sell_threshold >= buy_threshold.
        """
        # Validate thresholds
        if self.config.sell_threshold >= self.config.buy_threshold:
            raise ConfigError(
                f"sell_threshold ({self.config.sell_threshold}) phải nhỏ hơn "
                f"buy_threshold ({self.config.buy_threshold})",
                error_code="INVALID_THRESHOLD",
                details={
                    "buy": self.config.buy_threshold,
                    "sell": self.config.sell_threshold,
                },
            )

        # Xác định start_date và end_date từ test_df nếu không được cung cấp
        if start_date is None or end_date is None:
            dates = self._extract_dates(test_df)
            if start_date is None:
                start_date = str(dates.min().date())
            if end_date is None:
                end_date = str(dates.max().date())

        # Xây dựng index mapping: positional index trong test_df → index trong predictions
        index_mapping = self._build_index_mapping(test_df)

        # Tạo ModelStrategy
        strategy = ModelStrategy(
            predictions=predictions,
            index_mapping=index_mapping,
            buy_threshold=self.config.buy_threshold,
            sell_threshold=self.config.sell_threshold,
        )

        logger.info(
            f"Chạy backtest từ {start_date} đến {end_date} "
            f"với {len(predictions)} predictions"
        )

        # Chạy backtest qua BacktestEngine
        result = self.engine.run(
            strategy=strategy,
            df=test_df,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(
            f"Backtest hoàn tất: total_return={result.total_return_pct:.2f}%, "
            f"sharpe={result.sharpe_ratio:.2f}"
        )

        return result

    def _extract_dates(self, df: pd.DataFrame) -> pd.Series:
        """Trích xuất chuỗi ngày từ DataFrame."""
        if "time" in df.columns:
            return pd.to_datetime(df["time"])
        elif isinstance(df.index, pd.DatetimeIndex):
            return df.index.to_series()
        else:
            raise ConfigError(
                "DataFrame phải có cột 'time' hoặc DatetimeIndex",
                error_code="NO_DATE_COLUMN",
                details={"columns": list(df.columns)},
            )

    def _build_index_mapping(self, test_df: pd.DataFrame) -> Dict[int, int]:
        """
        Xây dựng mapping từ positional index trong DataFrame gốc
        sang index trong predictions array.

        BacktestEngine truyền positional index (df.index.get_loc) vào
        strategy.generate_signal(), nên ta cần map từ positional index
        của test_df về vị trí tương ứng trong predictions array.
        """
        mapping: Dict[int, int] = {}
        for pred_idx in range(len(test_df)):
            # Positional index trong test_df chính là pred_idx
            # vì BacktestEngine sẽ gọi df.index.get_loc(original_idx)
            # Khi test_df được truyền trực tiếp vào engine.run(),
            # original_pos sẽ là positional index trong test_df
            mapping[pred_idx] = pred_idx
        return mapping
