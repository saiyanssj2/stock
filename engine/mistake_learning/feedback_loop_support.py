"""
Hỗ trợ cho FeedbackLoopManager — strategy adapter và retrain helpers.

Tách ra để giữ feedback_loop.py dưới 700 dòng.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.config import BacktestResult, Trade

logger = logging.getLogger(__name__)


class _FeedbackAIStrategy:
    """Strategy adapter cho backtest trong feedback loop.

    Dùng model từ DecisionEngine để generate signal cho backtest.
    Pattern tương tự _AIStrategy trong decision_engine_backtest.py
    nhưng nhẹ hơn, không cần import circular.
    """

    def __init__(self, engine: Any, symbol: str, df: pd.DataFrame) -> None:
        """Khởi tạo strategy adapter.

        Args:
            engine: DecisionEngine instance.
            symbol: Tên symbol đang backtest.
            df: DataFrame đầy đủ cho symbol.
        """
        self._engine = engine
        self._symbol = symbol
        self._df = df
        self._lookback = getattr(
            getattr(engine, "config", None), "lookback", 60
        )

    def generate_signal(self, df: pd.DataFrame, index: int) -> Any:
        """Generate trading signal tại index cho backtest.

        Dùng model prediction với threshold ±0.3 để quyết định BUY/HOLD/SELL.
        Fallback HOLD nếu có lỗi.

        Args:
            df: DataFrame đầy đủ.
            index: Vị trí hiện tại.

        Returns:
            Action (BUY/HOLD/SELL).
        """
        from engine.config import Action
        from engine.market_state import MarketState

        # Cần đủ history cho lookback window
        if index < self._lookback:
            return Action.HOLD

        try:
            # Build MarketState từ slice data
            state = MarketState.from_dataframe(
                df.iloc[: index + 1],
                self._symbol,
                lookback=self._lookback,
                config=self._engine.config,
            )

            # Build feature array
            if state.indicators.shape[1] > 0:
                features = np.concatenate(
                    [state.ohlcv, state.indicators], axis=1
                ).astype(np.float32)
            else:
                features = state.ohlcv.astype(np.float32)

            # Xử lý NaN
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Lấy prediction từ model
            scores = self._engine.model_manager.predict(features)
            score = float(scores.flatten()[0])

            # Threshold-based decision
            if score > 0.3:
                return Action.BUY
            elif score < -0.3:
                return Action.SELL
            else:
                return Action.HOLD

        except Exception:
            return Action.HOLD


def run_backtest_via_engine(
    engine: Any, symbol_data: Dict[str, pd.DataFrame]
) -> List[Trade]:
    """Chạy backtest bằng BacktestEngine + AIStrategy pattern.

    Tạo strategy adapter từ engine, chạy trên symbol đầu tiên
    trong validation period (20% cuối dữ liệu).

    Args:
        engine: DecisionEngine instance.
        symbol_data: Dict symbol → DataFrame.

    Returns:
        Danh sách Trade từ backtest result.
    """
    if not symbol_data:
        return []

    # Chọn symbol đầu tiên có đủ dữ liệu
    symbol_name = next(iter(symbol_data))
    df = symbol_data[symbol_name]

    if len(df) < 60:
        return []

    # Xác định validation period: 20% cuối dữ liệu
    val_start_idx = int(len(df) * 0.8)
    if "time" in df.columns:
        start_date = str(pd.to_datetime(df.iloc[val_start_idx]["time"]).date())
        end_date = str(pd.to_datetime(df.iloc[-1]["time"]).date())
    elif isinstance(df.index, pd.DatetimeIndex):
        start_date = str(df.index[val_start_idx].date())
        end_date = str(df.index[-1].date())
    else:
        # Không xác định được date → fallback
        return []

    # Tạo AIStrategy adapter từ engine
    strategy = _FeedbackAIStrategy(engine, symbol_name, df)

    # Chạy backtest
    backtest_engine = engine._backtest_engine
    result = backtest_engine.run(strategy, df, start_date, end_date)

    if isinstance(result, BacktestResult):
        return result.trades
    return []


def run_backtest_via_api(
    engine: Any, symbol_data: Dict[str, pd.DataFrame]
) -> List[Trade]:
    """Chạy backtest qua engine.backtest() API cấp cao.

    Dùng khi engine có method backtest(symbol, start, end).

    Args:
        engine: DecisionEngine instance.
        symbol_data: Dict symbol → DataFrame.

    Returns:
        Danh sách Trade từ backtest result.
    """
    if not symbol_data:
        return []

    symbol_name = next(iter(symbol_data))
    df = symbol_data[symbol_name]

    if len(df) < 60:
        return []

    # Validation period: 20% cuối
    val_start_idx = int(len(df) * 0.8)
    if "time" in df.columns:
        start_date = str(pd.to_datetime(df.iloc[val_start_idx]["time"]).date())
        end_date = str(pd.to_datetime(df.iloc[-1]["time"]).date())
    else:
        return []

    try:
        result = engine.backtest(symbol_name, start_date, end_date)
        if isinstance(result, BacktestResult):
            return result.trades
    except Exception:
        pass

    return []


def retrain_via_pipeline(
    engine: Any,
    symbol_data: Dict[str, pd.DataFrame],
    max_epochs: int,
    hard_examples: Optional[Any] = None,
    sample_weights: Optional[np.ndarray] = None,
) -> Optional[Any]:
    """Retrain qua training pipeline với weighted data.

    Kết hợp hard examples vào symbol_data, áp dụng sample weights,
    rồi dùng train_incremental hoặc train_weighted.

    Args:
        engine: DecisionEngine instance.
        symbol_data: Dữ liệu training gốc.
        max_epochs: Epochs tối đa.
        hard_examples: HardExampleSet chứa features/labels/weights (optional).
        sample_weights: Mảng weights cho toàn bộ samples (optional).

    Returns:
        Kết quả retrain hoặc None.
    """
    try:
        bg_training = engine._background_training
        pipeline = getattr(bg_training, "_training_pipeline", None)

        if pipeline is None:
            return None

        # Ưu tiên train_weighted nếu pipeline hỗ trợ và có weights
        if (
            hard_examples is not None
            and hard_examples.count > 0
            and hasattr(pipeline, "train_weighted")
        ):
            result = pipeline.train_weighted(
                data=symbol_data,
                hard_features=hard_examples.features,
                hard_labels=hard_examples.labels,
                sample_weights=hard_examples.weights,
                max_epochs=max_epochs,
            )
            return result

        # Fallback: dùng train_incremental với max_epochs giới hạn
        if hasattr(pipeline, "train_incremental"):
            result = pipeline.train_incremental(
                new_data=symbol_data,
                max_epochs=max_epochs,
            )
            return result

    except Exception as e:
        logger.warning("Retrain via pipeline thất bại: %s", str(e))

    return None


def retrain_via_model_manager(
    engine: Any,
    hard_examples: Any,
    max_epochs: int,
) -> Optional[Any]:
    """Retrain trực tiếp qua model_manager với weighted samples.

    Tạo feature matrix từ hard examples, áp dụng sample weights,
    rồi fine-tune model.

    Args:
        engine: DecisionEngine instance.
        hard_examples: Hard examples với weights.
        max_epochs: Epochs tối đa.

    Returns:
        Dict chứa thông tin retrain hoặc None.
    """
    try:
        model_manager = engine.model_manager

        if not model_manager.is_loaded:
            logger.warning("Model chưa load, không thể retrain.")
            return None

        # Nếu hard_examples rỗng thì skip
        if hard_examples.count == 0:
            return {"status": "skipped", "reason": "no_hard_examples"}

        # Fine-tune model nếu có method train/fit
        if hasattr(model_manager, "fine_tune"):
            result = model_manager.fine_tune(
                features=hard_examples.features,
                labels=hard_examples.labels,
                sample_weights=hard_examples.weights,
                max_epochs=max_epochs,
            )
            return result

        # Fallback: log rằng retrain không khả thi
        logger.info(
            "Model manager không hỗ trợ fine_tune. "
            "Hard examples (%d) đã được ghi nhận nhưng không retrain.",
            hard_examples.count,
        )
        return {
            "status": "recorded_only",
            "hard_examples_count": hard_examples.count,
        }

    except Exception as e:
        logger.warning("Retrain via model_manager thất bại: %s", str(e))
        return None
