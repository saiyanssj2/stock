"""
Backtest and comparison helpers for DecisionEngine.

Extracted from decision_engine.py to keep the main module under 700 lines.
Contains:
- BacktestMixin: backtest(), compare(), and background training methods
- _AIStrategy: AI model wrapper compatible with the backtest engine interface
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from engine.backtest_engine import BacktestEngine
from engine.background_training import (
    BackgroundTrainingManager,
    GPUMemoryMonitor,
    SymbolQueue,
    TrainingState,
    TrainingStatus,
)
from engine.config import (
    Action,
    BacktestResult,
    ComparisonResult,
    ConfigError,
    DataError,
    EngineConfig,
)
from engine.market_state import MarketState
from engine.strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    TechnicalStrategy,
    WyckoffStrategy,
)

if TYPE_CHECKING:
    from engine.decision_engine import DecisionEngine

logger = logging.getLogger(__name__)


class _AIStrategy:
    """Wraps the DecisionEngine as a backtest-compatible strategy.

    Uses a simplified approach for backtesting: evaluates the market state
    at each point using the evaluation model directly, without running the
    full search tree (which would be prohibitively slow over many days).

    For backtesting, the AI strategy uses the model's position score:
    - score > 0.3 → BUY
    - score < -0.3 → SELL
    - otherwise → HOLD
    """

    def __init__(self, engine: "DecisionEngine", symbol: str, df: pd.DataFrame):
        self._engine = engine
        self._symbol = symbol
        self._df = df
        self._lookback = engine.config.lookback

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        """Generate a trading signal using the AI model.

        Parameters
        ----------
        df : pd.DataFrame
            Full DataFrame with OHLCV and indicators.
        index : int
            Current position index.

        Returns
        -------
        Action
            BUY, HOLD, or SELL.
        """
        # Need enough history for lookback window
        if index < self._lookback:
            return Action.HOLD

        try:
            # Extract lookback window
            window_df = df.iloc[index - self._lookback + 1 : index + 1].copy()

            if len(window_df) < self._lookback:
                return Action.HOLD

            # Build a quick feature vector and evaluate
            state = MarketState.from_dataframe(
                df.iloc[: index + 1],
                self._symbol,
                lookback=self._lookback,
                config=self._engine.config,
            )

            # Build feature array for model
            if state.indicators.shape[1] > 0:
                features = np.concatenate(
                    [state.ohlcv, state.indicators], axis=1
                ).astype(np.float32)
            else:
                features = state.ohlcv.astype(np.float32)

            # Handle NaN
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Get model prediction
            scores = self._engine.model_manager.predict(features)
            score = float(scores.flatten()[0])

            # Simple threshold-based decision
            if score > 0.3:
                return Action.BUY
            elif score < -0.3:
                return Action.SELL
            else:
                return Action.HOLD

        except Exception as e:
            logger.debug(f"AI strategy signal generation failed at index {index}: {e}")
            return Action.HOLD

    @property
    def name(self) -> str:
        """Strategy name."""
        return "AI Engine"


# ==============================================================================
# Backtest and Compare Functions
# ==============================================================================


def run_backtest(
    engine: "DecisionEngine",
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: Optional[float] = None,
) -> BacktestResult:
    """Run a backtest for a symbol using the AI decision engine strategy.

    Delegates to BacktestEngine with an AI-based strategy that uses
    the SearchModule for signal generation.

    Parameters
    ----------
    engine : DecisionEngine
        The parent engine instance.
    symbol : str
        Stock ticker symbol.
    start_date : str
        Start date (format: 'YYYY-MM-DD').
    end_date : str
        End date (format: 'YYYY-MM-DD').
    initial_capital : float, optional
        Starting capital in VND. Uses config default if not provided.

    Returns
    -------
    BacktestResult
        Backtest results with metrics and trade list.

    Raises
    ------
    DataError
        If CSV data is unavailable or malformed.
    ConfigError
        If date range is invalid.
    """
    csv_path = engine._resolve_csv_path(symbol)
    engine.validate_csv(csv_path)
    df = engine._load_csv(csv_path)

    # Create AI strategy wrapper
    ai_strategy = _AIStrategy(engine, symbol, df)

    # Configure backtest engine with capital
    if initial_capital is not None:
        bt_engine = BacktestEngine(
            initial_capital=initial_capital,
            max_position_pct=engine._config.max_position_pct,
            config=engine._config,
        )
    else:
        bt_engine = engine._backtest_engine

    return bt_engine.run(ai_strategy, df, start_date, end_date)


def run_compare(
    engine: "DecisionEngine",
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: Optional[float] = None,
) -> ComparisonResult:
    """Compare AI strategy against philosophy-based strategies.

    Runs identical backtests across all strategies:
    - AI Decision Engine
    - Wyckoff
    - Technical Indicators
    - Momentum
    - Mean Reversion

    Parameters
    ----------
    engine : DecisionEngine
        The parent engine instance.
    symbol : str
        Stock ticker symbol.
    start_date : str
        Start date (format: 'YYYY-MM-DD').
    end_date : str
        End date (format: 'YYYY-MM-DD').
    initial_capital : float, optional
        Starting capital in VND. Uses config default if not provided.

    Returns
    -------
    ComparisonResult
        Comparison results with all strategy metrics.

    Raises
    ------
    DataError
        If CSV data is unavailable or malformed.
    ConfigError
        If date range is invalid.
    """
    csv_path = engine._resolve_csv_path(symbol)
    engine.validate_csv(csv_path)
    df = engine._load_csv(csv_path)

    # Build strategy dict
    ai_strategy = _AIStrategy(engine, symbol, df)
    strategies: Dict[str, object] = {
        "AI Engine": ai_strategy,
        "Wyckoff": WyckoffStrategy(),
        "Technical": TechnicalStrategy(),
        "Momentum": MomentumStrategy(),
        "Mean Reversion": MeanReversionStrategy(),
    }

    # Configure backtest engine with capital
    if initial_capital is not None:
        bt_engine = BacktestEngine(
            initial_capital=initial_capital,
            max_position_pct=engine._config.max_position_pct,
            config=engine._config,
        )
    else:
        bt_engine = engine._backtest_engine

    return bt_engine.compare_strategies(strategies, df, start_date, end_date)


# ==============================================================================
# Background Training Functions
# ==============================================================================


def start_background_training(
    engine: "DecisionEngine",
    symbol_data: Optional[Dict[str, pd.DataFrame]] = None,
    mode: str = "incremental",
    symbols: Optional[List[str]] = None,
) -> bool:
    """Start background training without blocking inference.

    Inference continues to use the current model during training.
    When training completes, the model is hot-swapped within 3 seconds
    without dropping in-flight requests.

    If no symbol_data is provided, loads from base_dir for the given symbols
    (default: VN30 + VNINDEX).

    Requirements: 7.1, 7.2, 7.3, 6.11

    Parameters
    ----------
    engine : DecisionEngine
        The parent engine instance.
    symbol_data : Dict[str, pd.DataFrame], optional
        Pre-loaded DataFrames. If None, loads from CSV files.
    mode : str
        "full" for full retraining, "incremental" for fine-tuning.
    symbols : List[str], optional
        Symbols to train on. If None, uses VN30 + VNINDEX defaults.

    Returns
    -------
    bool
        True if training started, False if already running.
    """
    from engine.training_pipeline import TrainingPipeline

    # Set up training pipeline if not already configured
    if engine._background_training._training_pipeline is None:
        pipeline = TrainingPipeline()
        engine._background_training.set_training_pipeline(pipeline)

    # Load symbol data if not provided
    if symbol_data is None:
        symbol_data = {}
        if symbols is None:
            # Default VN30 + VNINDEX
            symbols = get_default_training_symbols()

        for symbol in symbols:
            csv_path = engine._resolve_csv_path(symbol)
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    symbol_data[symbol] = df
                except Exception as e:
                    logger.warning(f"Failed to load {symbol} for training: {e}")

    if not symbol_data:
        logger.error("No valid symbol data available for training")
        return False

    return engine._background_training.start_training(
        symbol_data=symbol_data,
        mode=mode,
        data_dir=str(engine._base_dir),
    )


def stop_background_training(engine: "DecisionEngine") -> None:
    """Stop background training gracefully.

    Training stops after the current epoch completes.
    Also signals the TrainingController for graceful stop (Req 14).
    """
    # Signal the TrainingController (if training is active via controller)
    if engine._training_controller.is_training:
        try:
            engine._training_controller.request_stop()
        except Exception:
            pass  # Controller may not have active training

    engine._background_training.stop_training()


def get_default_training_symbols() -> List[str]:
    """Get default list of training symbols (VN30 + VNINDEX).

    Returns
    -------
    List[str]
        Default symbol list for training.
    """
    vn30_symbols = [
        "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR",
        "HDB", "HPG", "MBB", "MSN", "MWG", "PLX", "POW", "SAB",
        "SHB", "SSB", "SSI", "STB", "TCB", "TPB", "VCB", "VHM",
        "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
    ]
    return ["VNINDEX"] + vn30_symbols
