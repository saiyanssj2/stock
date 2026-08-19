"""
Implementation helpers for DecisionEngine.

This module contains extracted implementation details to keep
decision_engine.py focused on the public API surface.
Thin delegates in DecisionEngine call into these helpers.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.config import (
    Action,
    BacktestResult,
    ComparisonResult,
    DataError,
    EngineConfig,
    SearchConfig,
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


# ==============================================================================
# AI Strategy Wrapper (for backtest integration)
# ==============================================================================


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
# Backtest / Compare Helpers
# ==============================================================================


def run_backtest(
    engine: "DecisionEngine",
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: Optional[float] = None,
) -> BacktestResult:
    """Implementation for DecisionEngine.backtest().

    Parameters
    ----------
    engine : DecisionEngine
        The engine instance (provides config, model, backtest engine).
    symbol : str
        Stock ticker symbol.
    start_date : str
        Start date (format: 'YYYY-MM-DD').
    end_date : str
        End date (format: 'YYYY-MM-DD').
    initial_capital : float, optional
        Starting capital in VND.

    Returns
    -------
    BacktestResult
        Backtest results with metrics and trade list.
    """
    from engine.backtest_engine import BacktestEngine

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
    """Implementation for DecisionEngine.compare().

    Parameters
    ----------
    engine : DecisionEngine
        The engine instance.
    symbol : str
        Stock ticker symbol.
    start_date : str
        Start date (format: 'YYYY-MM-DD').
    end_date : str
        End date (format: 'YYYY-MM-DD').
    initial_capital : float, optional
        Starting capital in VND.

    Returns
    -------
    ComparisonResult
        Comparison results with all strategy metrics.
    """
    from engine.backtest_engine import BacktestEngine

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


def run_compare_multi_symbol(
    engine: "DecisionEngine",
    start_date: str,
    end_date: str,
    initial_capital: Optional[float] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, ComparisonResult]:
    """Run each strategy across ALL available symbols and aggregate results.

    Instead of forcing all strategies onto a single symbol, this lets each
    strategy run on every symbol in the data directory. Returns per-strategy
    aggregated performance.

    Parameters
    ----------
    engine : DecisionEngine
        The engine instance.
    start_date : str
        Start date (format: 'YYYY-MM-DD').
    end_date : str
        End date (format: 'YYYY-MM-DD').
    initial_capital : float, optional
        Starting capital per symbol per strategy (default from config).
    data_dir : str, optional
        Data directory. If None uses engine base_dir.

    Returns
    -------
    Dict with keys:
        "per_strategy": Dict[strategy_name, aggregated BacktestResult]
        "per_symbol": Dict[symbol, Dict[strategy_name, BacktestResult]]
        "best_symbols": Dict[strategy_name, List[top symbols by return]]
        "metadata": Dict with date_range, capital, total_symbols
    """
    import os

    from engine.backtest_engine import BacktestEngine

    base_dir = Path(data_dir) if data_dir else engine._base_dir
    index_names = {"VNINDEX", "VN30", "HNX", "UPCOM", "HNX30"}

    # Collect all available CSV symbols
    symbols = []
    for f in os.listdir(base_dir):
        if f.endswith(".csv") and not f.startswith("."):
            sym = f.replace("_full.csv", "").replace(".csv", "")
            if sym.upper() not in index_names:
                symbols.append(sym)
    symbols = sorted(symbols)

    if not symbols:
        return {
            "per_strategy": {},
            "per_symbol": {},
            "best_symbols": {},
            "metadata": {"error": "No symbols found"},
        }

    # Configure backtest engine
    capital = initial_capital or engine._config.default_capital
    bt_engine = BacktestEngine(
        initial_capital=capital,
        max_position_pct=engine._config.max_position_pct,
        config=engine._config,
    )

    # Define strategies (no AI — too slow for multi-symbol scan)
    strategies_def = {
        "Wyckoff": WyckoffStrategy(),
        "Technical": TechnicalStrategy(),
        "Momentum": MomentumStrategy(),
        "Mean Reversion": MeanReversionStrategy(),
    }

    # Optionally include AI if model is loaded
    if engine.is_model_loaded:
        strategies_def["AI Engine"] = None  # Handled specially per symbol

    # Run backtests per symbol
    per_symbol: Dict[str, Dict[str, BacktestResult]] = {}
    per_strategy_results: Dict[str, List[BacktestResult]] = {
        name: [] for name in strategies_def
    }

    for sym in symbols:
        csv_path = base_dir / f"{sym}.csv"
        if not csv_path.exists():
            csv_path = base_dir / f"{sym}_full.csv"
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
            # Validate minimum data
            if len(df) < 60:
                continue
            # Check date column exists
            if "time" not in df.columns:
                continue
        except Exception:
            continue

        per_symbol[sym] = {}

        for strat_name, strat_instance in strategies_def.items():
            try:
                if strat_name == "AI Engine":
                    # Create AI strategy for this symbol
                    strat_instance = _AIStrategy(engine, sym, df)

                result, has_buy, has_sell = bt_engine._run_with_signal_tracking(
                    strat_instance, df, start_date, end_date
                )

                if has_buy and has_sell and len(result.trades) > 0:
                    per_symbol[sym][strat_name] = result
                    per_strategy_results[strat_name].append(result)
            except Exception:
                continue

    # Aggregate per-strategy results
    per_strategy_agg: Dict[str, dict] = {}
    best_symbols: Dict[str, List] = {}

    for strat_name, results in per_strategy_results.items():
        if not results:
            per_strategy_agg[strat_name] = {
                "total_return_pct": 0.0,
                "avg_return_pct": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "symbols_traded": 0,
                "total_trades": 0,
            }
            best_symbols[strat_name] = []
            continue

        returns = [r.total_return_pct for r in results]
        win_rates = [r.win_rate for r in results]
        drawdowns = [r.max_drawdown for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        trades_count = sum(len(r.trades) for r in results)

        per_strategy_agg[strat_name] = {
            "total_return_pct": sum(returns) / len(returns),
            "avg_return_pct": sum(returns) / len(returns),
            "win_rate": sum(win_rates) / len(win_rates),
            "max_drawdown": min(drawdowns),  # worst drawdown
            "sharpe_ratio": sum(sharpes) / len(sharpes),
            "symbols_traded": len(results),
            "total_trades": trades_count,
        }

        # Find best symbols for this strategy
        symbol_returns = []
        for sym, sym_results in per_symbol.items():
            if strat_name in sym_results:
                symbol_returns.append(
                    (sym, sym_results[strat_name].total_return_pct)
                )
        symbol_returns.sort(key=lambda x: x[1], reverse=True)
        best_symbols[strat_name] = symbol_returns[:5]

    return {
        "per_strategy": per_strategy_agg,
        "per_symbol": per_symbol,
        "best_symbols": best_symbols,
        "metadata": {
            "date_range_start": start_date,
            "date_range_end": end_date,
            "initial_capital": capital,
            "total_symbols": len(symbols),
            "symbols_with_data": len(per_symbol),
        },
    }


# ==============================================================================
# Background Training Helpers
# ==============================================================================


def start_background_training_impl(
    engine: "DecisionEngine",
    symbol_data: Optional[Dict[str, pd.DataFrame]] = None,
    mode: str = "incremental",
    symbols: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
    resume_from_checkpoint: bool = False,
    session_state_dict: Optional[Dict] = None,
) -> bool:
    """Implementation for DecisionEngine.start_background_training().

    Parameters
    ----------
    engine : DecisionEngine
        The engine instance.
    symbol_data : Dict[str, pd.DataFrame], optional
        Pre-loaded DataFrames. If None, loads from CSV files.
    mode : str
        "full" for full retraining, "incremental" for fine-tuning.
    symbols : List[str], optional
        Symbols to train on. If None, uses VN30 + VNINDEX defaults.
    data_dir : str, optional
        Base directory for data loading. If None, uses engine's base_dir.
    resume_from_checkpoint : bool
        If True, resume from existing checkpoint.
    session_state_dict : Dict, optional
        Reference to st.session_state for progress updates.

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
            symbols = engine._get_default_training_symbols()

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

    effective_data_dir = data_dir if data_dir is not None else str(engine._base_dir)

    return engine._background_training.start_training(
        symbol_data=symbol_data,
        mode=mode,
        data_dir=effective_data_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        session_state_dict=session_state_dict,
    )
