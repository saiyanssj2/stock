"""
Progressive Training - Phase C Logic.

Phase C: Train with simple labels (future return → tanh), validate with deep
search against philosophy strategies.

Extracted from progressive_trainer.py for maintainability.

Requirements: 16.2, 16.3, 16.8
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from engine.progressive_trainer import ProgressiveTrainer
    from engine.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


def run_phase_c(
    trainer: "ProgressiveTrainer",
    symbol_data: Dict[str, pd.DataFrame],
    data_dir: Optional[str] = None,
    validation_symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> "ValidationRound":
    """Phase C: Train with simple labels, validate with deep search.

    This function:
    1. Trains the model using simple labels (future return → tanh)
       via the standard training pipeline.
    2. Validates the model by running a backtest on out-of-sample data
       using a deep-search-based strategy (depth 5-7, 5 min/sample).
    3. Compares the model's Sharpe ratio against all 4 philosophy
       strategy Sharpe ratios on the same out-of-sample period.
    4. Selects the best model based on Sharpe ratio.
    5. Records the ValidationRound result to a local log file.

    Args:
        trainer: The ProgressiveTrainer instance.
        symbol_data: Dict mapping symbol names to DataFrames with
            OHLCV + indicators data for training.
        data_dir: Optional base directory for loading additional data.
        validation_symbols: Symbols to use for out-of-sample validation.
            If None, uses all available symbols from symbol_data.
        start_date: Start date for validation backtest period.
            If None, uses last 20% of data as out-of-sample period.
        end_date: End date for validation backtest period.
            If None, uses the last available date.

    Returns:
        ValidationRound with the results of this validation round.

    Raises:
        EngineError: If training pipeline is not set or no strategies
            are configured.

    Requirements: 16.2, 16.8
    """
    from engine.config import EngineError
    from engine.progressive_trainer import TrainingPhase, ValidationRound

    if trainer._pipeline is None:
        raise EngineError(
            "TrainingPipeline not configured for progressive training",
            error_code="PROGRESSIVE_NO_PIPELINE",
        )

    if not trainer._strategies:
        raise EngineError(
            "No philosophy strategies configured for comparison",
            error_code="PROGRESSIVE_NO_STRATEGIES",
        )

    # Step 1: Train model with simple labels (future return → tanh)
    logger.info("Phase C: Starting model training with simple labels...")
    training_result = trainer._pipeline.train_full(
        symbol_data=symbol_data,
        data_dir=data_dir,
        resume=True,
    )
    logger.info(
        f"Phase C: Training complete - {training_result.epochs_completed} epochs, "
        f"best_val_loss={training_result.best_val_loss:.6f}"
    )

    # Step 2: Load the best trained model for validation
    model_path = training_result.model_path
    if model_path is None:
        model_path = str(
            Path(trainer._pipeline.config.checkpoint_dir) / "stock_eval_net.pt"
        )

    model_version = f"phase_c_v{len(trainer._validation_history) + 1}"

    # Step 3: Run deep-search validation on out-of-sample data
    logger.info(
        f"Phase C: Validating with deep search "
        f"(depth={trainer._config.phase_c_search_depth}, "
        f"timeout={trainer._config.phase_c_search_timeout}s per sample)..."
    )

    # Determine validation period
    val_symbols = validation_symbols or list(symbol_data.keys())
    if not val_symbols:
        raise EngineError(
            "No symbols available for validation",
            error_code="PROGRESSIVE_NO_VAL_DATA",
        )

    # Use the first available symbol with sufficient data for backtesting
    val_symbol = val_symbols[0]
    val_df = symbol_data[val_symbol]

    # Determine date range for out-of-sample validation
    if start_date is None or end_date is None:
        dates = pd.to_datetime(val_df["time"]) if "time" in val_df.columns else val_df.index
        if hasattr(dates, "values"):
            dates = pd.Series(dates.values) if not isinstance(dates, pd.Series) else dates
        total_days = len(dates)
        # Use last 20% as out-of-sample period
        oos_start_idx = int(total_days * 0.80)
        if start_date is None:
            start_date = str(dates.iloc[oos_start_idx])[:10]
        if end_date is None:
            end_date = str(dates.iloc[-1])[:10]

    # Step 4: Compute model Sharpe via deep-search backtest
    model_sharpe = trainer._run_model_backtest(
        model_path=model_path,
        val_df=val_df,
        symbol=val_symbol,
        start_date=start_date,
        end_date=end_date,
    )

    # Step 5: Compute philosophy strategy Sharpes on same period
    strategy_sharpes = trainer._run_strategy_backtests(
        strategies=trainer._strategies,
        val_df=val_df,
        start_date=start_date,
        end_date=end_date,
    )

    # Step 6: Record ValidationRound
    round_number = len(trainer._validation_history) + 1
    validation_round = ValidationRound(
        round_number=round_number,
        model_version=model_version,
        model_sharpe=model_sharpe,
        strategy_sharpes=strategy_sharpes,
        phase=TrainingPhase.PHASE_C,
    )

    # Step 7: Log validation round to local file (Req 16.8)
    trainer._log_validation_round(validation_round)

    # Step 8: Check transition criteria
    transition_available = trainer.check_c_to_b_transition(validation_round)

    if transition_available:
        logger.info(
            f"Phase C: Transition to Phase B available! "
            f"Model Sharpe ({model_sharpe:.4f}) beats "
            f"{trainer.transition_criteria.required_strategies_beaten}/"
            f"{trainer.transition_criteria.total_strategies} strategies "
            f"for {trainer.transition_criteria.consecutive_wins} consecutive rounds."
        )
    else:
        strategies_beaten = sum(
            1 for s in strategy_sharpes if model_sharpe > s
        )
        logger.info(
            f"Phase C: Validation round {round_number} complete. "
            f"Model Sharpe={model_sharpe:.4f}, "
            f"beats {strategies_beaten}/{len(strategy_sharpes)} strategies. "
            f"Consecutive wins: {trainer.transition_criteria.consecutive_wins}/"
            f"{trainer.transition_criteria.required_consecutive_wins}"
        )

    return validation_round


def _run_model_backtest(
    trainer: "ProgressiveTrainer",
    model_path: str,
    val_df: pd.DataFrame,
    symbol: str,
    start_date: str,
    end_date: str,
) -> float:
    """Run a backtest using the trained model with deep search configuration.

    Creates a model-based strategy that uses the evaluation model for
    signal generation, configured with Phase C's deep search parameters.

    Args:
        trainer: The ProgressiveTrainer instance.
        model_path: Path to the trained model weights.
        val_df: DataFrame with OHLCV + indicators for backtesting.
        symbol: Stock symbol name.
        start_date: Backtest start date.
        end_date: Backtest end date.

    Returns:
        Sharpe ratio from the model backtest.
    """
    from engine.backtest_engine import BacktestEngine
    from engine.evaluation_model import ModelManager

    # Load model for validation
    model_manager = ModelManager()
    try:
        model_manager.load(model_path)
    except Exception as e:
        logger.warning(f"Phase C: Failed to load model for validation: {e}")
        return 0.0

    # Create a deep-search model strategy for backtesting
    model_strategy = _PhaseCModelStrategy(
        model_manager=model_manager,
        symbol=symbol,
        lookback=trainer._pipeline.model_config.lookback,
        search_depth=trainer._config.phase_c_search_depth,
        search_timeout=trainer._config.phase_c_search_timeout,
    )

    # Run backtest
    backtest_engine = BacktestEngine()
    try:
        result = backtest_engine.run(
            strategy=model_strategy,
            df=val_df,
            start_date=start_date,
            end_date=end_date,
        )
        return result.sharpe_ratio
    except Exception as e:
        logger.warning(f"Phase C: Model backtest failed: {e}")
        return 0.0


def _run_strategy_backtests(
    strategies: List["BaseStrategy"],
    val_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> List[float]:
    """Run backtests for all philosophy strategies on the same period.

    Args:
        strategies: List of philosophy strategy instances.
        val_df: DataFrame with OHLCV + indicators for backtesting.
        start_date: Backtest start date.
        end_date: Backtest end date.

    Returns:
        List of Sharpe ratios, one per strategy.
    """
    from engine.backtest_engine import BacktestEngine

    backtest_engine = BacktestEngine()
    sharpes: List[float] = []

    for strategy in strategies:
        try:
            result = backtest_engine.run(
                strategy=strategy,
                df=val_df,
                start_date=start_date,
                end_date=end_date,
            )
            sharpes.append(result.sharpe_ratio)
            logger.info(
                f"Phase C: Strategy '{strategy.name}' "
                f"Sharpe={result.sharpe_ratio:.4f}"
            )
        except Exception as e:
            logger.warning(
                f"Phase C: Strategy '{strategy.name}' backtest failed: {e}"
            )
            sharpes.append(0.0)

    return sharpes


def _log_validation_round(validation_round: "ValidationRound") -> None:
    """Record ValidationRound results to a local log file.

    Writes the validation round as a JSON line to the validation log file
    located in the models directory.

    Args:
        validation_round: The completed validation round to log.

    Requirements: 16.8
    """
    log_dir = Path("engine/models")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "validation_rounds.jsonl"

    entry = {
        "timestamp": validation_round.timestamp.isoformat(),
        "round_number": validation_round.round_number,
        "model_version": validation_round.model_version,
        "model_sharpe": float(validation_round.model_sharpe),
        "strategy_sharpes": [float(s) for s in validation_round.strategy_sharpes],
        "phase": validation_round.phase.value,
        "strategies_beaten": sum(
            1
            for s in validation_round.strategy_sharpes
            if validation_round.model_sharpe > s
        ),
        "total_strategies": len(validation_round.strategy_sharpes),
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(
        f"Phase C: ValidationRound {validation_round.round_number} logged to "
        f"{log_file}"
    )


def check_c_to_b_transition_impl(
    trainer: "ProgressiveTrainer",
    validation_result: "ValidationRound",
) -> bool:
    """Check Phase C → B transition criteria.

    Criteria: Model Sharpe > at least 2 of 4 strategy Sharpes for
    3 consecutive validation rounds.

    This function:
    1. Records the validation round in history
    2. Counts how many strategies the model beats
    3. Updates the consecutive win streak
    4. Flags transition as available if criteria are met

    Args:
        trainer: The ProgressiveTrainer instance.
        validation_result: The completed validation round to evaluate.

    Returns:
        True if transition criteria are now met (transition available).
        False if criteria are not yet met.

    Requirements: 16.3
    """
    # Record in history
    trainer._validation_history.append(validation_result)

    # Count strategies beaten (model Sharpe > strategy Sharpe)
    strategies_beaten = sum(
        1
        for strategy_sharpe in validation_result.strategy_sharpes
        if validation_result.model_sharpe > strategy_sharpe
    )

    # Update consecutive wins
    if strategies_beaten >= trainer.transition_criteria.required_strategies_beaten:
        trainer.transition_criteria.consecutive_wins += 1
    else:
        # Reset streak on failure
        trainer.transition_criteria.consecutive_wins = 0

    # Check if threshold reached
    if (
        trainer.transition_criteria.consecutive_wins
        >= trainer.transition_criteria.required_consecutive_wins
    ):
        trainer.transition_criteria.transition_available = True
        logger.info(
            "Phase C→B transition available: model beat %d/%d strategies "
            "for %d consecutive rounds",
            trainer.transition_criteria.required_strategies_beaten,
            trainer.transition_criteria.total_strategies,
            trainer.transition_criteria.consecutive_wins,
        )
        return True

    return False



# ==============================================================================
# Phase C Model Strategy (used internally for validation backtests)
# ==============================================================================


class _PhaseCModelStrategy:
    """Strategy adapter that uses the evaluation model + deep search for signals.

    This class wraps the trained model and search module to produce trading
    signals compatible with the BacktestEngine's Strategy protocol. It enables
    Phase C validation by running the model-based decision engine against
    philosophy strategies on the same out-of-sample data.

    The strategy performs deep search (depth 5-7, up to 5 min/sample) as
    specified by Phase C requirements to thoroughly evaluate each trading day.

    Requirements: 16.2
    """

    def __init__(
        self,
        model_manager,
        symbol: str,
        lookback: int = 60,
        search_depth: int = 5,
        search_timeout: float = 300.0,
    ):
        """Initialize the Phase C model strategy.

        Args:
            model_manager: ModelManager with a loaded evaluation model.
            symbol: Stock symbol being analyzed.
            lookback: Lookback window for MarketState construction.
            search_depth: Search tree depth for validation (5-7).
            search_timeout: Maximum time per search in seconds (5 min default).
        """
        self._model_manager = model_manager
        self._symbol = symbol
        self._lookback = lookback
        self._search_depth = search_depth
        self._search_timeout = search_timeout

        # Lazily initialized
        self._search_module = None
        self._scenario_generator = None
        self._feature_builder = None

    @property
    def name(self) -> str:
        """Human-readable name for the strategy."""
        return f"Model_PhaseC_d{self._search_depth}"

    def _ensure_initialized(self) -> None:
        """Lazily initialize the search module and scenario generator."""
        if self._search_module is not None:
            return

        from engine.config import SearchConfig
        from engine.scenario_generator import ScenarioGenerator
        from engine.search_module import SearchModule

        # Configure search for Phase C deep validation
        search_config = SearchConfig(
            default_depth=self._search_depth,
            max_depth=max(self._search_depth, 7),
            timeout_seconds=self._search_timeout,
        )

        self._scenario_generator = ScenarioGenerator(
            search_config=search_config,
        )

        self._search_module = SearchModule(
            model_manager=self._model_manager,
            scenario_generator=self._scenario_generator,
            config=search_config,
        )

    def generate_signal(self, df: pd.DataFrame, index: int) -> "Action":
        """Generate a trading signal using model + deep search.

        Constructs a MarketState from the data up to and including the given
        index, then runs the search module to determine the best action.

        For efficiency during backtesting, this uses the model's direct
        prediction when the search would be too slow (falls back to simple
        evaluation if search takes too long for a single day).

        Args:
            df: Full historical DataFrame with OHLCV and indicators.
            index: Current row index to evaluate.

        Returns:
            Action: BUY, HOLD, or SELL signal.
        """
        from engine.config import Action

        self._ensure_initialized()

        from engine.market_state import MarketState

        # Need enough lookback data
        if index < self._lookback:
            return Action.HOLD

        try:
            # Extract the window of data ending at current index
            window_start = index - self._lookback + 1
            window_df = df.iloc[window_start : index + 1].copy()

            if len(window_df) < self._lookback:
                return Action.HOLD

            # Build MarketState from the window
            state = MarketState.from_dataframe(
                df=df.iloc[: index + 1],
                symbol=self._symbol,
                lookback=self._lookback,
            )

            # Run search with the configured depth and timeout
            search_start = time.time()
            result = self._search_module.search(
                state=state,
                depth=self._search_depth,
            )
            search_elapsed = time.time() - search_start

            logger.debug(
                f"Phase C search: action={result.best_action.value}, "
                f"score={result.best_score:.4f}, "
                f"nodes={result.nodes_evaluated}, "
                f"time={search_elapsed:.1f}s"
            )

            return result.best_action

        except Exception as e:
            # On any error during search, fall back to model-only evaluation
            logger.debug(
                f"Phase C search fallback to model-only at index {index}: {e}"
            )
            return self._model_only_signal(df, index)

    def _model_only_signal(self, df: pd.DataFrame, index: int) -> "Action":
        """Fall back to using only the model's direct evaluation.

        When search fails (e.g., insufficient data for scenario generation),
        use the raw model score to generate a signal.

        Args:
            df: Full historical DataFrame.
            index: Current row index.

        Returns:
            Action based on model's position score.
        """
        from engine.config import Action
        from engine.market_state import FeatureVectorBuilder, MarketState

        try:
            state = MarketState.from_dataframe(
                df=df.iloc[: index + 1],
                symbol=self._symbol,
                lookback=self._lookback,
            )

            # Build feature vector and get model prediction
            if self._feature_builder is None:
                norm_params_path = str(
                    Path("engine/models") / "norm_params.json"
                )
                try:
                    self._feature_builder = FeatureVectorBuilder(
                        norm_params_path=norm_params_path
                    )
                except Exception:
                    return Action.HOLD

            feature_tensor = self._feature_builder.build(state)
            score = self._model_manager.predict(feature_tensor.numpy())

            # Convert score to action using simple thresholds
            score_val = float(score.flatten()[0])
            if score_val > 0.3:
                return Action.BUY
            elif score_val < -0.3:
                return Action.SELL
            else:
                return Action.HOLD

        except Exception as e:
            logger.debug(f"Model-only signal failed at index {index}: {e}")
            return Action.HOLD
