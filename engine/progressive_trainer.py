"""
Progressive Training Strategy - Phase C → B → A transition logic.

Implements the progressive training roadmap that evolves the model from
simple supervised learning (Phase C) through enhanced label generation
(Phase B) to self-play (Phase A). Phase transitions require meeting
specific performance criteria AND explicit user confirmation.

Components:
- TrainingPhase enum: PHASE_C, PHASE_B, PHASE_A
- PhaseTransitionCriteria: tracks progress toward next transition
- ValidationRound: records a single validation round result
- ProgressiveTrainingConfig: phase-specific search depth/timeout settings
- ProgressiveTrainer: main class orchestrating phase logic and transitions

Requirements: 16.1, 16.3, 16.5, 16.9, 16.10
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from engine.config import (
    Action,
    BacktestResult,
    EngineConfig,
    EngineError,
    ModelConfig,
    SearchConfig,
    TrainingConfig,
)

if TYPE_CHECKING:
    from engine.strategies.base import BaseStrategy
    from engine.training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


# ==============================================================================
# TrainingPhase Enum
# ==============================================================================


class TrainingPhase(Enum):
    """Progressive training phases from simplest to most advanced.

    Phase C (Validation): Train with simple labels (future return → tanh),
        validate with deep search against philosophy strategies.
    Phase B (Deep Labels): Use best model + deep search to generate
        enhanced training labels, retrain model.
    Phase A (Self-Play): Self-play training loop with unlimited search
        depth and no time constraint per move.

    Requirements: 16.1
    """

    PHASE_C = "phase_c_validation"
    PHASE_B = "phase_b_deep_labels"
    PHASE_A = "phase_a_self_play"


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class PhaseTransitionCriteria:
    """Tracks criteria for phase transitions.

    C → B transition: Model must achieve higher Sharpe ratio than at least
    2 of 4 philosophy strategies for 3 consecutive validation rounds.

    B → A transition: ScenarioGenerator must achieve >60% directional
    accuracy on at least 100 validation samples.

    Both transitions require explicit user confirmation before advancing.

    Requirements: 16.3, 16.5, 16.9
    """

    # C → B: Model beats 2/4 strategies for 3 consecutive rounds
    consecutive_wins: int = 0
    required_consecutive_wins: int = 3
    required_strategies_beaten: int = 2
    total_strategies: int = 4

    # B → A: ScenarioGenerator >60% accuracy on 100+ samples
    scenario_accuracy: float = 0.0
    required_accuracy: float = 0.60
    accuracy_sample_count: int = 0
    required_sample_count: int = 100

    # User confirmation gate
    transition_available: bool = False
    user_confirmed: bool = False


@dataclass
class ValidationRound:
    """Records a single validation round result.

    Each validation round compares the model's Sharpe ratio against the
    philosophy strategies' Sharpe ratios on out-of-sample data.

    Attributes:
        round_number: Sequential round number.
        model_version: Identifier for the model version tested.
        model_sharpe: Model's Sharpe ratio on validation data.
        strategy_sharpes: List of Sharpe ratios for each philosophy strategy.
        phase: The training phase during which this round was executed.
        timestamp: When this validation round was recorded.
    """

    round_number: int
    model_version: str
    model_sharpe: float
    strategy_sharpes: List[float]
    phase: TrainingPhase
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProgressiveTrainingConfig:
    """Configuration for each progressive training phase.

    Controls search depth and timeout settings per phase, allowing
    increasingly thorough evaluation as the model advances.

    Phase C: Moderate depth (5-7), 5 min/sample for validation.
    Phase B: Higher depth (5-7), up to 1 hour/sample for label generation.
    Phase A: Unlimited depth, no time constraint for self-play.

    Attributes:
        phase_c_search_depth: Search depth for Phase C validation (5-7).
        phase_c_search_timeout: Timeout per sample in Phase C (seconds).
        phase_b_search_depth: Search depth for Phase B label generation (5-7).
        phase_b_search_timeout: Timeout per sample in Phase B (seconds).
        phase_a_search_depth: Search depth for Phase A (-1 = unlimited).
        phase_a_search_timeout: Timeout per move in Phase A (-1 = no limit).
    """

    # Phase C config
    phase_c_search_depth: int = 5
    phase_c_search_timeout: float = 300.0  # 5 minutes per sample

    # Phase B config
    phase_b_search_depth: int = 7
    phase_b_search_timeout: float = 3600.0  # Up to 1 hour per sample

    # Phase A config
    phase_a_search_depth: int = -1  # Unlimited (-1)
    phase_a_search_timeout: float = -1  # No time constraint (-1)


# ==============================================================================
# ProgressiveTrainer
# ==============================================================================


class ProgressiveTrainer:
    """Implements the C → B → A progressive training strategy.

    The ProgressiveTrainer manages phase transitions based on performance
    criteria and user confirmation. It tracks validation history and
    provides transition state information for UI display.

    Key behaviors:
    - Starts in Phase C (simplest supervised learning)
    - Checks transition criteria after each validation round
    - Flags transition as available when criteria are met
    - Only advances phase after explicit user confirmation (Req 16.9)
    - Allows manual revert to earlier phases (Req 16.10)
    - Records all validation rounds for history tracking

    Requirements: 16.1, 16.3, 16.5, 16.9, 16.10
    """

    def __init__(
        self,
        config: Optional[ProgressiveTrainingConfig] = None,
        training_pipeline: Optional["TrainingPipeline"] = None,
        strategies: Optional[List["BaseStrategy"]] = None,
    ):
        """Initialize ProgressiveTrainer.

        Args:
            config: Phase-specific configuration. Uses defaults if None.
            training_pipeline: Reference to the training pipeline for
                executing training within each phase.
            strategies: List of philosophy strategies for comparison.
                Expected to contain 4 strategies (Wyckoff, Technical,
                Momentum, Mean Reversion).
        """
        self._config = config or ProgressiveTrainingConfig()
        self._pipeline = training_pipeline
        self._strategies = strategies or []

        # Phase state
        self.current_phase: TrainingPhase = TrainingPhase.PHASE_C
        self.transition_criteria: PhaseTransitionCriteria = PhaseTransitionCriteria()

        # Validation history
        self._validation_history: List[ValidationRound] = []

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def config(self) -> ProgressiveTrainingConfig:
        """Current progressive training configuration."""
        return self._config

    @property
    def validation_history(self) -> List[ValidationRound]:
        """Complete validation round history."""
        return list(self._validation_history)

    @property
    def strategies(self) -> List["BaseStrategy"]:
        """Philosophy strategies used for comparison."""
        return self._strategies

    # ==========================================================================
    # Phase Execution (thin delegates)
    # ==========================================================================

    def run_phase_b(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        data_dir: Optional[str] = None,
        num_label_samples: Optional[int] = None,
        validation_samples: int = 100,
    ) -> Dict[str, object]:
        """Phase B: Enhanced label generation with deep search + model retrain.

        Args:
            symbol_data: Dict mapping symbol names to DataFrames with
                OHLCV + indicators data.
            data_dir: Optional base directory for loading additional data.
            num_label_samples: Maximum number of samples to generate enhanced
                labels for. If None, processes all available samples.
            validation_samples: Number of samples for ScenarioGenerator
                directional accuracy validation (default 100).

        Returns:
            Dict with keys: enhanced_labels_generated, training_result,
            directional_accuracy, accuracy_samples, transition_available.

        Raises:
            EngineError: If pipeline not set, no model, or wrong phase.

        Requirements: 16.4, 16.5
        """
        from engine.progressive_phase_b import run_phase_b
        return run_phase_b(
            self,
            symbol_data=symbol_data,
            data_dir=data_dir,
            num_label_samples=num_label_samples,
            validation_samples=validation_samples,
        )

    def run_phase_a(
        self,
        symbol_data: Optional[Dict[str, pd.DataFrame]] = None,
        data_dir: Optional[str] = None,
        config: Optional["PhaseASelfPlayConfig"] = None,
    ) -> "PhaseAResult":
        """Phase A: Self-play training loop.

        In Phase A, the model trains against ScenarioGenerator market
        simulations with unlimited search depth and no time constraint
        per move. The model generates its own training labels by running
        deep iterative-deepening search, then retrains on these labels
        in an iterative self-improvement loop.

        This is the most advanced training phase, analogous to AlphaZero's
        self-play where the model bootstraps its own improvement.

        The search runs at "unlimited" depth via iterative deepening:
        starting at depth 3 and increasing until the evaluation score
        converges (change between depths < threshold) or a practical
        maximum cap is reached. No time constraint is imposed per move.

        Args:
            symbol_data: Dict mapping symbol names to DataFrames with
                OHLCV + indicator data. Required for self-play.
            data_dir: Optional base directory for loading additional data.
            config: Optional PhaseASelfPlayConfig to override defaults.
                Controls iterations, samples, convergence thresholds.

        Returns:
            PhaseAResult with complete self-play training results including
            per-iteration metrics, convergence status, and model path.

        Raises:
            EngineError: If training pipeline is not set, current phase
                is not PHASE_A, or no data is provided.

        Requirements: 16.6
        """
        if self._pipeline is None:
            raise EngineError(
                "TrainingPipeline not configured for progressive training",
                error_code="PROGRESSIVE_NO_PIPELINE",
            )

        if self.current_phase != TrainingPhase.PHASE_A:
            raise EngineError(
                f"Cannot run Phase A: current phase is {self.current_phase.value}. "
                "Transition to Phase A first.",
                error_code="PROGRESSIVE_WRONG_PHASE",
            )

        if symbol_data is None:
            raise EngineError(
                "Phase A requires symbol_data to be provided",
                error_code="PHASE_A_NO_DATA",
            )

        from engine.phase_a_selfplay import PhaseASelfPlayConfig, PhaseASelfPlayEngine

        # Use provided config or build from progressive training config
        if config is None:
            config = PhaseASelfPlayConfig(
                search_depth=self._config.phase_a_search_depth,
                search_timeout=self._config.phase_a_search_timeout,
            )

        engine = PhaseASelfPlayEngine(
            pipeline=self._pipeline,
            config=config,
        )

        logger.info("Phase A: Starting self-play training loop...")
        result = engine.run(symbol_data=symbol_data, data_dir=data_dir)
        logger.info(
            f"Phase A: Self-play complete - "
            f"{result.iterations_completed} iterations, "
            f"converged={result.converged}"
        )

        return result

    def run_phase_c(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        data_dir: Optional[str] = None,
        validation_symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> "ValidationRound":
        """Phase C: Train with simple labels, validate with deep search.

        Args:
            symbol_data: Dict mapping symbol names to DataFrames with
                OHLCV + indicators data for training.
            data_dir: Optional base directory for loading additional data.
            validation_symbols: Symbols for out-of-sample validation.
            start_date: Start date for validation backtest period.
            end_date: End date for validation backtest period.

        Returns:
            ValidationRound with the results of this validation round.

        Raises:
            EngineError: If pipeline not set or no strategies configured.

        Requirements: 16.2, 16.8
        """
        from engine.progressive_phase_c import run_phase_c
        return run_phase_c(
            self,
            symbol_data=symbol_data,
            data_dir=data_dir,
            validation_symbols=validation_symbols,
            start_date=start_date,
            end_date=end_date,
        )

    # ==========================================================================
    # Private helpers (thin delegates to keep tests working)
    # ==========================================================================

    def _generate_enhanced_labels(
        self,
        df: pd.DataFrame,
        symbol: str,
        search_module,
        lookback: int,
        num_samples: int,
    ) -> np.ndarray:
        """Generate enhanced labels for a symbol using deep search."""
        from engine.progressive_phase_b import _generate_enhanced_labels
        return _generate_enhanced_labels(
            trainer=self,
            df=df,
            symbol=symbol,
            search_module=search_module,
            lookback=lookback,
            num_samples=num_samples,
        )

    def _retrain_with_enhanced_labels(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        enhanced_labels_map: Dict[str, np.ndarray],
        valid_symbols: List[str],
    ):
        """Retrain the model using enhanced (deep search) labels."""
        from engine.progressive_phase_b import _retrain_with_enhanced_labels
        return _retrain_with_enhanced_labels(
            trainer=self,
            symbol_data=symbol_data,
            enhanced_labels_map=enhanced_labels_map,
            valid_symbols=valid_symbols,
        )

    def _validate_scenario_accuracy(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        valid_symbols: List[str],
        num_samples: int,
        lookback: int,
    ) -> tuple:
        """Validate ScenarioGenerator directional accuracy."""
        from engine.progressive_phase_b import _validate_scenario_accuracy
        return _validate_scenario_accuracy(
            trainer=self,
            symbol_data=symbol_data,
            valid_symbols=valid_symbols,
            num_samples=num_samples,
            lookback=lookback,
        )

    def _log_phase_b_result(
        self,
        total_labels_generated: int,
        training_result,
        directional_accuracy: float,
        accuracy_samples: int,
        transition_available: bool,
    ) -> None:
        """Log Phase B results to the validation log file."""
        from engine.progressive_phase_b import _log_phase_b_result
        _log_phase_b_result(
            total_labels_generated=total_labels_generated,
            training_result=training_result,
            directional_accuracy=directional_accuracy,
            accuracy_samples=accuracy_samples,
            transition_available=transition_available,
        )

    def _run_model_backtest(
        self,
        model_path: str,
        val_df: pd.DataFrame,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> float:
        """Run a backtest using the trained model with deep search configuration."""
        from engine.progressive_phase_c import _run_model_backtest
        return _run_model_backtest(
            trainer=self,
            model_path=model_path,
            val_df=val_df,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

    def _run_strategy_backtests(
        self,
        strategies: List["BaseStrategy"],
        val_df: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> List[float]:
        """Run backtests for all philosophy strategies on the same period."""
        from engine.progressive_phase_c import _run_strategy_backtests
        return _run_strategy_backtests(
            strategies=strategies,
            val_df=val_df,
            start_date=start_date,
            end_date=end_date,
        )

    def _log_validation_round(self, validation_round: ValidationRound) -> None:
        """Record ValidationRound results to a local log file."""
        from engine.progressive_phase_c import _log_validation_round
        _log_validation_round(validation_round)

    # ==========================================================================
    # Transition Checks
    # ==========================================================================

    def check_c_to_b_transition(self, validation_result: ValidationRound) -> bool:
        """Check Phase C → B transition criteria.

        Criteria: Model Sharpe > at least 2 of 4 strategy Sharpes for
        3 consecutive validation rounds.

        This method:
        1. Records the validation round in history
        2. Counts how many strategies the model beats
        3. Updates the consecutive win streak
        4. Flags transition as available if criteria are met

        The transition is flagged but NOT executed - user must call
        confirm_transition() to actually advance the phase.

        Args:
            validation_result: The completed validation round to evaluate.

        Returns:
            True if transition criteria are now met (transition available).
            False if criteria are not yet met.

        Requirements: 16.3
        """
        from engine.progressive_phase_c import check_c_to_b_transition_impl
        return check_c_to_b_transition_impl(self, validation_result)

    def check_b_to_a_transition(
        self, predictions: List[float], actuals: List[float]
    ) -> bool:
        """Check Phase B → A transition criteria.

        Criteria: ScenarioGenerator achieves >60% directional accuracy
        on at least 100 validation samples.

        Directional accuracy = proportion of samples where the sign of
        the prediction matches the sign of the actual value.
        Sign matching rules:
        - pred > 0 and actual > 0 → correct
        - pred < 0 and actual < 0 → correct
        - pred == 0 and actual == 0 → correct
        - All other combinations → incorrect

        Args:
            predictions: List of predicted values (e.g., predicted returns).
            actuals: List of actual values (e.g., actual returns).

        Returns:
            True if transition criteria are now met (transition available).
            False if criteria are not yet met (insufficient samples or
            accuracy below threshold).

        Requirements: 16.5
        """
        if len(predictions) < self.transition_criteria.required_sample_count:
            self.transition_criteria.accuracy_sample_count = len(predictions)
            return False

        # Compute directional accuracy
        correct_direction = sum(
            1
            for pred, actual in zip(predictions, actuals)
            if (pred > 0 and actual > 0)
            or (pred < 0 and actual < 0)
            or (pred == 0 and actual == 0)
        )
        accuracy = correct_direction / len(predictions)

        # Update criteria state
        self.transition_criteria.scenario_accuracy = accuracy
        self.transition_criteria.accuracy_sample_count = len(predictions)

        # Check threshold
        if accuracy > self.transition_criteria.required_accuracy:
            self.transition_criteria.transition_available = True
            logger.info(
                "Phase B→A transition available: %.1f%% directional accuracy "
                "on %d samples (threshold: >%.0f%%)",
                accuracy * 100,
                len(predictions),
                self.transition_criteria.required_accuracy * 100,
            )
            return True

        return False

    # ==========================================================================
    # Phase Transitions
    # ==========================================================================

    def confirm_transition(self) -> TrainingPhase:
        """Confirm and execute phase transition.

        Advances to the next phase only after transition criteria are met
        AND this method is explicitly called (representing user confirmation).

        Phase transitions:
        - PHASE_C → PHASE_B
        - PHASE_B → PHASE_A
        - PHASE_A → (no further transition possible)

        After transition, resets the transition criteria for the new phase.

        Returns:
            The new TrainingPhase after advancing.

        Raises:
            EngineError: If no transition is currently available
                (criteria not met or already at Phase A).
                Error code: NO_TRANSITION

        Requirements: 16.9
        """
        if not self.transition_criteria.transition_available:
            raise EngineError(
                "No phase transition available",
                error_code="NO_TRANSITION",
            )

        if self.current_phase == TrainingPhase.PHASE_C:
            self.current_phase = TrainingPhase.PHASE_B
            logger.info("Phase transition confirmed: C → B (Deep Labels)")
        elif self.current_phase == TrainingPhase.PHASE_B:
            self.current_phase = TrainingPhase.PHASE_A
            logger.info("Phase transition confirmed: B → A (Self-Play)")
        else:
            raise EngineError(
                "Already at Phase A, no further transitions possible",
                error_code="NO_TRANSITION",
            )

        # Reset criteria for new phase
        self.transition_criteria = PhaseTransitionCriteria()
        return self.current_phase

    def revert_phase(self, target_phase: TrainingPhase) -> None:
        """Manually revert to an earlier training phase.

        Allows the user to go back to a previous phase if the current
        phase shows degraded performance. Resets transition criteria.

        Valid reversions:
        - PHASE_B → PHASE_C
        - PHASE_A → PHASE_C or PHASE_B

        Args:
            target_phase: The phase to revert to. Must be earlier than
                the current phase.

        Raises:
            EngineError: If target_phase is not earlier than current phase.
                Error code: INVALID_REVERT

        Requirements: 16.10
        """
        # Define phase ordering for comparison
        phase_order = {
            TrainingPhase.PHASE_C: 0,
            TrainingPhase.PHASE_B: 1,
            TrainingPhase.PHASE_A: 2,
        }

        target_order = phase_order[target_phase]
        current_order = phase_order[self.current_phase]

        if target_order >= current_order:
            raise EngineError(
                f"Can only revert to earlier phases. "
                f"Current: {self.current_phase.value}, "
                f"Target: {target_phase.value}",
                error_code="INVALID_REVERT",
            )

        logger.info(
            "Phase revert: %s → %s",
            self.current_phase.value,
            target_phase.value,
        )
        self.current_phase = target_phase
        self.transition_criteria = PhaseTransitionCriteria()


# ==============================================================================
# Phase C Model Strategy - re-exported for backward compatibility
# ==============================================================================

# Import from progressive_phase_c to maintain the public API
# (tests import _PhaseCModelStrategy from this module)
from engine.progressive_phase_c import _PhaseCModelStrategy  # noqa: E402, F401
