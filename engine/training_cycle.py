"""
Training Cycle Orchestrator — "Càng train nhiều, càng tránh sai lầm".

Core concept: Thay vì train 1 lần rồi xong, hệ thống train theo CYCLES.
Mỗi cycle = 1 vòng train + validate + evaluate performance.
Model tích lũy kinh nghiệm qua nhiều cycles.

Phase tự động chuyển khi đủ điều kiện:
- Phase C: Simple labels + deep search validation
- Phase B: Enhanced labels from deep search
- Phase A: Self-play (model tự cải thiện)

Lifecycle:
    1. User bấm "Train" → start 1 cycle
    2. Cycle chạy background: train → validate → record result
    3. Kết thúc cycle → update stats, check transition criteria
    4. User có thể bấm "Train tiếp" hoặc auto-schedule

This module provides:
- TrainingCycleManager: orchestrator for cycle-based training
- CycleResult: result of a single training cycle
- TrainingHistory: persistent history of all cycles
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from engine.config import EngineError
from engine.progressive_trainer import TrainingPhase

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class CycleResult:
    """Result of a single training cycle.

    Records everything about what happened in this cycle:
    model performance, phase, duration, comparison vs strategies.
    """

    cycle_number: int
    phase: TrainingPhase
    timestamp: str  # ISO format
    duration_seconds: float = 0.0

    # Training metrics
    train_loss: float = 0.0
    val_loss: float = 0.0
    best_val_loss: float = float("inf")
    epochs_completed: int = 0

    # Performance vs strategies (Sharpe ratios)
    model_sharpe: float = 0.0
    strategy_sharpes: Dict[str, float] = field(default_factory=dict)
    strategies_beaten: int = 0
    total_strategies: int = 4

    # Meta
    symbols_trained: int = 0
    mode: str = "full"
    hardware_device: str = "cpu"
    model_path: Optional[str] = None

    @property
    def beats_market(self) -> bool:
        """True if model beats at least 2/4 strategies."""
        return self.strategies_beaten >= 2

    def to_dict(self) -> dict:
        """Serialize to dict for JSON persistence."""
        return {
            "cycle_number": self.cycle_number,
            "phase": self.phase.value,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_val_loss": self.best_val_loss,
            "epochs_completed": self.epochs_completed,
            "model_sharpe": self.model_sharpe,
            "strategy_sharpes": self.strategy_sharpes,
            "strategies_beaten": self.strategies_beaten,
            "total_strategies": self.total_strategies,
            "symbols_trained": self.symbols_trained,
            "mode": self.mode,
            "hardware_device": self.hardware_device,
            "model_path": self.model_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CycleResult":
        """Deserialize from dict."""
        phase_str = d.get("phase", "phase_c_validation")
        try:
            phase = TrainingPhase(phase_str)
        except ValueError:
            phase = TrainingPhase.PHASE_C

        return cls(
            cycle_number=d.get("cycle_number", 0),
            phase=phase,
            timestamp=d.get("timestamp", ""),
            duration_seconds=d.get("duration_seconds", 0.0),
            train_loss=d.get("train_loss", 0.0),
            val_loss=d.get("val_loss", 0.0),
            best_val_loss=d.get("best_val_loss", float("inf")),
            epochs_completed=d.get("epochs_completed", 0),
            model_sharpe=d.get("model_sharpe", 0.0),
            strategy_sharpes=d.get("strategy_sharpes", {}),
            strategies_beaten=d.get("strategies_beaten", 0),
            total_strategies=d.get("total_strategies", 4),
            symbols_trained=d.get("symbols_trained", 0),
            mode=d.get("mode", "full"),
            hardware_device=d.get("hardware_device", "cpu"),
            model_path=d.get("model_path"),
        )


# ==============================================================================
# Training History (persistent)
# ==============================================================================


class TrainingHistory:
    """Persistent history of all training cycles.

    Stores cycle results in a JSONL file for tracking model improvement
    over time. Provides query methods for UI display.
    """

    HISTORY_FILE = "engine/models/training_cycles.jsonl"

    def __init__(self, base_dir: str = "."):
        self._path = Path(base_dir) / self.HISTORY_FILE
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, result: CycleResult) -> None:
        """Append a cycle result to history."""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def load_all(self) -> List[CycleResult]:
        """Load all cycle results from history."""
        if not self._path.exists():
            return []
        results = []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(CycleResult.from_dict(json.loads(line)))
                        except (json.JSONDecodeError, KeyError):
                            continue
        except OSError:
            return []
        return results

    @property
    def total_cycles(self) -> int:
        """Total number of completed cycles."""
        return len(self.load_all())

    @property
    def last_cycle(self) -> Optional[CycleResult]:
        """Most recent cycle result."""
        all_cycles = self.load_all()
        return all_cycles[-1] if all_cycles else None

    def get_sharpe_trend(self) -> List[float]:
        """Get model Sharpe ratio across all cycles (for trend chart)."""
        return [c.model_sharpe for c in self.load_all()]

    def get_win_streak(self) -> int:
        """Count consecutive cycles where model beats >=2 strategies."""
        all_cycles = self.load_all()
        streak = 0
        for cycle in reversed(all_cycles):
            if cycle.beats_market:
                streak += 1
            else:
                break
        return streak

    def get_phase_summary(self) -> Dict[str, int]:
        """Count cycles per phase."""
        summary: Dict[str, int] = {}
        for cycle in self.load_all():
            phase_name = cycle.phase.value
            summary[phase_name] = summary.get(phase_name, 0) + 1
        return summary


# ==============================================================================
# Training Cycle Manager
# ==============================================================================


class TrainingCycleManager:
    """Orchestrates cycle-based training.

    Concept: "Càng train nhiều, càng tránh sai lầm"
    - Mỗi cycle tự chọn phương pháp training theo phase hiện tại
    - Sau mỗi cycle, validate performance vs 4 strategies
    - Tự track progress và suggest khi nào nên chuyển phase
    - Persistent history cho user thấy model đang tiến bộ
    - Tích hợp Mistake-Driven Learning feedback loop (optional)

    Usage:
        manager = TrainingCycleManager(engine)
        result = manager.run_cycle(symbol_data)
        # result contains all metrics for this cycle

        # Với feedback loop:
        from engine.mistake_learning.config import MistakeLearningConfig
        config = MistakeLearningConfig()
        manager = TrainingCycleManager(engine, mistake_learning_config=config)
    """

    def __init__(self, engine, base_dir: str = ".", mistake_learning_config=None):
        """Initialize.

        Args:
            engine: DecisionEngine instance.
            base_dir: Project root directory.
            mistake_learning_config: Optional MistakeLearningConfig.
                Nếu cung cấp, feedback loop sẽ tự động chạy sau mỗi cycle.
                Nếu None, hệ thống hoạt động như cũ (backward compatible).
        """
        self._engine = engine
        self._base_dir = base_dir
        self._history = TrainingHistory(base_dir)

        # Mistake-Driven Learning integration (optional)
        self._mistake_learning_config = mistake_learning_config
        self._feedback_loop_manager = None
        if mistake_learning_config is not None:
            from engine.mistake_learning.feedback_loop import FeedbackLoopManager

            self._feedback_loop_manager = FeedbackLoopManager(
                engine=self._engine,
                config=mistake_learning_config,
                base_dir=base_dir,
            )

    @property
    def history(self) -> TrainingHistory:
        """Access training history."""
        return self._history

    @property
    def current_phase(self) -> TrainingPhase:
        """Current training phase."""
        return self._engine.progressive_trainer.current_phase

    @property
    def total_cycles(self) -> int:
        """Total completed cycles."""
        return self._history.total_cycles

    @property
    def win_streak(self) -> int:
        """Current consecutive wins vs market."""
        return self._history.get_win_streak()

    def get_status_summary(self) -> dict:
        """Get a summary for UI display.

        Returns dict with:
            - total_cycles: int
            - current_phase: str
            - win_streak: int
            - last_sharpe: float or None
            - trend: "improving" | "declining" | "stable" | "no_data"
            - transition_available: bool
        """
        last = self._history.last_cycle
        sharpe_trend = self._history.get_sharpe_trend()

        # Determine trend
        trend = "no_data"
        if len(sharpe_trend) >= 3:
            recent = sharpe_trend[-3:]
            if recent[-1] > recent[0]:
                trend = "improving"
            elif recent[-1] < recent[0]:
                trend = "declining"
            else:
                trend = "stable"

        return {
            "total_cycles": self._history.total_cycles,
            "current_phase": self.current_phase.value,
            "win_streak": self.win_streak,
            "last_sharpe": last.model_sharpe if last else None,
            "trend": trend,
            "transition_available": (
                self._engine.progressive_trainer.transition_criteria.transition_available
            ),
        }

    def run_cycle(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        data_dir: Optional[str] = None,
        mode: str = "full",
        validation_symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> CycleResult:
        """Run a single training cycle.

        Orchestrates:
        1. Determine what to do based on current phase
        2. Execute training (phase-appropriate method)
        3. Validate performance vs strategies
        4. Record result
        5. Check phase transition criteria

        Args:
            symbol_data: Symbol DataFrames for training.
            data_dir: Data directory path.
            mode: "full" or "incremental".
            validation_symbol: Symbol for validation backtest (default: first available).
            start_date: Validation period start.
            end_date: Validation period end.

        Returns:
            CycleResult with all metrics from this cycle.
        """
        import time

        from engine.hardware_profile import HardwareProfile

        start_time = time.time()
        hw = HardwareProfile()

        cycle_number = self._history.total_cycles + 1
        phase = self.current_phase

        logger.info(
            f"Starting training cycle #{cycle_number} "
            f"(Phase: {phase.value}, mode: {mode})"
        )

        # Execute training based on phase
        training_result = self._execute_phase_training(
            phase, symbol_data, data_dir, mode
        )

        # Validate performance
        model_sharpe, strategy_sharpes, strategies_beaten = self._validate_cycle(
            symbol_data, data_dir, validation_symbol, start_date, end_date
        )

        duration = time.time() - start_time

        # Build result
        result = CycleResult(
            cycle_number=cycle_number,
            phase=phase,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            train_loss=getattr(training_result, "final_train_loss", 0.0),
            val_loss=getattr(training_result, "final_val_loss", 0.0),
            best_val_loss=getattr(training_result, "best_val_loss", float("inf")),
            epochs_completed=getattr(training_result, "epochs_completed", 0),
            model_sharpe=model_sharpe,
            strategy_sharpes=strategy_sharpes,
            strategies_beaten=strategies_beaten,
            total_strategies=4,
            symbols_trained=len(symbol_data),
            mode=mode,
            hardware_device=hw.device_type,
            model_path=getattr(training_result, "model_path", None),
        )

        # Persist result
        self._history.append(result)

        # Check phase transition
        self._check_transition(result)

        logger.info(
            f"Cycle #{cycle_number} complete: "
            f"Sharpe={model_sharpe:.3f}, "
            f"beats {strategies_beaten}/4 strategies, "
            f"duration={duration:.0f}s"
        )

        # Chạy feedback loop nếu được cấu hình (Mistake-Driven Learning)
        if self._feedback_loop_manager is not None:
            try:
                feedback_result = self._feedback_loop_manager.run_feedback_cycle(
                    symbol_data=symbol_data,
                    cycle_number=cycle_number,
                )
                if feedback_result.skipped:
                    logger.info(
                        f"Feedback cycle #{cycle_number} skipped: "
                        f"{feedback_result.reason}"
                    )
                else:
                    logger.info(
                        f"Feedback cycle #{cycle_number} complete: "
                        f"hard_examples={feedback_result.hard_examples_count}, "
                        f"corrections={feedback_result.corrections_count}"
                    )
            except Exception as e:
                logger.warning(
                    f"Feedback cycle #{cycle_number} failed: {e}. "
                    f"Training cycle result unaffected."
                )

        return result

    def _execute_phase_training(
        self,
        phase: TrainingPhase,
        symbol_data: Dict[str, pd.DataFrame],
        data_dir: Optional[str],
        mode: str,
    ):
        """Execute training appropriate for the current phase.

        Phase C: Standard supervised training
        Phase B: Enhanced label training (deep search labels)
        Phase A: Self-play training

        Returns training result object.
        """
        trainer = self._engine.progressive_trainer

        if phase == TrainingPhase.PHASE_C:
            # Phase C: Standard training (simple labels)
            # Use background training manager for per-symbol training
            return self._run_standard_training(symbol_data, data_dir, mode)

        elif phase == TrainingPhase.PHASE_B:
            # Phase B: Enhanced label generation + retrain
            try:
                result = trainer.run_phase_b(
                    symbol_data=symbol_data,
                    data_dir=data_dir,
                )
                return result.get("training_result")
            except Exception as e:
                logger.warning(f"Phase B training failed, falling back to standard: {e}")
                return self._run_standard_training(symbol_data, data_dir, mode)

        elif phase == TrainingPhase.PHASE_A:
            # Phase A: Self-play
            try:
                return trainer.run_phase_a(
                    symbol_data=symbol_data,
                    data_dir=data_dir,
                )
            except Exception as e:
                logger.warning(f"Phase A training failed, falling back to standard: {e}")
                return self._run_standard_training(symbol_data, data_dir, mode)

        return self._run_standard_training(symbol_data, data_dir, mode)

    def _run_standard_training(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        data_dir: Optional[str],
        mode: str,
    ):
        """Run standard supervised training (Phase C method)."""
        from engine.training_pipeline import TrainingPipeline

        pipeline = TrainingPipeline()

        if mode == "incremental":
            return pipeline.train_incremental(symbol_data)
        else:
            return pipeline.train_full(symbol_data, data_dir)

    def _validate_cycle(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        data_dir: Optional[str],
        validation_symbol: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> tuple:
        """Validate model performance vs philosophy strategies.

        Returns:
            (model_sharpe, strategy_sharpes_dict, strategies_beaten)
        """
        try:
            # Pick validation symbol
            if validation_symbol is None:
                # Use first symbol with enough data
                for sym in symbol_data:
                    if len(symbol_data[sym]) >= 252:
                        validation_symbol = sym
                        break
                if validation_symbol is None:
                    validation_symbol = list(symbol_data.keys())[0]

            # Determine date range
            if start_date is None or end_date is None:
                df = symbol_data.get(validation_symbol)
                if df is not None and "time" in df.columns:
                    df_sorted = df.sort_values("time")
                    end_date = str(df_sorted["time"].iloc[-1])[:10]
                    # Use last 252 trading days
                    start_idx = max(0, len(df_sorted) - 252)
                    start_date = str(df_sorted["time"].iloc[start_idx])[:10]
                else:
                    return 0.0, {}, 0

            # Run comparison
            comparison = self._engine.compare(
                validation_symbol, start_date, end_date
            )

            # Extract sharpes
            model_sharpe = 0.0
            strategy_sharpes = {}
            strategies_beaten = 0

            for name, result in comparison.results.items():
                if name == "AI Decision Engine":
                    model_sharpe = result.sharpe_ratio
                else:
                    strategy_sharpes[name] = result.sharpe_ratio

            for _, s_sharpe in strategy_sharpes.items():
                if model_sharpe > s_sharpe:
                    strategies_beaten += 1

            return model_sharpe, strategy_sharpes, strategies_beaten

        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return 0.0, {}, 0

    def _check_transition(self, result: CycleResult) -> None:
        """Check if phase transition criteria are met after this cycle."""
        trainer = self._engine.progressive_trainer

        if self.current_phase == TrainingPhase.PHASE_C:
            # Build a ValidationRound for the progressive trainer
            from engine.progressive_trainer import ValidationRound

            strategy_sharpe_list = list(result.strategy_sharpes.values())
            vr = ValidationRound(
                round_number=result.cycle_number,
                model_version=f"cycle_{result.cycle_number}",
                model_sharpe=result.model_sharpe,
                strategy_sharpes=strategy_sharpe_list,
                phase=TrainingPhase.PHASE_C,
                timestamp=datetime.now(),
            )
            trainer.check_c_to_b_transition(vr)
