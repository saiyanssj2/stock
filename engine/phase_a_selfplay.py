"""
Phase A Self-Play Training Module.

Implements the self-play training loop where the model trains against
ScenarioGenerator market simulations with unlimited search depth and
no time constraint per move. Uses iterative deepening until convergence.

Requirements: 16.6
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from engine.config import (
    Action,
    EngineError,
    ModelConfig,
    SearchConfig,
    TrainingConfig,
)

if TYPE_CHECKING:
    from engine.training_pipeline import TrainingPipeline, TrainingResult

logger = logging.getLogger(__name__)


# Configuration


@dataclass
class PhaseASelfPlayConfig:
    """Configuration for Phase A self-play training."""

    max_iterations: int = 10
    samples_per_iteration: int = 200
    search_depth: int = -1  # Unlimited
    search_timeout: float = -1.0  # No time constraint
    max_search_depth_cap: int = 15  # Practical cap for iterative deepening
    convergence_threshold: float = 0.001  # Score convergence between depths
    training_epochs_per_iter: int = 50
    early_stopping_patience: int = 5
    improvement_threshold: float = 1e-4  # Min improvement between iterations
    min_improvement_iters: int = 3  # Min iterations before checking convergence


@dataclass
class SelfPlayIterationResult:
    """Result of a single self-play iteration."""

    iteration: int = 0
    labels_generated: int = 0
    labels_failed: int = 0
    avg_search_depth: float = 0.0
    avg_search_time: float = 0.0
    training_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    best_val_loss: float = float("inf")
    elapsed_seconds: float = 0.0


@dataclass
class PhaseAResult:
    """Overall result of Phase A self-play training."""

    iterations_completed: int = 0
    total_labels_generated: int = 0
    final_val_loss: float = float("inf")
    best_val_loss: float = float("inf")
    converged: bool = False
    iteration_results: List[SelfPlayIterationResult] = field(default_factory=list)
    total_time_seconds: float = 0.0
    model_path: Optional[str] = None


# Phase A Self-Play Engine


class PhaseASelfPlayEngine:
    """Implements the Phase A self-play training loop.

    The model generates its own training labels by running deep
    (unlimited-depth) iterative-deepening search on market states.
    The minimax scores become new training targets, creating a cycle:
    better model → better evaluations → better labels → better model.

    Requirements: 16.6
    """

    def __init__(
        self,
        pipeline: "TrainingPipeline",
        config: Optional[PhaseASelfPlayConfig] = None,
    ):
        self._pipeline = pipeline
        self._config = config or PhaseASelfPlayConfig()

    @property
    def config(self) -> PhaseASelfPlayConfig:
        """Current self-play configuration."""
        return self._config

    def run(
        self,
        symbol_data: Optional[Dict[str, pd.DataFrame]] = None,
        data_dir: Optional[str] = None,
    ) -> PhaseAResult:
        """Execute the Phase A self-play training loop.

        Iteratively generates self-play labels via unlimited-depth search,
        retrains the model, and checks for convergence.

        Args:
            symbol_data: Dict mapping symbol names to DataFrames.
            data_dir: Base directory for loading data files.

        Returns:
            PhaseAResult with complete self-play training results.

        Raises:
            EngineError: If pipeline is not configured or no data available.
        """
        start_time = time.time()
        result = PhaseAResult()

        # Prepare data
        if symbol_data is None:
            raise EngineError(
                "Phase A requires symbol_data to be provided",
                error_code="PHASE_A_NO_DATA",
            )

        valid_symbols, valid_data = self._pipeline.prepare_training_data(
            symbol_data, data_dir
        )

        if not valid_symbols:
            raise EngineError(
                "Phase A: No valid symbols with sufficient data",
                error_code="PHASE_A_NO_VALID_DATA",
            )

        logger.info(
            f"Phase A: Starting self-play loop with {len(valid_symbols)} symbols, "
            f"max_iterations={self._config.max_iterations}, "
            f"samples_per_iter={self._config.samples_per_iteration}"
        )

        prev_best_val_loss = float("inf")

        for iteration in range(1, self._config.max_iterations + 1):
            # Check graceful stop
            if self._pipeline._training_controller is not None:
                if not self._pipeline._training_controller.should_continue_training():
                    logger.info(
                        f"Phase A: Graceful stop requested at iteration {iteration}"
                    )
                    break

            logger.info(f"Phase A: === Iteration {iteration}/{self._config.max_iterations} ===")
            iter_start = time.time()

            # Step 1: Generate self-play labels
            self_play_labels, iter_stats = self._generate_self_play_labels(
                valid_data=valid_data,
                valid_symbols=valid_symbols,
            )

            if iter_stats["labels_generated"] == 0:
                logger.warning(
                    f"Phase A: Iteration {iteration} generated no labels, stopping."
                )
                break

            # Step 2: Train model on self-play labels
            iter_training = self._train_iteration(
                symbol_data=valid_data,
                valid_symbols=valid_symbols,
                self_play_labels=self_play_labels,
            )

            # Record iteration result
            iter_result = SelfPlayIterationResult(
                iteration=iteration,
                labels_generated=iter_stats["labels_generated"],
                labels_failed=iter_stats["labels_failed"],
                avg_search_depth=iter_stats["avg_depth"],
                avg_search_time=iter_stats["avg_time"],
                training_epochs=iter_training["epochs_completed"],
                train_loss=iter_training["final_train_loss"],
                val_loss=iter_training["final_val_loss"],
                best_val_loss=iter_training["best_val_loss"],
                elapsed_seconds=time.time() - iter_start,
            )
            result.iteration_results.append(iter_result)
            result.iterations_completed = iteration
            result.total_labels_generated += iter_stats["labels_generated"]

            # Track best val loss
            if iter_training["best_val_loss"] < result.best_val_loss:
                result.best_val_loss = iter_training["best_val_loss"]

            result.final_val_loss = iter_training["final_val_loss"]

            logger.info(
                f"Phase A: Iteration {iteration} complete - "
                f"labels={iter_stats['labels_generated']}, "
                f"val_loss={iter_training['final_val_loss']:.6f}, "
                f"best_val_loss={iter_training['best_val_loss']:.6f}, "
                f"time={iter_result.elapsed_seconds:.1f}s"
            )

            # Step 3: Check convergence
            if iteration >= self._config.min_improvement_iters:
                improvement = prev_best_val_loss - iter_training["best_val_loss"]
                if improvement < self._config.improvement_threshold:
                    logger.info(
                        f"Phase A: Converged at iteration {iteration}. "
                        f"Improvement ({improvement:.6f}) below threshold "
                        f"({self._config.improvement_threshold})"
                    )
                    result.converged = True
                    break

            prev_best_val_loss = min(prev_best_val_loss, iter_training["best_val_loss"])

        result.total_time_seconds = time.time() - start_time
        result.model_path = str(
            Path(self._pipeline.config.checkpoint_dir) / "stock_eval_net.pt"
        )

        # Log final results
        self._log_phase_a_result(result)

        logger.info(
            f"Phase A: Self-play complete - {result.iterations_completed} iterations, "
            f"total_labels={result.total_labels_generated}, "
            f"best_val_loss={result.best_val_loss:.6f}, "
            f"converged={result.converged}, "
            f"total_time={result.total_time_seconds:.1f}s"
        )

        return result

    def _generate_self_play_labels(
        self,
        valid_data: Dict[str, pd.DataFrame],
        valid_symbols: List[str],
    ) -> tuple:
        """Generate training labels via unlimited-depth self-play search.

        For each sampled market state, runs iterative deepening search
        (no timeout) until the score converges or max depth is reached.

        Returns:
            Tuple of (self_play_labels dict, stats dict).
        """
        from engine.evaluation_model import ModelManager
        from engine.market_state import MarketState
        from engine.scenario_generator import ScenarioGenerator
        from engine.search_module import SearchModule

        lookback = self._pipeline.model_config.lookback

        # Load current model for search evaluation
        model_path = str(
            Path(self._pipeline.config.checkpoint_dir) / "stock_eval_net.pt"
        )
        model_manager = ModelManager()
        try:
            model_manager.load(model_path)
        except Exception as e:
            logger.warning(
                f"Phase A: Could not load model for self-play, using random init: {e}"
            )
            # If no model exists yet, self-play labels won't be meaningful
            # but we can still generate them to bootstrap
            from engine.evaluation_model import StockEvalNet

            import torch

            model = StockEvalNet(self._pipeline.model_config)
            model.eval()
            # Save initial model so ModelManager can load it
            self._pipeline._save_complete_model(
                model=model,
                optimizer=torch.optim.Adam(model.parameters()),
                epoch=0,
                model_config=self._pipeline.model_config,
                norm_params={"min_vals": [0.0] * 63, "max_vals": [1.0] * 63},
                path=model_path,
            )
            model_manager.load(model_path)

        # Configure search: unlimited depth (uses iterative deepening), no timeout
        search_config = SearchConfig(
            default_depth=self._config.max_search_depth_cap,
            max_depth=self._config.max_search_depth_cap,
            timeout_seconds=float("inf"),  # No time constraint
        )
        scenario_generator = ScenarioGenerator(search_config=search_config)
        search_module = SearchModule(
            model_manager=model_manager,
            scenario_generator=scenario_generator,
            config=search_config,
        )

        # Distribute samples across symbols
        samples_per_symbol = max(
            1, self._config.samples_per_iteration // len(valid_symbols)
        )

        self_play_labels: Dict[str, np.ndarray] = {}
        total_generated = 0
        total_failed = 0
        depths: List[float] = []
        times: List[float] = []

        for symbol in valid_symbols:
            df = valid_data[symbol]
            n_rows = len(df)

            if n_rows < lookback + 1:
                self_play_labels[symbol] = np.full(n_rows, np.nan)
                continue

            labels = np.full(n_rows, np.nan, dtype=np.float64)
            available = n_rows - lookback

            # Sample evenly across the data
            actual_samples = min(samples_per_symbol, available)
            if actual_samples <= 0:
                self_play_labels[symbol] = labels
                continue

            step = max(1, available // actual_samples)
            sample_indices = [
                lookback + i * step
                for i in range(actual_samples)
                if lookback + i * step < n_rows
            ]

            for idx in sample_indices:
                # Check graceful stop within label generation
                if self._pipeline._training_controller is not None:
                    if not self._pipeline._training_controller.should_continue_training():
                        break

                try:
                    search_start = time.time()

                    # Build MarketState
                    state = MarketState.from_dataframe(
                        df=df.iloc[: idx + 1],
                        symbol=symbol,
                        lookback=lookback,
                    )

                    # Run iterative deepening search (unlimited depth)
                    score, depth_reached = self._iterative_deepening_search(
                        state=state,
                        search_module=search_module,
                    )

                    search_elapsed = time.time() - search_start

                    # Store the converged score as the self-play label
                    labels[idx] = max(-1.0, min(1.0, score))
                    total_generated += 1
                    depths.append(depth_reached)
                    times.append(search_elapsed)

                except Exception as e:
                    logger.debug(
                        f"Phase A: Self-play label failed at idx {idx} "
                        f"for '{symbol}': {e}"
                    )
                    total_failed += 1

            self_play_labels[symbol] = labels

        stats = {
            "labels_generated": total_generated,
            "labels_failed": total_failed,
            "avg_depth": float(np.mean(depths)) if depths else 0.0,
            "avg_time": float(np.mean(times)) if times else 0.0,
        }

        logger.info(
            f"Phase A: Self-play label generation - "
            f"generated={total_generated}, failed={total_failed}, "
            f"avg_depth={stats['avg_depth']:.1f}, "
            f"avg_time={stats['avg_time']:.1f}s"
        )

        return self_play_labels, stats

    def _iterative_deepening_search(
        self,
        state,
        search_module,
    ) -> tuple:
        """Run iterative deepening search until score converges.

        Starts at depth 3, increases by 1 each step. Stops when score
        change < convergence_threshold or max depth cap reached.

        Returns:
            Tuple of (converged_score, depth_reached).
        """
        prev_score = None
        best_score = 0.0
        max_depth = self._config.max_search_depth_cap
        start_depth = 3

        for depth in range(start_depth, max_depth + 1):
            try:
                result = search_module.search(state=state, depth=depth)
                current_score = result.best_score

                # Check convergence
                if prev_score is not None:
                    score_change = abs(current_score - prev_score)
                    if score_change < self._config.convergence_threshold:
                        # Score has converged - further depth won't help
                        return current_score, depth

                prev_score = current_score
                best_score = current_score

            except Exception as e:
                # If search fails at this depth, return last good score
                logger.debug(
                    f"Phase A: Search failed at depth {depth}: {e}"
                )
                if prev_score is not None:
                    return prev_score, depth - 1
                break

        return best_score, max_depth

    def _train_iteration(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        valid_symbols: List[str],
        self_play_labels: Dict[str, np.ndarray],
    ) -> Dict[str, object]:
        """Train the model for one self-play iteration.

        Uses self-play labels where available, falls back to simple labels
        for positions without self-play labels (NaN).
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        from engine.evaluation_model import StockEvalNet
        from engine.hardware_profile import HardwareProfile
        from engine.market_state import FeatureVectorBuilder

        lookback = self._pipeline.model_config.lookback

        # Build feature matrices with self-play labels
        feature_builder = FeatureVectorBuilder.from_training_data(
            list(symbol_data.values())
        )
        nan_mask = (
            np.isnan(feature_builder.min_vals)
            | np.isnan(feature_builder.max_vals)
        )
        feature_builder.min_vals[nan_mask] = 0.0
        feature_builder.max_vals[nan_mask] = 1.0

        # Save normalization params
        norm_params_path = str(
            Path(self._pipeline.config.checkpoint_dir) / "norm_params.json"
        )
        feature_builder.save_params(norm_params_path)

        all_features = []
        all_labels = []

        for symbol in valid_symbols:
            df = symbol_data[symbol]
            features, simple_labels = self._pipeline._build_feature_matrix(df)

            if len(features) == 0:
                continue

            # Normalize features
            for i in range(len(features)):
                features[i] = feature_builder._normalize(features[i])

            # Replace simple labels with self-play labels where available
            sp_labels = self_play_labels.get(symbol)
            if sp_labels is not None:
                for i in range(len(simple_labels)):
                    df_idx = i + lookback - 1
                    if df_idx < len(sp_labels) and not np.isnan(sp_labels[df_idx]):
                        simple_labels[i] = sp_labels[df_idx]

            # Filter out NaN labels
            valid_mask = ~np.isnan(simple_labels)
            features = features[valid_mask]
            simple_labels = simple_labels[valid_mask]

            if len(features) > 0:
                all_features.append(features)
                all_labels.append(simple_labels)

        if not all_features:
            return {
                "epochs_completed": 0,
                "final_train_loss": float("inf"),
                "final_val_loss": float("inf"),
                "best_val_loss": float("inf"),
            }

        combined_features = np.concatenate(all_features, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)

        # Chronological split
        (train_data, train_labels), (val_data, val_labels), _ = (
            self._pipeline._chronological_split(combined_features, combined_labels)
        )

        # Setup model and training
        device = HardwareProfile().get_device()
        model = StockEvalNet(self._pipeline.model_config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self._pipeline.config.learning_rate,
            weight_decay=self._pipeline.config.weight_decay,
        )
        criterion = nn.MSELoss()
        batch_size = HardwareProfile().get_optimal_batch_size(mode="training")

        # Load current best model as starting point
        model_path = str(
            Path(self._pipeline.config.checkpoint_dir) / "stock_eval_net.pt"
        )
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                logger.info("Phase A: Loaded current model for fine-tuning")
            model.to(device)
        except Exception as e:
            logger.warning(
                f"Phase A: Could not load model, training from scratch: {e}"
            )

        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.from_numpy(train_data.astype(np.float32)),
            torch.from_numpy(train_labels.astype(np.float32)),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(val_data.astype(np.float32)),
            torch.from_numpy(val_labels.astype(np.float32)),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        final_train_loss = float("inf")
        final_val_loss = float("inf")
        epochs_completed = 0

        for epoch in range(self._config.training_epochs_per_iter):
            # Check graceful stop
            if self._pipeline._training_controller is not None:
                if not self._pipeline._training_controller.should_continue_training():
                    logger.info(f"Phase A: Graceful stop at epoch {epoch}")
                    break

            # Train
            model.train()
            train_loss_sum = 0.0
            train_count = 0
            for batch_features, batch_labels_t in train_loader:
                batch_features = batch_features.to(device)
                batch_labels_t = batch_labels_t.to(device).unsqueeze(1)

                optimizer.zero_grad()
                predictions = model(batch_features)
                loss = criterion(predictions, batch_labels_t)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * len(batch_features)
                train_count += len(batch_features)

            final_train_loss = train_loss_sum / max(train_count, 1)

            # Validate
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for batch_features, batch_labels_t in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels_t = batch_labels_t.to(device).unsqueeze(1)
                    predictions = model(batch_features)
                    loss = criterion(predictions, batch_labels_t)
                    val_loss_sum += loss.item() * len(batch_features)
                    val_count += len(batch_features)

            final_val_loss = val_loss_sum / max(val_count, 1)
            epochs_completed = epoch + 1

            # Early stopping
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                patience_counter = 0

                # Save best model
                self._pipeline._save_complete_model(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    model_config=self._pipeline.model_config,
                    norm_params={
                        "min_vals": feature_builder.min_vals.tolist(),
                        "max_vals": feature_builder.max_vals.tolist(),
                    },
                    path=model_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= self._config.early_stopping_patience:
                    logger.info(
                        f"Phase A: Early stopping at epoch {epoch} "
                        f"(patience={self._config.early_stopping_patience})"
                    )
                    break

            if epoch % 5 == 0:
                logger.info(
                    f"Phase A: Epoch {epoch}/{self._config.training_epochs_per_iter} - "
                    f"train_loss={final_train_loss:.6f}, "
                    f"val_loss={final_val_loss:.6f}"
                )

        return {
            "epochs_completed": epochs_completed,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
        }

    def _log_phase_a_result(self, result: PhaseAResult) -> None:
        """Log Phase A results to the validation log file."""
        log_dir = Path("engine/models")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "validation_rounds.jsonl"

        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": "phase_a_self_play",
            "event": "phase_a_complete",
            "iterations_completed": result.iterations_completed,
            "total_labels_generated": result.total_labels_generated,
            "final_val_loss": float(result.final_val_loss),
            "best_val_loss": float(result.best_val_loss),
            "converged": result.converged,
            "total_time_seconds": result.total_time_seconds,
            "iteration_details": [
                {
                    "iteration": ir.iteration,
                    "labels_generated": ir.labels_generated,
                    "avg_search_depth": ir.avg_search_depth,
                    "avg_search_time": ir.avg_search_time,
                    "val_loss": ir.val_loss,
                    "best_val_loss": ir.best_val_loss,
                    "elapsed_seconds": ir.elapsed_seconds,
                }
                for ir in result.iteration_results
            ],
        }

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"Phase A: Results logged to {log_file}")
