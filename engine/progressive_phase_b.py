"""
Progressive Training - Phase B Logic.

Phase B: Enhanced label generation with deep search + model retrain.
Uses the best model from Phase C + deep search (depth 5-7, up to 1 hour/sample)
to generate enhanced training labels, then retrains the model.

Extracted from progressive_trainer.py for maintainability.

Requirements: 16.4, 16.5
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import json
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from engine.progressive_trainer import ProgressiveTrainer

logger = logging.getLogger(__name__)


def run_phase_b(
    trainer: "ProgressiveTrainer",
    symbol_data: Dict[str, pd.DataFrame],
    data_dir: Optional[str] = None,
    num_label_samples: Optional[int] = None,
    validation_samples: int = 100,
) -> Dict[str, object]:
    """Phase B: Enhanced label generation with deep search + model retrain.

    This function:
    1. Loads the best model from Phase C training.
    2. Uses that model + deep search (depth 5-7, up to 1 hour/sample)
       to generate enhanced training labels for the training set.
    3. Retrains the model using these enhanced (search-derived) labels.
    4. Validates ScenarioGenerator directional accuracy on 100+ samples.
    5. Checks B→A transition criteria.

    Args:
        trainer: The ProgressiveTrainer instance.
        symbol_data: Dict mapping symbol names to DataFrames with
            OHLCV + indicators data.
        data_dir: Optional base directory for loading additional data.
        num_label_samples: Maximum number of samples to generate enhanced
            labels for. If None, processes all available samples.
        validation_samples: Number of samples to use for ScenarioGenerator
            directional accuracy validation (default 100).

    Returns:
        Dict with phase B results.

    Raises:
        EngineError: If training pipeline is not set, no model exists,
            or current phase is not PHASE_B.

    Requirements: 16.4, 16.5
    """
    from engine.config import EngineError, SearchConfig
    from engine.progressive_trainer import TrainingPhase

    if trainer._pipeline is None:
        raise EngineError(
            "TrainingPipeline not configured for progressive training",
            error_code="PROGRESSIVE_NO_PIPELINE",
        )

    if trainer.current_phase != TrainingPhase.PHASE_B:
        raise EngineError(
            f"Cannot run Phase B: current phase is {trainer.current_phase.value}. "
            "Transition to Phase B first.",
            error_code="PROGRESSIVE_WRONG_PHASE",
        )

    # Step 1: Load best model from Phase C
    logger.info("Phase B: Loading best model from Phase C...")
    model_path = str(
        Path(trainer._pipeline.config.checkpoint_dir) / "stock_eval_net.pt"
    )

    from engine.evaluation_model import ModelManager
    from engine.market_state import FeatureVectorBuilder, MarketState
    from engine.scenario_generator import ScenarioGenerator
    from engine.search_module import SearchModule

    model_manager = ModelManager()
    try:
        model_manager.load(model_path)
    except Exception as e:
        raise EngineError(
            f"Phase B: Failed to load best model from Phase C: {e}",
            error_code="PROGRESSIVE_NO_MODEL",
        )

    # Step 2: Configure deep search for enhanced label generation
    search_config = SearchConfig(
        default_depth=trainer._config.phase_b_search_depth,
        max_depth=max(trainer._config.phase_b_search_depth, 7),
        timeout_seconds=trainer._config.phase_b_search_timeout,
    )
    scenario_generator = ScenarioGenerator(search_config=search_config)
    search_module = SearchModule(
        model_manager=model_manager,
        scenario_generator=scenario_generator,
        config=search_config,
    )

    # Step 3: Prepare data and generate enhanced labels
    logger.info(
        f"Phase B: Generating enhanced labels with deep search "
        f"(depth={trainer._config.phase_b_search_depth}, "
        f"timeout={trainer._config.phase_b_search_timeout}s per sample)..."
    )

    valid_symbols, valid_data = trainer._pipeline.prepare_training_data(
        symbol_data, data_dir
    )

    enhanced_labels_map = {}
    total_labels_generated = 0
    lookback = trainer._pipeline.model_config.lookback

    for symbol in valid_symbols:
        df = valid_data[symbol]
        n_rows = len(df)

        if n_rows < lookback + 1:
            logger.warning(
                f"Phase B: Skipping '{symbol}' - insufficient data "
                f"({n_rows} rows, need {lookback + 1})"
            )
            continue

        # Determine how many samples to label for this symbol
        available_samples = n_rows - lookback
        if num_label_samples is not None:
            per_symbol_limit = max(
                1, num_label_samples // len(valid_symbols)
            )
            samples_to_generate = min(available_samples, per_symbol_limit)
        else:
            samples_to_generate = available_samples

        # Generate enhanced labels using deep search
        symbol_labels = trainer._generate_enhanced_labels(
            df=df,
            symbol=symbol,
            search_module=search_module,
            lookback=lookback,
            num_samples=samples_to_generate,
        )

        enhanced_labels_map[symbol] = symbol_labels
        total_labels_generated += len(symbol_labels)
        logger.info(
            f"Phase B: Generated {len(symbol_labels)} enhanced labels "
            f"for '{symbol}'"
        )

    if total_labels_generated == 0:
        raise EngineError(
            "Phase B: No enhanced labels could be generated",
            error_code="PROGRESSIVE_NO_LABELS",
        )

    logger.info(
        f"Phase B: Total enhanced labels generated: {total_labels_generated}"
    )

    # Step 4: Retrain model with enhanced labels
    logger.info("Phase B: Retraining model with enhanced labels...")
    training_result = trainer._retrain_with_enhanced_labels(
        symbol_data=valid_data,
        enhanced_labels_map=enhanced_labels_map,
        valid_symbols=valid_symbols,
    )
    logger.info(
        f"Phase B: Retraining complete - {training_result.epochs_completed} epochs, "
        f"best_val_loss={training_result.best_val_loss:.6f}"
    )

    # Step 5: Validate ScenarioGenerator directional accuracy
    logger.info(
        f"Phase B: Validating ScenarioGenerator directional accuracy "
        f"on {validation_samples} samples..."
    )
    predictions, actuals = trainer._validate_scenario_accuracy(
        symbol_data=valid_data,
        valid_symbols=valid_symbols,
        num_samples=validation_samples,
        lookback=lookback,
    )

    directional_accuracy = 0.0
    if len(predictions) > 0:
        correct = sum(
            1
            for pred, actual in zip(predictions, actuals)
            if (pred > 0 and actual > 0)
            or (pred < 0 and actual < 0)
            or (pred == 0 and actual == 0)
        )
        directional_accuracy = correct / len(predictions)

    logger.info(
        f"Phase B: ScenarioGenerator directional accuracy = "
        f"{directional_accuracy * 100:.1f}% on {len(predictions)} samples"
    )

    # Step 6: Check B→A transition criteria
    transition_available = trainer.check_b_to_a_transition(predictions, actuals)

    if transition_available:
        logger.info(
            "Phase B: Transition to Phase A available! "
            f"Accuracy {directional_accuracy * 100:.1f}% > "
            f"{trainer.transition_criteria.required_accuracy * 100:.0f}% "
            f"on {len(predictions)} samples."
        )
    else:
        logger.info(
            f"Phase B: Transition to Phase A not yet available. "
            f"Accuracy {directional_accuracy * 100:.1f}% "
            f"(need >{trainer.transition_criteria.required_accuracy * 100:.0f}% "
            f"on {trainer.transition_criteria.required_sample_count}+ samples)"
        )

    # Log Phase B results
    trainer._log_phase_b_result(
        total_labels_generated=total_labels_generated,
        training_result=training_result,
        directional_accuracy=directional_accuracy,
        accuracy_samples=len(predictions),
        transition_available=transition_available,
    )

    return {
        "enhanced_labels_generated": total_labels_generated,
        "training_result": training_result,
        "directional_accuracy": directional_accuracy,
        "accuracy_samples": len(predictions),
        "transition_available": transition_available,
    }


def _generate_enhanced_labels(
    trainer: "ProgressiveTrainer",
    df: pd.DataFrame,
    symbol: str,
    search_module,
    lookback: int,
    num_samples: int,
) -> np.ndarray:
    """Generate enhanced labels for a symbol using deep search.

    For each sample point in the data, constructs a MarketState and runs
    deep search to obtain the best position score.

    Args:
        trainer: The ProgressiveTrainer instance.
        df: DataFrame with OHLCV + indicators for the symbol.
        symbol: Stock symbol name.
        search_module: Configured SearchModule for deep search.
        lookback: Lookback window size for MarketState construction.
        num_samples: Number of enhanced labels to generate.

    Returns:
        np.ndarray of shape (n_rows,) with enhanced label values.
    """
    from engine.market_state import MarketState

    n_rows = len(df)
    total_windows = n_rows - lookback
    labels = np.full(n_rows, np.nan, dtype=np.float64)

    if total_windows <= 0 or num_samples <= 0:
        return labels

    # Sample evenly across available windows
    if num_samples >= total_windows:
        sample_indices = list(range(lookback, n_rows))
    else:
        step = total_windows / num_samples
        sample_indices = [
            int(lookback + i * step) for i in range(num_samples)
        ]

    successful = 0
    failed = 0

    for idx in sample_indices:
        try:
            state = MarketState.from_dataframe(
                df=df.iloc[: idx + 1],
                symbol=symbol,
                lookback=lookback,
            )

            result = search_module.search(
                state=state,
                depth=trainer._config.phase_b_search_depth,
            )

            score = max(-1.0, min(1.0, result.best_score))
            labels[idx] = score
            successful += 1

        except Exception as e:
            logger.debug(
                f"Phase B: Enhanced label generation failed at index {idx} "
                f"for '{symbol}': {e}"
            )
            failed += 1
            continue

    logger.info(
        f"Phase B: Symbol '{symbol}' - {successful} labels generated, "
        f"{failed} failed out of {len(sample_indices)} attempted"
    )
    return labels


def _retrain_with_enhanced_labels(
    trainer: "ProgressiveTrainer",
    symbol_data: Dict[str, pd.DataFrame],
    enhanced_labels_map: Dict[str, np.ndarray],
    valid_symbols: List[str],
) -> "TrainingResult":
    """Retrain the model using enhanced (deep search) labels.

    Replaces the standard future-return labels with deep-search-derived
    labels where available.

    Args:
        trainer: The ProgressiveTrainer instance.
        symbol_data: Dict mapping symbols to their DataFrames.
        enhanced_labels_map: Dict mapping symbols to enhanced label arrays.
        valid_symbols: List of valid symbols to include in training.

    Returns:
        TrainingResult from the retraining session.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from engine.config import EngineError
    from engine.evaluation_model import StockEvalNet
    from engine.hardware_profile import HardwareProfile
    from engine.market_state import FeatureVectorBuilder
    from engine.training_pipeline import TrainingResult

    lookback = trainer._pipeline.model_config.lookback

    # Build feature matrices with enhanced labels
    logger.info("Phase B: Building feature matrices with enhanced labels...")
    all_features = []
    all_labels = []

    # Compute normalization parameters
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
        Path(trainer._pipeline.config.checkpoint_dir) / "norm_params.json"
    )
    feature_builder.save_params(norm_params_path)

    for symbol in valid_symbols:
        df = symbol_data[symbol]
        features, simple_labels = trainer._pipeline._build_feature_matrix(df)

        if len(features) == 0:
            continue

        # Normalize features
        for i in range(len(features)):
            features[i] = feature_builder._normalize(features[i])

        # Replace simple labels with enhanced labels where available
        enhanced = enhanced_labels_map.get(symbol)
        if enhanced is not None:
            for i in range(len(simple_labels)):
                df_idx = i + lookback - 1
                if df_idx < len(enhanced) and not np.isnan(enhanced[df_idx]):
                    simple_labels[i] = enhanced[df_idx]

        # Filter out NaN labels
        valid_mask = ~np.isnan(simple_labels)
        features = features[valid_mask]
        simple_labels = simple_labels[valid_mask]

        if len(features) > 0:
            all_features.append(features)
            all_labels.append(simple_labels)

    if not all_features:
        raise EngineError(
            "Phase B: No valid training samples after enhanced label integration",
            error_code="PROGRESSIVE_NO_TRAINING_DATA",
        )

    combined_features = np.concatenate(all_features, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    logger.info(
        f"Phase B: Training data prepared - {len(combined_features)} samples, "
        f"feature shape: {combined_features.shape}"
    )

    # Chronological split
    (train_data, train_labels), (val_data, val_labels), _ = (
        trainer._pipeline._chronological_split(combined_features, combined_labels)
    )

    # Train model
    device = HardwareProfile().get_device()
    model = StockEvalNet(trainer._pipeline.model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trainer._pipeline.config.learning_rate,
        weight_decay=trainer._pipeline.config.weight_decay,
    )
    criterion = nn.MSELoss()
    batch_size = HardwareProfile().get_optimal_batch_size(mode="training")

    # Load best model weights as starting point for fine-tuning
    model_path = str(
        Path(trainer._pipeline.config.checkpoint_dir) / "stock_eval_net.pt"
    )
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            logger.info("Phase B: Loaded best model weights for fine-tuning")
        model.to(device)
    except Exception as e:
        logger.warning(
            f"Phase B: Could not load prior model weights, training from scratch: {e}"
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    max_epochs = trainer._pipeline.config.max_epochs_full
    patience = trainer._pipeline.config.early_stopping_patience
    patience_counter = 0
    result = TrainingResult(symbols_trained=valid_symbols)

    start_time = time.time()

    for epoch in range(max_epochs):
        # Check graceful stop
        if trainer._pipeline._training_controller is not None:
            if not trainer._pipeline._training_controller.should_continue_training():
                logger.info(f"Phase B: Graceful stop at epoch {epoch}")
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

        train_loss = train_loss_sum / max(train_count, 1)

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

        val_loss = val_loss_sum / max(val_count, 1)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            best_model_path = str(
                Path(trainer._pipeline.config.checkpoint_dir) / "stock_eval_net.pt"
            )
            trainer._pipeline._save_complete_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                model_config=trainer._pipeline.model_config,
                norm_params={
                    "min_vals": feature_builder.min_vals.tolist(),
                    "max_vals": feature_builder.max_vals.tolist(),
                },
                path=best_model_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Phase B: Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            logger.info(
                f"Phase B: Epoch {epoch}/{max_epochs} - "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
            )

    result.epochs_completed = epoch + 1
    result.best_val_loss = best_val_loss
    result.best_epoch = best_epoch
    result.final_val_loss = val_loss
    result.final_train_loss = train_loss
    result.total_time_seconds = time.time() - start_time
    result.model_path = str(
        Path(trainer._pipeline.config.checkpoint_dir) / "stock_eval_net.pt"
    )

    return result


def _validate_scenario_accuracy(
    trainer: "ProgressiveTrainer",
    symbol_data: Dict[str, pd.DataFrame],
    valid_symbols: List[str],
    num_samples: int,
    lookback: int,
) -> tuple:
    """Validate ScenarioGenerator directional accuracy.

    Compares the ScenarioGenerator's predicted price direction against
    the actual next-day price direction for validation samples.

    Args:
        trainer: The ProgressiveTrainer instance.
        symbol_data: Dict mapping symbols to DataFrames.
        valid_symbols: Symbols to use for validation.
        num_samples: Total number of samples to validate.
        lookback: Lookback window for MarketState construction.

    Returns:
        Tuple of (predictions, actuals) as lists of floats.
    """
    from engine.config import Action
    from engine.market_state import MarketState
    from engine.scenario_generator import ScenarioGenerator

    scenario_gen = ScenarioGenerator()
    predictions: List[float] = []
    actuals: List[float] = []

    # Distribute samples across symbols
    samples_per_symbol = max(1, num_samples // len(valid_symbols))

    for symbol in valid_symbols:
        df = symbol_data[symbol]
        n_rows = len(df)

        if n_rows < lookback + 2:
            continue

        close_col = "close" if "close" in df.columns else None
        if close_col is None:
            continue
        close_prices = df[close_col].values.astype(np.float64)

        available = n_rows - lookback - 1
        if available <= 0:
            continue

        step = max(1, available // samples_per_symbol)
        sample_indices = [
            lookback + i * step
            for i in range(min(samples_per_symbol, available))
            if lookback + i * step < n_rows - 1
        ]

        for idx in sample_indices:
            try:
                state = MarketState.from_dataframe(
                    df=df.iloc[: idx + 1],
                    symbol=symbol,
                    lookback=lookback,
                )

                scenarios = scenario_gen.generate(
                    state, Action.HOLD, num_scenarios=5
                )

                if scenarios:
                    median_idx = len(scenarios) // 2
                    median_scenario = scenarios[median_idx]
                    predicted_close = median_scenario.ohlcv[-1, 3]
                    current_close = close_prices[idx]

                    if current_close > 0:
                        predicted_return = (
                            predicted_close - current_close
                        ) / current_close
                    else:
                        continue

                    next_close = close_prices[idx + 1]
                    actual_return = (
                        next_close - current_close
                    ) / current_close

                    predictions.append(predicted_return)
                    actuals.append(actual_return)

            except Exception as e:
                logger.debug(
                    f"Phase B: Scenario validation failed at index {idx} "
                    f"for '{symbol}': {e}"
                )
                continue

        if len(predictions) >= num_samples:
            break

    logger.info(
        f"Phase B: Scenario accuracy validation collected "
        f"{len(predictions)} samples from {len(valid_symbols)} symbols"
    )
    return predictions, actuals


def _log_phase_b_result(
    total_labels_generated: int,
    training_result,
    directional_accuracy: float,
    accuracy_samples: int,
    transition_available: bool,
) -> None:
    """Log Phase B results to the validation log file."""
    from engine.progressive_trainer import TrainingPhase

    log_dir = Path("engine/models")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "validation_rounds.jsonl"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "phase": TrainingPhase.PHASE_B.value,
        "event": "phase_b_complete",
        "enhanced_labels_generated": total_labels_generated,
        "retraining_epochs": training_result.epochs_completed,
        "best_val_loss": float(training_result.best_val_loss),
        "directional_accuracy": float(directional_accuracy),
        "accuracy_samples": accuracy_samples,
        "transition_available": transition_available,
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Phase B: Results logged to {log_file}")
