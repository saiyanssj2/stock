"""
Unit tests for Progressive Trainer Phase A (Self-Play) implementation.

Tests the Phase A self-play training loop:
- run_phase_a() requires PHASE_A to be the current phase
- Self-play label generation via iterative deepening search
- Model retraining with self-play labels
- Convergence detection between iterations
- Graceful stop support during self-play
- Logging of Phase A results

Requirements: 16.6
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from engine.config import Action, EngineError, ModelConfig, SearchConfig, TrainingConfig
from engine.phase_a_selfplay import (
    PhaseAResult,
    PhaseASelfPlayConfig,
    PhaseASelfPlayEngine,
    SelfPlayIterationResult,
)
from engine.progressive_trainer import (
    ProgressiveTrainer,
    ProgressiveTrainingConfig,
    TrainingPhase,
)


# ==============================================================================
# Fixtures
# ==============================================================================


def _make_sample_df(rows: int = 300) -> pd.DataFrame:
    """Create a sample DataFrame with OHLCV + indicators data."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=rows, freq="B")
    base_price = 25000.0

    prices = base_price + np.cumsum(np.random.randn(rows) * 200)
    prices = np.maximum(prices, 1000)

    df = pd.DataFrame(
        {
            "time": dates,
            "open": prices + np.random.randn(rows) * 50,
            "high": prices + abs(np.random.randn(rows) * 100),
            "low": prices - abs(np.random.randn(rows) * 100),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, rows),
        }
    )

    # Add indicator columns
    df["EMA_9"] = df["close"].ewm(span=9).mean()
    df["EMA_20"] = df["close"].ewm(span=20).mean()
    df["RSI_14"] = 50.0 + np.random.randn(rows) * 10
    df["MACD"] = np.random.randn(rows) * 100
    df["MACD_signal"] = np.random.randn(rows) * 80
    df["ADX"] = 20 + np.random.rand(rows) * 30
    df["BB_upper"] = prices + 500
    df["BB_lower"] = prices - 500
    df["OBV"] = np.cumsum(np.random.randn(rows) * 10000)

    return df


def _make_mock_pipeline(tmp_dir: str):
    """Create a mock TrainingPipeline for Phase A testing."""
    pipeline = MagicMock()
    pipeline.config = TrainingConfig(checkpoint_dir=tmp_dir)
    pipeline.model_config = ModelConfig()
    pipeline._training_controller = None

    # Mock prepare_training_data
    def mock_prepare(symbol_data, data_dir=None):
        valid_symbols = list(symbol_data.keys())
        return valid_symbols, symbol_data

    pipeline.prepare_training_data.side_effect = mock_prepare

    # Mock _build_feature_matrix
    def mock_build_features(df, lookback=None):
        lb = lookback or 60
        n = len(df) - lb
        if n <= 0:
            return np.array([]).reshape(0, lb, 63), np.array([])
        features = np.random.randn(n, lb, 63).astype(np.float64)
        labels = np.tanh(np.random.randn(n) * 0.5).astype(np.float64)
        return features, labels

    pipeline._build_feature_matrix.side_effect = mock_build_features

    # Mock _chronological_split
    def mock_split(features, labels):
        n = len(features)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        return (
            (features[:train_end], labels[:train_end]),
            (features[train_end:val_end], labels[train_end:val_end]),
            (features[val_end:], labels[val_end:]),
        )

    pipeline._chronological_split.side_effect = mock_split

    # Mock _save_complete_model
    pipeline._save_complete_model = MagicMock()

    return pipeline


# ==============================================================================
# Tests for PhaseASelfPlayConfig
# ==============================================================================


class TestPhaseASelfPlayConfig:
    """Tests for Phase A self-play configuration."""

    def test_default_config(self):
        """Default config should have unlimited depth and no timeout."""
        config = PhaseASelfPlayConfig()
        assert config.search_depth == -1  # Unlimited
        assert config.search_timeout == -1.0  # No constraint
        assert config.max_iterations == 10
        assert config.samples_per_iteration == 200
        assert config.max_search_depth_cap == 15

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = PhaseASelfPlayConfig(
            max_iterations=5,
            samples_per_iteration=50,
            convergence_threshold=0.01,
        )
        assert config.max_iterations == 5
        assert config.samples_per_iteration == 50
        assert config.convergence_threshold == 0.01


# ==============================================================================
# Tests for run_phase_a - Phase validation
# ==============================================================================


class TestPhaseAPhaseValidation:
    """Tests that run_phase_a enforces being in PHASE_A."""

    def test_run_phase_a_requires_phase_a(self):
        """run_phase_a should raise if not in PHASE_A."""
        pipeline = MagicMock()
        trainer = ProgressiveTrainer(training_pipeline=pipeline)
        # Default phase is PHASE_C
        assert trainer.current_phase == TrainingPhase.PHASE_C

        symbol_data = {"VNM": _make_sample_df()}
        with pytest.raises(EngineError, match="Cannot run Phase A"):
            trainer.run_phase_a(symbol_data=symbol_data)

    def test_run_phase_a_requires_pipeline(self):
        """run_phase_a should raise if no pipeline is configured."""
        trainer = ProgressiveTrainer(training_pipeline=None)
        trainer.current_phase = TrainingPhase.PHASE_A

        symbol_data = {"VNM": _make_sample_df()}
        with pytest.raises(EngineError, match="TrainingPipeline not configured"):
            trainer.run_phase_a(symbol_data=symbol_data)

    def test_run_phase_a_requires_symbol_data(self):
        """run_phase_a should raise if no symbol_data provided."""
        pipeline = MagicMock()
        trainer = ProgressiveTrainer(training_pipeline=pipeline)
        trainer.current_phase = TrainingPhase.PHASE_A

        with pytest.raises(EngineError, match="requires symbol_data"):
            trainer.run_phase_a(symbol_data=None)

    def test_run_phase_a_in_phase_b_raises(self):
        """run_phase_a should raise if currently in PHASE_B."""
        pipeline = MagicMock()
        trainer = ProgressiveTrainer(training_pipeline=pipeline)
        trainer.current_phase = TrainingPhase.PHASE_B

        symbol_data = {"VNM": _make_sample_df()}
        with pytest.raises(EngineError, match="Cannot run Phase A"):
            trainer.run_phase_a(symbol_data=symbol_data)


# ==============================================================================
# Tests for PhaseASelfPlayEngine - Iterative deepening search
# ==============================================================================


class TestIterativeDeepeningSearch:
    """Tests for the iterative deepening search used in self-play."""

    def test_converges_when_scores_stabilize(self):
        """Search should stop deepening when score converges."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            config = PhaseASelfPlayConfig(
                convergence_threshold=0.01,
                max_search_depth_cap=10,
            )
            engine = PhaseASelfPlayEngine(pipeline=pipeline, config=config)

            # Mock search module that returns converging scores
            mock_search = MagicMock()
            scores = [0.5, 0.52, 0.525, 0.526]  # Converges at depth 5-6

            call_count = [0]

            def mock_search_fn(state, depth=None):
                idx = min(call_count[0], len(scores) - 1)
                call_count[0] += 1
                result = MagicMock()
                result.best_score = scores[idx]
                return result

            mock_search.search.side_effect = mock_search_fn

            mock_state = MagicMock()
            score, depth = engine._iterative_deepening_search(
                state=mock_state,
                search_module=mock_search,
            )

            # Should have converged (0.526 - 0.525 = 0.001 < 0.01)
            assert abs(score - 0.526) < 0.01
            assert depth <= 10

    def test_reaches_max_depth_without_convergence(self):
        """Search should stop at max depth cap if scores never converge."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            config = PhaseASelfPlayConfig(
                convergence_threshold=0.0001,  # Very tight threshold
                max_search_depth_cap=5,  # Low cap
            )
            engine = PhaseASelfPlayEngine(pipeline=pipeline, config=config)

            # Mock search module that returns diverging scores
            mock_search = MagicMock()
            call_count = [0]

            def mock_search_fn(state, depth=None):
                call_count[0] += 1
                result = MagicMock()
                result.best_score = 0.1 * call_count[0]  # Keeps changing
                return result

            mock_search.search.side_effect = mock_search_fn

            mock_state = MagicMock()
            score, depth = engine._iterative_deepening_search(
                state=mock_state,
                search_module=mock_search,
            )

            # Should have reached max depth cap
            assert depth == 5

    def test_handles_search_failure_at_depth(self):
        """Should return last good score when search fails at a depth."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            config = PhaseASelfPlayConfig(max_search_depth_cap=10)
            engine = PhaseASelfPlayEngine(pipeline=pipeline, config=config)

            # Search works at depths 3-4 but fails at 5+
            mock_search = MagicMock()
            call_count = [0]

            def mock_search_fn(state, depth=None):
                call_count[0] += 1
                if call_count[0] > 2:
                    raise Exception("OOM at higher depth")
                result = MagicMock()
                result.best_score = 0.3
                return result

            mock_search.search.side_effect = mock_search_fn

            mock_state = MagicMock()
            score, depth = engine._iterative_deepening_search(
                state=mock_state,
                search_module=mock_search,
            )

            # Should return last successful score
            assert abs(score - 0.3) < 0.01


# ==============================================================================
# Tests for PhaseASelfPlayEngine - Label generation
# ==============================================================================


class TestSelfPlayLabelGeneration:
    """Tests for self-play label generation."""

    def test_generates_labels_with_valid_data(self):
        """Should generate self-play labels for valid data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            config = PhaseASelfPlayConfig(
                samples_per_iteration=5,
                max_search_depth_cap=4,
                convergence_threshold=0.5,  # Easy convergence
            )
            engine = PhaseASelfPlayEngine(pipeline=pipeline, config=config)

            df = _make_sample_df(200)
            valid_data = {"VNM": df}

            # Mock dependencies at their actual module locations
            with patch("engine.evaluation_model.ModelManager") as mock_mm_cls, \
                 patch("engine.market_state.MarketState.from_dataframe") as mock_ms_from, \
                 patch("engine.scenario_generator.ScenarioGenerator") as mock_sg_cls, \
                 patch("engine.search_module.SearchModule") as mock_sm_cls:

                # Setup ModelManager
                mock_mm = MagicMock()
                mock_mm_cls.return_value = mock_mm

                # Setup SearchModule
                mock_sm = MagicMock()
                mock_sm_cls.return_value = mock_sm
                mock_result = MagicMock()
                mock_result.best_score = 0.4
                mock_sm.search.return_value = mock_result

                # Setup MarketState
                mock_state = MagicMock()
                mock_ms_from.return_value = mock_state

                # Setup ScenarioGenerator
                mock_sg = MagicMock()
                mock_sg_cls.return_value = mock_sg

                labels, stats = engine._generate_self_play_labels(
                    valid_data=valid_data,
                    valid_symbols=["VNM"],
                )

            assert "VNM" in labels
            assert stats["labels_generated"] > 0
            assert stats["labels_failed"] == 0
            # Labels should be in [-1, 1]
            non_nan = labels["VNM"][~np.isnan(labels["VNM"])]
            assert np.all(non_nan >= -1.0)
            assert np.all(non_nan <= 1.0)

    def test_distributes_samples_across_symbols(self):
        """Should distribute samples across symbols evenly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            config = PhaseASelfPlayConfig(
                samples_per_iteration=10,
                max_search_depth_cap=4,
                convergence_threshold=0.5,
            )
            engine = PhaseASelfPlayEngine(pipeline=pipeline, config=config)

            df1 = _make_sample_df(200)
            df2 = _make_sample_df(200)
            valid_data = {"VNM": df1, "FPT": df2}

            with patch("engine.evaluation_model.ModelManager") as mock_mm_cls, \
                 patch("engine.market_state.MarketState.from_dataframe") as mock_ms_from, \
                 patch("engine.scenario_generator.ScenarioGenerator") as mock_sg_cls, \
                 patch("engine.search_module.SearchModule") as mock_sm_cls:

                mock_mm = MagicMock()
                mock_mm_cls.return_value = mock_mm

                mock_sm = MagicMock()
                mock_sm_cls.return_value = mock_sm
                mock_result = MagicMock()
                mock_result.best_score = 0.3
                mock_sm.search.return_value = mock_result

                mock_state = MagicMock()
                mock_ms_from.return_value = mock_state

                mock_sg = MagicMock()
                mock_sg_cls.return_value = mock_sg

                labels, stats = engine._generate_self_play_labels(
                    valid_data=valid_data,
                    valid_symbols=["VNM", "FPT"],
                )

            # Both symbols should have labels
            assert "VNM" in labels
            assert "FPT" in labels
            # Labels should be distributed (5 per symbol for 10 total / 2 symbols)
            vnm_count = np.sum(~np.isnan(labels["VNM"]))
            fpt_count = np.sum(~np.isnan(labels["FPT"]))
            assert vnm_count == 5
            assert fpt_count == 5

    def test_handles_all_search_failures(self):
        """Should handle gracefully when all searches fail."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            config = PhaseASelfPlayConfig(
                samples_per_iteration=5,
                max_search_depth_cap=4,
            )
            engine = PhaseASelfPlayEngine(pipeline=pipeline, config=config)

            df = _make_sample_df(200)
            valid_data = {"VNM": df}

            with patch("engine.evaluation_model.ModelManager") as mock_mm_cls, \
                 patch("engine.market_state.MarketState.from_dataframe") as mock_ms_from, \
                 patch("engine.scenario_generator.ScenarioGenerator") as mock_sg_cls, \
                 patch("engine.search_module.SearchModule") as mock_sm_cls:

                mock_mm = MagicMock()
                mock_mm_cls.return_value = mock_mm

                mock_sm = MagicMock()
                mock_sm_cls.return_value = mock_sm
                mock_sm.search.side_effect = Exception("Search failed")

                mock_ms_from.side_effect = Exception("State failed")

                mock_sg = MagicMock()
                mock_sg_cls.return_value = mock_sg

                labels, stats = engine._generate_self_play_labels(
                    valid_data=valid_data,
                    valid_symbols=["VNM"],
                )

            assert stats["labels_generated"] == 0
            assert stats["labels_failed"] > 0


# ==============================================================================
# Tests for PhaseASelfPlayEngine - Full self-play loop
# ==============================================================================


class TestSelfPlayLoop:
    """Tests for the full self-play training loop."""

    def test_full_selfplay_loop_with_mocks(self):
        """Test the complete self-play loop returns expected structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            config = PhaseASelfPlayConfig(
                max_iterations=2,
                samples_per_iteration=5,
                max_search_depth_cap=4,
                convergence_threshold=0.5,
                training_epochs_per_iter=2,
                min_improvement_iters=3,  # Won't converge in 2 iterations
            )
            engine = PhaseASelfPlayEngine(pipeline=pipeline, config=config)

            symbol_data = {"VNM": _make_sample_df(300)}

            # Mock _generate_self_play_labels and _train_iteration
            def mock_gen_labels(valid_data, valid_symbols):
                labels = {"VNM": np.full(300, np.nan)}
                labels["VNM"][60:65] = 0.4
                stats = {
                    "labels_generated": 5,
                    "labels_failed": 0,
                    "avg_depth": 5.0,
                    "avg_time": 1.0,
                }
                return labels, stats

            def mock_train_iter(symbol_data, valid_symbols, self_play_labels):
                return {
                    "epochs_completed": 2,
                    "final_train_loss": 0.05,
                    "final_val_loss": 0.06,
                    "best_val_loss": 0.05,
                }

            with patch.object(engine, "_generate_self_play_labels", side_effect=mock_gen_labels), \
                 patch.object(engine, "_train_iteration", side_effect=mock_train_iter), \
                 patch.object(engine, "_log_phase_a_result"):

                result = engine.run(symbol_data=symbol_data)

            # Verify result structure
            assert isinstance(result, PhaseAResult)
            assert result.iterations_completed == 2
            assert result.total_labels_generated == 10  # 5 per iteration x 2
            assert len(result.iteration_results) == 2
            assert result.total_time_seconds >= 0
            assert result.model_path is not None

    def test_selfplay_converges_when_improvement_below_threshold(self):
        """Self-play should stop when improvement between iterations is tiny."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            config = PhaseASelfPlayConfig(
                max_iterations=10,
                samples_per_iteration=5,
                improvement_threshold=0.01,
                min_improvement_iters=2,
            )
            engine = PhaseASelfPlayEngine(pipeline=pipeline, config=config)

            symbol_data = {"VNM": _make_sample_df(300)}

            # Mock _generate_self_play_labels and _train_iteration
            call_count = [0]

            def mock_gen_labels(valid_data, valid_symbols):
                labels = {"VNM": np.full(300, np.nan)}
                labels["VNM"][60:65] = 0.4
                stats = {
                    "labels_generated": 5,
                    "labels_failed": 0,
                    "avg_depth": 5.0,
                    "avg_time": 1.0,
                }
                return labels, stats

            def mock_train_iter(symbol_data, valid_symbols, self_play_labels):
                call_count[0] += 1
                # Return converging losses: very small improvement each time
                return {
                    "epochs_completed": 5,
                    "final_train_loss": 0.05,
                    "final_val_loss": 0.051 - call_count[0] * 0.0001,
                    "best_val_loss": 0.05 - call_count[0] * 0.0001,
                }

            with patch.object(engine, "_generate_self_play_labels", side_effect=mock_gen_labels), \
                 patch.object(engine, "_train_iteration", side_effect=mock_train_iter), \
                 patch.object(engine, "_log_phase_a_result"):

                result = engine.run(symbol_data=symbol_data)

            # Should converge before max_iterations
            assert result.converged is True
            assert result.iterations_completed < 10

    def test_selfplay_stops_on_no_labels(self):
        """Self-play should stop if an iteration generates no labels."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            config = PhaseASelfPlayConfig(max_iterations=5, samples_per_iteration=5)
            engine = PhaseASelfPlayEngine(pipeline=pipeline, config=config)

            symbol_data = {"VNM": _make_sample_df(300)}

            def mock_gen_labels(valid_data, valid_symbols):
                labels = {"VNM": np.full(300, np.nan)}  # All NaN = no labels
                stats = {
                    "labels_generated": 0,
                    "labels_failed": 5,
                    "avg_depth": 0.0,
                    "avg_time": 0.0,
                }
                return labels, stats

            with patch.object(engine, "_generate_self_play_labels", side_effect=mock_gen_labels), \
                 patch.object(engine, "_log_phase_a_result"):

                result = engine.run(symbol_data=symbol_data)

            # Should stop after first iteration (no labels)
            assert result.iterations_completed == 0
            assert result.total_labels_generated == 0

    def test_selfplay_respects_graceful_stop(self):
        """Self-play should stop when graceful stop is requested."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)

            # Setup training controller that stops after first iteration
            mock_controller = MagicMock()
            call_count = [0]

            def should_continue():
                call_count[0] += 1
                return call_count[0] <= 1  # Allow first iteration only

            mock_controller.should_continue_training.side_effect = should_continue
            pipeline._training_controller = mock_controller

            config = PhaseASelfPlayConfig(max_iterations=5, samples_per_iteration=5)
            engine = PhaseASelfPlayEngine(pipeline=pipeline, config=config)

            symbol_data = {"VNM": _make_sample_df(300)}

            def mock_gen_labels(valid_data, valid_symbols):
                labels = {"VNM": np.full(300, np.nan)}
                labels["VNM"][60:65] = 0.3
                stats = {
                    "labels_generated": 5,
                    "labels_failed": 0,
                    "avg_depth": 4.0,
                    "avg_time": 1.0,
                }
                return labels, stats

            def mock_train_iter(symbol_data, valid_symbols, self_play_labels):
                return {
                    "epochs_completed": 5,
                    "final_train_loss": 0.05,
                    "final_val_loss": 0.06,
                    "best_val_loss": 0.05,
                }

            with patch.object(engine, "_generate_self_play_labels", side_effect=mock_gen_labels), \
                 patch.object(engine, "_train_iteration", side_effect=mock_train_iter), \
                 patch.object(engine, "_log_phase_a_result"):

                result = engine.run(symbol_data=symbol_data)

            # Should stop early due to graceful stop
            assert result.iterations_completed <= 2


# ==============================================================================
# Tests for run_phase_a via ProgressiveTrainer
# ==============================================================================


class TestRunPhaseAViaTrainer:
    """Tests for the run_phase_a method on ProgressiveTrainer."""

    def test_delegates_to_phase_a_engine(self):
        """run_phase_a should delegate to PhaseASelfPlayEngine."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            trainer = ProgressiveTrainer(training_pipeline=pipeline)
            trainer.current_phase = TrainingPhase.PHASE_A

            symbol_data = {"VNM": _make_sample_df(300)}

            # Mock the PhaseASelfPlayEngine at its source module
            mock_result = PhaseAResult(
                iterations_completed=3,
                total_labels_generated=50,
                final_val_loss=0.04,
                best_val_loss=0.03,
                converged=True,
            )

            with patch("engine.phase_a_selfplay.PhaseASelfPlayEngine") as mock_engine_cls:
                mock_engine = MagicMock()
                mock_engine.run.return_value = mock_result
                mock_engine_cls.return_value = mock_engine

                result = trainer.run_phase_a(symbol_data=symbol_data)

            assert result.iterations_completed == 3
            assert result.converged is True
            mock_engine_cls.assert_called_once()
            mock_engine.run.assert_called_once()

    def test_passes_custom_config(self):
        """run_phase_a should pass custom config to the engine."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            trainer = ProgressiveTrainer(training_pipeline=pipeline)
            trainer.current_phase = TrainingPhase.PHASE_A

            symbol_data = {"VNM": _make_sample_df(300)}
            custom_config = PhaseASelfPlayConfig(
                max_iterations=3,
                samples_per_iteration=20,
            )

            mock_result = PhaseAResult(iterations_completed=3, converged=False)

            with patch("engine.phase_a_selfplay.PhaseASelfPlayEngine") as mock_engine_cls:
                mock_engine = MagicMock()
                mock_engine.run.return_value = mock_result
                mock_engine_cls.return_value = mock_engine

                result = trainer.run_phase_a(
                    symbol_data=symbol_data, config=custom_config
                )

            # Verify the config was passed
            call_args = mock_engine_cls.call_args
            assert call_args[1]["config"] == custom_config


# ==============================================================================
# Tests for SelfPlayIterationResult
# ==============================================================================


class TestSelfPlayIterationResult:
    """Tests for the iteration result dataclass."""

    def test_default_values(self):
        """Default values should be zero/empty."""
        result = SelfPlayIterationResult()
        assert result.iteration == 0
        assert result.labels_generated == 0
        assert result.best_val_loss == float("inf")

    def test_custom_values(self):
        """Should store custom values correctly."""
        result = SelfPlayIterationResult(
            iteration=3,
            labels_generated=50,
            avg_search_depth=7.2,
            avg_search_time=3.5,
            training_epochs=20,
            val_loss=0.04,
            best_val_loss=0.035,
        )
        assert result.iteration == 3
        assert result.labels_generated == 50
        assert result.avg_search_depth == 7.2
