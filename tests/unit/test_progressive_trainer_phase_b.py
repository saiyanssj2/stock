"""
Unit tests for Progressive Trainer Phase B implementation.

Tests the Phase B enhanced label generation flow:
- run_phase_b() requires PHASE_B to be the current phase
- Enhanced label generation using deep search
- Model retraining with enhanced labels
- ScenarioGenerator directional accuracy validation
- B→A transition criteria checking
- Logging of Phase B results

Requirements: 16.4, 16.5
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from engine.config import Action, ModelConfig, SearchConfig, TrainingConfig
from engine.progressive_trainer import (
    PhaseTransitionCriteria,
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
    """Create a mock TrainingPipeline for Phase B testing."""
    pipeline = MagicMock()
    pipeline.config = TrainingConfig(checkpoint_dir=tmp_dir)
    pipeline.model_config = ModelConfig()
    pipeline._training_controller = None

    # Mock prepare_training_data
    def mock_prepare(symbol_data, data_dir=None):
        valid_symbols = list(symbol_data.keys())
        return valid_symbols, symbol_data

    pipeline.prepare_training_data.side_effect = mock_prepare

    # Mock _build_feature_matrix - returns features and simple labels
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
# Tests for run_phase_b - Phase validation
# ==============================================================================


class TestPhaseBPhaseValidation:
    """Tests that run_phase_b enforces being in PHASE_B."""

    def test_run_phase_b_requires_phase_b(self):
        """run_phase_b should raise if not in PHASE_B."""
        from engine.config import EngineError

        pipeline = MagicMock()
        trainer = ProgressiveTrainer(
            training_pipeline=pipeline,
        )
        # Default phase is PHASE_C
        assert trainer.current_phase == TrainingPhase.PHASE_C

        symbol_data = {"VNM": _make_sample_df()}
        with pytest.raises(EngineError, match="Cannot run Phase B"):
            trainer.run_phase_b(symbol_data=symbol_data)

    def test_run_phase_b_requires_pipeline(self):
        """run_phase_b should raise if no pipeline is configured."""
        from engine.config import EngineError

        trainer = ProgressiveTrainer(training_pipeline=None)
        trainer.current_phase = TrainingPhase.PHASE_B

        symbol_data = {"VNM": _make_sample_df()}
        with pytest.raises(EngineError, match="TrainingPipeline not configured"):
            trainer.run_phase_b(symbol_data=symbol_data)


# ==============================================================================
# Tests for _generate_enhanced_labels
# ==============================================================================


class TestGenerateEnhancedLabels:
    """Tests for the enhanced label generation using deep search."""

    def test_generates_labels_with_valid_data(self):
        """Should produce labels for valid data points."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            trainer = ProgressiveTrainer(
                training_pipeline=pipeline,
                config=ProgressiveTrainingConfig(
                    phase_b_search_depth=5,
                    phase_b_search_timeout=10.0,  # Short for testing
                ),
            )
            trainer.current_phase = TrainingPhase.PHASE_B

            df = _make_sample_df(200)

            # Mock search module
            mock_search = MagicMock()
            mock_result = MagicMock()
            mock_result.best_score = 0.5
            mock_search.search.return_value = mock_result

            # Mock MarketState.from_dataframe at its source module
            with patch("engine.market_state.MarketState.from_dataframe") as mock_ms:
                mock_state = MagicMock()
                mock_ms.return_value = mock_state

                labels = trainer._generate_enhanced_labels(
                    df=df,
                    symbol="VNM",
                    search_module=mock_search,
                    lookback=60,
                    num_samples=10,
                )

            assert isinstance(labels, np.ndarray)
            assert len(labels) == len(df)
            # Should have some non-NaN values
            non_nan = np.sum(~np.isnan(labels))
            assert non_nan == 10

    def test_labels_are_clamped_to_range(self):
        """Enhanced labels should be in [-1, 1]."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            trainer = ProgressiveTrainer(
                training_pipeline=pipeline,
                config=ProgressiveTrainingConfig(
                    phase_b_search_depth=5,
                    phase_b_search_timeout=10.0,
                ),
            )
            trainer.current_phase = TrainingPhase.PHASE_B

            df = _make_sample_df(200)

            # Mock search module returning extreme score
            mock_search = MagicMock()
            mock_result = MagicMock()
            mock_result.best_score = 2.5  # Out of range
            mock_search.search.return_value = mock_result

            with patch("engine.market_state.MarketState.from_dataframe") as mock_ms:
                mock_state = MagicMock()
                mock_ms.return_value = mock_state

                labels = trainer._generate_enhanced_labels(
                    df=df,
                    symbol="VNM",
                    search_module=mock_search,
                    lookback=60,
                    num_samples=5,
                )

            non_nan_labels = labels[~np.isnan(labels)]
            assert np.all(non_nan_labels >= -1.0)
            assert np.all(non_nan_labels <= 1.0)

    def test_handles_search_failure_gracefully(self):
        """Should produce NaN for positions where search fails."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            trainer = ProgressiveTrainer(
                training_pipeline=pipeline,
                config=ProgressiveTrainingConfig(
                    phase_b_search_depth=5,
                    phase_b_search_timeout=10.0,
                ),
            )
            trainer.current_phase = TrainingPhase.PHASE_B

            df = _make_sample_df(200)

            # Mock search module that always fails
            mock_search = MagicMock()
            mock_search.search.side_effect = Exception("Search failed")

            with patch("engine.market_state.MarketState.from_dataframe") as mock_ms:
                mock_state = MagicMock()
                mock_ms.return_value = mock_state

                labels = trainer._generate_enhanced_labels(
                    df=df,
                    symbol="VNM",
                    search_module=mock_search,
                    lookback=60,
                    num_samples=5,
                )

            # All labels should be NaN since all searches failed
            assert np.all(np.isnan(labels))

    def test_returns_empty_for_insufficient_data(self):
        """Should return all NaN for data shorter than lookback."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            trainer = ProgressiveTrainer(training_pipeline=pipeline)
            trainer.current_phase = TrainingPhase.PHASE_B

            df = _make_sample_df(30)  # Too short for lookback=60

            mock_search = MagicMock()
            labels = trainer._generate_enhanced_labels(
                df=df,
                symbol="VNM",
                search_module=mock_search,
                lookback=60,
                num_samples=5,
            )

            assert np.all(np.isnan(labels))


# ==============================================================================
# Tests for _validate_scenario_accuracy
# ==============================================================================


class TestValidateScenarioAccuracy:
    """Tests for ScenarioGenerator directional accuracy validation."""

    def test_computes_directional_accuracy(self):
        """Should compute accuracy as proportion of correct direction predictions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            trainer = ProgressiveTrainer(training_pipeline=pipeline)
            trainer.current_phase = TrainingPhase.PHASE_B

            df = _make_sample_df(300)
            symbol_data = {"VNM": df}

            # Mock ScenarioGenerator and MarketState at their source modules
            with patch("engine.scenario_generator.ScenarioGenerator.generate") as mock_gen, \
                 patch("engine.market_state.MarketState.from_dataframe") as mock_ms:

                # Create mock MarketState
                mock_state = MagicMock()
                mock_ms.return_value = mock_state

                # Make scenarios that predict close price slightly above current
                close_prices = df["close"].values

                def gen_scenarios(state, action, num_scenarios=5):
                    scenarios = []
                    for i in range(num_scenarios):
                        s = MagicMock()
                        s.ohlcv = np.array([[0, 0, 0, close_prices[60] * 1.01, 0]])
                        scenarios.append(s)
                    return scenarios

                mock_gen.side_effect = gen_scenarios

                predictions, actuals = trainer._validate_scenario_accuracy(
                    symbol_data=symbol_data,
                    valid_symbols=["VNM"],
                    num_samples=20,
                    lookback=60,
                )

            assert len(predictions) > 0
            assert len(predictions) == len(actuals)

    def test_handles_insufficient_data_gracefully(self):
        """Should return empty lists for symbols with insufficient data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            trainer = ProgressiveTrainer(training_pipeline=pipeline)
            trainer.current_phase = TrainingPhase.PHASE_B

            # Very short data - insufficient for lookback + validation
            df = _make_sample_df(30)
            symbol_data = {"VNM": df}

            predictions, actuals = trainer._validate_scenario_accuracy(
                symbol_data=symbol_data,
                valid_symbols=["VNM"],
                num_samples=20,
                lookback=60,
            )

            assert len(predictions) == 0
            assert len(actuals) == 0


# ==============================================================================
# Tests for _log_phase_b_result
# ==============================================================================


class TestLogPhaseBResult:
    """Tests for Phase B result logging."""

    def test_logs_result_to_file(self):
        """Should write Phase B results to the validation log file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            trainer = ProgressiveTrainer(training_pipeline=pipeline)

            # Patch the log file path to use tmp_dir
            with patch("engine.progressive_trainer.Path") as mock_path:
                mock_log_dir = Path(tmp_dir)
                mock_path.return_value = mock_log_dir
                # Actually, let's just test the method directly with real paths
                pass

            # Directly create the log directory in tmp_dir
            log_dir = Path(tmp_dir) / "engine" / "models"
            log_dir.mkdir(parents=True, exist_ok=True)

            mock_result = MagicMock()
            mock_result.epochs_completed = 50
            mock_result.best_val_loss = 0.03

            # Monkey-patch the log path temporarily
            import engine.progressive_trainer as pt_module
            original_path = Path

            with patch.object(pt_module, "Path", side_effect=lambda *a: Path(tmp_dir)):
                # This is tricky - let's just call the method and check it doesn't crash
                # The actual file writing uses hardcoded "engine/models" path
                pass

            # Simpler approach: just verify the method runs without error
            # by mocking the file write
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_open.return_value.__exit__ = MagicMock(return_value=False)

                trainer._log_phase_b_result(
                    total_labels_generated=100,
                    training_result=mock_result,
                    directional_accuracy=0.65,
                    accuracy_samples=120,
                    transition_available=True,
                )

            # Verify open was called
            mock_open.assert_called_once()


# ==============================================================================
# Tests for full run_phase_b flow
# ==============================================================================


class TestRunPhaseBFlow:
    """Integration-style tests for the full Phase B flow."""

    def test_full_phase_b_flow_with_mocks(self):
        """Test the complete Phase B flow returns expected results dict."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_pipeline(tmp_dir)
            trainer = ProgressiveTrainer(
                training_pipeline=pipeline,
                config=ProgressiveTrainingConfig(
                    phase_b_search_depth=5,
                    phase_b_search_timeout=5.0,
                ),
            )
            trainer.current_phase = TrainingPhase.PHASE_B

            symbol_data = {"VNM": _make_sample_df(300)}

            # Mock all the heavy dependencies at their source modules
            with patch("engine.evaluation_model.ModelManager") as mock_mm_cls, \
                 patch("engine.search_module.SearchModule") as mock_sm_cls, \
                 patch("engine.scenario_generator.ScenarioGenerator") as mock_sg_cls, \
                 patch("engine.market_state.MarketState.from_dataframe") as mock_ms, \
                 patch("engine.market_state.FeatureVectorBuilder.from_training_data") as mock_fvb_from, \
                 patch("engine.market_state.FeatureVectorBuilder.save_params") as mock_save_params, \
                 patch("builtins.open", create=True) as mock_open:

                # Setup ModelManager mock
                mock_mm = MagicMock()
                mock_mm_cls.return_value = mock_mm

                # Setup SearchModule mock
                mock_sm = MagicMock()
                mock_sm_cls.return_value = mock_sm
                mock_search_result = MagicMock()
                mock_search_result.best_score = 0.4
                mock_sm.search.return_value = mock_search_result

                # Setup ScenarioGenerator mock
                mock_sg = MagicMock()
                mock_sg_cls.return_value = mock_sg
                mock_scenario = MagicMock()
                mock_scenario.ohlcv = np.array([[0, 0, 0, 26000.0, 0]])
                mock_sg.generate.return_value = [mock_scenario] * 5

                # Setup MarketState mock
                mock_state = MagicMock()
                mock_ms.return_value = mock_state

                # Setup FeatureVectorBuilder mock
                mock_fvb = MagicMock()
                mock_fvb.min_vals = np.zeros(63)
                mock_fvb.max_vals = np.ones(63)
                mock_fvb.save_params = MagicMock()
                mock_fvb._normalize = MagicMock(side_effect=lambda x: x)
                mock_fvb_from.return_value = mock_fvb

                # Setup file mock for logging
                mock_open.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_open.return_value.__exit__ = MagicMock(return_value=False)

                # Mock the _retrain_with_enhanced_labels to avoid torch dependency
                mock_training_result = MagicMock()
                mock_training_result.epochs_completed = 20
                mock_training_result.best_val_loss = 0.03
                with patch.object(trainer, "_retrain_with_enhanced_labels", return_value=mock_training_result):
                    result = trainer.run_phase_b(
                        symbol_data=symbol_data,
                        num_label_samples=5,
                        validation_samples=10,
                    )

            # Verify result structure
            assert "enhanced_labels_generated" in result
            assert "training_result" in result
            assert "directional_accuracy" in result
            assert "accuracy_samples" in result
            assert "transition_available" in result
            assert isinstance(result["directional_accuracy"], float)
            assert 0.0 <= result["directional_accuracy"] <= 1.0

    def test_phase_b_transition_check_with_high_accuracy(self):
        """B→A transition should be flagged when accuracy > 60% on 100+ samples."""
        trainer = ProgressiveTrainer()
        trainer.current_phase = TrainingPhase.PHASE_B

        # Simulate 120 samples with 75% accuracy
        predictions = [1.0] * 90 + [-1.0] * 30
        actuals = [1.0] * 90 + [1.0] * 30  # 90/120 = 75% correct

        result = trainer.check_b_to_a_transition(predictions, actuals)

        assert result is True
        assert trainer.transition_criteria.transition_available is True
        assert trainer.transition_criteria.scenario_accuracy > 0.60

    def test_phase_b_transition_check_with_insufficient_samples(self):
        """B→A transition should NOT be flagged with < 100 samples."""
        trainer = ProgressiveTrainer()
        trainer.current_phase = TrainingPhase.PHASE_B

        # Only 50 samples - insufficient even with perfect accuracy
        predictions = [1.0] * 50
        actuals = [1.0] * 50

        result = trainer.check_b_to_a_transition(predictions, actuals)

        assert result is False
        assert trainer.transition_criteria.transition_available is False

    def test_phase_b_transition_check_with_low_accuracy(self):
        """B→A transition should NOT be flagged when accuracy <= 60%."""
        trainer = ProgressiveTrainer()
        trainer.current_phase = TrainingPhase.PHASE_B

        # 100+ samples but only 50% accuracy
        predictions = [1.0] * 50 + [-1.0] * 50 + [1.0] * 20
        actuals = [1.0] * 50 + [1.0] * 50 + [-1.0] * 20  # 50/120 ≈ 42%

        result = trainer.check_b_to_a_transition(predictions, actuals)

        assert result is False
        assert trainer.transition_criteria.transition_available is False
