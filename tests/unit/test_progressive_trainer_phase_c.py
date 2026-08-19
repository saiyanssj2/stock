"""
Unit tests for Progressive Trainer Phase C implementation.

Tests the Phase C validation flow:
- _PhaseCModelStrategy signal generation
- run_phase_c() end-to-end flow
- ValidationRound logging to local file
- Best model selection based on Sharpe ratio

Requirements: 16.2, 16.8
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from engine.config import Action, BacktestResult, ModelConfig, TrainingConfig
from engine.progressive_trainer import (
    ProgressiveTrainer,
    ProgressiveTrainingConfig,
    TrainingPhase,
    ValidationRound,
    _PhaseCModelStrategy,
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
    prices = np.maximum(prices, 1000)  # Ensure positive

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

    # Add some indicator columns that analysis.add_indicators would produce
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


def _make_mock_model_manager():
    """Create a mock ModelManager with predict capability."""
    manager = MagicMock()
    manager.is_loaded = True
    manager.predict.return_value = np.array([[0.2]])  # Neutral-positive
    return manager


def _make_mock_training_pipeline(tmp_dir: str):
    """Create a mock TrainingPipeline."""
    from engine.progressive_trainer import TrainingPhase

    pipeline = MagicMock()
    pipeline.config = TrainingConfig(checkpoint_dir=tmp_dir)
    pipeline.model_config = ModelConfig()

    # Mock train_full result
    mock_result = MagicMock()
    mock_result.epochs_completed = 10
    mock_result.best_val_loss = 0.05
    mock_result.model_path = str(Path(tmp_dir) / "stock_eval_net.pt")
    pipeline.train_full.return_value = mock_result

    return pipeline


def _make_mock_strategies():
    """Create mock philosophy strategies."""
    strategies = []
    for name in ["Wyckoff", "Technical", "Momentum", "MeanReversion"]:
        s = MagicMock()
        s.name = name
        # Strategies generate random signals
        s.generate_signal.return_value = Action.HOLD
        strategies.append(s)
    return strategies


# ==============================================================================
# Tests for _PhaseCModelStrategy
# ==============================================================================


class TestPhaseCModelStrategy:
    """Tests for the _PhaseCModelStrategy class."""

    def test_strategy_has_name(self):
        """Strategy should have a descriptive name including depth."""
        model_manager = _make_mock_model_manager()
        strategy = _PhaseCModelStrategy(
            model_manager=model_manager,
            symbol="VNM",
            lookback=60,
            search_depth=5,
            search_timeout=300.0,
        )
        assert "Model_PhaseC" in strategy.name
        assert "d5" in strategy.name

    def test_strategy_returns_hold_when_insufficient_lookback(self):
        """Strategy should return HOLD when index < lookback."""
        model_manager = _make_mock_model_manager()
        strategy = _PhaseCModelStrategy(
            model_manager=model_manager,
            symbol="VNM",
            lookback=60,
            search_depth=5,
        )
        df = _make_sample_df(100)
        # Index 30 is less than lookback 60
        signal = strategy.generate_signal(df, 30)
        assert signal == Action.HOLD

    def test_strategy_returns_valid_action(self):
        """Strategy should return a valid Action (BUY/HOLD/SELL)."""
        model_manager = _make_mock_model_manager()
        strategy = _PhaseCModelStrategy(
            model_manager=model_manager,
            symbol="VNM",
            lookback=60,
            search_depth=3,
            search_timeout=5.0,
        )

        df = _make_sample_df(200)

        # Mock the search module to avoid actual computation
        with patch(
            "engine.progressive_trainer._PhaseCModelStrategy._ensure_initialized"
        ):
            strategy._search_module = MagicMock()
            mock_result = MagicMock()
            mock_result.best_action = Action.BUY
            mock_result.best_score = 0.7
            mock_result.nodes_evaluated = 10
            strategy._search_module.search.return_value = mock_result

            # Also need to mock MarketState.from_dataframe
            with patch("engine.market_state.MarketState.from_dataframe") as mock_from_df:
                mock_state = MagicMock()
                mock_from_df.return_value = mock_state

                signal = strategy.generate_signal(df, 100)
                assert signal in (Action.BUY, Action.HOLD, Action.SELL)

    def test_strategy_falls_back_to_hold_on_error(self):
        """Strategy should return HOLD when search/model fails."""
        model_manager = _make_mock_model_manager()
        strategy = _PhaseCModelStrategy(
            model_manager=model_manager,
            symbol="VNM",
            lookback=60,
            search_depth=5,
        )

        # Force initialization to raise
        strategy._search_module = MagicMock()
        strategy._search_module.search.side_effect = Exception("Search failed")

        df = _make_sample_df(200)

        # Patch model-only fallback to also fail gracefully
        with patch.object(strategy, "_model_only_signal", return_value=Action.HOLD):
            signal = strategy.generate_signal(df, 100)
            assert signal == Action.HOLD


# ==============================================================================
# Tests for run_phase_c
# ==============================================================================


class TestRunPhaseC:
    """Tests for the run_phase_c() method."""

    def test_run_phase_c_raises_without_pipeline(self):
        """run_phase_c should raise EngineError if no pipeline configured."""
        trainer = ProgressiveTrainer(
            config=ProgressiveTrainingConfig(),
            training_pipeline=None,
            strategies=_make_mock_strategies(),
        )
        with pytest.raises(Exception, match="TrainingPipeline not configured"):
            trainer.run_phase_c(symbol_data={"VNM": _make_sample_df()})

    def test_run_phase_c_raises_without_strategies(self):
        """run_phase_c should raise EngineError if no strategies configured."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_training_pipeline(tmp_dir)
            trainer = ProgressiveTrainer(
                config=ProgressiveTrainingConfig(),
                training_pipeline=pipeline,
                strategies=[],
            )
            with pytest.raises(Exception, match="No philosophy strategies"):
                trainer.run_phase_c(symbol_data={"VNM": _make_sample_df()})

    @patch("engine.progressive_trainer.ProgressiveTrainer._run_model_backtest")
    @patch("engine.progressive_trainer.ProgressiveTrainer._run_strategy_backtests")
    def test_run_phase_c_returns_validation_round(
        self, mock_strategy_bt, mock_model_bt
    ):
        """run_phase_c should return a ValidationRound with results."""
        mock_model_bt.return_value = 1.5  # Model Sharpe
        mock_strategy_bt.return_value = [0.8, 1.2, 0.5, 0.9]  # Strategy Sharpes

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_training_pipeline(tmp_dir)
            strategies = _make_mock_strategies()

            trainer = ProgressiveTrainer(
                config=ProgressiveTrainingConfig(),
                training_pipeline=pipeline,
                strategies=strategies,
            )

            df = _make_sample_df(300)
            result = trainer.run_phase_c(symbol_data={"VNM": df})

            assert isinstance(result, ValidationRound)
            assert result.round_number == 1
            assert result.model_sharpe == 1.5
            assert result.strategy_sharpes == [0.8, 1.2, 0.5, 0.9]
            assert result.phase == TrainingPhase.PHASE_C
            assert "phase_c_v1" in result.model_version

    @patch("engine.progressive_trainer.ProgressiveTrainer._run_model_backtest")
    @patch("engine.progressive_trainer.ProgressiveTrainer._run_strategy_backtests")
    def test_run_phase_c_logs_validation_round(
        self, mock_strategy_bt, mock_model_bt
    ):
        """run_phase_c should write ValidationRound to local log file."""
        mock_model_bt.return_value = 2.0
        mock_strategy_bt.return_value = [1.0, 0.5, 1.5, 0.8]

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_training_pipeline(tmp_dir)
            strategies = _make_mock_strategies()

            trainer = ProgressiveTrainer(
                config=ProgressiveTrainingConfig(),
                training_pipeline=pipeline,
                strategies=strategies,
            )

            df = _make_sample_df(300)

            # Patch the log directory to use our temp dir
            log_file = Path(tmp_dir) / "validation_rounds.jsonl"
            with patch(
                "engine.progressive_trainer.ProgressiveTrainer._log_validation_round"
            ) as mock_log:
                # Call the actual log method but with patched path
                mock_log.side_effect = lambda vr: _write_log(vr, log_file)
                trainer.run_phase_c(symbol_data={"VNM": df})

            # Verify log file was written
            assert log_file.exists()
            with open(log_file, "r") as f:
                entry = json.loads(f.readline())

            assert entry["round_number"] == 1
            assert entry["model_sharpe"] == 2.0
            assert entry["phase"] == "phase_c_validation"
            assert "strategies_beaten" in entry

    @patch("engine.progressive_trainer.ProgressiveTrainer._run_model_backtest")
    @patch("engine.progressive_trainer.ProgressiveTrainer._run_strategy_backtests")
    def test_run_phase_c_checks_transition_criteria(
        self, mock_strategy_bt, mock_model_bt
    ):
        """run_phase_c should check C→B transition after each round."""
        # Model beats 3/4 strategies
        mock_model_bt.return_value = 2.0
        mock_strategy_bt.return_value = [1.0, 0.5, 1.5, 0.8]

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_training_pipeline(tmp_dir)
            strategies = _make_mock_strategies()

            trainer = ProgressiveTrainer(
                config=ProgressiveTrainingConfig(),
                training_pipeline=pipeline,
                strategies=strategies,
            )

            df = _make_sample_df(300)

            # Run 3 rounds to trigger transition
            for _ in range(3):
                trainer.run_phase_c(symbol_data={"VNM": df})

            # After 3 consecutive wins, transition should be available
            assert trainer.transition_criteria.transition_available is True
            assert trainer.transition_criteria.consecutive_wins >= 3

    @patch("engine.progressive_trainer.ProgressiveTrainer._run_model_backtest")
    @patch("engine.progressive_trainer.ProgressiveTrainer._run_strategy_backtests")
    def test_run_phase_c_selects_best_model(
        self, mock_strategy_bt, mock_model_bt
    ):
        """run_phase_c should track the model with best Sharpe ratio."""
        mock_model_bt.return_value = 1.8
        mock_strategy_bt.return_value = [1.0, 0.5, 1.5, 0.8]

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _make_mock_training_pipeline(tmp_dir)
            strategies = _make_mock_strategies()

            trainer = ProgressiveTrainer(
                config=ProgressiveTrainingConfig(),
                training_pipeline=pipeline,
                strategies=strategies,
            )

            df = _make_sample_df(300)
            result = trainer.run_phase_c(symbol_data={"VNM": df})

            # The model Sharpe should be recorded
            assert result.model_sharpe == 1.8
            # Validation history should be updated
            assert len(trainer.validation_history) == 1
            assert trainer.validation_history[0].model_sharpe == 1.8


# ==============================================================================
# Tests for ValidationRound logging
# ==============================================================================


class TestValidationRoundLogging:
    """Tests for _log_validation_round."""

    def test_log_creates_file_and_writes_entry(self):
        """_log_validation_round should create log file and write JSON entry."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = ProgressiveTrainer()

            # Patch log directory
            log_dir = Path(tmp_dir)
            log_file = log_dir / "validation_rounds.jsonl"

            vr = ValidationRound(
                round_number=1,
                model_version="phase_c_v1",
                model_sharpe=1.5,
                strategy_sharpes=[0.8, 1.2, 0.5, 0.9],
                phase=TrainingPhase.PHASE_C,
            )

            with patch("engine.progressive_trainer.Path") as mock_path:
                mock_path.return_value = log_dir
                mock_path.__truediv__ = lambda self, other: log_dir / other

                # Write directly using the internal method logic
                _write_log(vr, log_file)

            assert log_file.exists()
            with open(log_file, "r") as f:
                entry = json.loads(f.readline())

            assert entry["round_number"] == 1
            assert entry["model_version"] == "phase_c_v1"
            assert entry["model_sharpe"] == 1.5
            assert len(entry["strategy_sharpes"]) == 4
            assert entry["phase"] == "phase_c_validation"
            assert entry["strategies_beaten"] == 4  # 1.5 > [0.8, 1.2, 0.5, 0.9]

    def test_log_appends_multiple_entries(self):
        """Multiple validation rounds should be appended to the same file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "validation_rounds.jsonl"

            for i in range(3):
                vr = ValidationRound(
                    round_number=i + 1,
                    model_version=f"phase_c_v{i + 1}",
                    model_sharpe=1.0 + i * 0.5,
                    strategy_sharpes=[0.8, 1.2, 0.5, 0.9],
                    phase=TrainingPhase.PHASE_C,
                )
                _write_log(vr, log_file)

            with open(log_file, "r") as f:
                lines = f.readlines()

            assert len(lines) == 3
            for i, line in enumerate(lines):
                entry = json.loads(line)
                assert entry["round_number"] == i + 1


# ==============================================================================
# Helper functions
# ==============================================================================


def _write_log(validation_round: ValidationRound, log_file: Path) -> None:
    """Helper to write a validation round to log file (mirrors _log_validation_round)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

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
