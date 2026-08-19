"""
Unit tests cho engine/hard_example_trainer.py.

Kiểm tra convert_hard_examples_to_tensors và retrain_from_hard_examples
bao gồm happy path và edge cases.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from engine.config import ModelConfig, TrainingConfig
from engine.evaluation_model import StockEvalNet
from engine.hard_example_trainer import (
    convert_hard_examples_to_tensors,
    retrain_from_hard_examples,
    _find_checkpoint,
    _find_date_index,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def model_config():
    """ModelConfig mặc định cho tests."""
    return ModelConfig()


@pytest.fixture
def training_config():
    """TrainingConfig mặc định cho tests."""
    return TrainingConfig()


@pytest.fixture
def sample_hard_examples():
    """Danh sách hard examples mẫu dùng cho testing."""
    return [
        {
            "symbol": "VNM",
            "buy_date": "2024-01-15",
            "sell_date": "2024-02-01",
            "buy_price": 75.0,
            "sell_price": 73.5,
            "pnl": -1500,
            "pnl_pct": -0.02,
            "holding_days": 12,
            "outcome": "loss",
        },
        {
            "symbol": "FPT",
            "buy_date": "2024-03-01",
            "sell_date": "2024-03-15",
            "buy_price": 120.0,
            "sell_price": 115.0,
            "pnl": -5000,
            "pnl_pct": -0.042,
            "holding_days": 10,
            "outcome": "loss",
        },
    ]


@pytest.fixture
def checkpoint_dir():
    """Thư mục tạm cho checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def checkpoint_with_model(checkpoint_dir, model_config):
    """Tạo checkpoint có sẵn model để test loading."""
    from engine.hardware_profile import HardwareProfile

    model = StockEvalNet(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    hw = HardwareProfile()

    checkpoint_path = str(Path(checkpoint_dir) / "checkpoint_epoch_0000.pt")
    HardwareProfile.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=0,
        metrics={"train_loss": 0.0, "val_loss": 0.0, "best_val_loss": 0.0},
        hardware=hw,
        path=checkpoint_path,
        model_config=model_config,
    )
    return checkpoint_dir


# ==============================================================================
# Tests cho convert_hard_examples_to_tensors
# ==============================================================================


class TestConvertHardExamplesToTensors:
    """Tests cho hàm convert_hard_examples_to_tensors."""

    def test_happy_path_returns_correct_shapes(self, sample_hard_examples, model_config):
        """Verify tensor shapes đúng: (N, lookback, num_features) và (N, 1)."""
        features, labels = convert_hard_examples_to_tensors(
            sample_hard_examples, "data", model_config
        )

        assert features.shape == (2, model_config.lookback, model_config.num_features)
        assert labels.shape == (2, 1)

    def test_tensors_are_float32(self, sample_hard_examples, model_config):
        """Verify dtype là float32 — tương thích với model training."""
        features, labels = convert_hard_examples_to_tensors(
            sample_hard_examples, "data", model_config
        )

        assert features.dtype == torch.float32
        assert labels.dtype == torch.float32

    def test_labels_in_valid_range(self, sample_hard_examples, model_config):
        """Labels phải nằm trong [-1, 1] (output range của tanh)."""
        features, labels = convert_hard_examples_to_tensors(
            sample_hard_examples, "data", model_config
        )

        assert labels.min().item() >= -1.0
        assert labels.max().item() <= 1.0

    def test_features_no_nan(self, sample_hard_examples, model_config):
        """Features không chứa NaN sau preprocessing."""
        features, labels = convert_hard_examples_to_tensors(
            sample_hard_examples, "data", model_config
        )

        assert not torch.isnan(features).any()
        assert not torch.isnan(labels).any()

    def test_missing_symbol_skipped_gracefully(self, model_config):
        """Symbol không tồn tại bị skip — không crash."""
        examples = [
            {
                "symbol": "NONEXIST_XYZ",
                "buy_date": "2024-01-15",
                "sell_date": "2024-02-01",
                "buy_price": 75.0,
                "sell_price": 73.5,
                "pnl": -1500,
                "pnl_pct": -0.02,
                "holding_days": 12,
                "outcome": "loss",
            },
            {
                "symbol": "VNM",
                "buy_date": "2024-01-15",
                "sell_date": "2024-02-01",
                "buy_price": 75.0,
                "sell_price": 73.5,
                "pnl": -1500,
                "pnl_pct": -0.02,
                "holding_days": 12,
                "outcome": "loss",
            },
        ]

        features, labels = convert_hard_examples_to_tensors(examples, "data", model_config)

        # Chỉ VNM valid → 1 sample
        assert features.shape[0] == 1

    def test_all_invalid_symbols_raises_valueerror(self, model_config):
        """Khi tất cả symbols không tồn tại → raise ValueError."""
        examples = [
            {
                "symbol": "FAKE_ABC",
                "buy_date": "2024-01-15",
                "sell_date": "2024-02-01",
                "buy_price": 75.0,
                "sell_price": 73.5,
                "pnl": -1500,
                "pnl_pct": -0.02,
                "holding_days": 12,
                "outcome": "loss",
            },
        ]

        with pytest.raises(ValueError, match="Không tạo được training data"):
            convert_hard_examples_to_tensors(examples, "data", model_config)


# ==============================================================================
# Tests cho retrain_from_hard_examples
# ==============================================================================


class TestRetrainFromHardExamples:
    """Tests cho hàm retrain_from_hard_examples."""

    def test_empty_hard_examples_returns_false(self, checkpoint_dir, model_config):
        """Hard examples rỗng → skip training, return False."""
        result = retrain_from_hard_examples([], checkpoint_dir, "data", model_config)
        assert result is False

    def test_successful_retrain_returns_true(
        self, sample_hard_examples, checkpoint_dir, model_config
    ):
        """Retrain thành công → return True."""
        result = retrain_from_hard_examples(
            sample_hard_examples, checkpoint_dir, "data", model_config
        )
        assert result is True

    def test_checkpoint_saved_after_retrain(
        self, sample_hard_examples, checkpoint_dir, model_config
    ):
        """Checkpoint file được tạo sau retrain."""
        retrain_from_hard_examples(
            sample_hard_examples, checkpoint_dir, "data", model_config
        )

        checkpoint_file = Path(checkpoint_dir) / "stock_eval_net.pt"
        assert checkpoint_file.exists()

    def test_model_weights_change_after_retrain(
        self, sample_hard_examples, checkpoint_with_model, model_config
    ):
        """Model weights phải thay đổi sau retrain (core requirement)."""
        from engine.hardware_profile import HardwareProfile

        # Load weights trước retrain
        initial_checkpoint_path = str(
            Path(checkpoint_with_model) / "checkpoint_epoch_0000.pt"
        )
        initial_cp = HardwareProfile.load_checkpoint(
            path=initial_checkpoint_path,
            target_device=torch.device("cpu"),
        )
        before_weights = {k: v.clone() for k, v in initial_cp["model_state_dict"].items()}

        # Retrain
        retrain_from_hard_examples(
            sample_hard_examples, checkpoint_with_model, "data", model_config
        )

        # Load weights sau retrain
        finetuned_path = str(Path(checkpoint_with_model) / "stock_eval_net.pt")
        after_cp = HardwareProfile.load_checkpoint(
            path=finetuned_path,
            target_device=torch.device("cpu"),
        )
        after_weights = after_cp["model_state_dict"]

        # Ít nhất một số weights phải thay đổi
        any_changed = any(
            not torch.equal(before_weights[k], after_weights[k])
            for k in before_weights
        )
        assert any_changed, "Model weights phải thay đổi sau retrain"

    def test_no_checkpoint_uses_fresh_model(
        self, sample_hard_examples, model_config
    ):
        """Khi không có checkpoint → dùng fresh model, vẫn train thành công."""
        with tempfile.TemporaryDirectory() as empty_dir:
            result = retrain_from_hard_examples(
                sample_hard_examples, empty_dir, "data", model_config
            )
            assert result is True

    def test_invalid_symbols_returns_false(self, checkpoint_dir, model_config):
        """Tất cả symbols invalid → không train được, return False."""
        bad_examples = [
            {
                "symbol": "FAKE_999",
                "buy_date": "2024-01-15",
                "sell_date": "2024-02-01",
                "buy_price": 75.0,
                "sell_price": 73.5,
                "pnl": -1500,
                "pnl_pct": -0.02,
                "holding_days": 12,
                "outcome": "loss",
            },
        ]
        result = retrain_from_hard_examples(
            bad_examples, checkpoint_dir, "data", model_config
        )
        assert result is False


# ==============================================================================
# Tests cho helper functions
# ==============================================================================


class TestFindCheckpoint:
    """Tests cho _find_checkpoint."""

    def test_returns_none_for_empty_dir(self):
        """Thư mục rỗng → None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert _find_checkpoint(tmpdir) is None

    def test_returns_none_for_nonexistent_dir(self):
        """Thư mục không tồn tại → None."""
        assert _find_checkpoint("/nonexistent/path/xyz") is None

    def test_finds_epoch_checkpoint(self):
        """Tìm được checkpoint_epoch_*.pt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Tạo dummy checkpoint files
            Path(tmpdir, "checkpoint_epoch_0001.pt").touch()
            Path(tmpdir, "checkpoint_epoch_0002.pt").touch()

            result = _find_checkpoint(tmpdir)
            assert result is not None
            assert "checkpoint_epoch_0002.pt" in result

    def test_finds_stock_eval_net_fallback(self):
        """Fallback tìm stock_eval_net.pt khi không có epoch checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "stock_eval_net.pt").touch()

            result = _find_checkpoint(tmpdir)
            assert result is not None
            assert "stock_eval_net.pt" in result


class TestFindDateIndex:
    """Tests cho _find_date_index."""

    def test_finds_exact_date(self):
        """Tìm đúng index khi date match chính xác."""
        import pandas as pd

        df = pd.DataFrame({
            "time": ["2024-01-10", "2024-01-11", "2024-01-12"],
            "close": [100, 101, 102],
        })

        result = _find_date_index(df, "2024-01-11")
        assert result == 1

    def test_returns_none_for_empty_date(self):
        """Date string rỗng → None."""
        import pandas as pd

        df = pd.DataFrame({"time": ["2024-01-10"], "close": [100]})
        assert _find_date_index(df, "") is None

    def test_returns_none_for_missing_time_column(self):
        """DataFrame không có cột 'time' → None."""
        import pandas as pd

        df = pd.DataFrame({"date": ["2024-01-10"], "close": [100]})
        assert _find_date_index(df, "2024-01-10") is None
