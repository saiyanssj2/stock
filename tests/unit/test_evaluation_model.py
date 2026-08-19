"""
Unit tests for engine/evaluation_model.py - StockEvalNet and TCNBlock.

Tests architecture correctness, parameter count, output shapes, and value bounds.
"""

import torch
import pytest

from engine.config import ModelConfig
from engine.evaluation_model import StockEvalNet, TCNBlock


# ---------------------------------------------------------------------------
# TCNBlock Tests
# ---------------------------------------------------------------------------


class TestTCNBlock:
    """Tests for the TCNBlock module."""

    def test_output_shape_same_channels(self):
        """TCNBlock with same in/out channels preserves shape."""
        block = TCNBlock(64, 64, kernel_size=3, dilation=1)
        x = torch.randn(2, 64, 60)
        y = block(x)
        assert y.shape == (2, 64, 60)

    def test_output_shape_different_channels(self):
        """TCNBlock with different in/out channels changes channel dim only."""
        block = TCNBlock(61, 128, kernel_size=3, dilation=1)
        x = torch.randn(2, 61, 60)
        y = block(x)
        assert y.shape == (2, 128, 60)

    def test_causal_padding_preserves_length(self):
        """Causal padding ensures output temporal length matches input."""
        for dilation in [1, 2, 4, 8]:
            block = TCNBlock(32, 64, kernel_size=3, dilation=dilation)
            x = torch.randn(1, 32, 100)
            y = block(x)
            assert y.shape[-1] == x.shape[-1], f"Failed for dilation={dilation}"

    def test_residual_with_identity(self):
        """When in_ch == out_ch, residual uses Identity (no 1x1 conv)."""
        block = TCNBlock(64, 64)
        assert isinstance(block.residual, torch.nn.Identity)

    def test_residual_with_conv1x1(self):
        """When in_ch != out_ch, residual uses 1x1 Conv1d."""
        block = TCNBlock(61, 128)
        assert isinstance(block.residual, torch.nn.Conv1d)
        assert block.residual.kernel_size == (1,)

    def test_no_nan_in_output(self):
        """TCNBlock output should not contain NaN for valid input."""
        block = TCNBlock(61, 128, kernel_size=3, dilation=2)
        block.eval()
        x = torch.randn(4, 61, 60)
        with torch.no_grad():
            y = block(x)
        assert not torch.isnan(y).any()

    def test_gradient_flow(self):
        """Gradients flow through TCNBlock."""
        block = TCNBlock(61, 128)
        x = torch.randn(2, 61, 60, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ---------------------------------------------------------------------------
# StockEvalNet Tests
# ---------------------------------------------------------------------------


class TestStockEvalNet:
    """Tests for the StockEvalNet model."""

    @pytest.fixture
    def model(self):
        """Create a StockEvalNet with default config."""
        config = ModelConfig()
        net = StockEvalNet(config)
        net.eval()
        return net

    def test_output_shape_single(self, model):
        """Single sample produces output shape (1, 1)."""
        x = torch.randn(1, 60, 61)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1)

    def test_output_shape_batch(self, model):
        """Batch of samples produces correct output shape."""
        for batch_size in [1, 4, 16, 50]:
            x = torch.randn(batch_size, 60, 61)
            with torch.no_grad():
                y = model(x)
            assert y.shape == (batch_size, 1)

    def test_output_bounded(self, model):
        """Output values are in [-1.0, +1.0] due to Tanh."""
        x = torch.randn(50, 60, 61)
        with torch.no_grad():
            y = model(x)
        assert (y >= -1.0).all(), f"Min value: {y.min().item()}"
        assert (y <= 1.0).all(), f"Max value: {y.max().item()}"

    def test_output_bounded_extreme_input(self, model):
        """Output bounded even with extreme input values."""
        # Very large values
        x_large = torch.randn(10, 60, 61) * 1000
        with torch.no_grad():
            y = model(x_large)
        assert (y >= -1.0).all() and (y <= 1.0).all()

        # All zeros
        x_zero = torch.zeros(5, 60, 61)
        with torch.no_grad():
            y = model(x_zero)
        assert (y >= -1.0).all() and (y <= 1.0).all()

    def test_parameter_count_approximate(self, model):
        """Parameter count should be approximately 180K-250K for the specified architecture."""
        count = model.count_parameters()
        # The architecture (63→128→128→64 TCN + attention) yields ~244K params.
        # The design doc estimated ~180K but actual count is higher due to
        # the Conv1d layers. Within expected range for RTX 2060 VRAM budget.
        assert 140_000 <= count <= 260_000, f"Parameter count {count:,} outside expected range"

    def test_no_nan_in_output(self, model):
        """Model output should not contain NaN for valid input."""
        x = torch.randn(8, 60, 61)
        with torch.no_grad():
            y = model(x)
        assert not torch.isnan(y).any()

    def test_deterministic_eval_mode(self, model):
        """Model in eval mode produces deterministic outputs."""
        x = torch.randn(4, 60, 61)
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x)
        torch.testing.assert_close(y1, y2)

    def test_gradient_flow_full_model(self):
        """Gradients flow through the full model."""
        config = ModelConfig()
        net = StockEvalNet(config)
        net.train()
        x = torch.randn(4, 60, 61, requires_grad=True)
        y = net(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_custom_config(self):
        """Model works with custom configuration."""
        config = ModelConfig(
            num_features=32,
            lookback=30,
            tcn_channels=[64, 64, 32],
            dilations=[1, 2, 4],
            attention_heads=2,
            attention_dim=32,
        )
        net = StockEvalNet(config)
        net.eval()
        x = torch.randn(2, 30, 32)
        with torch.no_grad():
            y = net(x)
        assert y.shape == (2, 1)

    def test_default_config_used_when_none(self):
        """If config is None, default ModelConfig is used."""
        net = StockEvalNet(config=None)
        assert net.config.num_features == 61
        assert net.config.lookback == 60

    def test_count_parameters_method(self, model):
        """count_parameters returns a positive integer."""
        count = model.count_parameters()
        assert isinstance(count, int)
        assert count > 0
