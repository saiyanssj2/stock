"""
Property-based tests for EvaluationModel (StockEvalNet).

Tests the following correctness properties from the design document:
- Property 5: Model save/load inference consistency (outputs within ±1e-6 after save/load on same device)
- Property 7: Evaluation model output bounded for any batch size (shape (batch, 1) with values in [-1.0, +1.0])

**Validates: Requirements 3.1, 3.5, 13.2**
"""

import tempfile
from pathlib import Path

import torch
import numpy as np
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from engine.config import ModelConfig
from engine.evaluation_model import StockEvalNet


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_FEATURES = 61
DEFAULT_LOOKBACK = 60


# ---------------------------------------------------------------------------
# Custom Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def batch_size_strategy(draw):
    """Generate a valid batch size in [1, 50]."""
    return draw(st.integers(min_value=1, max_value=50))


@st.composite
def random_seed_strategy(draw):
    """Generate a random seed for reproducible tensor generation."""
    return draw(st.integers(min_value=0, max_value=2**31 - 1))


@st.composite
def input_scale_strategy(draw):
    """Generate an input scale factor for varying input magnitudes."""
    return draw(st.sampled_from([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]))


@st.composite
def input_distribution_strategy(draw):
    """
    Generate a strategy name for input tensor distribution.
    Used to test model robustness across different input patterns.
    """
    return draw(st.sampled_from(["normal", "uniform", "extreme_positive", "extreme_negative", "zeros", "ones"]))


# ---------------------------------------------------------------------------
# Property 5: Model save/load inference consistency
# ---------------------------------------------------------------------------


class TestProperty5SaveLoadInferenceConsistency:
    """
    Property 5: Model save/load inference consistency.

    For any trained EvaluationModel and any valid Feature_Vector input,
    saving the model to disk and loading it on the same device type SHALL
    produce Position_Score outputs within ±1e-6 tolerance of the original
    model's outputs.

    **Validates: Requirements 13.2**
    """

    @given(
        batch_size=batch_size_strategy(),
        seed=random_seed_strategy(),
        scale=input_scale_strategy(),
    )
    @settings(
        max_examples=15,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_save_load_produces_identical_outputs(self, batch_size, seed, scale):
        """
        Save model state_dict, load into a new instance, and verify
        inference outputs match within ±1e-6 for the same input.

        **Validates: Requirements 13.2**
        """
        # Set seed for reproducible input generation
        torch.manual_seed(seed)

        config = ModelConfig()
        model_original = StockEvalNet(config)
        model_original.eval()

        # Generate random input with varying scale
        x = torch.randn(batch_size, DEFAULT_LOOKBACK, NUM_FEATURES) * scale

        # Get original model output
        with torch.no_grad():
            output_original = model_original(x)

        # Save model state_dict to a temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / "model_checkpoint.pt")
            torch.save(model_original.state_dict(), model_path)

            # Create a new model instance and load the state_dict
            model_loaded = StockEvalNet(config)
            model_loaded.load_state_dict(torch.load(model_path, weights_only=True))
            model_loaded.eval()

            # Get loaded model output with the same input
            with torch.no_grad():
                output_loaded = model_loaded(x)

        # Assert outputs match within ±1e-6
        max_diff = (output_original - output_loaded).abs().max().item()
        assert max_diff <= 1e-6, (
            f"Max difference {max_diff:.2e} exceeds ±1e-6 tolerance. "
            f"batch_size={batch_size}, seed={seed}, scale={scale}"
        )

    @given(
        seed=random_seed_strategy(),
        batch_size=st.integers(min_value=1, max_value=20),
    )
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_save_load_consistency_multiple_inferences(self, seed, batch_size):
        """
        After save/load, multiple inferences with different inputs all
        match the original model within ±1e-6.

        **Validates: Requirements 13.2**
        """
        torch.manual_seed(seed)

        config = ModelConfig()
        model_original = StockEvalNet(config)
        model_original.eval()

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / "model_multi_check.pt")
            torch.save(model_original.state_dict(), model_path)

            # Load into new instance
            model_loaded = StockEvalNet(config)
            model_loaded.load_state_dict(torch.load(model_path, weights_only=True))
            model_loaded.eval()

            # Run multiple inferences with different inputs
            for i in range(5):
                x = torch.randn(batch_size, DEFAULT_LOOKBACK, NUM_FEATURES) * (i + 1)
                with torch.no_grad():
                    out_orig = model_original(x)
                    out_loaded = model_loaded(x)

                max_diff = (out_orig - out_loaded).abs().max().item()
                assert max_diff <= 1e-6, (
                    f"Inference {i}: max diff {max_diff:.2e} exceeds ±1e-6. "
                    f"seed={seed}, batch_size={batch_size}"
                )


# ---------------------------------------------------------------------------
# Property 7: Evaluation model output bounded for any batch size
# ---------------------------------------------------------------------------


class TestProperty7OutputBounded:
    """
    Property 7: Evaluation model output bounded for any batch size.

    For any batch of 1 to 50 valid Feature_Vector inputs (including edge cases
    like all-zero tensors), the EvaluationModel SHALL produce output of shape
    (batch_size, 1) with all values in the range [-1.0, +1.0].

    **Validates: Requirements 3.1, 3.5**
    """

    @given(
        batch_size=batch_size_strategy(),
        distribution=input_distribution_strategy(),
    )
    @settings(
        max_examples=15,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_output_shape_and_bounds_various_distributions(self, batch_size, distribution):
        """
        For batch sizes 1-50 and various input distributions, output shape
        is (batch, 1) and all values are in [-1.0, +1.0].

        **Validates: Requirements 3.1, 3.5**
        """
        config = ModelConfig()
        model = StockEvalNet(config)
        model.eval()

        # Generate input tensor based on the distribution type
        if distribution == "normal":
            x = torch.randn(batch_size, DEFAULT_LOOKBACK, NUM_FEATURES)
        elif distribution == "uniform":
            x = torch.rand(batch_size, DEFAULT_LOOKBACK, NUM_FEATURES)
        elif distribution == "extreme_positive":
            x = torch.ones(batch_size, DEFAULT_LOOKBACK, NUM_FEATURES) * 1000.0
        elif distribution == "extreme_negative":
            x = torch.ones(batch_size, DEFAULT_LOOKBACK, NUM_FEATURES) * -1000.0
        elif distribution == "zeros":
            x = torch.zeros(batch_size, DEFAULT_LOOKBACK, NUM_FEATURES)
        elif distribution == "ones":
            x = torch.ones(batch_size, DEFAULT_LOOKBACK, NUM_FEATURES)
        else:
            x = torch.randn(batch_size, DEFAULT_LOOKBACK, NUM_FEATURES)

        with torch.no_grad():
            output = model(x)

        # Assert shape is (batch_size, 1)
        assert output.shape == (batch_size, 1), (
            f"Expected shape ({batch_size}, 1), got {output.shape}. "
            f"distribution={distribution}"
        )

        # Assert all values in [-1.0, +1.0]
        assert (output >= -1.0).all(), (
            f"Found value below -1.0: min={output.min().item():.6f}. "
            f"batch_size={batch_size}, distribution={distribution}"
        )
        assert (output <= 1.0).all(), (
            f"Found value above +1.0: max={output.max().item():.6f}. "
            f"batch_size={batch_size}, distribution={distribution}"
        )

    @given(
        batch_size=batch_size_strategy(),
        scale=st.floats(min_value=0.001, max_value=10000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(
        max_examples=15,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_output_bounded_with_scaled_random_input(self, batch_size, scale):
        """
        Random input scaled by arbitrary positive factor still produces
        output in [-1.0, +1.0] with correct shape.

        **Validates: Requirements 3.1, 3.5**
        """
        config = ModelConfig()
        model = StockEvalNet(config)
        model.eval()

        x = torch.randn(batch_size, DEFAULT_LOOKBACK, NUM_FEATURES) * scale

        with torch.no_grad():
            output = model(x)

        # Assert shape
        assert output.shape == (batch_size, 1), (
            f"Expected shape ({batch_size}, 1), got {output.shape}. "
            f"scale={scale}"
        )

        # Assert bounds
        assert (output >= -1.0).all(), (
            f"Value below -1.0: min={output.min().item():.6f}. "
            f"batch_size={batch_size}, scale={scale}"
        )
        assert (output <= 1.0).all(), (
            f"Value above +1.0: max={output.max().item():.6f}. "
            f"batch_size={batch_size}, scale={scale}"
        )

    @given(batch_size=batch_size_strategy())
    @settings(
        max_examples=15,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_output_no_nan_or_inf(self, batch_size):
        """
        Model output should never contain NaN or Inf for any valid
        finite input tensor.

        **Validates: Requirements 3.1, 3.5**
        """
        config = ModelConfig()
        model = StockEvalNet(config)
        model.eval()

        # Mix of different value ranges in the same batch
        x = torch.randn(batch_size, DEFAULT_LOOKBACK, NUM_FEATURES)
        # Add some extreme values (but finite)
        x[:, :10, :5] = 500.0
        x[:, 10:20, 5:10] = -500.0

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output).any(), (
            f"Found NaN in output. batch_size={batch_size}"
        )
        assert not torch.isinf(output).any(), (
            f"Found Inf in output. batch_size={batch_size}"
        )
