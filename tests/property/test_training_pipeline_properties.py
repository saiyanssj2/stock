"""
Property-based tests for TrainingPipeline.

Tests the following correctness properties from the design document:
- Property 10: Training label generation bounded - labels in [-1.0, +1.0],
  monotonically increasing with return
- Property 11: Chronological split ordering - all train timestamps < all val
  timestamps < all test timestamps
- Property 12: Checkpoint save/resume consistency - resume produces same
  parameter updates within 1e-6

**Validates: Requirements 6.2, 6.3, 6.8**
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.config import ModelConfig, TrainingConfig
from engine.training_pipeline import CheckpointManager, TrainingPipeline


# ---------------------------------------------------------------------------
# Custom strategies for TrainingPipeline property tests
# ---------------------------------------------------------------------------


@st.composite
def price_series_strategy(draw, min_length=10, max_length=500):
    """
    Generate a valid DataFrame with a 'close' column containing
    realistic price data suitable for label generation.

    Prices follow a random walk from a base price with daily returns
    within the Vietnamese market's ±7% limit.
    """
    n = draw(st.integers(min_value=min_length, max_value=max_length))
    base_price = draw(st.floats(min_value=5.0, max_value=500.0))

    closes = np.zeros(n, dtype=np.float64)
    closes[0] = base_price

    for i in range(1, n):
        daily_return = draw(st.floats(min_value=-0.069, max_value=0.069))
        closes[i] = closes[i - 1] * (1.0 + daily_return)
        # Ensure positive prices
        closes[i] = max(closes[i], 0.01)

    return pd.DataFrame({"close": closes})


@st.composite
def chronological_data_strategy(draw, min_length=20, max_length=300):
    """
    Generate data and labels arrays suitable for chronological splitting.

    Data uses sequential integer values (simulating timestamps/indices),
    labels are valid floats in [-1, 1] with some NaN at the end
    (simulating the label horizon).
    """
    n = draw(st.integers(min_value=min_length, max_value=max_length))
    num_features = draw(st.integers(min_value=1, max_value=10))

    # Data is sequential indices (so we can verify ordering)
    data = np.arange(n * num_features, dtype=np.float64).reshape(n, num_features)

    # Labels: valid values with some NaN at the end (like label generation)
    nan_count = draw(st.integers(min_value=0, max_value=min(n // 4, 20)))
    labels = np.array(draw(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0,
                      allow_nan=False, allow_infinity=False),
            min_size=n - nan_count,
            max_size=n - nan_count,
        )
    ))
    # Append NaN labels at the end (simulating insufficient future data)
    if nan_count > 0:
        labels = np.concatenate([labels, np.full(nan_count, np.nan)])

    return data, labels


# ---------------------------------------------------------------------------
# Property 10: Training label generation bounded
# ---------------------------------------------------------------------------


class TestProperty10LabelGenerationBounded:
    """
    Property 10: Training label generation bounded.

    For any valid OHLCV price series and any horizon value, the generated
    training labels SHALL all be in the range [-1.0, +1.0], and the mapping
    SHALL be monotonically increasing with respect to the underlying
    percentage return.

    **Validates: Requirements 6.2**
    """

    @given(
        df=price_series_strategy(min_length=10, max_length=200),
        horizon=st.integers(min_value=1, max_value=20),
        sensitivity=st.floats(min_value=0.1, max_value=100.0),
    )
    @settings(max_examples=20)
    def test_labels_bounded_in_minus_one_plus_one(self, df, horizon, sensitivity):
        """
        All non-NaN labels produced by _generate_labels are in [-1.0, +1.0].

        **Validates: Requirements 6.2**
        """
        pipeline = TrainingPipeline()
        labels = pipeline._generate_labels(df, horizon=horizon, sensitivity=sensitivity)

        # Filter out NaN labels
        valid_labels = labels[~np.isnan(labels)]

        # All valid labels must be bounded
        if len(valid_labels) > 0:
            assert np.all(valid_labels >= -1.0), (
                f"Found label below -1.0: min={valid_labels.min()}"
            )
            assert np.all(valid_labels <= 1.0), (
                f"Found label above +1.0: max={valid_labels.max()}"
            )

    @given(
        base_price=st.floats(min_value=10.0, max_value=200.0),
        horizon=st.integers(min_value=1, max_value=10),
        sensitivity=st.floats(min_value=1.0, max_value=50.0),
    )
    @settings(max_examples=20)
    def test_labels_monotonically_increasing_with_return(
        self, base_price, horizon, sensitivity
    ):
        """
        Labels increase monotonically as the percentage return increases.
        A higher future return should always produce a higher (or equal) label.

        **Validates: Requirements 6.2**
        """
        pipeline = TrainingPipeline()

        # Generate a set of scenarios with known returns
        returns = [-0.20, -0.10, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10, 0.20]
        labels = []

        for ret in returns:
            future_price = base_price * (1.0 + ret)
            # Create a simple DataFrame: base_price for `horizon` entries,
            # then future_price
            closes = [base_price] * horizon + [future_price]
            df = pd.DataFrame({"close": closes})
            lab = pipeline._generate_labels(df, horizon=horizon, sensitivity=sensitivity)
            labels.append(lab[0])

        # Verify monotonically increasing
        for i in range(len(labels) - 1):
            assert labels[i] <= labels[i + 1] + 1e-12, (
                f"Monotonicity violation: label[{i}]={labels[i]:.8f} > "
                f"label[{i+1}]={labels[i+1]:.8f} "
                f"(returns: {returns[i]:.4f} vs {returns[i+1]:.4f})"
            )

    @given(
        df=price_series_strategy(min_length=10, max_length=100),
        horizon=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=15)
    def test_last_horizon_entries_are_nan(self, df, horizon):
        """
        The last `horizon` entries should be NaN since there's no future
        data available for label computation.

        **Validates: Requirements 6.2**
        """
        assume(len(df) > horizon)

        pipeline = TrainingPipeline()
        labels = pipeline._generate_labels(df, horizon=horizon)

        # Last `horizon` entries must be NaN
        assert all(np.isnan(labels[-horizon:])), (
            f"Expected last {horizon} labels to be NaN, "
            f"got: {labels[-horizon:]}"
        )


# ---------------------------------------------------------------------------
# Property 11: Chronological split ordering
# ---------------------------------------------------------------------------


class TestProperty11ChronologicalSplitOrdering:
    """
    Property 11: Chronological split ordering.

    For any valid data/label arrays, the chronological split should ensure
    all indices in train < all indices in val < all indices in test
    (no future leakage).

    **Validates: Requirements 6.3**
    """

    @given(data_and_labels=chronological_data_strategy(min_length=20, max_length=300))
    @settings(max_examples=20)
    def test_train_indices_before_val_before_test(self, data_and_labels):
        """
        All data points in the training set appear chronologically before
        all data points in the validation set, and all validation data points
        appear before all test data points.

        **Validates: Requirements 6.3**
        """
        data, labels = data_and_labels

        # Need at least some valid (non-NaN) labels for a meaningful split
        valid_count = np.sum(~np.isnan(labels))
        assume(valid_count >= 5)

        pipeline = TrainingPipeline()
        (train_d, train_l), (val_d, val_l), (test_d, test_l) = \
            pipeline._chronological_split(data, labels)

        # Each split must be non-empty
        assume(len(train_d) > 0 and len(val_d) > 0 and len(test_d) > 0)

        # Chronological ordering: max value in train < min value in val
        # Since data is sequential, the first column encodes position
        train_max = train_d[:, 0].max()
        val_min = val_d[:, 0].min()
        val_max = val_d[:, 0].max()
        test_min = test_d[:, 0].min()

        assert train_max < val_min, (
            f"Train/val overlap: train_max={train_max}, val_min={val_min}"
        )
        assert val_max < test_min, (
            f"Val/test overlap: val_max={val_max}, test_min={test_min}"
        )

    @given(data_and_labels=chronological_data_strategy(min_length=20, max_length=300))
    @settings(max_examples=20)
    def test_no_nan_labels_in_splits(self, data_and_labels):
        """
        No NaN labels appear in any of the split sets.

        **Validates: Requirements 6.3**
        """
        data, labels = data_and_labels

        valid_count = np.sum(~np.isnan(labels))
        assume(valid_count >= 5)

        pipeline = TrainingPipeline()
        (_, train_l), (_, val_l), (_, test_l) = \
            pipeline._chronological_split(data, labels)

        assert not np.any(np.isnan(train_l)), "Found NaN in train labels"
        assert not np.any(np.isnan(val_l)), "Found NaN in val labels"
        assert not np.any(np.isnan(test_l)), "Found NaN in test labels"

    @given(data_and_labels=chronological_data_strategy(min_length=20, max_length=300))
    @settings(max_examples=20)
    def test_split_covers_all_valid_data(self, data_and_labels):
        """
        The total number of samples across all splits equals the number
        of valid (non-NaN) labels in the original data.

        **Validates: Requirements 6.3**
        """
        data, labels = data_and_labels

        valid_count = int(np.sum(~np.isnan(labels)))
        assume(valid_count >= 5)

        pipeline = TrainingPipeline()
        (train_d, _), (val_d, _), (test_d, _) = \
            pipeline._chronological_split(data, labels)

        total_in_splits = len(train_d) + len(val_d) + len(test_d)
        assert total_in_splits == valid_count, (
            f"Split total {total_in_splits} != valid labels {valid_count}"
        )


# ---------------------------------------------------------------------------
# Property 12: Checkpoint save/resume consistency
# ---------------------------------------------------------------------------


class TestProperty12CheckpointSaveResumeConsistency:
    """
    Property 12: Checkpoint save/resume consistency.

    For any model state, saving a checkpoint and resuming should produce
    identical model parameters within ±1e-6.

    **Validates: Requirements 6.8**
    """

    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=20, deadline=None)
    def test_checkpoint_save_resume_produces_same_params(self, seed):
        """
        Saving a model checkpoint and then loading it into a fresh model
        produces identical parameters within ±1e-6.

        **Validates: Requirements 6.8**
        """
        torch.manual_seed(seed)

        from engine.evaluation_model import StockEvalNet

        config = ModelConfig(num_features=61, lookback=60)
        model = StockEvalNet(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Perform a forward/backward pass to give optimizer state
        dummy_input = torch.randn(2, 60, 61)
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Capture original parameters
        original_params = {
            k: v.clone().detach() for k, v in model.state_dict().items()
        }
        original_optimizer_state = {
            k: v.clone().detach() if isinstance(v, torch.Tensor) else v
            for group in optimizer.state_dict()["state"].values()
            for k, v in group.items()
            if isinstance(v, torch.Tensor)
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            # Save checkpoint
            checkpoint_path = manager.save(
                model=model,
                optimizer=optimizer,
                epoch=5,
                train_loss=0.3,
                val_loss=0.4,
                best_val_loss=0.35,
            )

            # Load into a fresh model
            new_model = StockEvalNet(config)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
            info = manager.load(new_model, new_optimizer, checkpoint_path)

            # Verify model parameters match within 1e-6
            for key in original_params:
                orig = original_params[key]
                loaded = new_model.state_dict()[key]
                max_diff = (orig - loaded).abs().max().item()
                assert max_diff < 1e-6, (
                    f"Parameter '{key}' differs by {max_diff:.2e} "
                    f"(exceeds 1e-6 tolerance)"
                )

            # Verify epoch and losses are preserved
            assert info["epoch"] == 5
            assert abs(info["train_loss"] - 0.3) < 1e-6
            assert abs(info["val_loss"] - 0.4) < 1e-6

    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=10, deadline=None)
    def test_resumed_training_step_produces_same_update(self, seed):
        """
        After saving and resuming, performing one training step on the
        same data produces the same parameter updates within ±1e-6 as
        continuing without interruption.

        **Validates: Requirements 6.8**
        """
        torch.manual_seed(seed)

        from engine.evaluation_model import StockEvalNet

        config = ModelConfig(num_features=61, lookback=60)

        # Create model and optimizer
        model_a = StockEvalNet(config)
        optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)

        # Generate deterministic training data
        torch.manual_seed(seed + 1000)
        train_input = torch.randn(4, 60, 61)
        train_target = torch.randn(4, 1)

        # Do initial training step to populate optimizer state
        model_a.train()
        output_a = model_a(train_input)
        loss_a = torch.nn.functional.mse_loss(output_a, train_target)
        loss_a.backward()
        optimizer_a.step()
        optimizer_a.zero_grad()

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            # Save checkpoint at this point
            checkpoint_path = manager.save(
                model=model_a,
                optimizer=optimizer_a,
                epoch=1,
                train_loss=loss_a.item(),
                val_loss=0.5,
                best_val_loss=0.5,
            )

            # Path A: Continue training without interruption
            torch.manual_seed(seed + 2000)
            next_input = torch.randn(4, 60, 61)
            next_target = torch.randn(4, 1)

            model_a.train()
            output_a2 = model_a(next_input)
            loss_a2 = torch.nn.functional.mse_loss(output_a2, next_target)
            loss_a2.backward()
            optimizer_a.step()

            params_after_a = {
                k: v.clone().detach() for k, v in model_a.state_dict().items()
            }

            # Path B: Resume from checkpoint and do same step
            model_b = StockEvalNet(config)
            optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
            manager.load(model_b, optimizer_b, checkpoint_path)

            # Same deterministic input
            torch.manual_seed(seed + 2000)
            next_input_b = torch.randn(4, 60, 61)
            next_target_b = torch.randn(4, 1)

            model_b.train()
            output_b2 = model_b(next_input_b)
            loss_b2 = torch.nn.functional.mse_loss(output_b2, next_target_b)
            loss_b2.backward()
            optimizer_b.step()

            params_after_b = {
                k: v.clone().detach() for k, v in model_b.state_dict().items()
            }

            # Compare: parameters should match within 1e-6
            for key in params_after_a:
                diff = (params_after_a[key] - params_after_b[key]).abs().max().item()
                assert diff < 1e-6, (
                    f"After one training step, parameter '{key}' differs by "
                    f"{diff:.2e} between continued and resumed paths "
                    f"(exceeds 1e-6 tolerance)"
                )
