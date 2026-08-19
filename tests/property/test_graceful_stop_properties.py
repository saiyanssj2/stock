"""
Property-based tests for Graceful Stop Training.

Tests the following correctness property from the design document:
- Property 20: Graceful stop checkpoint validity - stop request completes
  current epoch, does NOT begin next epoch, saves valid checkpoint, and
  resume produces identical parameter updates (±1e-6)

**Validates: Requirements 14.2, 14.3, 14.5, 14.6**
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.config import ModelConfig, TrainingConfig
from engine.training_controller import TrainingController
from engine.training_pipeline import CheckpointManager, TrainingPipeline


# ---------------------------------------------------------------------------
# Custom strategies for Graceful Stop property tests
# ---------------------------------------------------------------------------


@st.composite
def graceful_stop_scenario_strategy(draw):
    """
    Generate a graceful stop scenario with:
    - A seed for deterministic model initialization
    - The epoch at which stop is requested (1-indexed)
    - Total epochs planned for the training session

    Ensures stop_at_epoch <= total_epochs.
    """
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    total_epochs = draw(st.integers(min_value=3, max_value=8))
    stop_at_epoch = draw(st.integers(min_value=1, max_value=total_epochs - 1))

    return {
        "seed": seed,
        "total_epochs": total_epochs,
        "stop_at_epoch": stop_at_epoch,
    }


# ---------------------------------------------------------------------------
# Property 20: Graceful stop checkpoint validity
# ---------------------------------------------------------------------------


class TestProperty20GracefulStopCheckpointValidity:
    """
    Property 20: Graceful stop checkpoint validity.

    For any valid training session and any epoch at which stop is requested:
    1. The current epoch completes fully (all batches processed)
    2. No new epoch begins after stop is requested
    3. A valid checkpoint is saved at the stop point
    4. Resuming from that checkpoint and running one more training step
       produces identical parameter updates (±1e-6) as if training had
       continued without interruption

    **Validates: Requirements 14.2, 14.3, 14.5, 14.6**
    """

    @given(scenario=graceful_stop_scenario_strategy())
    @settings(max_examples=20, deadline=None)
    def test_graceful_stop_completes_current_epoch_and_saves_checkpoint(self, scenario):
        """
        When a graceful stop is requested during training, the current epoch
        completes fully and a valid checkpoint is saved. No new epoch begins
        after the stop request.

        Verifies:
        - The training loop runs exactly stop_at_epoch epochs (not more)
        - A checkpoint file exists for the stopped epoch
        - The checkpoint contains valid model parameters and optimizer state

        **Validates: Requirements 14.2, 14.3, 14.5**
        """
        seed = scenario["seed"]
        total_epochs = scenario["total_epochs"]
        stop_at_epoch = scenario["stop_at_epoch"]

        torch.manual_seed(seed)

        from engine.evaluation_model import StockEvalNet

        config = ModelConfig(num_features=61, lookback=60)
        model = StockEvalNet(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = CheckpointManager(checkpoint_dir=tmpdir)

            # Track which epochs actually ran
            epochs_completed = []
            checkpoint_saved_epochs = []

            def checkpoint_save_fn(epoch, metrics):
                checkpoint_saved_epochs.append(epoch)

            controller = TrainingController(checkpoint_save_fn=checkpoint_save_fn)
            controller.start_training(total_epochs=total_epochs)

            # Generate deterministic training data
            torch.manual_seed(seed + 100)
            train_input = torch.randn(8, 60, 61)
            train_target = torch.randn(8, 1)

            # Simulate training loop (mirrors TrainingPipeline logic)
            for epoch in range(1, total_epochs + 1):
                # Check should_continue BEFORE beginning new epoch (Req 14.5)
                if not controller.should_continue_training():
                    break

                controller.begin_epoch(epoch)

                # Simulate full epoch training (forward + backward + step)
                model.train()
                optimizer.zero_grad()
                output = model(train_input)
                loss = criterion(output, train_target)
                loss.backward()
                optimizer.step()

                epochs_completed.append(epoch)

                # Save checkpoint after epoch (Req 14.3)
                checkpoint_manager.save(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=loss.item(),
                    val_loss=loss.item() * 1.1,
                    best_val_loss=loss.item(),
                )

                # Notify controller epoch is complete
                metrics = {"train_loss": loss.item(), "val_loss": loss.item() * 1.1}
                should_continue = controller._on_epoch_complete(epoch, metrics)

                if not should_continue:
                    break

                # Request stop after the designated epoch completes
                if epoch == stop_at_epoch:
                    controller.request_stop()

            # ASSERTIONS:

            # 1. Current epoch completed (Req 14.2)
            assert stop_at_epoch in epochs_completed, (
                f"Epoch {stop_at_epoch} should have completed but didn't. "
                f"Completed epochs: {epochs_completed}"
            )

            # 2. No new epoch began after stop (Req 14.5)
            assert max(epochs_completed) == stop_at_epoch, (
                f"Expected training to stop at epoch {stop_at_epoch}, "
                f"but ran through epoch {max(epochs_completed)}. "
                f"Epochs completed: {epochs_completed}"
            )

            # 3. Checkpoint was saved for the stopped epoch (Req 14.3)
            checkpoint_path = Path(tmpdir) / f"checkpoint_epoch_{stop_at_epoch:04d}.pt"
            assert checkpoint_path.exists(), (
                f"Checkpoint for epoch {stop_at_epoch} not found at {checkpoint_path}"
            )

            # 4. Checkpoint is valid and loadable
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu",
                                    weights_only=False)
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert checkpoint["epoch"] == stop_at_epoch

            # Verify model parameters can be loaded
            new_model = StockEvalNet(config)
            new_model.load_state_dict(checkpoint["model_state_dict"])

    @given(scenario=graceful_stop_scenario_strategy())
    @settings(max_examples=20, deadline=None)
    def test_graceful_stop_does_not_begin_next_epoch(self, scenario):
        """
        After stop is requested, should_continue_training() returns False,
        preventing any subsequent epoch from starting.

        **Validates: Requirements 14.5**
        """
        seed = scenario["seed"]
        total_epochs = scenario["total_epochs"]
        stop_at_epoch = scenario["stop_at_epoch"]

        controller = TrainingController()
        controller.start_training(total_epochs=total_epochs)

        epochs_started = []

        for epoch in range(1, total_epochs + 1):
            if not controller.should_continue_training():
                break

            epochs_started.append(epoch)
            controller.begin_epoch(epoch)

            # Simulate epoch work...
            metrics = {"train_loss": 0.5, "val_loss": 0.6}
            controller._on_epoch_complete(epoch, metrics)

            # Request stop after designated epoch
            if epoch == stop_at_epoch:
                controller.request_stop()

        # The epoch after stop_at_epoch should NOT have started
        assert stop_at_epoch in epochs_started, (
            f"Epoch {stop_at_epoch} should have started"
        )
        for e in epochs_started:
            assert e <= stop_at_epoch, (
                f"Epoch {e} started after stop was requested at epoch {stop_at_epoch}"
            )

    @given(scenario=graceful_stop_scenario_strategy())
    @settings(max_examples=20, deadline=None)
    def test_resume_from_graceful_stop_produces_identical_updates(self, scenario):
        """
        After a graceful stop, resuming from the saved checkpoint and
        performing the next training step produces parameter updates
        identical (±1e-6) to those from continuing without interruption.

        This validates that:
        - The checkpoint captures complete training state (weights + optimizer)
        - Resume is seamless (epoch N+1 after stop is identical to epoch N+1
          without stop)

        **Validates: Requirements 14.6**
        """
        seed = scenario["seed"]
        total_epochs = scenario["total_epochs"]
        stop_at_epoch = scenario["stop_at_epoch"]

        from engine.evaluation_model import StockEvalNet

        config = ModelConfig(num_features=61, lookback=60)

        with tempfile.TemporaryDirectory() as tmpdir:
            # ================================================================
            # Path A: Training with graceful stop, then resume for one more step
            # ================================================================
            torch.manual_seed(seed)
            model_a = StockEvalNet(config)
            optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            checkpoint_manager_a = CheckpointManager(checkpoint_dir=tmpdir)
            controller = TrainingController()
            controller.start_training(total_epochs=total_epochs)

            # Generate deterministic training data for each epoch
            for epoch in range(1, total_epochs + 1):
                if not controller.should_continue_training():
                    break

                controller.begin_epoch(epoch)

                # Deterministic data per epoch
                torch.manual_seed(seed + epoch * 1000)
                train_input = torch.randn(4, 60, 61)
                train_target = torch.randn(4, 1)

                model_a.train()
                optimizer_a.zero_grad()
                output = model_a(train_input)
                loss = criterion(output, train_target)
                loss.backward()
                optimizer_a.step()

                # Save checkpoint
                cp_path = checkpoint_manager_a.save(
                    model=model_a,
                    optimizer=optimizer_a,
                    epoch=epoch,
                    train_loss=loss.item(),
                    val_loss=loss.item() * 1.1,
                    best_val_loss=loss.item(),
                )

                metrics = {"train_loss": loss.item()}
                should_continue = controller._on_epoch_complete(epoch, metrics)
                if not should_continue:
                    break

                # Request stop after designated epoch
                if epoch == stop_at_epoch:
                    controller.request_stop()

            # Now resume from the graceful stop checkpoint
            stop_checkpoint_path = str(
                Path(tmpdir) / f"checkpoint_epoch_{stop_at_epoch:04d}.pt"
            )
            assert Path(stop_checkpoint_path).exists()

            # Load checkpoint into fresh model
            model_a_resumed = StockEvalNet(config)
            optimizer_a_resumed = torch.optim.Adam(
                model_a_resumed.parameters(), lr=1e-3
            )
            checkpoint_manager_a.load(
                model_a_resumed, optimizer_a_resumed, stop_checkpoint_path
            )

            # Do one more training step (the epoch after stop)
            next_epoch = stop_at_epoch + 1
            torch.manual_seed(seed + next_epoch * 1000)
            next_input = torch.randn(4, 60, 61)
            next_target = torch.randn(4, 1)

            model_a_resumed.train()
            optimizer_a_resumed.zero_grad()
            output_a = model_a_resumed(next_input)
            loss_a = criterion(output_a, next_target)
            loss_a.backward()
            optimizer_a_resumed.step()

            params_after_resume = {
                k: v.clone().detach()
                for k, v in model_a_resumed.state_dict().items()
            }

            # ================================================================
            # Path B: Training WITHOUT interruption through stop_at_epoch + 1
            # ================================================================
            torch.manual_seed(seed)
            model_b = StockEvalNet(config)
            optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)

            # Run through all epochs up to and including next_epoch
            for epoch in range(1, next_epoch + 1):
                torch.manual_seed(seed + epoch * 1000)
                train_input_b = torch.randn(4, 60, 61)
                train_target_b = torch.randn(4, 1)

                model_b.train()
                optimizer_b.zero_grad()
                output_b = model_b(train_input_b)
                loss_b = criterion(output_b, train_target_b)
                loss_b.backward()
                optimizer_b.step()

            params_continuous = {
                k: v.clone().detach()
                for k, v in model_b.state_dict().items()
            }

            # ================================================================
            # Comparison: resumed path should match continuous path within 1e-6
            # ================================================================
            for key in params_after_resume:
                diff = (
                    params_after_resume[key] - params_continuous[key]
                ).abs().max().item()
                assert diff < 1e-6, (
                    f"Parameter '{key}' differs by {diff:.2e} between "
                    f"resumed (after graceful stop at epoch {stop_at_epoch}) "
                    f"and continuous training paths (exceeds ±1e-6 tolerance)"
                )

    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        total_epochs=st.integers(min_value=3, max_value=10),
    )
    @settings(max_examples=20, deadline=None)
    def test_graceful_stop_checkpoint_includes_complete_state(self, seed, total_epochs):
        """
        The checkpoint saved during graceful stop contains all required
        training state: model weights, optimizer state, epoch number,
        and training metrics.

        **Validates: Requirements 14.3**
        """
        torch.manual_seed(seed)

        from engine.evaluation_model import StockEvalNet

        config = ModelConfig(num_features=61, lookback=60)
        model = StockEvalNet(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Stop after epoch 1
        stop_at_epoch = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = CheckpointManager(checkpoint_dir=tmpdir)
            controller = TrainingController()
            controller.start_training(total_epochs=total_epochs)

            # Train one epoch then stop
            controller.begin_epoch(1)

            torch.manual_seed(seed + 1000)
            train_input = torch.randn(4, 60, 61)
            train_target = torch.randn(4, 1)

            model.train()
            optimizer.zero_grad()
            output = model(train_input)
            loss = criterion(output, train_target)
            loss.backward()
            optimizer.step()

            # Save checkpoint (mirrors what train_full does)
            train_loss = loss.item()
            val_loss = train_loss * 1.1
            checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=stop_at_epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=train_loss,
            )

            # Request stop and verify controller acknowledges it
            controller.request_stop()
            metrics = {"train_loss": train_loss, "val_loss": val_loss}
            should_continue = controller._on_epoch_complete(stop_at_epoch, metrics)
            assert not should_continue, "Controller should signal stop after request"

            # Verify checkpoint contents
            cp_path = Path(tmpdir) / f"checkpoint_epoch_{stop_at_epoch:04d}.pt"
            assert cp_path.exists(), "Checkpoint file must exist"

            checkpoint = torch.load(str(cp_path), map_location="cpu",
                                    weights_only=False)

            # Required fields
            assert "model_state_dict" in checkpoint, (
                "Checkpoint missing model_state_dict"
            )
            assert "optimizer_state_dict" in checkpoint, (
                "Checkpoint missing optimizer_state_dict"
            )
            assert "epoch" in checkpoint, "Checkpoint missing epoch"
            # train_loss/val_loss may be top-level (legacy) or in metrics dict (portable format)
            has_train_loss = (
                "train_loss" in checkpoint
                or ("metrics" in checkpoint and "train_loss" in checkpoint["metrics"])
            )
            has_val_loss = (
                "val_loss" in checkpoint
                or ("metrics" in checkpoint and "val_loss" in checkpoint["metrics"])
            )
            assert has_train_loss, "Checkpoint missing train_loss"
            assert has_val_loss, "Checkpoint missing val_loss"

            # Verify epoch matches
            assert checkpoint["epoch"] == stop_at_epoch

            # Verify model state has all expected keys
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(checkpoint["model_state_dict"].keys())
            assert model_keys == checkpoint_keys, (
                f"Model state keys mismatch: "
                f"missing={model_keys - checkpoint_keys}, "
                f"extra={checkpoint_keys - model_keys}"
            )

            # Verify optimizer state is non-empty (has parameter groups)
            opt_state = checkpoint["optimizer_state_dict"]
            assert "param_groups" in opt_state, (
                "Optimizer state missing param_groups"
            )
            assert len(opt_state["param_groups"]) > 0, (
                "Optimizer state has empty param_groups"
            )
