"""
Property-based tests for Multi-Machine Portable Training checkpoints.

Tests the following correctness property from the design document:
- Property 21: Portable checkpoint cross-device consistency - checkpoint saved
  on device A, loaded on device B produces Position_Score within ±1e-5; verify
  CPU storage format; verify metadata includes device_type, gpu_model, pytorch_version

**Validates: Requirements 15.2, 15.3, 15.4**
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.config import ModelConfig
from engine.evaluation_model import StockEvalNet
from engine.hardware_profile import HardwareProfile, _optimizer_state_to_cpu


# ---------------------------------------------------------------------------
# Custom strategies for Portable Checkpoint property tests
# ---------------------------------------------------------------------------


@st.composite
def portable_checkpoint_scenario_strategy(draw):
    """
    Generate a portable checkpoint scenario with:
    - A seed for deterministic model initialization and training
    - An epoch number (simulated checkpoint after N epochs)
    - A batch of random input features for inference comparison
    - Number of training steps before saving checkpoint

    This enables testing cross-device checkpoint portability.
    """
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    epoch = draw(st.integers(min_value=1, max_value=20))
    num_train_steps = draw(st.integers(min_value=1, max_value=5))
    batch_size = draw(st.integers(min_value=1, max_value=10))

    return {
        "seed": seed,
        "epoch": epoch,
        "num_train_steps": num_train_steps,
        "batch_size": batch_size,
    }


@st.composite
def hardware_metadata_strategy(draw):
    """
    Generate realistic hardware metadata combinations simulating
    different machines (GPU machine, CPU-only machine).
    """
    device_type = draw(st.sampled_from(["cuda", "cpu"]))

    if device_type == "cuda":
        gpu_model = draw(st.sampled_from([
            "NVIDIA GeForce RTX 2060",
            "NVIDIA GeForce RTX 3060",
            "NVIDIA GeForce GTX 1080 Ti",
            "NVIDIA GeForce RTX 4090",
        ]))
        vram_mb = draw(st.sampled_from([6144, 8192, 11264, 24576]))
    else:
        gpu_model = None
        vram_mb = None

    cpu_model = draw(st.sampled_from([
        "Intel(R) Core(TM) i5-10400F",
        "Intel(R) Core(TM) i7-12700",
        "AMD Ryzen 5 5600X",
        "Intel(R) Core(TM) i9-13900K",
    ]))

    pytorch_version = draw(st.sampled_from([
        "2.0.1", "2.1.0", "2.2.0", "2.3.0", "2.4.0",
    ]))

    return {
        "device_type": device_type,
        "gpu_model": gpu_model,
        "cpu_model": cpu_model,
        "pytorch_version": pytorch_version,
        "vram_mb": vram_mb,
    }


# ---------------------------------------------------------------------------
# Helper: create a trained model with deterministic state
# ---------------------------------------------------------------------------


def _create_trained_model(seed: int, num_train_steps: int, config: ModelConfig):
    """Create a model with deterministic training state for reproducibility."""
    torch.manual_seed(seed)
    model = StockEvalNet(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for step in range(num_train_steps):
        torch.manual_seed(seed + step * 100)
        x = torch.randn(4, config.lookback, config.num_features)
        target = torch.randn(4, 1)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    return model, optimizer


def _create_hardware_profile_mock(device_type="cpu", gpu_model=None, vram_mb=None,
                                   cpu_model="TestCPU"):
    """Create a HardwareProfile with overridden attributes for testing.

    Since HardwareProfile auto-detects hardware in __init__, we override
    the attributes after construction to simulate different machines.
    """
    profile = HardwareProfile.__new__(HardwareProfile)
    profile.device_type = device_type
    profile.gpu_model = gpu_model
    profile.gpu_vram_mb = vram_mb
    profile.cpu_model = cpu_model
    profile.pytorch_version = torch.__version__
    return profile


# ---------------------------------------------------------------------------
# Property 21: Portable checkpoint cross-device consistency
# ---------------------------------------------------------------------------


class TestProperty21PortableCheckpointCrossDeviceConsistency:
    """
    Property 21: Portable checkpoint cross-device consistency.

    For any trained model state:
    1. Checkpoint saved on device A and loaded on device B produces
       Position_Score within ±1e-5 for the same input
    2. All tensors in the checkpoint file are stored on CPU
    3. Checkpoint metadata includes device_type, gpu_model, pytorch_version

    **Validates: Requirements 15.2, 15.3, 15.4**
    """

    @given(scenario=portable_checkpoint_scenario_strategy())
    @settings(max_examples=20, deadline=None)
    def test_checkpoint_cross_device_inference_consistency(self, scenario):
        """
        A checkpoint saved on one device (simulated as device A) and loaded on
        another device (simulated as device B, here CPU since GPU may not be
        available in test env) produces Position_Score within ±1e-5.

        This simulates the cross-device scenario by:
        1. Training and saving on CPU (simulating device A)
        2. Loading on CPU with a different HardwareProfile (simulating device B)
        3. Comparing inference outputs

        **Validates: Requirements 15.2, 15.3**
        """
        seed = scenario["seed"]
        epoch = scenario["epoch"]
        num_train_steps = scenario["num_train_steps"]
        batch_size = scenario["batch_size"]

        config = ModelConfig(num_features=63, lookback=60)

        # Create a trained model (simulating "device A")
        model_a, optimizer_a = _create_trained_model(seed, num_train_steps, config)
        model_a.eval()

        # Generate deterministic test input
        torch.manual_seed(seed + 9999)
        test_input = torch.randn(batch_size, config.lookback, config.num_features)

        # Get inference output from original model (device A)
        with torch.no_grad():
            output_a = model_a(test_input).clone()

        # Save portable checkpoint (device A)
        hardware_a = _create_hardware_profile_mock(
            device_type="cpu",
            gpu_model="NVIDIA GeForce RTX 2060",
            vram_mb=6144,
            cpu_model="Intel(R) Core(TM) i5-10400F",
        )
        metrics = {"train_loss": 0.5, "val_loss": 0.6}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = str(Path(tmpdir) / "portable_checkpoint.pt")

            HardwareProfile.save_checkpoint(
                model=model_a,
                optimizer=optimizer_a,
                epoch=epoch,
                metrics=metrics,
                hardware=hardware_a,
                path=checkpoint_path,
                model_config=config,
            )

            # Load checkpoint on "device B" (simulated as different CPU machine)
            target_device = torch.device("cpu")
            checkpoint = HardwareProfile.load_checkpoint(
                path=checkpoint_path,
                target_device=target_device,
                expected_config=config,
            )

            # Reconstruct model on device B
            model_b = StockEvalNet(config)
            model_b.load_state_dict(checkpoint["model_state_dict"])
            model_b.to(target_device)
            model_b.eval()

            # Get inference output from loaded model (device B)
            test_input_b = test_input.to(target_device)
            with torch.no_grad():
                output_b = model_b(test_input_b)

            # Compare: Position_Score within ±1e-5
            diff = (output_a - output_b).abs().max().item()
            assert diff < 1e-5, (
                f"Cross-device inference differs by {diff:.2e} "
                f"(exceeds ±1e-5 tolerance). "
                f"Batch size={batch_size}, epoch={epoch}, "
                f"train_steps={num_train_steps}"
            )

    @given(scenario=portable_checkpoint_scenario_strategy())
    @settings(max_examples=20, deadline=None)
    def test_checkpoint_tensors_stored_on_cpu(self, scenario):
        """
        All tensors in the saved checkpoint file are stored in CPU format,
        regardless of the device used during training. This ensures
        portability across machines with different hardware.

        **Validates: Requirements 15.2**
        """
        seed = scenario["seed"]
        epoch = scenario["epoch"]
        num_train_steps = scenario["num_train_steps"]

        config = ModelConfig(num_features=63, lookback=60)
        model, optimizer = _create_trained_model(seed, num_train_steps, config)

        # Simulate saving from a "GPU" machine
        hardware = _create_hardware_profile_mock(
            device_type="cuda",
            gpu_model="NVIDIA GeForce RTX 2060",
            vram_mb=6144,
            cpu_model="Intel(R) Core(TM) i5-10400F",
        )
        metrics = {"train_loss": 0.3, "val_loss": 0.4}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = str(Path(tmpdir) / "checkpoint_cpu_verify.pt")

            HardwareProfile.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                hardware=hardware,
                path=checkpoint_path,
                model_config=config,
            )

            # Load raw checkpoint (on CPU) and verify all tensors are CPU
            checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                    weights_only=False)

            # Verify model state_dict tensors are on CPU
            for key, tensor in checkpoint["model_state_dict"].items():
                assert tensor.device == torch.device("cpu"), (
                    f"Model tensor '{key}' is on {tensor.device}, expected cpu"
                )

            # Verify optimizer state_dict tensors are on CPU
            opt_state = checkpoint["optimizer_state_dict"]
            if "state" in opt_state:
                for param_id, param_state in opt_state["state"].items():
                    for state_key, state_val in param_state.items():
                        if isinstance(state_val, torch.Tensor):
                            assert state_val.device == torch.device("cpu"), (
                                f"Optimizer state tensor "
                                f"[param={param_id}, key={state_key}] "
                                f"is on {state_val.device}, expected cpu"
                            )

    @given(
        scenario=portable_checkpoint_scenario_strategy(),
        hw_metadata=hardware_metadata_strategy(),
    )
    @settings(max_examples=20, deadline=None)
    def test_checkpoint_metadata_includes_required_fields(self, scenario, hw_metadata):
        """
        The saved checkpoint includes hardware metadata with at minimum:
        device_type, gpu_model, and pytorch_version fields. These are
        required for traceability across multiple machines.

        **Validates: Requirements 15.4**
        """
        seed = scenario["seed"]
        epoch = scenario["epoch"]
        num_train_steps = scenario["num_train_steps"]

        config = ModelConfig(num_features=63, lookback=60)
        model, optimizer = _create_trained_model(seed, num_train_steps, config)

        # Create hardware profile with the generated metadata
        hardware = _create_hardware_profile_mock(
            device_type=hw_metadata["device_type"],
            gpu_model=hw_metadata["gpu_model"],
            vram_mb=hw_metadata["vram_mb"],
            cpu_model=hw_metadata["cpu_model"],
        )
        # Override pytorch_version to match generated metadata
        hardware.pytorch_version = hw_metadata["pytorch_version"]

        metrics = {"train_loss": 0.25, "val_loss": 0.35}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = str(Path(tmpdir) / "checkpoint_metadata.pt")

            HardwareProfile.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                hardware=hardware,
                path=checkpoint_path,
                model_config=config,
            )

            # Load and verify metadata
            checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                    weights_only=False)

            # Metadata must be present
            assert "hardware_metadata" in checkpoint, (
                "Checkpoint missing 'hardware_metadata' field"
            )

            metadata = checkpoint["hardware_metadata"]

            # Required fields per Requirement 15.4
            assert "device_type" in metadata, (
                "Metadata missing 'device_type'"
            )
            assert "gpu_model" in metadata, (
                "Metadata missing 'gpu_model'"
            )
            assert "pytorch_version" in metadata, (
                "Metadata missing 'pytorch_version'"
            )

            # Verify values match what was passed
            assert metadata["device_type"] == hw_metadata["device_type"], (
                f"device_type mismatch: got {metadata['device_type']}, "
                f"expected {hw_metadata['device_type']}"
            )
            assert metadata["gpu_model"] == hw_metadata["gpu_model"], (
                f"gpu_model mismatch: got {metadata['gpu_model']}, "
                f"expected {hw_metadata['gpu_model']}"
            )
            assert metadata["pytorch_version"] == hw_metadata["pytorch_version"], (
                f"pytorch_version mismatch: got {metadata['pytorch_version']}, "
                f"expected {hw_metadata['pytorch_version']}"
            )

    @given(scenario=portable_checkpoint_scenario_strategy())
    @settings(max_examples=20, deadline=None)
    def test_checkpoint_version_and_architecture_preserved(self, scenario):
        """
        The portable checkpoint preserves checkpoint_version and model
        architecture information, enabling compatibility validation when
        loading on a different machine.

        **Validates: Requirements 15.2, 15.4**
        """
        seed = scenario["seed"]
        epoch = scenario["epoch"]
        num_train_steps = scenario["num_train_steps"]

        config = ModelConfig(num_features=63, lookback=60)
        model, optimizer = _create_trained_model(seed, num_train_steps, config)

        hardware = _create_hardware_profile_mock(
            device_type="cpu",
            cpu_model="Intel(R) Core(TM) i7-12700",
        )
        metrics = {"train_loss": 0.4, "val_loss": 0.5}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = str(Path(tmpdir) / "checkpoint_version.pt")

            HardwareProfile.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                hardware=hardware,
                path=checkpoint_path,
                model_config=config,
            )

            # Load and verify structure
            checkpoint = HardwareProfile.load_checkpoint(
                path=checkpoint_path,
                target_device=torch.device("cpu"),
                expected_config=config,
            )

            # checkpoint_version must be present
            assert "checkpoint_version" in checkpoint, (
                "Checkpoint missing 'checkpoint_version'"
            )
            assert checkpoint["checkpoint_version"] == "1.0", (
                f"Unexpected checkpoint_version: {checkpoint['checkpoint_version']}"
            )

            # model_architecture must be preserved
            assert "model_architecture" in checkpoint, (
                "Checkpoint missing 'model_architecture'"
            )
            arch = checkpoint["model_architecture"]
            assert arch["num_features"] == config.num_features
            assert arch["lookback"] == config.lookback
            assert arch["tcn_channels"] == config.tcn_channels
            assert arch["attention_heads"] == config.attention_heads
            assert arch["attention_dim"] == config.attention_dim

            # Epoch must match
            assert checkpoint["epoch"] == epoch

    @given(scenario=portable_checkpoint_scenario_strategy())
    @settings(max_examples=20, deadline=None)
    def test_optimizer_state_portable_across_devices(self, scenario):
        """
        The optimizer state is saved in CPU format and can be loaded on
        any device, enabling seamless training resumption across machines.

        Verifies that after loading the checkpoint on a different "device",
        the optimizer state can be used to continue training and produces
        the same parameter update as the original optimizer would.

        **Validates: Requirements 15.2, 15.3**
        """
        seed = scenario["seed"]
        epoch = scenario["epoch"]
        num_train_steps = scenario["num_train_steps"]

        config = ModelConfig(num_features=63, lookback=60)

        # Path A: Train, save, load on "device B", do one more step
        model_a, optimizer_a = _create_trained_model(seed, num_train_steps, config)

        hardware = _create_hardware_profile_mock(
            device_type="cpu",
            gpu_model="NVIDIA GeForce RTX 2060",
            vram_mb=6144,
            cpu_model="Intel(R) Core(TM) i5-10400F",
        )
        metrics = {"train_loss": 0.2}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = str(Path(tmpdir) / "opt_state_test.pt")

            HardwareProfile.save_checkpoint(
                model=model_a,
                optimizer=optimizer_a,
                epoch=epoch,
                metrics=metrics,
                hardware=hardware,
                path=checkpoint_path,
                model_config=config,
            )

            # Load on "device B"
            target_device = torch.device("cpu")
            checkpoint = HardwareProfile.load_checkpoint(
                path=checkpoint_path,
                target_device=target_device,
                expected_config=config,
            )

            # Reconstruct model and optimizer on device B
            model_b = StockEvalNet(config)
            model_b.load_state_dict(checkpoint["model_state_dict"])
            model_b.to(target_device)

            optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
            optimizer_b.load_state_dict(checkpoint["optimizer_state_dict"])

            # Do one training step on device B
            criterion = nn.MSELoss()
            torch.manual_seed(seed + 50000)
            x = torch.randn(4, config.lookback, config.num_features)
            target = torch.randn(4, 1)

            model_b.train()
            optimizer_b.zero_grad()
            output_b = model_b(x)
            loss_b = criterion(output_b, target)
            loss_b.backward()
            optimizer_b.step()

            params_b = {k: v.clone().detach() for k, v in model_b.state_dict().items()}

            # Path A continued: do the same step on original model
            torch.manual_seed(seed + 50000)
            x_a = torch.randn(4, config.lookback, config.num_features)
            target_a = torch.randn(4, 1)

            model_a.train()
            optimizer_a.zero_grad()
            output_a = model_a(x_a)
            loss_a = criterion(output_a, target_a)
            loss_a.backward()
            optimizer_a.step()

            params_a = {k: v.clone().detach() for k, v in model_a.state_dict().items()}

            # Compare: parameters should be identical within tolerance
            for key in params_a:
                diff = (params_a[key] - params_b[key]).abs().max().item()
                assert diff < 1e-5, (
                    f"Parameter '{key}' differs by {diff:.2e} between "
                    f"original and resumed-on-device-B paths "
                    f"(exceeds ±1e-5 tolerance)"
                )
