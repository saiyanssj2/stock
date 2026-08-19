"""
Unit tests for HardwareProfile class.

Tests auto-detection, batch size optimization, device selection,
metadata serialization, portable checkpoint save/load,
optimizer state CPU mapping, and checkpoint compatibility validation.
"""

import platform
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import pytest

from engine.config import ModelConfig, ModelError
from engine.hardware_profile import HardwareProfile, _optimizer_state_to_cpu


class TestHardwareProfileAutoDetection:
    """Tests for hardware auto-detection at construction."""

    def test_device_type_detected(self):
        """HardwareProfile detects device type (cuda or cpu)."""
        profile = HardwareProfile()
        assert profile.device_type in ("cuda", "cpu")

    def test_cpu_model_detected(self):
        """HardwareProfile detects CPU model via platform.processor()."""
        profile = HardwareProfile()
        # cpu_model should be a string (may be empty on some systems)
        assert isinstance(profile.cpu_model, str)
        assert profile.cpu_model == platform.processor()

    def test_pytorch_version_detected(self):
        """HardwareProfile records PyTorch version."""
        profile = HardwareProfile()
        assert profile.pytorch_version == torch.__version__

    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_only_profile(self, mock_cuda):
        """When CUDA is unavailable, GPU fields are None."""
        profile = HardwareProfile()
        assert profile.device_type == "cpu"
        assert profile.gpu_model is None
        assert profile.gpu_vram_mb is None

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 2060")
    @patch("torch.cuda.get_device_properties")
    def test_gpu_profile(self, mock_props, mock_name, mock_cuda):
        """When CUDA is available, GPU model and VRAM are detected."""
        # Mock device properties with total_mem attribute
        mock_props.return_value.total_mem = 6 * 1024 * 1024 * 1024  # 6GB
        profile = HardwareProfile()
        assert profile.device_type == "cuda"
        assert profile.gpu_model == "NVIDIA GeForce RTX 2060"
        assert profile.gpu_vram_mb == 6144  # 6GB in MB


class TestGetOptimalBatchSize:
    """Tests for batch size optimization based on hardware."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 2060")
    @patch("torch.cuda.get_device_properties")
    def test_gpu_training_batch_size(self, mock_props, mock_name, mock_cuda):
        """GPU training mode returns batch size 64."""
        mock_props.return_value.total_mem = 6 * 1024 * 1024 * 1024
        profile = HardwareProfile()
        assert profile.get_optimal_batch_size("training") == 64

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 2060")
    @patch("torch.cuda.get_device_properties")
    def test_gpu_inference_batch_size(self, mock_props, mock_name, mock_cuda):
        """GPU inference mode returns batch size 50."""
        mock_props.return_value.total_mem = 6 * 1024 * 1024 * 1024
        profile = HardwareProfile()
        assert profile.get_optimal_batch_size("inference") == 50

    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_training_batch_size(self, mock_cuda):
        """CPU training mode returns batch size 16."""
        profile = HardwareProfile()
        assert profile.get_optimal_batch_size("training") == 16

    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_inference_batch_size(self, mock_cuda):
        """CPU inference mode returns batch size 10."""
        profile = HardwareProfile()
        assert profile.get_optimal_batch_size("inference") == 10


class TestGetDevice:
    """Tests for device selection."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 2060")
    @patch("torch.cuda.get_device_properties")
    def test_gpu_device(self, mock_props, mock_name, mock_cuda):
        """Returns cuda device when GPU is available."""
        mock_props.return_value.total_mem = 6 * 1024 * 1024 * 1024
        profile = HardwareProfile()
        device = profile.get_device()
        assert device == torch.device("cuda")

    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_device(self, mock_cuda):
        """Returns cpu device when GPU is not available."""
        profile = HardwareProfile()
        device = profile.get_device()
        assert device == torch.device("cpu")


class TestToMetadata:
    """Tests for metadata serialization."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_metadata_keys(self, mock_cuda):
        """Metadata dict contains all required keys."""
        profile = HardwareProfile()
        metadata = profile.to_metadata()
        assert "device_type" in metadata
        assert "gpu_model" in metadata
        assert "cpu_model" in metadata
        assert "pytorch_version" in metadata
        assert "vram_mb" in metadata

    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_metadata_values(self, mock_cuda):
        """CPU profile metadata has correct values."""
        profile = HardwareProfile()
        metadata = profile.to_metadata()
        assert metadata["device_type"] == "cpu"
        assert metadata["gpu_model"] is None
        assert metadata["cpu_model"] == platform.processor()
        assert metadata["pytorch_version"] == torch.__version__
        assert metadata["vram_mb"] is None

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 2060")
    @patch("torch.cuda.get_device_properties")
    def test_gpu_metadata_values(self, mock_props, mock_name, mock_cuda):
        """GPU profile metadata has correct values."""
        mock_props.return_value.total_mem = 6 * 1024 * 1024 * 1024
        profile = HardwareProfile()
        metadata = profile.to_metadata()
        assert metadata["device_type"] == "cuda"
        assert metadata["gpu_model"] == "NVIDIA GeForce RTX 2060"
        assert metadata["cpu_model"] == platform.processor()
        assert metadata["pytorch_version"] == torch.__version__
        assert metadata["vram_mb"] == 6144

    @patch("torch.cuda.is_available", return_value=False)
    def test_metadata_is_json_serializable(self, mock_cuda):
        """Metadata dict is JSON-serializable."""
        import json

        profile = HardwareProfile()
        metadata = profile.to_metadata()
        # Should not raise
        json_str = json.dumps(metadata)
        assert isinstance(json_str, str)


class TestRepr:
    """Tests for string representation."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_repr(self, mock_cuda):
        """CPU profile repr includes device type and cpu model."""
        profile = HardwareProfile()
        repr_str = repr(profile)
        assert "cpu" in repr_str
        assert "pytorch" in repr_str

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 2060")
    @patch("torch.cuda.get_device_properties")
    def test_gpu_repr(self, mock_props, mock_name, mock_cuda):
        """GPU profile repr includes GPU model and VRAM."""
        mock_props.return_value.total_mem = 6 * 1024 * 1024 * 1024
        profile = HardwareProfile()
        repr_str = repr(profile)
        assert "cuda" in repr_str
        assert "RTX 2060" in repr_str
        assert "6144MB" in repr_str



# =============================================================================
# Helper: Simple model for testing save/load
# =============================================================================


class _SimpleModel(nn.Module):
    """Minimal model for checkpoint tests."""

    def __init__(self, in_features: int = 61, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# =============================================================================
# Tests: _optimizer_state_to_cpu
# =============================================================================


class TestOptimizerStateToCpu:
    """Tests for the _optimizer_state_to_cpu helper function."""

    def test_maps_tensors_to_cpu(self):
        """All tensors in optimizer state are mapped to CPU."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # Do a forward/backward pass to populate optimizer state
        x = torch.randn(4, 61)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        state_dict = optimizer.state_dict()
        cpu_state = _optimizer_state_to_cpu(state_dict)

        # All tensors in 'state' should be on CPU
        for param_id, param_state in cpu_state["state"].items():
            for key, val in param_state.items():
                if isinstance(val, torch.Tensor):
                    assert val.device == torch.device("cpu"), (
                        f"Tensor state[{param_id}][{key}] not on CPU"
                    )

    def test_preserves_non_tensor_values(self):
        """Non-tensor values (step count, etc.) are preserved unchanged."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        x = torch.randn(4, 61)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        state_dict = optimizer.state_dict()
        cpu_state = _optimizer_state_to_cpu(state_dict)

        # step count should be preserved
        for param_id in cpu_state["state"]:
            if "step" in cpu_state["state"][param_id]:
                original_step = state_dict["state"][param_id]["step"]
                cpu_step = cpu_state["state"][param_id]["step"]
                if isinstance(original_step, torch.Tensor):
                    assert cpu_step.item() == original_step.item()
                else:
                    assert cpu_step == original_step

    def test_preserves_param_groups(self):
        """param_groups are copied as-is."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        x = torch.randn(4, 61)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        state_dict = optimizer.state_dict()
        cpu_state = _optimizer_state_to_cpu(state_dict)

        assert cpu_state["param_groups"] == state_dict["param_groups"]

    def test_empty_optimizer_state(self):
        """Handles optimizer with no accumulated state (no step yet)."""
        model = _SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # No step performed, state is empty
        state_dict = optimizer.state_dict()
        cpu_state = _optimizer_state_to_cpu(state_dict)

        assert "state" in cpu_state
        assert "param_groups" in cpu_state
        assert cpu_state["state"] == {}


# =============================================================================
# Tests: HardwareProfile.save_checkpoint
# =============================================================================


class TestSaveCheckpoint:
    """Tests for HardwareProfile.save_checkpoint()."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_saves_checkpoint_file(self, mock_cuda):
        """Saves a checkpoint file to the specified path."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=5,
                metrics={"loss": 0.5, "val_loss": 0.6},
                hardware=hardware,
                path=path,
            )
            assert Path(path).exists()

    @patch("torch.cuda.is_available", return_value=False)
    def test_checkpoint_contains_required_keys(self, mock_cuda):
        """Checkpoint contains all required keys."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=3,
                metrics={"loss": 0.1},
                hardware=hardware,
                path=path,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "epoch" in checkpoint
            assert "metrics" in checkpoint
            assert "hardware_metadata" in checkpoint
            assert "checkpoint_version" in checkpoint

    @patch("torch.cuda.is_available", return_value=False)
    def test_checkpoint_version_is_1_0(self, mock_cuda):
        """checkpoint_version is set to '1.0'."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=1,
                metrics={},
                hardware=hardware,
                path=path,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            assert checkpoint["checkpoint_version"] == "1.0"

    @patch("torch.cuda.is_available", return_value=False)
    def test_model_state_dict_on_cpu(self, mock_cuda):
        """All model tensors in checkpoint are on CPU."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=1,
                metrics={},
                hardware=hardware,
                path=path,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            for key, tensor in checkpoint["model_state_dict"].items():
                assert tensor.device == torch.device("cpu"), (
                    f"model_state_dict['{key}'] not on CPU"
                )

    @patch("torch.cuda.is_available", return_value=False)
    def test_hardware_metadata_included(self, mock_cuda):
        """hardware_metadata is correctly included from HardwareProfile."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=1,
                metrics={},
                hardware=hardware,
                path=path,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            metadata = checkpoint["hardware_metadata"]
            assert metadata["device_type"] == "cpu"
            assert metadata["pytorch_version"] == torch.__version__
            assert "cpu_model" in metadata
            assert "gpu_model" in metadata
            assert "vram_mb" in metadata

    @patch("torch.cuda.is_available", return_value=False)
    def test_model_architecture_included_when_config_provided(self, mock_cuda):
        """model_architecture is stored when ModelConfig is provided."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()
        config = ModelConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=1,
                metrics={},
                hardware=hardware,
                path=path,
                model_config=config,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            assert "model_architecture" in checkpoint
            arch = checkpoint["model_architecture"]
            assert arch["num_features"] == 61
            assert arch["tcn_channels"] == [128, 128, 64]
            assert arch["attention_heads"] == 4
            assert arch["attention_dim"] == 64

    @patch("torch.cuda.is_available", return_value=False)
    def test_creates_parent_directories(self, mock_cuda):
        """Creates parent directories if they don't exist."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "nested" / "dir" / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=1,
                metrics={},
                hardware=hardware,
                path=path,
            )
            assert Path(path).exists()

    @patch("torch.cuda.is_available", return_value=False)
    def test_epoch_and_metrics_preserved(self, mock_cuda):
        """Epoch number and metrics are preserved in checkpoint."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()
        metrics = {"train_loss": 0.123, "val_loss": 0.456, "mae": 0.05}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=42,
                metrics=metrics,
                hardware=hardware,
                path=path,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            assert checkpoint["epoch"] == 42
            assert checkpoint["metrics"] == metrics


# =============================================================================
# Tests: HardwareProfile.load_checkpoint
# =============================================================================


class TestLoadCheckpoint:
    """Tests for HardwareProfile.load_checkpoint()."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_loads_checkpoint_onto_cpu(self, mock_cuda):
        """Loads checkpoint and maps model_state_dict to CPU device."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model, optimizer=optimizer, epoch=1,
                metrics={}, hardware=hardware, path=path,
            )

            loaded = HardwareProfile.load_checkpoint(path, torch.device("cpu"))
            for key, tensor in loaded["model_state_dict"].items():
                assert tensor.device == torch.device("cpu")

    @patch("torch.cuda.is_available", return_value=False)
    def test_model_weights_match_after_load(self, mock_cuda):
        """Model weights loaded from checkpoint match original weights."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()

        # Save original weights
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model, optimizer=optimizer, epoch=1,
                metrics={}, hardware=hardware, path=path,
            )

            loaded = HardwareProfile.load_checkpoint(path, torch.device("cpu"))

            for key in original_state:
                assert torch.allclose(
                    original_state[key], loaded["model_state_dict"][key], atol=1e-7
                ), f"Weight mismatch for key '{key}'"

    @patch("torch.cuda.is_available", return_value=False)
    def test_raises_on_missing_file(self, mock_cuda):
        """Raises ModelError when checkpoint file doesn't exist."""
        with pytest.raises(ModelError) as exc_info:
            HardwareProfile.load_checkpoint(
                "/nonexistent/path/checkpoint.pt", torch.device("cpu")
            )
        assert exc_info.value.error_code == "CHECKPOINT_NOT_FOUND"

    @patch("torch.cuda.is_available", return_value=False)
    def test_raises_on_corrupted_file(self, mock_cuda):
        """Raises ModelError when checkpoint file is corrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "corrupted.pt")
            Path(path).write_text("this is not a valid checkpoint")

            with pytest.raises(ModelError) as exc_info:
                HardwareProfile.load_checkpoint(path, torch.device("cpu"))
            assert exc_info.value.error_code == "CHECKPOINT_CORRUPTED"

    @patch("torch.cuda.is_available", return_value=False)
    def test_raises_on_missing_required_keys(self, mock_cuda):
        """Raises ModelError when checkpoint is missing required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "incomplete.pt")
            # Save a dict without required keys
            torch.save({"some_key": "some_value"}, path)

            with pytest.raises(ModelError) as exc_info:
                HardwareProfile.load_checkpoint(path, torch.device("cpu"))
            assert exc_info.value.error_code == "CHECKPOINT_CORRUPTED"

    @patch("torch.cuda.is_available", return_value=False)
    def test_returns_all_checkpoint_fields(self, mock_cuda):
        """Loaded checkpoint contains all saved fields."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model, optimizer=optimizer, epoch=7,
                metrics={"loss": 0.3}, hardware=hardware, path=path,
                model_config=ModelConfig(),
            )

            loaded = HardwareProfile.load_checkpoint(path, torch.device("cpu"))
            assert loaded["epoch"] == 7
            assert loaded["metrics"] == {"loss": 0.3}
            assert loaded["checkpoint_version"] == "1.0"
            assert "hardware_metadata" in loaded
            assert "model_architecture" in loaded
            assert "optimizer_state_dict" in loaded

    @patch("torch.cuda.is_available", return_value=False)
    def test_save_load_roundtrip_inference_consistency(self, mock_cuda):
        """Model produces same output after save/load roundtrip."""
        model = _SimpleModel()
        model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()

        # Generate test input and get original output
        test_input = torch.randn(2, 61)
        with torch.no_grad():
            original_output = model(test_input).clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model, optimizer=optimizer, epoch=1,
                metrics={}, hardware=hardware, path=path,
            )

            loaded = HardwareProfile.load_checkpoint(path, torch.device("cpu"))

            # Create new model and load state
            new_model = _SimpleModel()
            new_model.load_state_dict(loaded["model_state_dict"])
            new_model.eval()

            with torch.no_grad():
                loaded_output = new_model(test_input)

            assert torch.allclose(original_output, loaded_output, atol=1e-6)


# =============================================================================
# Tests: Checkpoint Compatibility Validation
# =============================================================================


class TestCheckpointCompatibilityValidation:
    """Tests for checkpoint compatibility validation on load."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_passes_with_matching_config(self, mock_cuda):
        """No error when checkpoint architecture matches expected config."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()
        config = ModelConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model, optimizer=optimizer, epoch=1,
                metrics={}, hardware=hardware, path=path,
                model_config=config,
            )

            # Should not raise
            loaded = HardwareProfile.load_checkpoint(
                path, torch.device("cpu"), expected_config=config
            )
            assert "model_state_dict" in loaded

    @patch("torch.cuda.is_available", return_value=False)
    def test_raises_on_num_features_mismatch(self, mock_cuda):
        """Raises ModelError when feature vector dimension doesn't match."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()
        save_config = ModelConfig(num_features=61)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model, optimizer=optimizer, epoch=1,
                metrics={}, hardware=hardware, path=path,
                model_config=save_config,
            )

            # Try to load with different num_features
            different_config = ModelConfig(num_features=100)
            with pytest.raises(ModelError) as exc_info:
                HardwareProfile.load_checkpoint(
                    path, torch.device("cpu"), expected_config=different_config
                )
            assert exc_info.value.error_code == "CHECKPOINT_INCOMPATIBLE"
            assert "feature vector dimension" in str(exc_info.value.message)

    @patch("torch.cuda.is_available", return_value=False)
    def test_raises_on_tcn_channels_mismatch(self, mock_cuda):
        """Raises ModelError when TCN architecture doesn't match."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()
        save_config = ModelConfig(tcn_channels=[128, 128, 64])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model, optimizer=optimizer, epoch=1,
                metrics={}, hardware=hardware, path=path,
                model_config=save_config,
            )

            different_config = ModelConfig(tcn_channels=[256, 256, 128])
            with pytest.raises(ModelError) as exc_info:
                HardwareProfile.load_checkpoint(
                    path, torch.device("cpu"), expected_config=different_config
                )
            assert exc_info.value.error_code == "CHECKPOINT_INCOMPATIBLE"
            assert "tcn_channels" in str(exc_info.value.message)

    @patch("torch.cuda.is_available", return_value=False)
    def test_raises_on_attention_heads_mismatch(self, mock_cuda):
        """Raises ModelError when attention heads count doesn't match."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()
        save_config = ModelConfig(attention_heads=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model, optimizer=optimizer, epoch=1,
                metrics={}, hardware=hardware, path=path,
                model_config=save_config,
            )

            different_config = ModelConfig(attention_heads=8)
            with pytest.raises(ModelError) as exc_info:
                HardwareProfile.load_checkpoint(
                    path, torch.device("cpu"), expected_config=different_config
                )
            assert exc_info.value.error_code == "CHECKPOINT_INCOMPATIBLE"

    @patch("torch.cuda.is_available", return_value=False)
    def test_raises_on_attention_dim_mismatch(self, mock_cuda):
        """Raises ModelError when attention dimension doesn't match."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()
        save_config = ModelConfig(attention_dim=64, tcn_channels=[128, 128, 64])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model, optimizer=optimizer, epoch=1,
                metrics={}, hardware=hardware, path=path,
                model_config=save_config,
            )

            different_config = ModelConfig(attention_dim=128, tcn_channels=[128, 128, 128])
            with pytest.raises(ModelError) as exc_info:
                HardwareProfile.load_checkpoint(
                    path, torch.device("cpu"), expected_config=different_config
                )
            assert exc_info.value.error_code == "CHECKPOINT_INCOMPATIBLE"

    @patch("torch.cuda.is_available", return_value=False)
    def test_no_validation_without_expected_config(self, mock_cuda):
        """No validation error when expected_config is not provided."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()
        # Save with one config
        save_config = ModelConfig(num_features=61)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            HardwareProfile.save_checkpoint(
                model=model, optimizer=optimizer, epoch=1,
                metrics={}, hardware=hardware, path=path,
                model_config=save_config,
            )

            # Load without expected_config - should not raise
            loaded = HardwareProfile.load_checkpoint(
                path, torch.device("cpu"), expected_config=None
            )
            assert "model_state_dict" in loaded

    @patch("torch.cuda.is_available", return_value=False)
    def test_no_validation_when_checkpoint_lacks_architecture(self, mock_cuda):
        """No validation error when checkpoint has no model_architecture."""
        model = _SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        hardware = HardwareProfile()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint.pt")
            # Save WITHOUT model_config (no model_architecture in checkpoint)
            HardwareProfile.save_checkpoint(
                model=model, optimizer=optimizer, epoch=1,
                metrics={}, hardware=hardware, path=path,
                model_config=None,
            )

            # Load with expected_config - should not raise since checkpoint
            # has no architecture info to validate against
            loaded = HardwareProfile.load_checkpoint(
                path, torch.device("cpu"),
                expected_config=ModelConfig(num_features=999),
            )
            assert "model_state_dict" in loaded
