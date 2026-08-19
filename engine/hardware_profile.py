"""
Hardware profile auto-detection and configuration optimization.

Detects available hardware (GPU/CPU), provides optimal batch sizes,
device selection, metadata serialization, and portable checkpoint
save/load for cross-device training.

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7
"""

import platform
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from engine.config import ModelConfig, ModelError


def _optimizer_state_to_cpu(optimizer_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Map all tensors in optimizer state_dict to CPU.

    Recursively traverses the optimizer state dict and moves any
    torch.Tensor values to CPU. This ensures portable checkpoints
    can be loaded on any device.

    Args:
        optimizer_state_dict: The optimizer's state_dict() output.

    Returns:
        A new state dict with all tensors on CPU.
    """
    cpu_state = {}
    for key, value in optimizer_state_dict.items():
        if key == "state":
            cpu_state[key] = {}
            for param_id, param_state in value.items():
                cpu_state[key][param_id] = {}
                for state_key, state_val in param_state.items():
                    if isinstance(state_val, torch.Tensor):
                        cpu_state[key][param_id][state_key] = state_val.cpu()
                    else:
                        cpu_state[key][param_id][state_key] = state_val
        elif key == "param_groups":
            # param_groups contain config, not tensors - copy as-is
            cpu_state[key] = value
        else:
            if isinstance(value, torch.Tensor):
                cpu_state[key] = value.cpu()
            else:
                cpu_state[key] = value
    return cpu_state


class HardwareProfile:
    """Auto-detects hardware capabilities and optimizes configuration.

    Detects GPU availability, VRAM, CPU model, and PyTorch version at
    construction time. Provides optimal batch sizes for training/inference
    based on detected hardware, and serializes profile to metadata dict
    for checkpoint portability.
    """

    def __init__(self) -> None:
        self.device_type: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_model: Optional[str] = None
        self.gpu_vram_mb: Optional[int] = None
        self.cpu_model: Optional[str] = None
        self.pytorch_version: str = torch.__version__
        self._detect_hardware()

    def _detect_hardware(self) -> None:
        """Auto-detect hardware at startup.

        - Check torch.cuda.is_available()
        - If GPU: read torch.cuda.get_device_name(), total memory
        - Read CPU info from platform module
        """
        if self.device_type == "cuda":
            self.gpu_model = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            self.gpu_vram_mb = props.total_mem // (1024 * 1024)
        self.cpu_model = platform.processor()

    def get_optimal_batch_size(self, mode: str = "training") -> int:
        """Determine batch size based on hardware profile.

        GPU (e.g., RTX 2060, 6GB):
          - Training: batch_size = 64
          - Inference: batch_size = 50

        CPU-only (e.g., i7-12700):
          - Training: batch_size = 16
          - Inference: batch_size = 10

        Args:
            mode: Either "training" or "inference".

        Returns:
            Optimal batch size for the detected hardware and mode.
        """
        if self.device_type == "cuda":
            return 64 if mode == "training" else 50
        else:
            return 16 if mode == "training" else 10

    def get_device(self) -> torch.device:
        """Return the appropriate torch device based on detected hardware.

        Returns:
            torch.device("cuda") if GPU available, else torch.device("cpu").
        """
        return torch.device(self.device_type)

    def to_metadata(self) -> dict:
        """Serialize hardware profile for checkpoint metadata.

        Returns:
            Dictionary with keys: device_type, gpu_model, cpu_model,
            pytorch_version, vram_mb.
        """
        return {
            "device_type": self.device_type,
            "gpu_model": self.gpu_model,
            "cpu_model": self.cpu_model,
            "pytorch_version": self.pytorch_version,
            "vram_mb": self.gpu_vram_mb,
        }

    def __repr__(self) -> str:
        if self.device_type == "cuda":
            return (
                f"HardwareProfile(device={self.device_type}, "
                f"gpu={self.gpu_model}, vram={self.gpu_vram_mb}MB, "
                f"cpu={self.cpu_model}, pytorch={self.pytorch_version})"
            )
        return (
            f"HardwareProfile(device={self.device_type}, "
            f"cpu={self.cpu_model}, pytorch={self.pytorch_version})"
        )

    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        hardware: "HardwareProfile",
        path: str,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        """Save portable checkpoint. Always maps tensors to CPU for portability.

        Includes hardware metadata for traceability and checkpoint_version
        for future format compatibility. Also stores model architecture info
        (num_features, tcn_channels, etc.) for compatibility validation on load.

        Args:
            model: The trained model (weights will be saved on CPU).
            optimizer: The optimizer (state will be saved on CPU).
            epoch: Current epoch number.
            metrics: Training metrics dict (loss, val_loss, etc.).
            hardware: HardwareProfile of the saving machine.
            path: File path to save the checkpoint.
            model_config: Optional ModelConfig for architecture validation on load.

        Requirements: 15.2, 15.3, 15.4
        """
        checkpoint: Dict[str, Any] = {
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": _optimizer_state_to_cpu(optimizer.state_dict()),
            "epoch": epoch,
            "metrics": metrics,
            "hardware_metadata": hardware.to_metadata(),
            "checkpoint_version": "1.0",
        }

        # Include model architecture info for compatibility validation
        if model_config is not None:
            checkpoint["model_architecture"] = {
                "num_features": model_config.num_features,
                "lookback": model_config.lookback,
                "tcn_channels": model_config.tcn_channels,
                "kernel_size": model_config.kernel_size,
                "dilations": model_config.dilations,
                "attention_heads": model_config.attention_heads,
                "attention_dim": model_config.attention_dim,
            }

        # Ensure parent directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(
        path: str,
        target_device: torch.device,
        expected_config: Optional[ModelConfig] = None,
    ) -> dict:
        """Load portable checkpoint onto target device.

        Maps all tensors from CPU storage to target_device. Validates
        checkpoint compatibility if expected_config is provided.

        Args:
            path: Path to the checkpoint file.
            target_device: Device to map model tensors to (cpu or cuda).
            expected_config: Optional ModelConfig to validate architecture
                compatibility. If provided, raises ModelError on mismatch.

        Returns:
            Dict containing: model_state_dict (on target_device),
            optimizer_state_dict (on CPU), epoch, metrics,
            hardware_metadata, checkpoint_version, and optionally
            model_architecture.

        Raises:
            ModelError: If checkpoint file not found, corrupted, or
                incompatible architecture/feature vector dimension.

        Requirements: 15.3, 15.4, 15.7
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise ModelError(
                f"Checkpoint file not found: {path}",
                error_code="CHECKPOINT_NOT_FOUND",
                details={"path": path},
            )

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise ModelError(
                f"Failed to load checkpoint: {e}",
                error_code="CHECKPOINT_CORRUPTED",
                details={"path": path, "error": str(e)},
            )

        # Validate checkpoint has required keys
        required_keys = ["model_state_dict", "checkpoint_version"]
        for key in required_keys:
            if key not in checkpoint:
                raise ModelError(
                    f"Invalid checkpoint format: missing key '{key}'",
                    error_code="CHECKPOINT_CORRUPTED",
                    details={"path": path, "missing_key": key},
                )

        # Validate compatibility if expected_config is provided
        if expected_config is not None and "model_architecture" in checkpoint:
            _validate_checkpoint_compatibility(
                checkpoint["model_architecture"], expected_config, path
            )

        # Move model state_dict tensors to target device
        checkpoint["model_state_dict"] = {
            k: v.to(target_device)
            for k, v in checkpoint["model_state_dict"].items()
        }

        return checkpoint



def _validate_checkpoint_compatibility(
    saved_arch: Dict[str, Any],
    expected_config: ModelConfig,
    path: str,
) -> None:
    """Validate checkpoint architecture matches expected configuration.

    Checks num_features (feature vector dimension) and model architecture
    parameters to ensure the checkpoint is compatible with the current system.

    Args:
        saved_arch: Architecture dict from the checkpoint.
        expected_config: The ModelConfig the current system expects.
        path: Checkpoint path (for error reporting).

    Raises:
        ModelError: If architecture is incompatible.

    Requirements: 15.7
    """
    # Check feature vector dimension
    if saved_arch.get("num_features") != expected_config.num_features:
        raise ModelError(
            f"Checkpoint feature vector dimension mismatch: "
            f"checkpoint has {saved_arch.get('num_features')}, "
            f"expected {expected_config.num_features}",
            error_code="CHECKPOINT_INCOMPATIBLE",
            details={
                "path": path,
                "checkpoint_num_features": saved_arch.get("num_features"),
                "expected_num_features": expected_config.num_features,
            },
        )

    # Check TCN channels (model architecture)
    if saved_arch.get("tcn_channels") != expected_config.tcn_channels:
        raise ModelError(
            f"Checkpoint model architecture mismatch: "
            f"checkpoint tcn_channels={saved_arch.get('tcn_channels')}, "
            f"expected {expected_config.tcn_channels}",
            error_code="CHECKPOINT_INCOMPATIBLE",
            details={
                "path": path,
                "checkpoint_tcn_channels": saved_arch.get("tcn_channels"),
                "expected_tcn_channels": expected_config.tcn_channels,
            },
        )

    # Check attention configuration
    if saved_arch.get("attention_heads") != expected_config.attention_heads:
        raise ModelError(
            f"Checkpoint attention heads mismatch: "
            f"checkpoint has {saved_arch.get('attention_heads')}, "
            f"expected {expected_config.attention_heads}",
            error_code="CHECKPOINT_INCOMPATIBLE",
            details={
                "path": path,
                "checkpoint_attention_heads": saved_arch.get("attention_heads"),
                "expected_attention_heads": expected_config.attention_heads,
            },
        )

    if saved_arch.get("attention_dim") != expected_config.attention_dim:
        raise ModelError(
            f"Checkpoint attention dim mismatch: "
            f"checkpoint has {saved_arch.get('attention_dim')}, "
            f"expected {expected_config.attention_dim}",
            error_code="CHECKPOINT_INCOMPATIBLE",
            details={
                "path": path,
                "checkpoint_attention_dim": saved_arch.get("attention_dim"),
                "expected_attention_dim": expected_config.attention_dim,
            },
        )
