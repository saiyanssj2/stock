"""
Evaluation model: TCN + Attention neural network for market state scoring.

Implements StockEvalNet, a Temporal Convolutional Network with multi-head
self-attention that evaluates market states and outputs a Position_Score
in [-1.0, +1.0].

Also implements ModelManager for production model lifecycle: loading with
checksum validation, hot-swap (atomic model replacement), version tracking,
GPU/CPU device management, and high-level predict() API with OOM retry.

Architecture:
    Input (batch, lookback=60, num_features=63)
    → Transpose to (batch, 63, 60)
    → TCN Block 1: 63→128, kernel=3, dilation=1
    → TCN Block 2: 128→128, kernel=3, dilation=2
    → TCN Block 3: 128→64, kernel=3, dilation=4
    → Transpose to (batch, 60, 64)
    → MultiheadAttention (4 heads, embed_dim=64) + residual + LayerNorm
    → Transpose to (batch, 64, 60)
    → AdaptiveAvgPool1d → (batch, 64)
    → Linear(64, 1) → Tanh → output in [-1, 1]

Target parameter count: ~180K (within ±20K).
"""

import hashlib
import logging
import math
import threading
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from engine.config import ModelConfig, ModelError, ResourceError

logger = logging.getLogger(__name__)


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Block with dilated causal convolutions and residual connection.

    Structure:
        Conv1d (causal, dilated) → BatchNorm → GELU → Dropout
        → Conv1d (causal, dilated) → BatchNorm → Residual add
        → GELU

    Uses causal padding to prevent information leakage from future timesteps.
    If in_channels != out_channels, a 1x1 convolution adapts the residual path.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Causal padding: (kernel_size - 1) * dilation on the left side only
        self.causal_padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=self.causal_padding
        )
        self.norm1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, dilation=dilation, padding=self.causal_padding
        )
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection: 1x1 conv if channel dimensions differ, else identity
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)

        Returns:
            (batch, out_channels, seq_len)
        """
        seq_len = x.size(-1)
        res = self.residual(x)

        # First conv block: Conv1d → trim causal → BatchNorm → GELU → Dropout
        out = self.conv1(x)
        out = out[..., :seq_len]  # Trim causal padding to maintain seq_len
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Second conv block: Conv1d → trim causal → BatchNorm
        out = self.conv2(out)
        out = out[..., :seq_len]  # Trim causal padding
        out = self.norm2(out)

        # Residual add + final activation
        out = out + res
        out = self.activation(out)

        return out


class StockEvalNet(nn.Module):
    """
    TCN + Attention model for market state evaluation.

    Evaluates a market state represented as a feature tensor and outputs
    a Position_Score in [-1.0, +1.0].

    Parameters: ~180K (well within 4GB VRAM constraint)
    Inference: ~5ms per sample on RTX 2060
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = ModelConfig()

        self.config = config
        num_features = config.num_features
        tcn_channels = config.tcn_channels
        dilations = config.dilations
        kernel_size = config.kernel_size
        attention_heads = config.attention_heads
        attention_dim = config.attention_dim
        dropout = config.dropout

        # Validate: last TCN output channel must match attention embed_dim
        if tcn_channels[-1] != attention_dim:
            raise ValueError(
                f"Last TCN channel ({tcn_channels[-1]}) must equal "
                f"attention_dim ({attention_dim})"
            )

        # Build TCN backbone
        # Input channels: num_features (63)
        # Channel progression: 63 → 128 → 128 → 64
        tcn_layers = []
        in_ch = num_features
        for out_ch, dil in zip(tcn_channels, dilations):
            tcn_layers.append(
                TCNBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=dil, dropout=dropout)
            )
            in_ch = out_ch
        self.tcn = nn.Sequential(*tcn_layers)

        # Multi-head self-attention
        # embed_dim = last TCN channel = attention_dim = 64
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim, num_heads=attention_heads, batch_first=True, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(attention_dim)

        # Output head: AdaptiveAvgPool1d → Flatten → Linear → Tanh
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(attention_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, lookback, num_features)

        Returns:
            Position scores of shape (batch, 1) with values in [-1.0, +1.0]
        """
        # Transpose for Conv1d: (batch, num_features, lookback)
        x = x.transpose(1, 2)

        # TCN backbone: (batch, 64, lookback)
        x = self.tcn(x)

        # Transpose for attention: (batch, lookback, 64)
        x = x.transpose(1, 2)

        # Multi-head self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)

        # Transpose for pooling: (batch, 64, lookback)
        x = x.transpose(1, 2)

        # Global average pooling: (batch, 64, 1) → (batch, 64)
        x = self.pool(x)
        x = x.squeeze(-1)

        # Linear head with Tanh: (batch, 1)
        x = self.head(x)

        return x

    def count_parameters(self) -> int:
        """Count total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# ModelManager - Production Model Lifecycle
# ==============================================================================


def _compute_state_dict_checksum(state_dict: dict) -> str:
    """Compute SHA256 checksum of model state_dict weights.

    Deterministically hashes all tensor parameters sorted by key name.
    """
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        if isinstance(state_dict[key], torch.Tensor):
            hasher.update(key.encode("utf-8"))
            hasher.update(state_dict[key].cpu().numpy().tobytes())
    return hasher.hexdigest()


class ModelManager:
    """Manages model loading, validation, hot-swap, and inference lifecycle.

    Provides:
    - load(): Load model from file with SHA256 checksum validation
    - hot_swap(): Atomically replace active model (thread-safe)
    - predict(): High-level inference API with batching, device placement,
      OOM retry (halve batch 3x then CPU fallback), and memory cleanup
    - Device management: prefer CUDA, fallback to CPU, allow manual override
    - Version tracking: increments on each load/hot_swap

    Thread Safety:
        All model access is protected by a threading.Lock to support
        concurrent inference and hot-swap operations.
    """

    def __init__(self, config: Optional[ModelConfig] = None, device: Optional[str] = None):
        """Initialize ModelManager.

        Args:
            config: Model architecture configuration. Uses default if None.
            device: Target device ('cuda', 'cpu', or None for auto-detect).
        """
        self._config = config or ModelConfig()
        self._model: Optional[StockEvalNet] = None
        self._lock = threading.Lock()
        self._version: int = 0
        self._last_loaded: Optional[datetime] = None
        self._checksum: Optional[str] = None
        self._model_path: Optional[str] = None

        # Device management
        if device is not None:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def version(self) -> int:
        """Current model version (increments on each load/hot_swap)."""
        return self._version

    @property
    def last_loaded(self) -> Optional[datetime]:
        """Timestamp of last model load or hot_swap."""
        return self._last_loaded

    @property
    def checksum(self) -> Optional[str]:
        """SHA256 checksum of currently loaded model weights."""
        return self._checksum

    @property
    def device(self) -> torch.device:
        """Current device for inference."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Whether a model is currently loaded and ready for inference."""
        return self._model is not None

    @property
    def model(self) -> Optional[StockEvalNet]:
        """Direct access to the underlying model (use with caution)."""
        return self._model

    # --------------------------------------------------------------------------
    # Load & Validation
    # --------------------------------------------------------------------------

    def load(self, path: str, expected_checksum: Optional[str] = None) -> None:
        """Load model from file with checksum validation.

        The saved file is expected to be a dict with keys:
        - 'state_dict': model weights
        - 'model_config': ModelConfig fields dict
        - 'checksum': SHA256 of state_dict at save time
        - 'version' (optional): saved version number

        Args:
            path: Path to saved model .pt file.
            expected_checksum: If provided, validate against this checksum.
                If None, validates against checksum stored in the file.

        Raises:
            ModelError: If file cannot be loaded, checksum fails, or
                architecture is incompatible.
        """
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            raise ModelError(
                f"Model file not found: {path}",
                error_code="MODEL_NOT_FOUND",
                details={"path": path},
            )
        except Exception as e:
            raise ModelError(
                f"Failed to load model file: {e}",
                error_code="MODEL_LOAD_FAILED",
                details={"path": path, "error": str(e)},
            )

        # Extract components
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            saved_checksum = checkpoint.get("checksum")
            saved_config = checkpoint.get("model_config")
        else:
            # Legacy format: raw state_dict
            state_dict = checkpoint
            saved_checksum = None
            saved_config = None

        # Validate checksum
        computed_checksum = _compute_state_dict_checksum(state_dict)
        validate_against = expected_checksum or saved_checksum

        if validate_against is not None and computed_checksum != validate_against:
            raise ModelError(
                "Model checksum validation failed - file may be corrupted",
                error_code="MODEL_CHECKSUM_FAILED",
                details={
                    "path": path,
                    "expected": validate_against,
                    "computed": computed_checksum,
                },
            )

        # Validate architecture compatibility if config saved
        if saved_config is not None:
            self._validate_config_compatibility(saved_config)

        # Create model and load weights
        new_model = StockEvalNet(self._config)
        try:
            new_model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise ModelError(
                f"Model architecture incompatible with saved weights: {e}",
                error_code="MODEL_INCOMPATIBLE",
                details={"path": path, "error": str(e)},
            )

        new_model.to(self._device)
        new_model.eval()

        # Atomic swap under lock
        with self._lock:
            self._model = new_model
            self._version += 1
            self._last_loaded = datetime.now()
            self._checksum = computed_checksum
            self._model_path = path

        logger.info(
            f"Model loaded: version={self._version}, "
            f"checksum={computed_checksum[:12]}..., device={self._device}"
        )

    def _validate_config_compatibility(self, saved_config: dict) -> None:
        """Check that saved model config is compatible with current config.

        Raises:
            ModelError: If dimensions are incompatible.
        """
        current = self._config
        num_features = saved_config.get("num_features", current.num_features)
        attention_dim = saved_config.get("attention_dim", current.attention_dim)

        if num_features != current.num_features:
            raise ModelError(
                f"Feature dimension mismatch: saved={num_features}, "
                f"current={current.num_features}",
                error_code="MODEL_INCOMPATIBLE",
                details={
                    "saved_num_features": num_features,
                    "current_num_features": current.num_features,
                },
            )

        if attention_dim != current.attention_dim:
            raise ModelError(
                f"Attention dimension mismatch: saved={attention_dim}, "
                f"current={current.attention_dim}",
                error_code="MODEL_INCOMPATIBLE",
                details={
                    "saved_attention_dim": attention_dim,
                    "current_attention_dim": current.attention_dim,
                },
            )

    # --------------------------------------------------------------------------
    # Hot-Swap
    # --------------------------------------------------------------------------

    def hot_swap(self, new_model_path: str, expected_checksum: Optional[str] = None) -> None:
        """Atomically replace the active model with a new one.

        Loads the new model, validates it, then atomically swaps under lock.
        If the new model fails to load, the old model remains active.

        Args:
            new_model_path: Path to the new model .pt file.
            expected_checksum: Optional checksum for validation.

        Raises:
            ModelError: If new model fails to load or validate.
        """
        try:
            checkpoint = torch.load(new_model_path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            raise ModelError(
                f"Hot-swap model file not found: {new_model_path}",
                error_code="MODEL_NOT_FOUND",
                details={"path": new_model_path},
            )
        except Exception as e:
            raise ModelError(
                f"Failed to load hot-swap model: {e}",
                error_code="MODEL_LOAD_FAILED",
                details={"path": new_model_path, "error": str(e)},
            )

        # Extract state dict
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            saved_checksum = checkpoint.get("checksum")
            saved_config = checkpoint.get("model_config")
        else:
            state_dict = checkpoint
            saved_checksum = None
            saved_config = None

        # Validate checksum
        computed_checksum = _compute_state_dict_checksum(state_dict)
        validate_against = expected_checksum or saved_checksum

        if validate_against is not None and computed_checksum != validate_against:
            raise ModelError(
                "Hot-swap model checksum validation failed",
                error_code="MODEL_CHECKSUM_FAILED",
                details={
                    "path": new_model_path,
                    "expected": validate_against,
                    "computed": computed_checksum,
                },
            )

        # Validate config compatibility
        if saved_config is not None:
            self._validate_config_compatibility(saved_config)

        # Build new model
        new_model = StockEvalNet(self._config)
        try:
            new_model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise ModelError(
                f"Hot-swap model incompatible: {e}",
                error_code="MODEL_INCOMPATIBLE",
                details={"path": new_model_path, "error": str(e)},
            )

        new_model.to(self._device)
        new_model.eval()

        # Atomic swap under lock
        with self._lock:
            old_model = self._model
            self._model = new_model
            self._version += 1
            self._last_loaded = datetime.now()
            self._checksum = computed_checksum
            self._model_path = new_model_path

        # Cleanup old model (release GPU memory)
        if old_model is not None:
            del old_model
            if self._device.type == "cuda":
                torch.cuda.empty_cache()

        logger.info(
            f"Hot-swap complete: version={self._version}, "
            f"checksum={computed_checksum[:12]}..., device={self._device}"
        )

    # --------------------------------------------------------------------------
    # Save (utility for training pipeline)
    # --------------------------------------------------------------------------

    def save(self, path: str) -> str:
        """Save current model state with checksum metadata.

        Args:
            path: Destination file path.

        Returns:
            SHA256 checksum of the saved state_dict.

        Raises:
            ModelError: If no model is loaded.
        """
        with self._lock:
            if self._model is None:
                raise ModelError(
                    "No model loaded to save",
                    error_code="MODEL_NOT_LOADED",
                )
            state_dict = self._model.state_dict()

        checksum = _compute_state_dict_checksum(state_dict)

        checkpoint = {
            "state_dict": state_dict,
            "checksum": checksum,
            "model_config": {
                "num_features": self._config.num_features,
                "lookback": self._config.lookback,
                "tcn_channels": self._config.tcn_channels,
                "kernel_size": self._config.kernel_size,
                "dilations": self._config.dilations,
                "attention_heads": self._config.attention_heads,
                "attention_dim": self._config.attention_dim,
                "dropout": self._config.dropout,
            },
            "version": self._version,
            "saved_at": datetime.now().isoformat(),
        }

        torch.save(checkpoint, path)
        logger.info(f"Model saved: path={path}, checksum={checksum[:12]}...")
        return checksum

    # --------------------------------------------------------------------------
    # Predict (High-Level Inference API)
    # --------------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> np.ndarray:
        """High-level inference API with batching, device placement, and OOM retry.

        Handles:
        - Input validation and shape normalization
        - Conversion to tensor and device placement
        - Eval mode and no_grad context
        - Batch size retry on OOM (halve 3x, then CPU fallback)
        - GPU memory cleanup after inference

        Args:
            features: Input array of shape (batch, lookback, num_features) or
                (lookback, num_features) for single sample.

        Returns:
            Scores array of shape (batch, 1) with values in [-1, 1].

        Raises:
            ModelError: If no model is loaded.
            ResourceError: If inference fails after all retries and CPU fallback.
        """
        if self._model is None:
            raise ModelError(
                "No model loaded for inference",
                error_code="MODEL_NOT_LOADED",
            )

        # Normalize input shape: (lookback, features) → (1, lookback, features)
        if features.ndim == 2:
            features = features[np.newaxis, :]
        elif features.ndim != 3:
            raise ModelError(
                f"Expected 2D or 3D input, got shape {features.shape}",
                error_code="MODEL_INPUT_INVALID",
                details={"shape": features.shape},
            )

        # Try inference with OOM retry strategy
        return self._predict_with_retry(features)

    def _predict_with_retry(self, features: np.ndarray) -> np.ndarray:
        """Attempt inference with batch-halving OOM retry and CPU fallback.

        Strategy:
        1. Try full batch on current device
        2. On OOM: halve batch size and retry (up to 3 retries)
        3. If all GPU retries fail: fallback to CPU
        4. If CPU also fails: raise ResourceError

        Args:
            features: Input array of shape (batch, lookback, num_features).

        Returns:
            Scores array of shape (batch, 1).
        """
        batch_size = features.shape[0]
        max_retries = 3

        # Attempt on primary device
        for attempt in range(max_retries + 1):
            current_batch_size = batch_size if attempt == 0 else max(1, batch_size // (2**attempt))
            try:
                return self._run_inference(features, self._device, current_batch_size)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    logger.warning(
                        f"OOM on {self._device} (attempt {attempt + 1}/{max_retries + 1}), "
                        f"batch_size={current_batch_size}"
                    )
                    if self._device.type == "cuda":
                        torch.cuda.empty_cache()
                    if attempt == max_retries:
                        break  # Fall through to CPU fallback
                else:
                    raise

        # CPU fallback (only if primary device was GPU)
        if self._device.type == "cuda":
            logger.warning("All GPU retries exhausted, falling back to CPU")
            try:
                return self._run_inference(features, torch.device("cpu"), batch_size)
            except RuntimeError as e:
                raise ResourceError(
                    f"Inference failed on both GPU and CPU: {e}",
                    error_code="RESOURCE_EXHAUSTED",
                    details={"error": str(e), "batch_size": batch_size},
                )

        # Already on CPU and all retries failed
        raise ResourceError(
            "Inference failed: insufficient resources on CPU",
            error_code="RESOURCE_EXHAUSTED",
            details={"batch_size": batch_size},
        )

    def _run_inference(
        self, features: np.ndarray, device: torch.device, batch_size: int
    ) -> np.ndarray:
        """Run inference in batches on specified device.

        Args:
            features: Full input array (batch, lookback, num_features).
            device: Target device for computation.
            batch_size: Maximum batch size per forward pass.

        Returns:
            Scores array (total_batch, 1).
        """
        total = features.shape[0]
        all_scores = []

        with self._lock:
            model = self._model
            if model is None:
                raise ModelError("Model was unloaded during inference", error_code="MODEL_NOT_LOADED")

        # Move model to target device if different from current
        if device != self._device:
            model = model.to(device)

        model.eval()

        num_batches = math.ceil(total / batch_size)

        try:
            with torch.no_grad():
                for i in range(num_batches):
                    start = i * batch_size
                    end = min(start + batch_size, total)
                    batch = features[start:end]

                    tensor = torch.from_numpy(batch.astype(np.float32)).to(device)
                    scores = model(tensor)
                    all_scores.append(scores.cpu().numpy())

                    # Clean up batch tensors
                    del tensor, scores
        finally:
            # Memory cleanup
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Move model back to primary device if we moved it
            if device != self._device:
                model.to(self._device)

        return np.concatenate(all_scores, axis=0)

    # --------------------------------------------------------------------------
    # Device Management
    # --------------------------------------------------------------------------

    def set_device(self, device: str) -> None:
        """Change the inference device.

        Args:
            device: New device string ('cuda', 'cpu', 'cuda:0', etc.).
        """
        new_device = torch.device(device)
        with self._lock:
            if self._model is not None:
                self._model.to(new_device)
            self._device = new_device
        logger.info(f"Device changed to: {new_device}")

    # --------------------------------------------------------------------------
    # Utility
    # --------------------------------------------------------------------------

    def create_new_model(self) -> None:
        """Create a new randomly initialized model (for training from scratch).

        Increments version and sets the model as current.
        """
        new_model = StockEvalNet(self._config)
        new_model.to(self._device)
        new_model.eval()

        with self._lock:
            self._model = new_model
            self._version += 1
            self._last_loaded = datetime.now()
            self._checksum = None
            self._model_path = None

        logger.info(f"New model created: version={self._version}, device={self._device}")
