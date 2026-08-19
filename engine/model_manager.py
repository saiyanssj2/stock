"""
ModelManager - Production Model Lifecycle Management.

Handles model loading with checksum validation, hot-swap (atomic model
replacement), version tracking, GPU/CPU device management, and high-level
predict() API with OOM retry.
"""

import hashlib
import logging
import math
import threading
from datetime import datetime
from typing import Optional

import numpy as np
import torch

from engine.config import ModelConfig, ModelError, ResourceError
from engine.evaluation_model import StockEvalNet

logger = logging.getLogger(__name__)


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
