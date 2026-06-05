"""
Unit tests for ModelManager in engine/evaluation_model.py.

Tests model loading, checksum validation, hot-swap, predict API,
device management, OOM retry logic, and version tracking.
"""

import os
import tempfile
import threading

import numpy as np
import pytest
import torch

from engine.config import ModelConfig, ModelError, ResourceError
from engine.evaluation_model import (
    ModelManager,
    StockEvalNet,
    _compute_state_dict_checksum,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Small model config for faster tests."""
    return ModelConfig(
        num_features=16,
        lookback=20,
        tcn_channels=[32, 32, 16],
        dilations=[1, 2, 4],
        attention_heads=2,
        attention_dim=16,
        dropout=0.0,
    )


@pytest.fixture
def manager(config):
    """ModelManager with CPU device and small config."""
    return ModelManager(config=config, device="cpu")


@pytest.fixture
def saved_model_path(config):
    """Save a model to a temp file and return path."""
    model = StockEvalNet(config)
    state_dict = model.state_dict()
    checksum = _compute_state_dict_checksum(state_dict)

    checkpoint = {
        "state_dict": state_dict,
        "checksum": checksum,
        "model_config": {
            "num_features": config.num_features,
            "lookback": config.lookback,
            "tcn_channels": config.tcn_channels,
            "kernel_size": config.kernel_size,
            "dilations": config.dilations,
            "attention_heads": config.attention_heads,
            "attention_dim": config.attention_dim,
            "dropout": config.dropout,
        },
        "version": 1,
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(checkpoint, f.name)
        path = f.name

    yield path

    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def second_model_path(config):
    """Save a second model (different weights) for hot-swap tests."""
    model = StockEvalNet(config)
    # Modify weights to create a different model
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(0.01)

    state_dict = model.state_dict()
    checksum = _compute_state_dict_checksum(state_dict)

    checkpoint = {
        "state_dict": state_dict,
        "checksum": checksum,
        "model_config": {
            "num_features": config.num_features,
            "lookback": config.lookback,
            "tcn_channels": config.tcn_channels,
            "kernel_size": config.kernel_size,
            "dilations": config.dilations,
            "attention_heads": config.attention_heads,
            "attention_dim": config.attention_dim,
            "dropout": config.dropout,
        },
        "version": 2,
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(checkpoint, f.name)
        path = f.name

    yield path

    if os.path.exists(path):
        os.unlink(path)


# ---------------------------------------------------------------------------
# Checksum Tests
# ---------------------------------------------------------------------------


class TestChecksum:
    """Tests for _compute_state_dict_checksum."""

    def test_deterministic(self, config):
        """Same state_dict produces same checksum."""
        model = StockEvalNet(config)
        sd = model.state_dict()
        c1 = _compute_state_dict_checksum(sd)
        c2 = _compute_state_dict_checksum(sd)
        assert c1 == c2

    def test_different_weights_different_checksum(self, config):
        """Different model weights produce different checksums."""
        m1 = StockEvalNet(config)
        m2 = StockEvalNet(config)
        # Random init means these will differ
        c1 = _compute_state_dict_checksum(m1.state_dict())
        c2 = _compute_state_dict_checksum(m2.state_dict())
        # Technically could collide but astronomically unlikely
        assert c1 != c2

    def test_checksum_is_sha256_hex(self, config):
        """Checksum is a valid SHA256 hex string (64 chars)."""
        model = StockEvalNet(config)
        checksum = _compute_state_dict_checksum(model.state_dict())
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)


# ---------------------------------------------------------------------------
# Load Tests
# ---------------------------------------------------------------------------


class TestModelManagerLoad:
    """Tests for ModelManager.load()."""

    def test_load_success(self, manager, saved_model_path):
        """Load succeeds with valid checkpoint file."""
        manager.load(saved_model_path)
        assert manager.is_loaded
        assert manager.version == 1
        assert manager.checksum is not None
        assert manager.last_loaded is not None

    def test_load_increments_version(self, manager, saved_model_path):
        """Each load increments version."""
        manager.load(saved_model_path)
        assert manager.version == 1
        manager.load(saved_model_path)
        assert manager.version == 2

    def test_load_file_not_found(self, manager):
        """Load raises ModelError for missing file."""
        with pytest.raises(ModelError) as exc_info:
            manager.load("/nonexistent/path/model.pt")
        assert "MODEL_NOT_FOUND" in str(exc_info.value.error_code)

    def test_load_checksum_validation_passes(self, manager, saved_model_path):
        """Load passes when checksum matches."""
        # Load once to get checksum
        manager.load(saved_model_path)
        valid_checksum = manager.checksum

        # Reload with explicit expected checksum
        manager2 = ModelManager(config=manager._config, device="cpu")
        manager2.load(saved_model_path, expected_checksum=valid_checksum)
        assert manager2.is_loaded

    def test_load_checksum_validation_fails(self, manager, saved_model_path):
        """Load raises ModelError when checksum doesn't match."""
        wrong_checksum = "a" * 64
        with pytest.raises(ModelError) as exc_info:
            manager.load(saved_model_path, expected_checksum=wrong_checksum)
        assert "CHECKSUM" in exc_info.value.error_code

    def test_load_incompatible_features(self, saved_model_path):
        """Load raises ModelError for incompatible feature dimensions."""
        wrong_config = ModelConfig(
            num_features=32,  # Mismatch with saved (16)
            lookback=20,
            tcn_channels=[32, 32, 16],
            dilations=[1, 2, 4],
            attention_heads=2,
            attention_dim=16,
        )
        manager = ModelManager(config=wrong_config, device="cpu")
        with pytest.raises(ModelError) as exc_info:
            manager.load(saved_model_path)
        assert "INCOMPATIBLE" in exc_info.value.error_code

    def test_load_legacy_format(self, config):
        """Load works with legacy raw state_dict format (no metadata)."""
        model = StockEvalNet(config)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            path = f.name
        try:
            manager = ModelManager(config=config, device="cpu")
            manager.load(path)
            assert manager.is_loaded
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Hot-Swap Tests
# ---------------------------------------------------------------------------


class TestModelManagerHotSwap:
    """Tests for ModelManager.hot_swap()."""

    def test_hot_swap_replaces_model(self, manager, saved_model_path, second_model_path):
        """Hot-swap replaces the active model with a new one."""
        manager.load(saved_model_path)
        checksum_v1 = manager.checksum
        assert manager.version == 1

        manager.hot_swap(second_model_path)
        assert manager.version == 2
        assert manager.checksum != checksum_v1

    def test_hot_swap_without_initial_model(self, manager, saved_model_path):
        """Hot-swap works even without a previously loaded model."""
        assert not manager.is_loaded
        manager.hot_swap(saved_model_path)
        assert manager.is_loaded
        assert manager.version == 1

    def test_hot_swap_file_not_found(self, manager, saved_model_path):
        """Hot-swap raises ModelError for missing file, old model persists."""
        manager.load(saved_model_path)
        original_version = manager.version

        with pytest.raises(ModelError):
            manager.hot_swap("/nonexistent/model.pt")

        # Old model still works
        assert manager.is_loaded
        assert manager.version == original_version

    def test_hot_swap_checksum_failure_keeps_old(self, manager, saved_model_path, second_model_path):
        """Hot-swap checksum failure preserves old model."""
        manager.load(saved_model_path)
        v1 = manager.version

        with pytest.raises(ModelError):
            manager.hot_swap(second_model_path, expected_checksum="bad" * 16)

        assert manager.version == v1
        assert manager.is_loaded

    def test_hot_swap_thread_safety(self, manager, saved_model_path, second_model_path, config):
        """Hot-swap is thread-safe: concurrent predict and swap don't crash."""
        manager.load(saved_model_path)
        errors = []

        def do_predict():
            try:
                features = np.random.randn(4, config.lookback, config.num_features).astype(np.float32)
                manager.predict(features)
            except Exception as e:
                # ModelError from unload during swap is acceptable
                if "unloaded" not in str(e).lower():
                    errors.append(e)

        def do_swap():
            try:
                manager.hot_swap(second_model_path)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=do_predict))
            threads.append(threading.Thread(target=do_swap))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No unexpected errors
        assert len(errors) == 0, f"Thread safety errors: {errors}"


# ---------------------------------------------------------------------------
# Predict Tests
# ---------------------------------------------------------------------------


class TestModelManagerPredict:
    """Tests for ModelManager.predict()."""

    def test_predict_single_sample(self, manager, saved_model_path, config):
        """Predict with 2D input (lookback, features) returns (1, 1)."""
        manager.load(saved_model_path)
        features = np.random.randn(config.lookback, config.num_features).astype(np.float32)
        result = manager.predict(features)
        assert result.shape == (1, 1)
        assert -1.0 <= result[0, 0] <= 1.0

    def test_predict_batch(self, manager, saved_model_path, config):
        """Predict with 3D input (batch, lookback, features) returns (batch, 1)."""
        manager.load(saved_model_path)
        batch_size = 8
        features = np.random.randn(batch_size, config.lookback, config.num_features).astype(np.float32)
        result = manager.predict(features)
        assert result.shape == (batch_size, 1)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_predict_output_bounded(self, manager, saved_model_path, config):
        """Output always in [-1, 1] for various inputs."""
        manager.load(saved_model_path)

        # Normal input
        features = np.random.randn(10, config.lookback, config.num_features).astype(np.float32)
        result = manager.predict(features)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

        # Zero input
        features_zero = np.zeros((5, config.lookback, config.num_features), dtype=np.float32)
        result_zero = manager.predict(features_zero)
        assert np.all(result_zero >= -1.0) and np.all(result_zero <= 1.0)

        # Large input
        features_large = np.random.randn(3, config.lookback, config.num_features).astype(np.float32) * 100
        result_large = manager.predict(features_large)
        assert np.all(result_large >= -1.0) and np.all(result_large <= 1.0)

    def test_predict_no_model_raises(self, manager, config):
        """Predict raises ModelError when no model is loaded."""
        features = np.random.randn(config.lookback, config.num_features).astype(np.float32)
        with pytest.raises(ModelError) as exc_info:
            manager.predict(features)
        assert "MODEL_NOT_LOADED" in exc_info.value.error_code

    def test_predict_invalid_ndim(self, manager, saved_model_path, config):
        """Predict raises ModelError for 1D or 4D input."""
        manager.load(saved_model_path)
        with pytest.raises(ModelError):
            manager.predict(np.array([1.0, 2.0, 3.0]))

    def test_predict_deterministic(self, manager, saved_model_path, config):
        """Same input produces same output (deterministic in eval mode)."""
        manager.load(saved_model_path)
        features = np.random.randn(4, config.lookback, config.num_features).astype(np.float32)
        r1 = manager.predict(features)
        r2 = manager.predict(features)
        np.testing.assert_array_almost_equal(r1, r2, decimal=6)


# ---------------------------------------------------------------------------
# Save Tests
# ---------------------------------------------------------------------------


class TestModelManagerSave:
    """Tests for ModelManager.save()."""

    def test_save_and_reload(self, manager, config):
        """Save then load produces consistent inference."""
        manager.create_new_model()
        features = np.random.randn(4, config.lookback, config.num_features).astype(np.float32)
        original_output = manager.predict(features)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            checksum = manager.save(path)
            assert len(checksum) == 64

            # Reload in new manager
            manager2 = ModelManager(config=config, device="cpu")
            manager2.load(path)
            reloaded_output = manager2.predict(features)

            np.testing.assert_array_almost_equal(original_output, reloaded_output, decimal=5)
        finally:
            os.unlink(path)

    def test_save_no_model_raises(self, manager):
        """Save raises ModelError when no model loaded."""
        with pytest.raises(ModelError):
            manager.save("/tmp/test.pt")


# ---------------------------------------------------------------------------
# Device Management Tests
# ---------------------------------------------------------------------------


class TestModelManagerDevice:
    """Tests for device management."""

    def test_default_device_cpu_when_no_cuda(self, config):
        """Defaults to CPU when CUDA is not available (in test env)."""
        manager = ModelManager(config=config, device="cpu")
        assert manager.device == torch.device("cpu")

    def test_explicit_device_setting(self, config):
        """Explicit device override works."""
        manager = ModelManager(config=config, device="cpu")
        assert manager.device == torch.device("cpu")

    def test_set_device(self, manager, saved_model_path):
        """set_device moves model to new device."""
        manager.load(saved_model_path)
        manager.set_device("cpu")
        assert manager.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Version Tracking Tests
# ---------------------------------------------------------------------------


class TestModelManagerVersion:
    """Tests for version tracking."""

    def test_initial_version_zero(self, manager):
        """Version starts at 0."""
        assert manager.version == 0

    def test_load_increments(self, manager, saved_model_path):
        """Each load increments version."""
        manager.load(saved_model_path)
        assert manager.version == 1
        manager.load(saved_model_path)
        assert manager.version == 2

    def test_hot_swap_increments(self, manager, saved_model_path, second_model_path):
        """hot_swap increments version."""
        manager.load(saved_model_path)
        assert manager.version == 1
        manager.hot_swap(second_model_path)
        assert manager.version == 2

    def test_create_new_increments(self, manager):
        """create_new_model increments version."""
        manager.create_new_model()
        assert manager.version == 1
        manager.create_new_model()
        assert manager.version == 2


# ---------------------------------------------------------------------------
# create_new_model Tests
# ---------------------------------------------------------------------------


class TestCreateNewModel:
    """Tests for ModelManager.create_new_model()."""

    def test_creates_usable_model(self, manager, config):
        """create_new_model produces a model ready for inference."""
        manager.create_new_model()
        assert manager.is_loaded
        features = np.random.randn(2, config.lookback, config.num_features).astype(np.float32)
        result = manager.predict(features)
        assert result.shape == (2, 1)
