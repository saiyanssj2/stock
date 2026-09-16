# -*- coding: utf-8 -*-
"""
Unit tests cho module engine/feature_scaler.py

Kiểm tra các functions:
- FeatureScaler.fit()
- FeatureScaler.transform()
- FeatureScaler.save() / load()
- FeatureScaler.partial_fit()
- FeatureScaler.fit_transform()
- FeatureScaler.inverse_transform()
- Edge cases và error handling
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Thêm root vào path để import được engine module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.feature_scaler import (
    FeatureScaler,
    ScalerParams,
    scaler_exists,
    get_default_scaler_path,
)


# === Fixtures ===

@pytest.fixture
def sample_windows():
    """Tạo sample windows cho testing."""
    np.random.seed(42)
    # 10 windows, mỗi window (60, 78) - lookback x num_features
    windows = []
    for _ in range(10):
        # Tạo data với range khác nhau cho mỗi feature
        window = np.random.randn(60, 78) * 100 + 50
        windows.append(window)
    return windows


@pytest.fixture
def fitted_scaler(sample_windows):
    """Tạo scaler đã fit."""
    scaler = FeatureScaler()
    scaler.fit(sample_windows, symbols=["VNM", "FPT", "HPG"])
    return scaler


@pytest.fixture
def temp_scaler_file():
    """Tạo temp file cho save/load tests."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        yield f.name
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


# === Tests cho ScalerParams ===

class TestScalerParams:
    """Tests cho ScalerParams dataclass."""
    
    def test_to_dict(self):
        """Kiểm tra serialize to dict."""
        params = ScalerParams(
            min_vals=np.array([0.0, 1.0, 2.0]),
            max_vals=np.array([10.0, 11.0, 12.0]),
            num_features=3,
            feature_names=["a", "b", "c"],
            fitted_at="2026-08-19T10:00:00",
            num_samples=1000,
            symbols_used=["VNM", "FPT"],
        )
        
        d = params.to_dict()
        
        assert d["min_vals"] == [0.0, 1.0, 2.0]
        assert d["max_vals"] == [10.0, 11.0, 12.0]
        assert d["num_features"] == 3
        assert d["feature_names"] == ["a", "b", "c"]
        assert d["symbols_used"] == ["VNM", "FPT"]
    
    def test_from_dict(self):
        """Kiểm tra deserialize from dict."""
        d = {
            "min_vals": [0.0, 1.0],
            "max_vals": [10.0, 11.0],
            "num_features": 2,
            "feature_names": ["x", "y"],
            "fitted_at": "2026-08-19",
            "num_samples": 500,
            "symbols_used": ["ACB"],
        }
        
        params = ScalerParams.from_dict(d)
        
        assert np.array_equal(params.min_vals, np.array([0.0, 1.0]))
        assert np.array_equal(params.max_vals, np.array([10.0, 11.0]))
        assert params.num_features == 2
        assert params.symbols_used == ["ACB"]


# === Tests cho FeatureScaler.fit() ===

class TestFit:
    """Tests cho FeatureScaler.fit()."""
    
    def test_fit_creates_params(self, sample_windows):
        """Kiểm tra fit() tạo params."""
        scaler = FeatureScaler()
        assert not scaler.is_fitted
        
        scaler.fit(sample_windows)
        
        assert scaler.is_fitted
        assert scaler.params is not None
        assert scaler.params.num_features == 78
    
    def test_fit_computes_global_minmax(self, sample_windows):
        """Kiểm tra fit() tính global min/max đúng."""
        scaler = FeatureScaler()
        scaler.fit(sample_windows)
        
        # Tự tính global min/max
        all_data = np.concatenate(sample_windows, axis=0)
        expected_min = np.min(all_data, axis=0)
        expected_max = np.max(all_data, axis=0)
        
        np.testing.assert_array_almost_equal(scaler.params.min_vals, expected_min, decimal=5)
        np.testing.assert_array_almost_equal(scaler.params.max_vals, expected_max, decimal=5)
    
    def test_fit_with_symbols(self, sample_windows):
        """Kiểm tra fit() lưu symbols."""
        scaler = FeatureScaler()
        symbols = ["VNM", "FPT", "HPG"]
        
        scaler.fit(sample_windows, symbols=symbols)
        
        assert scaler.params.symbols_used == symbols
    
    def test_fit_empty_raises_error(self):
        """Kiểm tra fit() với empty list raises error."""
        scaler = FeatureScaler()
        
        with pytest.raises(ValueError, match="Cần ít nhất 1 window"):
            scaler.fit([])
    
    def test_fit_handles_nan(self):
        """Kiểm tra fit() xử lý NaN đúng."""
        windows = [np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])]
        
        scaler = FeatureScaler()
        scaler.fit(windows)
        
        # NaN được thay bằng 0, nên min feature 1 = 0 (không phải nan)
        assert scaler.params.min_vals[1] == 0.0
    
    def test_fit_handles_degenerate_features(self):
        """Kiểm tra fit() xử lý features có min==max."""
        windows = [np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])]  # Feature 0 và 1 đều constant
        
        scaler = FeatureScaler()
        scaler.fit(windows)
        
        # max được điều chỉnh để tránh chia 0
        assert scaler.params.max_vals[0] > scaler.params.min_vals[0]
        assert scaler.params.max_vals[1] > scaler.params.min_vals[1]


# === Tests cho FeatureScaler.transform() ===

class TestTransform:
    """Tests cho FeatureScaler.transform()."""
    
    def test_transform_output_range(self, fitted_scaler, sample_windows):
        """Kiểm tra transform() output trong [0, 1]."""
        result = fitted_scaler.transform(sample_windows[0])
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_transform_shape_preserved(self, fitted_scaler, sample_windows):
        """Kiểm tra transform() giữ nguyên shape."""
        window = sample_windows[0]
        result = fitted_scaler.transform(window)
        
        assert result.shape == window.shape
    
    def test_transform_not_fitted_raises(self):
        """Kiểm tra transform() khi chưa fit raises error."""
        scaler = FeatureScaler()
        window = np.random.randn(60, 78)
        
        with pytest.raises(RuntimeError, match="chưa được fit"):
            scaler.transform(window)
    
    def test_transform_1d_input(self, fitted_scaler):
        """Kiểm tra transform() với 1D input."""
        window_1d = np.random.randn(78)
        
        result = fitted_scaler.transform(window_1d)
        
        assert result.shape == (78,)
    
    def test_transform_clips_out_of_range(self, fitted_scaler):
        """Kiểm tra transform() clips values ngoài range training."""
        # Tạo window với giá trị extreme
        window = np.ones((60, 78)) * 10000  # Rất cao
        
        result = fitted_scaler.transform(window, clip=True)
        
        # Tất cả values phải <= 1.0
        assert result.max() <= 1.0
    
    def test_transform_no_clip_option(self, fitted_scaler):
        """Kiểm tra transform() với clip=False."""
        window = np.ones((60, 78)) * 10000
        
        result = fitted_scaler.transform(window, clip=False)
        
        # Values có thể > 1.0
        assert result.max() > 1.0


# === Tests cho FeatureScaler.save() / load() ===

class TestSaveLoad:
    """Tests cho save() và load()."""
    
    def test_save_creates_file(self, fitted_scaler, temp_scaler_file):
        """Kiểm tra save() tạo file."""
        fitted_scaler.save(temp_scaler_file)
        
        assert Path(temp_scaler_file).exists()
    
    def test_save_valid_json(self, fitted_scaler, temp_scaler_file):
        """Kiểm tra save() tạo valid JSON."""
        fitted_scaler.save(temp_scaler_file)
        
        with open(temp_scaler_file, "r") as f:
            data = json.load(f)
        
        assert "min_vals" in data
        assert "max_vals" in data
        assert "num_features" in data
    
    def test_load_restores_params(self, fitted_scaler, temp_scaler_file):
        """Kiểm tra load() khôi phục params đúng."""
        fitted_scaler.save(temp_scaler_file)
        
        loaded = FeatureScaler.load(temp_scaler_file)
        
        assert loaded.is_fitted
        np.testing.assert_array_equal(
            loaded.params.min_vals, 
            fitted_scaler.params.min_vals
        )
        np.testing.assert_array_equal(
            loaded.params.max_vals, 
            fitted_scaler.params.max_vals
        )
    
    def test_load_nonexistent_raises(self):
        """Kiểm tra load() file không tồn tại raises error."""
        with pytest.raises(FileNotFoundError):
            FeatureScaler.load("/nonexistent/path/scaler.json")
    
    def test_load_invalid_json_raises(self, temp_scaler_file):
        """Kiểm tra load() invalid JSON raises error."""
        with open(temp_scaler_file, "w") as f:
            f.write("not valid json {{{")
        
        with pytest.raises(ValueError):
            FeatureScaler.load(temp_scaler_file)
    
    def test_save_not_fitted_raises(self, temp_scaler_file):
        """Kiểm tra save() khi chưa fit raises error."""
        scaler = FeatureScaler()
        
        with pytest.raises(RuntimeError, match="chưa được fit"):
            scaler.save(temp_scaler_file)
    
    def test_load_or_create_existing(self, fitted_scaler, temp_scaler_file):
        """Kiểm tra load_or_create() với file có sẵn."""
        fitted_scaler.save(temp_scaler_file)
        
        scaler, is_loaded = FeatureScaler.load_or_create(temp_scaler_file)
        
        assert is_loaded is True
        assert scaler.is_fitted
    
    def test_load_or_create_missing(self):
        """Kiểm tra load_or_create() với file không có."""
        scaler, is_loaded = FeatureScaler.load_or_create("/nonexistent/scaler.json")
        
        assert is_loaded is False
        assert not scaler.is_fitted


# === Tests cho FeatureScaler.partial_fit() ===

class TestPartialFit:
    """Tests cho partial_fit()."""
    
    def test_partial_fit_expands_range(self, fitted_scaler):
        """Kiểm tra partial_fit() mở rộng range."""
        original_min = fitted_scaler.params.min_vals.copy()
        original_max = fitted_scaler.params.max_vals.copy()
        
        # Tạo windows với values ngoài range hiện tại
        new_windows = [np.ones((60, 78)) * -1000]  # Rất thấp
        
        fitted_scaler.partial_fit(new_windows)
        
        # min phải giảm
        assert np.all(fitted_scaler.params.min_vals <= original_min)
    
    def test_partial_fit_on_unfitted_calls_fit(self, sample_windows):
        """Kiểm tra partial_fit() trên unfitted scaler gọi fit()."""
        scaler = FeatureScaler()
        
        scaler.partial_fit(sample_windows)
        
        assert scaler.is_fitted
    
    def test_partial_fit_adds_symbols(self, fitted_scaler):
        """Kiểm tra partial_fit() thêm symbols mới."""
        original_symbols = list(fitted_scaler.params.symbols_used)
        
        fitted_scaler.partial_fit([np.random.randn(60, 78)], symbols=["NEW_SYM"])
        
        assert "NEW_SYM" in fitted_scaler.params.symbols_used
        for sym in original_symbols:
            assert sym in fitted_scaler.params.symbols_used


# === Tests cho fit_transform() và inverse_transform() ===

class TestFitTransformAndInverse:
    """Tests cho fit_transform() và inverse_transform()."""
    
    def test_fit_transform(self, sample_windows):
        """Kiểm tra fit_transform() works."""
        scaler = FeatureScaler()
        
        results = scaler.fit_transform(sample_windows)
        
        assert scaler.is_fitted
        assert len(results) == len(sample_windows)
        assert results[0].shape == sample_windows[0].shape
    
    def test_inverse_transform_recovers_original(self, fitted_scaler, sample_windows):
        """Kiểm tra inverse_transform() khôi phục giá trị gốc."""
        window = sample_windows[0]
        
        normalized = fitted_scaler.transform(window, clip=False)
        recovered = fitted_scaler.inverse_transform(normalized)
        
        # Dùng decimal=3 vì có floating point precision loss
        np.testing.assert_array_almost_equal(recovered, window, decimal=3)
    
    def test_inverse_transform_not_fitted_raises(self):
        """Kiểm tra inverse_transform() khi chưa fit raises error."""
        scaler = FeatureScaler()
        
        with pytest.raises(RuntimeError):
            scaler.inverse_transform(np.random.randn(60, 78))


# === Tests cho helper functions ===

class TestHelperFunctions:
    """Tests cho helper functions."""
    
    def test_get_default_scaler_path(self):
        """Kiểm tra get_default_scaler_path()."""
        path = get_default_scaler_path()
        
        assert path.name == "scaler_params.json"
        # Windows dùng backslash, Unix dùng forward slash
        assert "engine" in str(path) and "models" in str(path)
    
    def test_scaler_exists_false(self):
        """Kiểm tra scaler_exists() với path không tồn tại."""
        result = scaler_exists("/nonexistent/path.json")
        assert result is False
    
    def test_scaler_exists_true(self, fitted_scaler, temp_scaler_file):
        """Kiểm tra scaler_exists() với file có sẵn."""
        fitted_scaler.save(temp_scaler_file)
        
        result = scaler_exists(temp_scaler_file)
        assert result is True


# === Integration Tests ===

class TestIntegration:
    """Integration tests cho workflow đầy đủ."""
    
    def test_full_workflow(self, sample_windows, temp_scaler_file):
        """Test workflow: fit → save → load → transform."""
        # 1. Fit
        scaler = FeatureScaler()
        scaler.fit(sample_windows, symbols=["VNM", "FPT"])
        
        # 2. Transform
        transformed = scaler.transform(sample_windows[0])
        assert transformed.min() >= 0.0
        assert transformed.max() <= 1.0
        
        # 3. Save
        scaler.save(temp_scaler_file)
        
        # 4. Load trong process mới
        loaded_scaler = FeatureScaler.load(temp_scaler_file)
        
        # 5. Transform với loaded scaler → kết quả phải giống
        transformed_2 = loaded_scaler.transform(sample_windows[0])
        
        np.testing.assert_array_equal(transformed, transformed_2)
    
    def test_consistency_across_sessions(self, sample_windows, temp_scaler_file):
        """Test consistency giữa training và inference sessions."""
        # Simulate training session
        train_scaler = FeatureScaler()
        train_scaler.fit(sample_windows[:5])
        train_scaler.save(temp_scaler_file)
        
        train_result = train_scaler.transform(sample_windows[5])
        
        # Simulate inference session (load từ file)
        infer_scaler = FeatureScaler.load(temp_scaler_file)
        infer_result = infer_scaler.transform(sample_windows[5])
        
        # Kết quả phải giống nhau
        np.testing.assert_array_equal(train_result, infer_result)
    
    def test_incremental_learning_scenario(self, temp_scaler_file):
        """Test scenario training liên tục (incremental)."""
        np.random.seed(123)
        
        # Cycle 1: fit trên batch đầu
        windows_1 = [np.random.randn(60, 78) * 10 for _ in range(5)]
        scaler = FeatureScaler()
        scaler.fit(windows_1)
        scaler.save(temp_scaler_file)
        
        # Cycle 2: load và partial_fit với batch mới
        scaler_2 = FeatureScaler.load(temp_scaler_file)
        windows_2 = [np.random.randn(60, 78) * 20 for _ in range(5)]  # Khác range
        scaler_2.partial_fit(windows_2)
        scaler_2.save(temp_scaler_file)
        
        # Cycle 3: load → range phải đã mở rộng
        scaler_3 = FeatureScaler.load(temp_scaler_file)
        
        # Transform window có range lớn → vẫn clip được
        big_window = np.ones((60, 78)) * 15
        result = scaler_3.transform(big_window)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
