# -*- coding: utf-8 -*-
"""
Feature Scaler - Quản lý normalization params cho training/inference consistency.

Module này giải quyết vấn đề Training-Serving Skew:
- Training: Tính global min/max từ toàn bộ training data
- Inference: Dùng exact params đã lưu thay vì tính lại per-window

Best Practice:
- Scaler được fit 1 lần trên training data
- Params được lưu vào JSON file
- Inference load params và transform với cùng scale
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Đường dẫn mặc định cho scaler params
DEFAULT_SCALER_PATH = Path("engine/models/scaler_params.json")


@dataclass
class ScalerParams:
    """
    Normalization parameters cho MinMax scaling.
    
    Attributes:
        min_vals: Giá trị min mỗi feature, shape (num_features,)
        max_vals: Giá trị max mỗi feature, shape (num_features,)
        num_features: Số features
        feature_names: Tên các features (optional, để debug)
        fitted_at: Timestamp khi fit
        num_samples: Số samples dùng để fit
        symbols_used: Danh sách symbols dùng để fit
    """
    min_vals: np.ndarray
    max_vals: np.ndarray
    num_features: int
    feature_names: List[str] = field(default_factory=list)
    fitted_at: str = ""
    num_samples: int = 0
    symbols_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Chuyển thành dict để serialize."""
        return {
            "min_vals": self.min_vals.tolist(),
            "max_vals": self.max_vals.tolist(),
            "num_features": self.num_features,
            "feature_names": self.feature_names,
            "fitted_at": self.fitted_at,
            "num_samples": self.num_samples,
            "symbols_used": self.symbols_used,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ScalerParams":
        """Load từ dict."""
        return cls(
            min_vals=np.array(data["min_vals"], dtype=np.float64),
            max_vals=np.array(data["max_vals"], dtype=np.float64),
            num_features=data["num_features"],
            feature_names=data.get("feature_names", []),
            fitted_at=data.get("fitted_at", ""),
            num_samples=data.get("num_samples", 0),
            symbols_used=data.get("symbols_used", []),
        )


class FeatureScaler:
    """
    MinMax Scaler với save/load capabilities.
    
    Đảm bảo consistency giữa training và inference bằng cách:
    1. fit() trên training data → tính global min/max
    2. save() params vào JSON file
    3. load() params khi inference
    4. transform() dùng saved params
    
    Usage:
        # Training time
        scaler = FeatureScaler()
        scaler.fit(training_windows)  # List of (lookback, num_features) arrays
        scaler.save("engine/models/scaler_params.json")
        
        # Inference time
        scaler = FeatureScaler.load("engine/models/scaler_params.json")
        normalized = scaler.transform(window)
    """
    
    def __init__(self, params: Optional[ScalerParams] = None):
        """
        Khởi tạo scaler.
        
        Args:
            params: ScalerParams đã có (optional). Nếu None, cần gọi fit() trước.
        """
        self._params = params
        self._is_fitted = params is not None
    
    @property
    def is_fitted(self) -> bool:
        """Kiểm tra scaler đã được fit chưa."""
        return self._is_fitted
    
    @property
    def params(self) -> Optional[ScalerParams]:
        """Trả về params hiện tại."""
        return self._params
    
    def fit(
        self,
        windows: List[np.ndarray],
        feature_names: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
    ) -> "FeatureScaler":
        """
        Fit scaler trên training data.
        
        Tính global min/max từ tất cả windows, KHÔNG phải per-window.
        Điều này đảm bảo inference sẽ dùng cùng scale.
        
        Args:
            windows: List các numpy arrays, mỗi array shape (lookback, num_features)
            feature_names: Tên các features (optional)
            symbols: Danh sách symbols dùng để fit (optional, để tracking)
        
        Returns:
            self (để chaining)
        
        Raises:
            ValueError: Nếu windows rỗng hoặc không hợp lệ
        """
        if not windows:
            raise ValueError("Cần ít nhất 1 window để fit scaler")
        
        # Xác định num_features từ window đầu tiên
        num_features = windows[0].shape[1]
        
        # Khởi tạo global min/max
        global_min = np.full(num_features, np.inf, dtype=np.float64)
        global_max = np.full(num_features, -np.inf, dtype=np.float64)
        
        total_samples = 0
        
        for window in windows:
            if window.shape[1] != num_features:
                logger.warning(
                    f"Window có {window.shape[1]} features, expected {num_features}. Skipping."
                )
                continue
            
            # Xử lý NaN/Inf trước khi tính min/max
            window_clean = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Cập nhật global min/max
            window_min = np.min(window_clean, axis=0)
            window_max = np.max(window_clean, axis=0)
            
            global_min = np.minimum(global_min, window_min)
            global_max = np.maximum(global_max, window_max)
            
            total_samples += window.shape[0]
        
        # Handle features không có data (vẫn là inf)
        invalid_mask = np.isinf(global_min) | np.isinf(global_max)
        global_min[invalid_mask] = 0.0
        global_max[invalid_mask] = 1.0
        
        # Handle degenerate features (min == max)
        degenerate_mask = global_min == global_max
        global_max[degenerate_mask] = global_min[degenerate_mask] + 1.0
        
        self._params = ScalerParams(
            min_vals=global_min,
            max_vals=global_max,
            num_features=num_features,
            feature_names=feature_names or [],
            fitted_at=datetime.now().isoformat(),
            num_samples=total_samples,
            symbols_used=symbols or [],
        )
        self._is_fitted = True
        
        logger.info(
            f"[Scaler] Fitted: {num_features} features, {total_samples} samples, "
            f"{len(symbols or [])} symbols"
        )
        
        return self
    
    def partial_fit(
        self,
        windows: List[np.ndarray],
        symbols: Optional[List[str]] = None,
    ) -> "FeatureScaler":
        """
        Incremental fit - cập nhật min/max với data mới.
        
        Dùng khi training tiếp tục và muốn mở rộng range mà không reset.
        
        Args:
            windows: List các numpy arrays mới
            symbols: Danh sách symbols mới
        
        Returns:
            self
        """
        if not self._is_fitted:
            return self.fit(windows, symbols=symbols)
        
        if not windows:
            return self
        
        num_features = self._params.num_features
        global_min = self._params.min_vals.copy()
        global_max = self._params.max_vals.copy()
        total_samples = self._params.num_samples
        all_symbols = list(self._params.symbols_used)
        
        for window in windows:
            if window.shape[1] != num_features:
                continue
            
            window_clean = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
            window_min = np.min(window_clean, axis=0)
            window_max = np.max(window_clean, axis=0)
            
            global_min = np.minimum(global_min, window_min)
            global_max = np.maximum(global_max, window_max)
            total_samples += window.shape[0]
        
        # Thêm symbols mới
        if symbols:
            for s in symbols:
                if s not in all_symbols:
                    all_symbols.append(s)
        
        # Handle degenerate
        degenerate_mask = global_min == global_max
        global_max[degenerate_mask] = global_min[degenerate_mask] + 1.0
        
        self._params = ScalerParams(
            min_vals=global_min,
            max_vals=global_max,
            num_features=num_features,
            feature_names=self._params.feature_names,
            fitted_at=datetime.now().isoformat(),
            num_samples=total_samples,
            symbols_used=all_symbols,
        )
        
        return self
    
    def transform(self, window: np.ndarray, clip: bool = True) -> np.ndarray:
        """
        Transform window về [0, 1] dùng saved params.
        
        Args:
            window: Array shape (lookback, num_features) hoặc (num_features,)
            clip: Có clip về [0, 1] không (default True)
        
        Returns:
            Normalized array cùng shape
        
        Raises:
            RuntimeError: Nếu scaler chưa được fit
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler chưa được fit. Gọi fit() hoặc load() trước.")
        
        # Xử lý input 1D
        is_1d = window.ndim == 1
        if is_1d:
            window = window.reshape(1, -1)
        
        # Kiểm tra số features
        if window.shape[1] != self._params.num_features:
            logger.warning(
                f"Window có {window.shape[1]} features, scaler có {self._params.num_features}. "
                "Padding/trimming."
            )
            window = self._adjust_features(window)
        
        # Xử lý NaN/Inf
        window_clean = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        
        # MinMax normalize
        range_vals = self._params.max_vals - self._params.min_vals
        # Tránh chia cho 0
        range_vals = np.where(range_vals == 0, 1.0, range_vals)
        
        normalized = (window_clean - self._params.min_vals) / range_vals
        
        if clip:
            normalized = np.clip(normalized, 0.0, 1.0)
        
        if is_1d:
            normalized = normalized.flatten()
        
        return normalized.astype(np.float32)
    
    def fit_transform(
        self,
        windows: List[np.ndarray],
        feature_names: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """
        Fit rồi transform tất cả windows.
        
        Args:
            windows: List các windows
            feature_names: Tên features
            symbols: Danh sách symbols
        
        Returns:
            List các normalized windows
        """
        self.fit(windows, feature_names=feature_names, symbols=symbols)
        return [self.transform(w) for w in windows]
    
    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """
        Chuyển từ [0, 1] về giá trị gốc.
        
        Args:
            normalized: Array đã normalize
        
        Returns:
            Array giá trị gốc
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler chưa được fit.")
        
        range_vals = self._params.max_vals - self._params.min_vals
        return normalized * range_vals + self._params.min_vals
    
    def save(self, path: Optional[str] = None) -> Path:
        """
        Lưu scaler params vào JSON file.
        
        Args:
            path: Đường dẫn file (default: engine/models/scaler_params.json)
        
        Returns:
            Path đã lưu
        
        Raises:
            RuntimeError: Nếu scaler chưa được fit
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler chưa được fit. Không có gì để save.")
        
        filepath = Path(path) if path else DEFAULT_SCALER_PATH
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._params.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"[Scaler] Saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, path: Optional[str] = None) -> "FeatureScaler":
        """
        Load scaler từ JSON file.
        
        Args:
            path: Đường dẫn file (default: engine/models/scaler_params.json)
        
        Returns:
            FeatureScaler đã load
        
        Raises:
            FileNotFoundError: Nếu file không tồn tại
            ValueError: Nếu file không hợp lệ
        """
        filepath = Path(path) if path else DEFAULT_SCALER_PATH
        
        if not filepath.exists():
            raise FileNotFoundError(f"Scaler file không tồn tại: {filepath}")
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            params = ScalerParams.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Scaler file không hợp lệ: {e}")
        
        logger.info(
            f"[Scaler] Loaded: {params.num_features} features, "
            f"fitted at {params.fitted_at}, {params.num_samples} samples"
        )
        
        return cls(params=params)
    
    @classmethod
    def load_or_create(cls, path: Optional[str] = None) -> Tuple["FeatureScaler", bool]:
        """
        Load scaler nếu có, hoặc tạo mới (unfitted).
        
        Args:
            path: Đường dẫn file
        
        Returns:
            (scaler, is_loaded) - is_loaded=True nếu load thành công
        """
        filepath = Path(path) if path else DEFAULT_SCALER_PATH
        
        if filepath.exists():
            try:
                scaler = cls.load(path)
                return scaler, True
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"[Scaler] Load failed ({e}), creating new")
        
        return cls(), False
    
    def _adjust_features(self, window: np.ndarray) -> np.ndarray:
        """Pad hoặc trim window để khớp num_features."""
        target = self._params.num_features
        current = window.shape[1]
        
        if current < target:
            padding = np.zeros((window.shape[0], target - current), dtype=np.float64)
            return np.concatenate([window, padding], axis=1)
        else:
            return window[:, :target]


def get_default_scaler_path() -> Path:
    """Trả về đường dẫn mặc định của scaler file."""
    return DEFAULT_SCALER_PATH


def scaler_exists(path: Optional[str] = None) -> bool:
    """Kiểm tra scaler file có tồn tại không."""
    filepath = Path(path) if path else DEFAULT_SCALER_PATH
    return filepath.exists()
