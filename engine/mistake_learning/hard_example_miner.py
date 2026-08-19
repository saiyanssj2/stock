"""
Hard Example Miner cho Mistake-Driven Learning.

Trích xuất feature vectors từ bad trades, tạo hard examples với sample weights
tăng cường để model tập trung vào các trường hợp khó trong quá trình retrain.

Chức năng chính:
- mine(): trích xuất 60-day lookback feature windows tại entry_date mỗi bad trade
- compute_sample_weights(): tính toán mảng weights cho toàn bộ training set
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from engine.mistake_learning.models import HardExampleSet, TradeRecord

logger = logging.getLogger(__name__)

# Các cột OHLCV cơ bản không phải feature
_OHLCV_COLUMNS = {"time", "open", "high", "low", "close", "volume"}

# Số chiều feature cố định
_FEATURE_DIM = 61


class HardExampleMiner:
    """Trích xuất hard examples từ bad trades với sample weights tăng cường.

    Hard examples là các feature windows tại thời điểm model ra signal sai,
    được gán weight cao hơn để model chú ý nhiều hơn khi retrain.

    Attributes:
        DEFAULT_WEIGHT_MULTIPLIER: Hệ số weight mặc định cho hard examples.
        weight_multiplier: Hệ số weight thực tế đang dùng.
        max_hard_examples_ratio: Tỷ lệ tối đa hard examples trong training set.
    """

    DEFAULT_WEIGHT_MULTIPLIER: float = 3.0  # Range: 2-5x

    def __init__(
        self,
        weight_multiplier: float = 3.0,
        max_hard_examples_ratio: float = 0.3,
    ) -> None:
        """Khởi tạo HardExampleMiner.

        Args:
            weight_multiplier: Hệ số weight cho hard examples (2.0 - 5.0).
            max_hard_examples_ratio: Tỷ lệ tối đa hard examples so với
                total training samples (0.05 - 0.50).
        """
        self.weight_multiplier = weight_multiplier
        self.max_hard_examples_ratio = max_hard_examples_ratio

    def mine(
        self,
        bad_trades: List[TradeRecord],
        feature_data: Dict[str, pd.DataFrame],
        lookback: int = 60,
    ) -> HardExampleSet:
        """Trích xuất hard examples từ danh sách bad trades.

        Với mỗi bad trade, trích xuất cửa sổ lookback ngày feature data
        kết thúc tại entry_date. Gán weight_multiplier cho tất cả hard examples.
        Cap số lượng tại max_hard_examples_ratio × total_training_samples.

        Args:
            bad_trades: Danh sách trades bị lỗ nặng (pnl_pct < -5%).
            feature_data: Dict mapping symbol → DataFrame chứa OHLCV + indicators.
            lookback: Số ngày lookback (mặc định 60).

        Returns:
            HardExampleSet với features shape (N, lookback, 61).
            Trả về empty set nếu bad_trades rỗng hoặc không có trade hợp lệ.
        """
        # Xử lý trường hợp bad_trades rỗng
        if not bad_trades:
            return HardExampleSet(
                features=np.array([]).reshape(0, lookback, _FEATURE_DIM),
                labels=np.array([]),
                weights=np.array([]),
                source_trades=[],
            )

        # Xử lý trường hợp feature_data rỗng
        if not feature_data:
            logger.warning("feature_data rỗng, không thể mine hard examples")
            return HardExampleSet(
                features=np.array([]).reshape(0, lookback, _FEATURE_DIM),
                labels=np.array([]),
                weights=np.array([]),
                source_trades=[],
            )

        # Lấy DataFrame đầu tiên làm reference
        first_symbol = next(iter(feature_data))
        ref_df = feature_data[first_symbol]

        # Ước tính total_training_samples từ số hàng của DataFrame đầu tiên
        total_training_samples = len(ref_df)

        # Xác định feature columns (loại bỏ OHLCV)
        feature_cols = self._get_feature_columns(ref_df)

        # Trích xuất feature windows cho mỗi bad trade
        features_list: List[np.ndarray] = []
        labels_list: List[float] = []
        source_trades: List[TradeRecord] = []

        for trade in bad_trades:
            try:
                feature_window = self._extract_feature_window(
                    trade, feature_data, feature_cols, lookback
                )
                if feature_window is None:
                    continue

                # Tính corrected label dựa trên trade outcome
                corrected_label = self._compute_corrected_label(trade)

                features_list.append(feature_window)
                labels_list.append(corrected_label)
                source_trades.append(trade)
            except Exception as e:
                logger.warning(
                    f"Lỗi khi xử lý trade entry_date={trade.entry_date}: {e}"
                )
                continue

        # Trường hợp không có trade hợp lệ nào
        if not features_list:
            return HardExampleSet(
                features=np.array([]).reshape(0, lookback, _FEATURE_DIM),
                labels=np.array([]),
                weights=np.array([]),
                source_trades=[],
            )

        # Stack thành arrays
        features = np.stack(features_list)  # (N, lookback, 61)
        labels = np.array(labels_list)
        weights = np.full(len(labels), self.weight_multiplier)

        # Cap tại max_hard_examples_ratio × total_training_samples
        max_count = int(self.max_hard_examples_ratio * total_training_samples)
        if max_count < 1:
            max_count = 1

        if len(labels) > max_count:
            # Giữ lại trades có pnl_pct thấp nhất (worst losses)
            pnl_values = [t.pnl_pct for t in source_trades]
            loss_order = np.argsort(pnl_values)  # Ascending = worst first
            indices = loss_order[:max_count]

            features = features[indices]
            labels = labels[indices]
            weights = weights[indices]
            source_trades = [source_trades[i] for i in indices]

        return HardExampleSet(
            features=features,
            labels=labels,
            weights=weights,
            source_trades=source_trades,
        )

    def compute_sample_weights(
        self,
        total_samples: int,
        hard_example_indices: List[int],
    ) -> np.ndarray:
        """Tính toán mảng sample weights cho toàn bộ training set.

        Hard examples nhận weight_multiplier, các samples còn lại nhận 1.0.

        Args:
            total_samples: Tổng số samples trong training set.
            hard_example_indices: Danh sách indices của hard examples.

        Returns:
            Mảng weights shape (total_samples,) với giá trị 1.0 hoặc weight_multiplier.
        """
        weights = np.ones(total_samples, dtype=np.float64)

        for idx in hard_example_indices:
            if 0 <= idx < total_samples:
                weights[idx] = self.weight_multiplier

        return weights

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Xác định danh sách feature columns (loại bỏ OHLCV).

        Args:
            df: DataFrame chứa tất cả columns.

        Returns:
            Danh sách tên feature columns.
        """
        feature_cols = [
            col for col in df.columns if col.lower() not in _OHLCV_COLUMNS
        ]
        return feature_cols

    def _extract_feature_window(
        self,
        trade: TradeRecord,
        feature_data: Dict[str, pd.DataFrame],
        feature_cols: List[str],
        lookback: int,
    ) -> np.ndarray | None:
        """Trích xuất cửa sổ feature lookback ngày kết thúc tại entry_date.

        Args:
            trade: Trade cần trích xuất features.
            feature_data: Dict symbol → DataFrame.
            feature_cols: Danh sách tên feature columns.
            lookback: Số ngày lookback.

        Returns:
            Feature window shape (lookback, 61) hoặc None nếu không đủ dữ liệu.
        """
        if trade.entry_date is None:
            return None

        # Lấy DataFrame đầu tiên
        first_symbol = next(iter(feature_data))
        df = feature_data[first_symbol]

        # Tìm vị trí entry_date trong DataFrame
        entry_idx = self._find_entry_index(df, trade.entry_date)
        if entry_idx is None:
            return None

        # Kiểm tra đủ lookback rows
        if entry_idx < lookback - 1:
            logger.debug(
                f"Không đủ {lookback} ngày history trước entry_date={trade.entry_date}, "
                f"chỉ có {entry_idx + 1} ngày"
            )
            return None

        # Trích xuất window: lookback rows kết thúc tại entry_idx (inclusive)
        start_idx = entry_idx - lookback + 1
        end_idx = entry_idx + 1  # exclusive
        window_df = df.iloc[start_idx:end_idx]

        # Lấy feature columns
        available_cols = [col for col in feature_cols if col in window_df.columns]
        window_features = window_df[available_cols].copy()

        # Đảm bảo đúng 61 feature columns
        window_features = self._ensure_feature_dim(window_features)

        # Forward-fill rồi zero-fill cho NaN
        window_features = window_features.ffill()
        window_features = window_features.fillna(0.0)

        # Chuyển sang numpy array
        feature_array = window_features.values.astype(np.float64)

        # Kiểm tra NaN/Inf sau khi fill (phòng trường hợp Inf còn tồn tại)
        if not np.isfinite(feature_array).all():
            logger.debug(
                f"Feature window chứa NaN/Inf sau fill, skip trade entry_date={trade.entry_date}"
            )
            return None

        return feature_array

    def _find_entry_index(
        self, df: pd.DataFrame, entry_date: pd.Timestamp
    ) -> int | None:
        """Tìm index của entry_date trong DataFrame.

        Hỗ trợ tìm kiếm cả khi DataFrame có column 'time' hoặc dùng index.

        Args:
            df: DataFrame cần tìm.
            entry_date: Ngày cần tìm.

        Returns:
            Integer index hoặc None nếu không tìm thấy.
        """
        # Chuẩn hoá entry_date thành Timestamp
        entry_ts = pd.Timestamp(entry_date)

        # Thử tìm trong column 'time' trước
        if "time" in df.columns:
            time_col = pd.to_datetime(df["time"])
            # Tìm exact match (normalize về ngày)
            matches = time_col.dt.normalize() == entry_ts.normalize()
            if matches.any():
                matched_positions = np.where(matches.values)[0]
                if len(matched_positions) > 0:
                    return int(matched_positions[-1])

            # Nếu không exact match, tìm ngày gần nhất <= entry_date
            before_mask = time_col.dt.normalize() <= entry_ts.normalize()
            if before_mask.any():
                positions = np.where(before_mask.values)[0]
                if len(positions) > 0:
                    return int(positions[-1])
            return None

        # Fallback: tìm trong index
        if isinstance(df.index, pd.DatetimeIndex):
            normalized_idx = df.index.normalize()
            matches = normalized_idx == entry_ts.normalize()
            if matches.any():
                positions = np.where(matches)[0]
                if len(positions) > 0:
                    return int(positions[-1])

            # Tìm ngày gần nhất <= entry_date
            before_mask = normalized_idx <= entry_ts.normalize()
            if before_mask.any():
                positions = np.where(before_mask)[0]
                if len(positions) > 0:
                    return int(positions[-1])

        return None

    def _ensure_feature_dim(self, df: pd.DataFrame) -> pd.DataFrame:
        """Đảm bảo DataFrame có đúng 61 feature columns.

        Nếu thiếu columns → pad thêm columns 0.
        Nếu thừa columns → truncate lấy 61 columns đầu.

        Args:
            df: DataFrame features.

        Returns:
            DataFrame với đúng 61 columns.
        """
        current_cols = len(df.columns)

        if current_cols == _FEATURE_DIM:
            return df

        if current_cols > _FEATURE_DIM:
            # Truncate: lấy 61 columns đầu
            return df.iloc[:, :_FEATURE_DIM]

        # Pad: thêm columns 0 cho đủ 61
        pad_count = _FEATURE_DIM - current_cols
        for i in range(pad_count):
            df[f"_pad_{i}"] = 0.0

        return df

    def _compute_corrected_label(self, trade: TradeRecord) -> float:
        """Tính corrected label dựa trên trade outcome.

        Logic:
        - pnl_pct < -5% và signal > 0 (BUY) → -0.8 (SELL)
        - pnl_pct < -5% và signal < 0 (SELL) → +0.8 (BUY)
        - Mặc định → 0.0 (HOLD)

        Args:
            trade: TradeRecord cần tính label.

        Returns:
            Corrected label trong [-1.0, 1.0].
        """
        if trade.pnl_pct < -0.05:
            if trade.signal_strength > 0:
                return -0.8  # Was BUY, should SELL
            elif trade.signal_strength < 0:
                return 0.8  # Was SELL, should BUY
        return 0.0  # Default: HOLD
