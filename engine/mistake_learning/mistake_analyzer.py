"""
MistakeAnalyzer — Phân loại trades và phát hiện cơ hội bị bỏ lỡ.

Phân loại mỗi trade thành GOOD hoặc BAD dựa trên PnL percentage,
phát hiện missed opportunities khi model giữ HOLD nhưng giá tăng mạnh,
và tạo MistakeReport tổng hợp cho mỗi feedback cycle.
"""

import logging
from typing import List, Optional, Set

import numpy as np
import pandas as pd

from engine.config import Trade
from engine.mistake_learning.config import MistakeLearningConfig
from engine.mistake_learning.models import (
    MissedOpportunity,
    MistakeReport,
    TradeClassification,
    TradeRecord,
)

logger = logging.getLogger(__name__)

# Số cột cơ bản OHLCV (time/open/high/low/close/volume)
_OHLCV_COLUMNS = {"time", "open", "high", "low", "close", "volume"}
_FEATURE_VECTOR_LENGTH = 61


class MistakeAnalyzer:
    """Phân tích trades để phân loại mistakes và phát hiện patterns.

    Phân loại trades thành GOOD/BAD dựa trên ngưỡng PnL,
    phát hiện missed opportunities qua hindsight analysis,
    và tạo report tổng hợp cho feedback loop.
    """

    BAD_TRADE_THRESHOLD: float = -0.05  # -5% PnL

    def __init__(self, config: Optional[MistakeLearningConfig] = None) -> None:
        """Khởi tạo MistakeAnalyzer với config tùy chọn.

        Args:
            config: Cấu hình cho mistake learning. Dùng default nếu None.
        """
        if config is None:
            config = MistakeLearningConfig()
        self._config = config
        self._bad_trade_threshold = config.bad_trade_threshold
        self._missed_opportunity_threshold = config.missed_opportunity_threshold
        self._lookforward_days = config.lookforward_days

    @property
    def config(self) -> MistakeLearningConfig:
        """Trả về config hiện tại."""
        return self._config

    def analyze_trades(
        self,
        trades: List[Trade],
        df: pd.DataFrame,
        signal_history: List[float],
    ) -> MistakeReport:
        """Phân loại tất cả trades, phát hiện missed opportunities, tạo MistakeReport.

        Args:
            trades: Danh sách trades từ backtest.
            df: DataFrame chứa OHLCV và indicator columns.
            signal_history: Lịch sử tín hiệu model (chưa sử dụng trực tiếp).

        Returns:
            MistakeReport tổng hợp kết quả phân tích.
        """
        # Trường hợp danh sách trades rỗng
        if not trades:
            return MistakeReport(
                cycle_number=0,
                total_trades=0,
                good_trades=[],
                bad_trades=[],
                missed_opportunities=[],
                bad_trade_ratio=0.0,
                avg_bad_loss=0.0,
            )

        good_trades: List[TradeRecord] = []
        bad_trades: List[TradeRecord] = []

        for trade in trades:
            classification = self.classify_trade(trade)
            record = self._enrich_trade(trade, df, classification)

            if classification == TradeClassification.BAD:
                bad_trades.append(record)
            else:
                good_trades.append(record)

        # Phát hiện missed opportunities
        missed = self.detect_missed_opportunities(
            df=df,
            trades=trades,
            lookforward=self._lookforward_days,
        )

        total_trades = len(trades)
        bad_trade_ratio = len(bad_trades) / max(total_trades, 1)
        avg_bad_loss = (
            float(np.mean([t.pnl_pct for t in bad_trades]))
            if bad_trades
            else 0.0
        )

        return MistakeReport(
            cycle_number=0,
            total_trades=total_trades,
            good_trades=good_trades,
            bad_trades=bad_trades,
            missed_opportunities=missed,
            bad_trade_ratio=bad_trade_ratio,
            avg_bad_loss=avg_bad_loss,
        )

    def classify_trade(self, trade: Trade) -> TradeClassification:
        """Phân loại trade dựa trên PnL percentage.

        BAD nếu pnl_pct < bad_trade_threshold (default -5%).
        GOOD trong mọi trường hợp còn lại.

        Args:
            trade: Trade object từ backtest (có attribute pnl_pct).

        Returns:
            TradeClassification.BAD hoặc TradeClassification.GOOD.
        """
        if trade.pnl_pct < self._bad_trade_threshold:
            return TradeClassification.BAD
        return TradeClassification.GOOD

    def detect_missed_opportunities(
        self,
        df: pd.DataFrame,
        trades: List[Trade],
        lookforward: int = 5,
    ) -> List[MissedOpportunity]:
        """Phát hiện các cơ hội bị bỏ lỡ khi model giữ HOLD.

        Scan qua các ngày mà model không có trade active,
        kiểm tra nếu max gain từ close tại ngày đó đến highest close
        trong lookforward ngày tiếp theo >= missed_opportunity_threshold.

        Args:
            df: DataFrame chứa cột 'close' và 'time' (hoặc index dạng datetime).
            trades: Danh sách trades đã thực hiện.
            lookforward: Số ngày nhìn trước (default 5).

        Returns:
            Danh sách MissedOpportunity.
        """
        if df.empty or lookforward <= 0:
            return []

        # Lấy cột close
        close_col = self._get_close_series(df)
        if close_col is None or len(close_col) == 0:
            return []

        # Xác định các ngày đang có trade active
        active_dates = self._get_active_trade_dates(trades, df)

        # Scan từng ngày không có trade active
        missed: List[MissedOpportunity] = []
        dates = close_col.index

        for i in range(len(dates) - 1):
            date = dates[i]

            # Bỏ qua ngày đang có trade
            if date in active_dates:
                continue

            close_at_date = close_col.iloc[i]
            if close_at_date <= 0:
                continue

            # Lấy window lookforward ngày tiếp theo
            end_idx = min(i + 1 + lookforward, len(close_col))
            if end_idx <= i + 1:
                continue

            future_window = close_col.iloc[i + 1: end_idx]
            if future_window.empty:
                continue

            max_close_in_window = future_window.max()
            potential_gain_pct = (max_close_in_window - close_at_date) / close_at_date

            if potential_gain_pct >= self._missed_opportunity_threshold:
                # Trích xuất feature vector tại ngày này
                feature_vector = self._extract_feature_vector_at_index(df, i)

                missed.append(
                    MissedOpportunity(
                        date=pd.Timestamp(date),
                        actual_action="HOLD",
                        optimal_action="BUY",
                        potential_gain_pct=float(potential_gain_pct),
                        feature_vector=feature_vector,
                    )
                )

        return missed

    # =========================================================================
    # Private methods
    # =========================================================================

    def _enrich_trade(
        self,
        trade: Trade,
        df: pd.DataFrame,
        classification: TradeClassification,
    ) -> TradeRecord:
        """Chuyển Trade thành TradeRecord, thêm feature vector và market context.

        Trích xuất feature_vector (61 features) từ df tại entry_date,
        cùng market_volatility, market_trend, volume_ratio.

        Args:
            trade: Trade gốc từ backtest.
            df: DataFrame chứa OHLCV + indicators.
            classification: Kết quả phân loại (GOOD/BAD).

        Returns:
            TradeRecord đã enriched.
        """
        feature_vector = None
        market_volatility = 0.0
        market_trend = 0.0
        volume_ratio = 0.0

        if trade.entry_date is not None and not df.empty:
            # Tìm index của entry_date trong df
            idx = self._find_date_index(trade.entry_date, df)
            if idx is not None:
                feature_vector = self._extract_feature_vector_at_index(df, idx)
                market_volatility = self._compute_market_volatility(df, idx)
                market_trend = self._compute_market_trend(df, idx)
                volume_ratio = self._compute_volume_ratio(df, idx)

        # Xác định action từ trade
        action_taken = "BUY"
        if hasattr(trade, "action"):
            action_taken = str(trade.action) if trade.action else "BUY"

        # Tạo classification string
        classification_str = classification.value

        return TradeRecord(
            entry_date=trade.entry_date,
            exit_date=trade.exit_date,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            shares=trade.shares,
            pnl=trade.pnl,
            pnl_pct=trade.pnl_pct,
            cycle_number=0,
            signal_strength=0.0,
            action_taken=action_taken,
            classification=classification_str,
            feature_vector=feature_vector,
            market_volatility=market_volatility,
            market_trend=market_trend,
            volume_ratio=volume_ratio,
        )

    def _extract_feature_vector_at_index(
        self, df: pd.DataFrame, idx: int
    ) -> Optional[List[float]]:
        """Trích xuất feature vector (61 features) tại index cho trước.

        Lấy tất cả cột sau OHLCV cơ bản làm features.
        Nếu không đủ 61 features, pad zeros. Nếu thừa, cắt bớt.

        Args:
            df: DataFrame nguồn.
            idx: Index vị trí trong DataFrame.

        Returns:
            List 61 float values hoặc None nếu không extract được.
        """
        if idx < 0 or idx >= len(df):
            return None

        row = df.iloc[idx]

        # Lấy các cột không phải OHLCV cơ bản
        feature_cols = [
            col for col in df.columns
            if col.lower() not in _OHLCV_COLUMNS
        ]

        if not feature_cols:
            return None

        # Trích xuất giá trị
        values = []
        for col in feature_cols:
            val = row[col]
            if pd.isna(val):
                values.append(0.0)
            else:
                values.append(float(val))

        # Đảm bảo đúng 61 phần tử
        if len(values) >= _FEATURE_VECTOR_LENGTH:
            return values[:_FEATURE_VECTOR_LENGTH]
        else:
            # Pad zeros nếu thiếu
            values.extend([0.0] * (_FEATURE_VECTOR_LENGTH - len(values)))
            return values

    def _find_date_index(
        self, target_date: pd.Timestamp, df: pd.DataFrame
    ) -> Optional[int]:
        """Tìm index của ngày trong DataFrame.

        Tìm theo cột 'time' hoặc index của df.

        Args:
            target_date: Ngày cần tìm.
            df: DataFrame nguồn.

        Returns:
            Index (integer) hoặc None nếu không tìm thấy.
        """
        if df.empty:
            return None

        # Thử tìm trong cột 'time' trước
        if "time" in df.columns:
            time_col = pd.to_datetime(df["time"], errors="coerce")
            target_normalized = pd.Timestamp(target_date).normalize()
            matches = time_col.dt.normalize() == target_normalized
            if matches.any():
                return int(matches.idxmax()) if isinstance(df.index, pd.RangeIndex) else int(
                    df.index.get_loc(matches.idxmax())
                )

        # Thử tìm trong index nếu index là DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            target_normalized = pd.Timestamp(target_date).normalize()
            normalized_index = df.index.normalize()
            matches = normalized_index == target_normalized
            if matches.any():
                return int(np.where(matches)[0][0])

        return None

    def _get_close_series(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Lấy series close price từ DataFrame.

        Returns:
            Series close prices hoặc None.
        """
        if "close" in df.columns:
            return df["close"].reset_index(drop=True)
        elif "Close" in df.columns:
            return df["Close"].reset_index(drop=True)
        return None

    def _get_active_trade_dates(
        self, trades: List[Trade], df: pd.DataFrame
    ) -> Set[int]:
        """Xác định set các index ngày đang có trade active.

        Trade active từ entry_date đến exit_date.

        Args:
            trades: Danh sách trades.
            df: DataFrame nguồn để map dates → indices.

        Returns:
            Set các integer index đang có trade.
        """
        active_indices: Set[int] = set()

        close_series = self._get_close_series(df)
        if close_series is None:
            return active_indices

        num_rows = len(close_series)

        for trade in trades:
            entry_idx = None
            exit_idx = None

            if trade.entry_date is not None:
                entry_idx = self._find_date_index_in_reset_df(trade.entry_date, df)
            if trade.exit_date is not None:
                exit_idx = self._find_date_index_in_reset_df(trade.exit_date, df)

            if entry_idx is not None and exit_idx is not None:
                for i in range(entry_idx, min(exit_idx + 1, num_rows)):
                    active_indices.add(i)
            elif entry_idx is not None:
                active_indices.add(entry_idx)

        return active_indices

    def _find_date_index_in_reset_df(
        self, target_date: pd.Timestamp, df: pd.DataFrame
    ) -> Optional[int]:
        """Tìm index (positional) của ngày trong DataFrame sau reset_index.

        Args:
            target_date: Ngày cần tìm.
            df: DataFrame gốc.

        Returns:
            Integer positional index hoặc None.
        """
        if df.empty or target_date is None:
            return None

        target_normalized = pd.Timestamp(target_date).normalize()

        # Tìm trong cột 'time'
        if "time" in df.columns:
            time_col = pd.to_datetime(df["time"], errors="coerce")
            for i, t in enumerate(time_col):
                if pd.notna(t) and t.normalize() == target_normalized:
                    return i

        # Tìm trong index nếu là DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            normalized_index = df.index.normalize()
            matches = normalized_index == target_normalized
            if matches.any():
                return int(np.where(matches)[0][0])

        return None

    def _compute_market_volatility(self, df: pd.DataFrame, idx: int) -> float:
        """Tính market volatility tại index (std của returns 20 ngày gần nhất).

        Args:
            df: DataFrame chứa cột close.
            idx: Index hiện tại.

        Returns:
            Volatility (std of daily returns) hoặc 0.0.
        """
        if "close" not in df.columns and "Close" not in df.columns:
            return 0.0

        close_col = "close" if "close" in df.columns else "Close"
        start_idx = max(0, idx - 20)
        close_window = df[close_col].iloc[start_idx: idx + 1]

        if len(close_window) < 2:
            return 0.0

        returns = close_window.pct_change().dropna()
        if returns.empty:
            return 0.0

        return float(returns.std())

    def _compute_market_trend(self, df: pd.DataFrame, idx: int) -> float:
        """Tính market trend tại index (% thay đổi close 20 ngày).

        Dương = uptrend, âm = downtrend.

        Args:
            df: DataFrame chứa cột close.
            idx: Index hiện tại.

        Returns:
            Trend value hoặc 0.0.
        """
        if "close" not in df.columns and "Close" not in df.columns:
            return 0.0

        close_col = "close" if "close" in df.columns else "Close"
        lookback = 20
        start_idx = max(0, idx - lookback)

        if start_idx >= idx:
            return 0.0

        close_start = df[close_col].iloc[start_idx]
        close_end = df[close_col].iloc[idx]

        if close_start <= 0:
            return 0.0

        return float((close_end - close_start) / close_start)

    def _compute_volume_ratio(self, df: pd.DataFrame, idx: int) -> float:
        """Tính volume ratio (volume hiện tại / trung bình 20 ngày).

        Args:
            df: DataFrame chứa cột volume.
            idx: Index hiện tại.

        Returns:
            Volume ratio hoặc 0.0.
        """
        if "volume" not in df.columns and "Volume" not in df.columns:
            return 0.0

        vol_col = "volume" if "volume" in df.columns else "Volume"
        lookback = 20
        start_idx = max(0, idx - lookback)

        avg_volume = df[vol_col].iloc[start_idx: idx + 1].mean()
        current_volume = df[vol_col].iloc[idx]

        if avg_volume <= 0 or pd.isna(avg_volume):
            return 0.0

        return float(current_volume / avg_volume)
