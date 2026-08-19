"""
TradeHistoryStore — Persistent storage cho trade history qua các cycles.

Lưu trữ dưới dạng JSONL (append-only), hỗ trợ query theo cycle,
filter bad trades, và đếm tổng records. Xử lý graceful khi file
không tồn tại hoặc bị corrupt.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.mistake_learning.models import TradeRecord

logger = logging.getLogger(__name__)


class TradeHistoryStore:
    """Persistent store cho trade history across training cycles.

    Dữ liệu lưu dưới dạng JSONL (một JSON object mỗi dòng).
    Hỗ trợ append-only write và các query operations.
    """

    HISTORY_FILE = "engine/models/trade_history.jsonl"

    def __init__(self, base_dir: str = ".") -> None:
        """Khởi tạo store với base directory.

        Args:
            base_dir: Thư mục gốc, HISTORY_FILE sẽ relative to đây.
        """
        self._base_dir = base_dir
        self._file_path = os.path.join(base_dir, self.HISTORY_FILE)

    @property
    def file_path(self) -> str:
        """Đường dẫn đầy đủ tới file JSONL."""
        return self._file_path

    def persist_trades(self, cycle_number: int, trades: List[TradeRecord]) -> None:
        """Ghi danh sách TradeRecord vào file JSONL (append mode).

        Tạo file và thư mục cha nếu chưa tồn tại.

        Args:
            cycle_number: Số cycle hiện tại.
            trades: Danh sách TradeRecord cần lưu.
        """
        # Tạo thư mục cha nếu chưa tồn tại
        dir_path = os.path.dirname(self._file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(self._file_path, "a", encoding="utf-8") as f:
            for trade in trades:
                record_dict = self._serialize_trade(trade, cycle_number)
                line = json.dumps(record_dict, ensure_ascii=False)
                f.write(line + "\n")

    def load_all(self) -> List[TradeRecord]:
        """Load tất cả records, sắp xếp theo entry_date tăng dần.

        Returns:
            Danh sách TradeRecord sorted by entry_date.
            Trả về list rỗng nếu file không tồn tại hoặc bị lỗi.
        """
        records = self._load_raw_records()
        # Sắp xếp theo entry_date (None đặt cuối)
        records.sort(key=lambda r: r.entry_date if r.entry_date is not None else pd.Timestamp.max)
        return records

    def load_by_cycle(self, cycle_number: int) -> List[TradeRecord]:
        """Load records theo cycle number.

        Args:
            cycle_number: Số cycle cần filter.

        Returns:
            Danh sách TradeRecord thuộc cycle đó.
        """
        records = self._load_raw_records()
        return [r for r in records if r.cycle_number == cycle_number]

    def load_bad_trades(self, min_loss_pct: float = -5.0) -> List[TradeRecord]:
        """Load các trades có pnl_pct nhỏ hơn ngưỡng (bad trades).

        Args:
            min_loss_pct: Ngưỡng PnL phần trăm (dạng percentage, vd: -5.0 nghĩa là -5%).
                         Records có pnl_pct < min_loss_pct/100 sẽ được trả về.

        Returns:
            Danh sách bad trades.
        """
        records = self._load_raw_records()
        threshold = min_loss_pct / 100.0
        return [r for r in records if r.pnl_pct < threshold]

    def get_trade_count(self) -> int:
        """Đếm tổng số records trong file.

        Returns:
            Số lượng records. Trả về 0 nếu file không tồn tại.
        """
        records = self._load_raw_records()
        return len(records)

    # =========================================================================
    # Private methods
    # =========================================================================

    def _load_raw_records(self) -> List[TradeRecord]:
        """Đọc tất cả records từ JSONL file.

        Xử lý graceful:
        - File không tồn tại → trả về list rỗng
        - Dòng JSON bị corrupt → skip dòng đó, log warning
        - Lỗi đọc file → trả về list rỗng, log warning

        Returns:
            Danh sách TradeRecord đã deserialize.
        """
        if not os.path.exists(self._file_path):
            return []

        records: List[TradeRecord] = []
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        record = self._deserialize_trade(data)
                        records.append(record)
                    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                        logger.warning(
                            "Dòng %d trong %s bị corrupt, bỏ qua: %s",
                            line_num,
                            self._file_path,
                            str(e),
                        )
                        continue
        except (IOError, OSError, PermissionError) as e:
            logger.warning(
                "Không thể đọc file %s, trả về list rỗng: %s",
                self._file_path,
                str(e),
            )
            return []

        return records

    def _serialize_trade(self, trade: TradeRecord, cycle_number: int) -> Dict[str, Any]:
        """Chuyển TradeRecord thành dict để serialize JSON.

        Xử lý:
        - pd.Timestamp → ISO format string
        - numpy types (np.float64, np.int64) → native Python types
        - feature_vector → list of floats
        """
        return {
            "entry_date": self._serialize_timestamp(trade.entry_date),
            "exit_date": self._serialize_timestamp(trade.exit_date),
            "entry_price": self._to_native(trade.entry_price),
            "exit_price": self._to_native(trade.exit_price),
            "shares": self._to_native(trade.shares),
            "pnl": self._to_native(trade.pnl),
            "pnl_pct": self._to_native(trade.pnl_pct),
            "cycle_number": self._to_native(cycle_number),
            "signal_strength": self._to_native(trade.signal_strength),
            "action_taken": trade.action_taken,
            "classification": trade.classification,
            "feature_vector": self._serialize_feature_vector(trade.feature_vector),
            "market_volatility": self._to_native(trade.market_volatility),
            "market_trend": self._to_native(trade.market_trend),
            "volume_ratio": self._to_native(trade.volume_ratio),
        }

    def _deserialize_trade(self, data: Dict[str, Any]) -> TradeRecord:
        """Chuyển dict từ JSON thành TradeRecord.

        Xử lý:
        - ISO format string → pd.Timestamp
        - list → feature_vector (list of floats)
        """
        return TradeRecord(
            entry_date=self._deserialize_timestamp(data.get("entry_date")),
            exit_date=self._deserialize_timestamp(data.get("exit_date")),
            entry_price=float(data.get("entry_price", 0.0)),
            exit_price=float(data.get("exit_price", 0.0)),
            shares=int(data.get("shares", 0)),
            pnl=float(data.get("pnl", 0.0)),
            pnl_pct=float(data.get("pnl_pct", 0.0)),
            cycle_number=int(data.get("cycle_number", 0)),
            signal_strength=float(data.get("signal_strength", 0.0)),
            action_taken=str(data.get("action_taken", "HOLD")),
            classification=str(data.get("classification", "UNKNOWN")),
            feature_vector=self._deserialize_feature_vector(data.get("feature_vector")),
            market_volatility=float(data.get("market_volatility", 0.0)),
            market_trend=float(data.get("market_trend", 0.0)),
            volume_ratio=float(data.get("volume_ratio", 0.0)),
        )

    @staticmethod
    def _serialize_timestamp(ts: Optional[pd.Timestamp]) -> Optional[str]:
        """Chuyển pd.Timestamp thành ISO format string."""
        if ts is None:
            return None
        if isinstance(ts, pd.Timestamp):
            return ts.isoformat()
        return str(ts)

    @staticmethod
    def _deserialize_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
        """Chuyển ISO format string thành pd.Timestamp."""
        if value is None:
            return None
        try:
            return pd.Timestamp(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _to_native(value: Any) -> Any:
        """Chuyển numpy types thành native Python types."""
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    @staticmethod
    def _serialize_feature_vector(fv: Optional[List[float]]) -> Optional[List[float]]:
        """Serialize feature_vector thành list of native floats."""
        if fv is None:
            return None
        return [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in fv]

    @staticmethod
    def _deserialize_feature_vector(value: Any) -> Optional[List[float]]:
        """Deserialize feature_vector từ JSON."""
        if value is None:
            return None
        if isinstance(value, list):
            return [float(x) for x in value]
        return None
