# -*- coding: utf-8 -*-
"""
Data Update Wiring - Kết nối DataPipeline với RecommendationEngine.

Module này đảm nhận:
1. Sau khi DataPipeline.update_all() hoàn tất → trigger RecommendationEngine.refresh()
2. DataScheduler.auto_update_at_time() → data fetch → recommendations update
3. Lưu last_update_timestamp ra file JSON để Dashboard hiển thị

Đảm bảo partial failures vẫn trigger recommendation refresh (Req 8.3):
ngay cả khi một số symbols fetch fail, recommendations vẫn được refresh.

References:
- Req 8.3: Data update xong → recommendation refresh (kể cả partial failure)
- Req 8.5: Last update timestamp hiển thị trên Dashboard
- Req 6.7: Recommendations refresh mỗi Trading_Day sau data update
"""

import json
import os
import tempfile
from datetime import datetime
from typing import Optional

from engine.data_pipeline import DataPipeline
from engine.data_scheduler import DataScheduler
from engine.recommendation_engine import RecommendationEngine
from models.data_models import UpdateResult


# Đường dẫn file lưu timestamp update cuối cùng
LAST_UPDATE_STATUS_PATH: str = "data/engine/status/last_data_update.json"


def save_last_update_timestamp(
    result: UpdateResult,
    path: str = LAST_UPDATE_STATUS_PATH,
    override_timestamp: Optional[datetime] = None,
    trading_weekday: Optional[int] = None,
    slot: Optional[str] = None,
) -> None:
    """
    Lưu kết quả update cuối cùng ra file JSON.

    Lưu: ngày giao dịch (weekday T2-T6), trạng thái slot (M/N/P),
    và thống kê update.

    Args:
        result: Kết quả từ DataPipeline.update_all()
        path: Đường dẫn file lưu (mặc định: LAST_UPDATE_STATUS_PATH)
        override_timestamp: Timestamp ghi nhận (dùng cho log). Mặc định = now().
        trading_weekday: Ngày giao dịch (0=T2, 4=T6). Nếu None → tự tính.
        slot: Trạng thái M/N/P. Nếu None → tự tính từ giờ hiện tại.
    """
    now = override_timestamp if override_timestamp is not None else datetime.now()

    # Tự tính trading_weekday nếu không truyền vào
    if trading_weekday is None:
        wd = now.weekday()
        if wd >= 5:
            # T7/CN → coi = thứ 6 (weekday=4)
            trading_weekday = 4
        else:
            trading_weekday = wd

    # Tự tính slot nếu không truyền vào — dựa trên thời điểm BẮT ĐẦU update
    if slot is None:
        h, m = now.hour, now.minute
        if h < 9 or (h == 9 and m < 15):
            slot = "X"  # Trước 9:15
        elif (h == 9 and m >= 15) or (h == 10) or (h == 11 and m < 30):
            slot = "M"  # 9:15 - 11:30
        elif (h == 11 and m >= 30) or (h >= 12 and h < 15):
            slot = "N"  # 11:30 - 15:00
        else:
            slot = "P"  # >= 15:00
        # Nếu T7/CN → luôn lưu slot P (đã hoàn tất thứ 6)
        if now.weekday() >= 5:
            slot = "P"

    data = {
        "trading_weekday": trading_weekday,  # 0=T2, 4=T6
        "slot": slot,  # M, N, P
        "timestamp": now.isoformat(),  # Để log/debug
        "total_symbols": result.total_symbols,
        "success_count": result.success_count,
        "failed_count": len(result.failed_symbols),
        "failed_symbols": result.failed_symbols,
        "duration_seconds": result.duration_seconds,
    }

    # Tạo thư mục nếu chưa có
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # Atomic write
    json_content = json.dumps(data, ensure_ascii=False, indent=2)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix="last_update_",
        dir=dir_path if dir_path else ".",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json_content)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_last_update_timestamp(
    path: str = LAST_UPDATE_STATUS_PATH,
) -> Optional[dict]:
    """
    Đọc thông tin update cuối cùng từ file JSON.

    Args:
        path: Đường dẫn file (mặc định: LAST_UPDATE_STATUS_PATH)

    Returns:
        Dictionary chứa timestamp, thống kê, hoặc None nếu chưa có update.
        Keys: timestamp, total_symbols, success_count, failed_count,
              failed_symbols, duration_seconds
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except (json.JSONDecodeError, OSError):
        return None


def run_data_update_with_refresh(
    pipeline: DataPipeline,
    recommendation_engine: RecommendationEngine,
    status_path: str = LAST_UPDATE_STATUS_PATH,
) -> UpdateResult:
    """
    Chạy data update và tự động refresh recommendations sau khi hoàn tất.

    Flow:
    1. DataPipeline.update_all() — fetch data cho tất cả tracked symbols
    2. Lưu last_update_timestamp (kể cả khi có partial failure)
    3. RecommendationEngine.refresh() — re-generate recommendations với data mới

    Req 8.3: Kể cả khi có symbol fetch thất bại, recommendations
    vẫn được refresh với data hiện có.

    Args:
        pipeline: DataPipeline instance
        recommendation_engine: RecommendationEngine instance
        status_path: Đường dẫn file lưu timestamp

    Returns:
        UpdateResult chứa thống kê success/failure
    """
    # Bước 1: Fetch data
    result = pipeline.update_all()

    # Bước 2: Lưu timestamp (luôn lưu, kể cả partial failure)
    save_last_update_timestamp(result, path=status_path)

    # Bước 3: Refresh recommendations (Req 8.3 — ngay cả khi có failures)
    # Partial failure không ngăn recommendation refresh
    recommendation_engine.refresh()

    return result


def wire_scheduler_to_recommendation(
    scheduler: DataScheduler,
    recommendation_engine: RecommendationEngine,
    status_path: str = LAST_UPDATE_STATUS_PATH,
) -> None:
    """
    Wire DataScheduler callback để sau mỗi auto-update sẽ refresh recommendations.

    Kết nối:
    - DataScheduler.auto_update_at_time() hoàn tất
    - → Lưu last_update_timestamp
    - → RecommendationEngine.refresh()

    Gọi hàm này khi khởi tạo hệ thống để thiết lập callback.

    Args:
        scheduler: DataScheduler instance
        recommendation_engine: RecommendationEngine instance
        status_path: Đường dẫn file lưu timestamp
    """

    def _on_update_complete(result: UpdateResult) -> None:
        """Callback: sau data update → save timestamp + refresh recommendations."""
        # Lưu timestamp
        save_last_update_timestamp(result, path=status_path)
        # Refresh recommendations với data mới (Req 6.7)
        recommendation_engine.refresh()

    scheduler.set_on_update_complete(_on_update_complete)
