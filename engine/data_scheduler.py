"""
DataScheduler - Quản lý lịch tự động cập nhật dữ liệu thị trường.

Lên lịch auto-update data sau khi thị trường đóng cửa (mặc định 15:30).
Khi đến giờ, trigger DataPipeline.update_all() và gọi callback
để refresh recommendations.

Thiết kế đơn giản (polling-based) phù hợp với Streamlit:
- Không dùng threading/async scheduler
- Streamlit gọi auto_update_at_time() mỗi lần render để kiểm tra

References:
- Req 8.3: Scheduled auto-update
- Req 8.5: Trigger recommendation refresh sau data update
- Req 8.7: Configurable update time
"""

import re
from datetime import datetime
from typing import Callable, Optional

from config.settings import DEFAULT_UPDATE_TIME
from engine.data_pipeline import DataPipeline
from models.data_models import UpdateResult


class DataScheduler:
    """
    Scheduler tự động cập nhật dữ liệu thị trường.

    Polling-based: Streamlit gọi auto_update_at_time() mỗi lần render.
    Nếu thời gian hiện tại nằm trong khoảng ±1 phút so với scheduled time
    và chưa update hôm nay → chạy update.

    Attributes:
        data_pipeline: DataPipeline instance để thực hiện update
        _update_time: Giờ scheduled update (HH:MM format)
        _on_update_complete: Callback sau khi update xong
        _last_update_date: Ngày cuối cùng đã chạy update (tránh chạy lại)
    """

    def __init__(
        self,
        data_pipeline: DataPipeline,
        update_time: str = DEFAULT_UPDATE_TIME,
    ) -> None:
        """
        Khởi tạo DataScheduler.

        Args:
            data_pipeline: DataPipeline instance để fetch data
            update_time: Giờ auto-update dạng HH:MM (mặc định: "15:30")

        Raises:
            ValueError: Nếu update_time không đúng format HH:MM
        """
        self.data_pipeline: DataPipeline = data_pipeline
        self._on_update_complete: Optional[Callable[[UpdateResult], None]] = None
        self._last_update_date: Optional[str] = None

        # Validate và set update time
        self._update_time: str = ""
        self.set_update_time(update_time)

    def schedule_update(self, update_time: Optional[str] = None) -> None:
        """
        Đặt lịch update tiếp theo.

        Nếu không truyền update_time, dùng giá trị hiện tại.
        Reset _last_update_date để cho phép chạy update lần tiếp theo.

        Args:
            update_time: Giờ update mới dạng HH:MM, hoặc None để giữ nguyên

        Raises:
            ValueError: Nếu update_time không đúng format HH:MM
        """
        if update_time is not None:
            self.set_update_time(update_time)

        # Reset last update date để cho phép chạy update lại
        self._last_update_date = None

    def auto_update_at_time(self) -> Optional[UpdateResult]:
        """
        Kiểm tra và chạy update nếu đúng giờ.

        Gọi hàm này mỗi lần Streamlit render. Nếu should_update_now()
        trả về True → chạy run_update() và trả về kết quả.

        Returns:
            UpdateResult nếu đã chạy update, None nếu chưa đến giờ
        """
        if self.should_update_now():
            return self.run_update()
        return None

    def should_update_now(self) -> bool:
        """
        Kiểm tra có cần chạy update không.

        Điều kiện:
        1. Chưa chạy update hôm nay (tránh chạy nhiều lần)
        2. Thời gian hiện tại >= scheduled time (đã qua giờ update)
           Hỗ trợ cả trường hợp app bật sau giờ scheduled (missed update).

        Returns:
            True nếu cần chạy update ngay, False nếu không
        """
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        # Đã update hôm nay rồi → skip
        if self._last_update_date == today_str:
            return False

        # Parse scheduled time
        hour, minute = self._parse_time(self._update_time)
        scheduled_minutes = hour * 60 + minute
        current_minutes = now.hour * 60 + now.minute

        # Chạy update nếu thời gian hiện tại đã qua giờ scheduled
        # (hỗ trợ cả app bật muộn — missed update)
        return current_minutes >= scheduled_minutes

    def run_update(self) -> UpdateResult:
        """
        Thực hiện data update qua DataPipeline.

        Gọi data_pipeline.update_all(), đánh dấu đã update hôm nay,
        và trigger callback nếu có.

        Returns:
            UpdateResult chứa thống kê success/failure
        """
        result = self.data_pipeline.update_all()

        # Đánh dấu đã update hôm nay
        self._last_update_date = datetime.now().strftime("%Y-%m-%d")

        # Trigger callback (recommendation refresh)
        if self._on_update_complete is not None:
            self._on_update_complete(result)

        return result

    def set_on_update_complete(
        self, callback: Callable[[UpdateResult], None]
    ) -> None:
        """
        Đăng ký callback chạy sau khi update hoàn tất.

        Callback nhận UpdateResult để có thể trigger recommendation refresh
        hoặc bất kỳ action nào sau data update.

        Args:
            callback: Hàm callback(UpdateResult) → None
        """
        self._on_update_complete = callback

    def get_next_update_time(self) -> str:
        """
        Trả về giờ update đã lên lịch.

        Returns:
            Thời gian dạng HH:MM (ví dụ: "15:30")
        """
        return self._update_time

    def set_update_time(self, time_str: str) -> None:
        """
        Thay đổi giờ update.

        Args:
            time_str: Giờ update mới dạng HH:MM (00:00 - 23:59)

        Raises:
            ValueError: Nếu time_str không đúng format HH:MM hoặc giá trị không hợp lệ
        """
        # Validate format HH:MM
        if not re.match(r"^\d{2}:\d{2}$", time_str):
            raise ValueError(
                f"Invalid time format: '{time_str}'. Expected HH:MM (e.g., '15:30')"
            )

        hour, minute = self._parse_time(time_str)

        if not (0 <= hour <= 23):
            raise ValueError(
                f"Invalid hour: {hour}. Must be between 00 and 23"
            )
        if not (0 <= minute <= 59):
            raise ValueError(
                f"Invalid minute: {minute}. Must be between 00 and 59"
            )

        self._update_time = time_str

    @staticmethod
    def _parse_time(time_str: str) -> tuple:
        """
        Parse chuỗi HH:MM thành tuple (hour, minute).

        Args:
            time_str: Chuỗi thời gian dạng HH:MM

        Returns:
            Tuple (hour: int, minute: int)
        """
        parts = time_str.split(":")
        return int(parts[0]), int(parts[1])
