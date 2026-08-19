"""
Rate Limiter dùng thuật toán Token Bucket cho API requests.

Giới hạn số lượng request trong sliding window để tuân thủ
rate limit của API (vnstock). Mặc định: 20 requests / 60 giây.

Validates: Requirements 8.6, 8.7, 8.8
"""

import math
import time
from collections import deque
from typing import Deque

from config.settings import RATE_LIMIT, RATE_LIMIT_WINDOW


class TokenBucketRateLimiter:
    """
    Rate limiter sử dụng sliding window với token bucket.

    Theo dõi timestamps của các request gần đây trong một deque.
    Cho phép tối đa max_tokens requests trong mỗi window_seconds giây.
    """

    def __init__(
        self,
        max_tokens: int = RATE_LIMIT,
        window_seconds: float = RATE_LIMIT_WINDOW,
    ) -> None:
        """
        Khởi tạo rate limiter.

        Args:
            max_tokens: Số request tối đa trong window (mặc định: 20)
            window_seconds: Kích thước sliding window tính bằng giây (mặc định: 60)
        """
        self.max_tokens: int = max_tokens
        self.window_seconds: float = window_seconds
        self._timestamps: Deque[float] = deque()

    def _cleanup_expired(self, now: float) -> None:
        """Xóa các timestamps đã hết hạn (ngoài sliding window)."""
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def acquire(self) -> bool:
        """
        Thử lấy một token. Trả về True nếu được phép, False nếu bị rate limit.

        Returns:
            True nếu request được chấp nhận, False nếu đã đạt giới hạn.
        """
        now = time.time()
        self._cleanup_expired(now)

        if len(self._timestamps) < self.max_tokens:
            self._timestamps.append(now)
            return True

        return False

    def wait_for_token(self) -> float:
        """
        Block cho đến khi có token khả dụng, trả về thời gian đã chờ (giây).

        Returns:
            Thời gian chờ tính bằng giây. 0.0 nếu token khả dụng ngay.
        """
        if self.acquire():
            return 0.0

        # Tính thời gian chờ: thời điểm request cũ nhất hết hạn
        now = time.time()
        oldest = self._timestamps[0]
        wait_time = (oldest + self.window_seconds) - now

        if wait_time > 0:
            time.sleep(wait_time)

        # Sau khi chờ, acquire lại
        # Đảm bảo acquire thành công sau khi đã sleep đủ
        acquired = self.acquire()
        if not acquired:
            # Trường hợp hiếm: retry thêm một lần nhỏ
            time.sleep(0.01)
            self.acquire()

        return wait_time if wait_time > 0 else 0.0

    def get_remaining_tokens(self) -> int:
        """
        Trả về số tokens còn khả dụng tại thời điểm hiện tại.

        Returns:
            Số tokens còn lại (0 đến max_tokens).
        """
        now = time.time()
        self._cleanup_expired(now)
        return max(0, self.max_tokens - len(self._timestamps))

    def estimate_duration(self, num_requests: int) -> float:
        """
        Ước tính tổng thời gian cần thiết cho N requests (tính bằng phút).

        Công thức: ceil(num_requests / max_tokens) phút.

        Args:
            num_requests: Số lượng requests cần thực hiện.

        Returns:
            Thời gian ước tính tính bằng phút.
        """
        if num_requests <= 0:
            return 0.0
        return float(math.ceil(num_requests / self.max_tokens))

    def reset(self) -> None:
        """Xóa toàn bộ timestamps đã theo dõi (dùng cho testing)."""
        self._timestamps.clear()
