"""
Unit tests cho TokenBucketRateLimiter.

Kiểm tra các chức năng chính: acquire, wait_for_token,
get_remaining_tokens, estimate_duration, và reset.
"""

import time
from unittest.mock import patch

import pytest

from engine.rate_limiter import TokenBucketRateLimiter


class TestAcquireUnderLimit:
    """Test acquire() thành công khi chưa đạt giới hạn."""

    def test_acquire_first_request_succeeds(self) -> None:
        """Request đầu tiên luôn thành công."""
        limiter = TokenBucketRateLimiter(max_tokens=20, window_seconds=60.0)
        assert limiter.acquire() is True

    def test_acquire_multiple_under_limit(self) -> None:
        """Nhiều requests dưới giới hạn đều thành công."""
        limiter = TokenBucketRateLimiter(max_tokens=5, window_seconds=60.0)
        for _ in range(5):
            assert limiter.acquire() is True

    def test_acquire_exactly_at_limit(self) -> None:
        """Request thứ max_tokens vẫn thành công."""
        limiter = TokenBucketRateLimiter(max_tokens=3, window_seconds=60.0)
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is True


class TestAcquireAtLimit:
    """Test acquire() thất bại khi đạt giới hạn."""

    def test_acquire_exceeds_limit(self) -> None:
        """Request vượt quá max_tokens bị từ chối."""
        limiter = TokenBucketRateLimiter(max_tokens=3, window_seconds=60.0)
        for _ in range(3):
            limiter.acquire()
        assert limiter.acquire() is False

    def test_acquire_multiple_over_limit(self) -> None:
        """Nhiều requests sau giới hạn đều bị từ chối."""
        limiter = TokenBucketRateLimiter(max_tokens=2, window_seconds=60.0)
        limiter.acquire()
        limiter.acquire()
        assert limiter.acquire() is False
        assert limiter.acquire() is False


class TestTokensReplenish:
    """Test tokens được phục hồi sau khi window hết hạn."""

    def test_tokens_replenish_after_window(self) -> None:
        """Sau khi window expires, tokens được phục hồi."""
        limiter = TokenBucketRateLimiter(max_tokens=2, window_seconds=1.0)

        # Dùng hết tokens
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is False

        # Chờ window hết hạn
        time.sleep(1.1)

        # Tokens được phục hồi
        assert limiter.acquire() is True

    def test_partial_replenish_sliding_window(self) -> None:
        """Sliding window: chỉ các request cũ nhất hết hạn trước."""
        limiter = TokenBucketRateLimiter(max_tokens=2, window_seconds=1.0)

        limiter.acquire()
        time.sleep(0.5)
        limiter.acquire()

        # Đầy, không thể acquire
        assert limiter.acquire() is False

        # Chờ cho request đầu hết hạn (tổng ~1.1s từ request đầu)
        time.sleep(0.7)

        # Request đầu đã hết hạn, có thể acquire lại
        assert limiter.acquire() is True


class TestEstimateDuration:
    """Test estimate_duration() tính đúng ETA."""

    def test_zero_requests(self) -> None:
        """0 requests → 0 phút."""
        limiter = TokenBucketRateLimiter(max_tokens=20, window_seconds=60.0)
        assert limiter.estimate_duration(0) == 0.0

    def test_negative_requests(self) -> None:
        """Số requests âm → 0 phút."""
        limiter = TokenBucketRateLimiter(max_tokens=20, window_seconds=60.0)
        assert limiter.estimate_duration(-5) == 0.0

    def test_under_one_window(self) -> None:
        """Requests ít hơn max_tokens → 1 phút."""
        limiter = TokenBucketRateLimiter(max_tokens=20, window_seconds=60.0)
        assert limiter.estimate_duration(15) == 1.0

    def test_exactly_one_window(self) -> None:
        """Requests bằng max_tokens → 1 phút."""
        limiter = TokenBucketRateLimiter(max_tokens=20, window_seconds=60.0)
        assert limiter.estimate_duration(20) == 1.0

    def test_over_one_window(self) -> None:
        """Requests vượt max_tokens → ceil(N/R) phút."""
        limiter = TokenBucketRateLimiter(max_tokens=20, window_seconds=60.0)
        assert limiter.estimate_duration(21) == 2.0

    def test_large_number(self) -> None:
        """Nhiều requests: ceil(100/20) = 5 phút."""
        limiter = TokenBucketRateLimiter(max_tokens=20, window_seconds=60.0)
        assert limiter.estimate_duration(100) == 5.0

    def test_non_divisible(self) -> None:
        """Không chia hết: ceil(45/20) = 3 phút."""
        limiter = TokenBucketRateLimiter(max_tokens=20, window_seconds=60.0)
        assert limiter.estimate_duration(45) == 3.0


class TestGetRemainingTokens:
    """Test get_remaining_tokens() trả về chính xác."""

    def test_full_tokens_initially(self) -> None:
        """Ban đầu có đầy đủ tokens."""
        limiter = TokenBucketRateLimiter(max_tokens=20, window_seconds=60.0)
        assert limiter.get_remaining_tokens() == 20

    def test_tokens_decrease_after_acquire(self) -> None:
        """Mỗi acquire giảm 1 token."""
        limiter = TokenBucketRateLimiter(max_tokens=5, window_seconds=60.0)
        limiter.acquire()
        assert limiter.get_remaining_tokens() == 4
        limiter.acquire()
        assert limiter.get_remaining_tokens() == 3

    def test_zero_remaining_at_limit(self) -> None:
        """Hết tokens khi đạt giới hạn."""
        limiter = TokenBucketRateLimiter(max_tokens=3, window_seconds=60.0)
        for _ in range(3):
            limiter.acquire()
        assert limiter.get_remaining_tokens() == 0

    def test_tokens_recover_after_window(self) -> None:
        """Tokens phục hồi sau khi window expires."""
        limiter = TokenBucketRateLimiter(max_tokens=2, window_seconds=1.0)
        limiter.acquire()
        limiter.acquire()
        assert limiter.get_remaining_tokens() == 0

        time.sleep(1.1)
        assert limiter.get_remaining_tokens() == 2


class TestReset:
    """Test reset() xóa sạch timestamps."""

    def test_reset_clears_state(self) -> None:
        """Reset phục hồi toàn bộ tokens."""
        limiter = TokenBucketRateLimiter(max_tokens=5, window_seconds=60.0)
        for _ in range(5):
            limiter.acquire()
        assert limiter.get_remaining_tokens() == 0

        limiter.reset()
        assert limiter.get_remaining_tokens() == 5

    def test_acquire_after_reset(self) -> None:
        """Sau reset có thể acquire lại bình thường."""
        limiter = TokenBucketRateLimiter(max_tokens=2, window_seconds=60.0)
        limiter.acquire()
        limiter.acquire()
        assert limiter.acquire() is False

        limiter.reset()
        assert limiter.acquire() is True


class TestWaitForToken:
    """Test wait_for_token() blocking behavior."""

    def test_wait_returns_zero_when_available(self) -> None:
        """Trả về 0.0 nếu token khả dụng ngay."""
        limiter = TokenBucketRateLimiter(max_tokens=5, window_seconds=60.0)
        wait_time = limiter.wait_for_token()
        assert wait_time == 0.0

    def test_wait_blocks_when_exhausted(self) -> None:
        """Block và trả về thời gian chờ khi hết token."""
        limiter = TokenBucketRateLimiter(max_tokens=2, window_seconds=1.0)
        limiter.acquire()
        limiter.acquire()

        start = time.time()
        wait_time = limiter.wait_for_token()
        elapsed = time.time() - start

        # Phải chờ khoảng 1 giây (window_seconds)
        assert wait_time > 0.0
        assert elapsed >= 0.9  # Cho phép sai số nhỏ


class TestDefaultConfig:
    """Test giá trị mặc định từ config/settings.py."""

    def test_default_max_tokens(self) -> None:
        """Mặc định max_tokens = RATE_LIMIT = 20."""
        limiter = TokenBucketRateLimiter()
        assert limiter.max_tokens == 20

    def test_default_window_seconds(self) -> None:
        """Mặc định window_seconds = RATE_LIMIT_WINDOW = 60."""
        limiter = TokenBucketRateLimiter()
        assert limiter.window_seconds == 60.0
