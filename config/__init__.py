"""
Config package - Cấu hình trung tâm cho toàn hệ thống stock trading platform.

Cung cấp:
- settings: các hằng số hệ thống (heartbeat, polling, resource limits, rate limit)
- vn_market_rules: luật giao dịch chứng khoán Việt Nam
"""

from config.settings import (
    CONFIDENCE_THRESHOLD,
    GPU_MAX_FRACTION,
    HEARTBEAT_INTERVAL,
    POLL_INTERVAL,
    RATE_LIMIT,
)
from config.vn_market_rules import (
    LOT_SIZE,
    PRICE_LIMIT_PCT,
    SETTLEMENT_DAYS,
    T_PLUS_BUY,
)

__all__ = [
    # System settings
    "HEARTBEAT_INTERVAL",
    "POLL_INTERVAL",
    "GPU_MAX_FRACTION",
    "RATE_LIMIT",
    "CONFIDENCE_THRESHOLD",
    # Vietnamese market rules
    "T_PLUS_BUY",
    "SETTLEMENT_DAYS",
    "PRICE_LIMIT_PCT",
    "LOT_SIZE",
]
