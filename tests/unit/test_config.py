"""
Unit tests cho config package - settings và vn_market_rules.

Kiểm tra:
- Các constant có giá trị đúng theo requirements
- Type annotations chính xác
- Import hoạt động bình thường
"""

import pytest


class TestSettings:
    """Test config/settings.py constants."""

    def test_heartbeat_interval_value(self) -> None:
        """HEARTBEAT_INTERVAL phải = 30 giây (Req 1.7)."""
        from config.settings import HEARTBEAT_INTERVAL

        assert HEARTBEAT_INTERVAL == 30

    def test_poll_interval_value(self) -> None:
        """POLL_INTERVAL phải = 10 giây (Req 2.2)."""
        from config.settings import POLL_INTERVAL

        assert POLL_INTERVAL == 10

    def test_gpu_max_fraction_value(self) -> None:
        """GPU_MAX_FRACTION phải = 0.7 (Req 1.4)."""
        from config.settings import GPU_MAX_FRACTION

        assert GPU_MAX_FRACTION == 0.7

    def test_gpu_max_fraction_range(self) -> None:
        """GPU_MAX_FRACTION phải nằm trong (0, 1]."""
        from config.settings import GPU_MAX_FRACTION

        assert 0.0 < GPU_MAX_FRACTION <= 1.0

    def test_rate_limit_value(self) -> None:
        """RATE_LIMIT phải = 20 requests/minute (Req 8.6)."""
        from config.settings import RATE_LIMIT

        assert RATE_LIMIT == 20

    def test_confidence_threshold_value(self) -> None:
        """CONFIDENCE_THRESHOLD phải = 0.5 (Req 6.4)."""
        from config.settings import CONFIDENCE_THRESHOLD

        assert CONFIDENCE_THRESHOLD == 0.5

    def test_confidence_threshold_range(self) -> None:
        """CONFIDENCE_THRESHOLD phải nằm trong [0, 1]."""
        from config.settings import CONFIDENCE_THRESHOLD

        assert 0.0 <= CONFIDENCE_THRESHOLD <= 1.0

    def test_heartbeat_timeout_threshold(self) -> None:
        """HEARTBEAT_TIMEOUT_THRESHOLD phải = 60 giây."""
        from config.settings import HEARTBEAT_TIMEOUT_THRESHOLD

        assert HEARTBEAT_TIMEOUT_THRESHOLD == 60

    def test_heartbeat_timeout_greater_than_interval(self) -> None:
        """Timeout phải lớn hơn interval để tránh false positive."""
        from config.settings import HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT_THRESHOLD

        assert HEARTBEAT_TIMEOUT_THRESHOLD > HEARTBEAT_INTERVAL

    def test_rate_limit_window(self) -> None:
        """RATE_LIMIT_WINDOW phải = 60 giây."""
        from config.settings import RATE_LIMIT_WINDOW

        assert RATE_LIMIT_WINDOW == 60

    def test_rate_limit_positive(self) -> None:
        """RATE_LIMIT phải là số dương."""
        from config.settings import RATE_LIMIT

        assert RATE_LIMIT > 0

    def test_types_are_correct(self) -> None:
        """Kiểm tra kiểu dữ liệu các constants."""
        from config.settings import (
            CONFIDENCE_THRESHOLD,
            GPU_MAX_FRACTION,
            HEARTBEAT_INTERVAL,
            POLL_INTERVAL,
            RATE_LIMIT,
        )

        assert isinstance(HEARTBEAT_INTERVAL, int)
        assert isinstance(POLL_INTERVAL, int)
        assert isinstance(GPU_MAX_FRACTION, float)
        assert isinstance(RATE_LIMIT, int)
        assert isinstance(CONFIDENCE_THRESHOLD, float)


class TestVnMarketRules:
    """Test config/vn_market_rules.py constants."""

    def test_t_plus_buy_value(self) -> None:
        """T_PLUS_BUY phải = 1 (phân tích T → mua T+1)."""
        from config.vn_market_rules import T_PLUS_BUY

        assert T_PLUS_BUY == 1

    def test_settlement_days_value(self) -> None:
        """SETTLEMENT_DAYS phải = 3 (ceiling T+2.5)."""
        from config.vn_market_rules import SETTLEMENT_DAYS

        assert SETTLEMENT_DAYS == 3

    def test_price_limit_pct_value(self) -> None:
        """PRICE_LIMIT_PCT phải = 7.0 (±7% sàn HOSE)."""
        from config.vn_market_rules import PRICE_LIMIT_PCT

        assert PRICE_LIMIT_PCT == 7.0

    def test_lot_size_value(self) -> None:
        """LOT_SIZE phải = 100 (lô tròn)."""
        from config.vn_market_rules import LOT_SIZE

        assert LOT_SIZE == 100

    def test_lot_size_positive(self) -> None:
        """LOT_SIZE phải là số dương."""
        from config.vn_market_rules import LOT_SIZE

        assert LOT_SIZE > 0

    def test_settlement_days_positive(self) -> None:
        """SETTLEMENT_DAYS phải là số dương."""
        from config.vn_market_rules import SETTLEMENT_DAYS

        assert SETTLEMENT_DAYS > 0

    def test_t_plus_buy_positive(self) -> None:
        """T_PLUS_BUY phải là số dương."""
        from config.vn_market_rules import T_PLUS_BUY

        assert T_PLUS_BUY > 0

    def test_price_limit_positive(self) -> None:
        """PRICE_LIMIT_PCT phải là số dương."""
        from config.vn_market_rules import PRICE_LIMIT_PCT

        assert PRICE_LIMIT_PCT > 0.0

    def test_settlement_greater_than_buy(self) -> None:
        """SETTLEMENT_DAYS phải lớn hơn T_PLUS_BUY (bán sau mua)."""
        from config.vn_market_rules import SETTLEMENT_DAYS, T_PLUS_BUY

        assert SETTLEMENT_DAYS > T_PLUS_BUY

    def test_vn30_symbols_count(self) -> None:
        """VN30_SYMBOLS phải có đúng 30 mã."""
        from config.vn_market_rules import VN30_SYMBOLS

        assert len(VN30_SYMBOLS) == 30

    def test_vn30_symbols_unique(self) -> None:
        """VN30_SYMBOLS không được có mã trùng lặp."""
        from config.vn_market_rules import VN30_SYMBOLS

        assert len(VN30_SYMBOLS) == len(set(VN30_SYMBOLS))

    def test_vn30_symbols_uppercase(self) -> None:
        """Tất cả mã VN30 phải viết hoa."""
        from config.vn_market_rules import VN30_SYMBOLS

        for symbol in VN30_SYMBOLS:
            assert symbol == symbol.upper()

    def test_trading_weekdays(self) -> None:
        """TRADING_WEEKDAYS phải là Mon-Fri (0-4)."""
        from config.vn_market_rules import TRADING_WEEKDAYS

        assert TRADING_WEEKDAYS == [0, 1, 2, 3, 4]

    def test_types_are_correct(self) -> None:
        """Kiểm tra kiểu dữ liệu các constants."""
        from config.vn_market_rules import (
            LOT_SIZE,
            PRICE_LIMIT_PCT,
            SETTLEMENT_DAYS,
            T_PLUS_BUY,
        )

        assert isinstance(T_PLUS_BUY, int)
        assert isinstance(SETTLEMENT_DAYS, int)
        assert isinstance(PRICE_LIMIT_PCT, float)
        assert isinstance(LOT_SIZE, int)


class TestConfigPackageImport:
    """Test import từ config package __init__.py."""

    def test_import_all_settings(self) -> None:
        """Import tất cả settings constants từ config package."""
        from config import (
            CONFIDENCE_THRESHOLD,
            GPU_MAX_FRACTION,
            HEARTBEAT_INTERVAL,
            POLL_INTERVAL,
            RATE_LIMIT,
        )

        assert HEARTBEAT_INTERVAL == 30
        assert POLL_INTERVAL == 10
        assert GPU_MAX_FRACTION == 0.7
        assert RATE_LIMIT == 20
        assert CONFIDENCE_THRESHOLD == 0.5

    def test_import_all_vn_rules(self) -> None:
        """Import tất cả VN market rules từ config package."""
        from config import LOT_SIZE, PRICE_LIMIT_PCT, SETTLEMENT_DAYS, T_PLUS_BUY

        assert T_PLUS_BUY == 1
        assert SETTLEMENT_DAYS == 3
        assert PRICE_LIMIT_PCT == 7.0
        assert LOT_SIZE == 100
