"""
Unit tests cho Vietnamese market rules enforcement.

Test các function trong engine/workers/vn_rules.py:
- is_vn_holiday(): kiểm tra ngày lễ VN
- is_trading_day(): kiểm tra ngày giao dịch hợp lệ
- get_next_trading_day(): tìm ngày giao dịch tiếp theo
- add_trading_days(): cộng N ngày giao dịch
- compute_buy_date(): tính ngày mua T+1
- compute_earliest_sell_date(): tính ngày bán sớm nhất (T+3 trading days)
- validate_lot_size(): kiểm tra lot size chia hết 100
- validate_price_limit(): kiểm tra biên độ ±7%
- is_within_settlement_period(): kiểm tra settlement period
- enforce_vn_rules(): áp dụng toàn bộ rules cho Trade

References: Req 5.1, 5.2, 5.4, 5.5, 7.5
"""

from datetime import date

import pytest

from engine.workers.vn_rules import (
    add_trading_days,
    compute_buy_date,
    compute_earliest_sell_date,
    enforce_vn_rules,
    get_next_trading_day,
    is_trading_day,
    is_vn_holiday,
    is_within_settlement_period,
    validate_lot_size,
    validate_price_limit,
)
from models.backtest_models import Trade


# ==============================================================================
# Tests cho is_vn_holiday
# ==============================================================================


class TestIsVnHoliday:
    """Tests cho hàm is_vn_holiday()."""

    def test_new_year(self):
        """Ngày 1/1 là ngày lễ."""
        assert is_vn_holiday(date(2024, 1, 1)) is True

    def test_liberation_day(self):
        """Ngày 30/4 là ngày lễ."""
        assert is_vn_holiday(date(2024, 4, 30)) is True

    def test_labor_day(self):
        """Ngày 1/5 là ngày lễ."""
        assert is_vn_holiday(date(2024, 5, 1)) is True

    def test_national_day(self):
        """Ngày 2/9 là ngày lễ."""
        assert is_vn_holiday(date(2024, 9, 2)) is True

    def test_normal_day_not_holiday(self):
        """Ngày thường không phải ngày lễ."""
        assert is_vn_holiday(date(2024, 3, 15)) is False

    def test_weekend_not_holiday(self):
        """Weekend không tính là holiday (holiday là concept riêng)."""
        # 2024-01-06 là thứ 7
        assert is_vn_holiday(date(2024, 1, 6)) is False


# ==============================================================================
# Tests cho is_trading_day
# ==============================================================================


class TestIsTradingDay:
    """Tests cho hàm is_trading_day()."""

    def test_monday_is_trading_day(self):
        """Thứ 2 bình thường là ngày giao dịch."""
        # 2024-01-08 là thứ 2
        assert is_trading_day(date(2024, 1, 8)) is True

    def test_friday_is_trading_day(self):
        """Thứ 6 bình thường là ngày giao dịch."""
        # 2024-01-12 là thứ 6
        assert is_trading_day(date(2024, 1, 12)) is True

    def test_saturday_not_trading_day(self):
        """Thứ 7 không phải ngày giao dịch."""
        # 2024-01-13 là thứ 7
        assert is_trading_day(date(2024, 1, 13)) is False

    def test_sunday_not_trading_day(self):
        """Chủ nhật không phải ngày giao dịch."""
        # 2024-01-14 là chủ nhật
        assert is_trading_day(date(2024, 1, 14)) is False

    def test_holiday_on_weekday_not_trading_day(self):
        """Ngày lễ rơi vào ngày trong tuần thì không giao dịch."""
        # 2024-09-02 là thứ 2 và là ngày Quốc khánh
        assert is_trading_day(date(2024, 9, 2)) is False

    def test_new_year_not_trading_day(self):
        """Ngày 1/1/2024 là thứ 2, nhưng là ngày lễ."""
        assert is_trading_day(date(2024, 1, 1)) is False


# ==============================================================================
# Tests cho get_next_trading_day
# ==============================================================================


class TestGetNextTradingDay:
    """Tests cho hàm get_next_trading_day()."""

    def test_weekday_to_next_weekday(self):
        """Từ thứ 2 → thứ 3."""
        # 2024-01-08 (Mon) → 2024-01-09 (Tue)
        assert get_next_trading_day(date(2024, 1, 8)) == date(2024, 1, 9)

    def test_friday_to_monday(self):
        """Từ thứ 6 → thứ 2 (skip weekend)."""
        # 2024-01-12 (Fri) → 2024-01-15 (Mon)
        assert get_next_trading_day(date(2024, 1, 12)) == date(2024, 1, 15)

    def test_saturday_to_monday(self):
        """Từ thứ 7 → thứ 2."""
        # 2024-01-13 (Sat) → 2024-01-15 (Mon)
        assert get_next_trading_day(date(2024, 1, 13)) == date(2024, 1, 15)

    def test_sunday_to_monday(self):
        """Từ chủ nhật → thứ 2."""
        # 2024-01-14 (Sun) → 2024-01-15 (Mon)
        assert get_next_trading_day(date(2024, 1, 14)) == date(2024, 1, 15)

    def test_skip_holiday(self):
        """Bỏ qua ngày lễ, lấy ngày giao dịch tiếp theo."""
        # 2024-04-29 (Mon) → 2024-04-30 là lễ (Tue), 2024-05-01 là lễ (Wed)
        # → 2024-05-02 (Thu)
        assert get_next_trading_day(date(2024, 4, 29)) == date(2024, 5, 2)


# ==============================================================================
# Tests cho add_trading_days
# ==============================================================================


class TestAddTradingDays:
    """Tests cho hàm add_trading_days()."""

    def test_add_1_day_normal(self):
        """Cộng 1 ngày giao dịch bình thường."""
        # 2024-01-08 (Mon) + 1 = 2024-01-09 (Tue)
        assert add_trading_days(date(2024, 1, 8), 1) == date(2024, 1, 9)

    def test_add_3_days_normal(self):
        """Cộng 3 ngày giao dịch không qua weekend."""
        # 2024-01-08 (Mon) + 3 = 2024-01-11 (Thu)
        assert add_trading_days(date(2024, 1, 8), 3) == date(2024, 1, 11)

    def test_add_3_days_over_weekend(self):
        """Cộng 3 ngày giao dịch qua weekend."""
        # 2024-01-11 (Thu) + 3 = Fri, Mon, Tue → 2024-01-16 (Tue)
        assert add_trading_days(date(2024, 1, 11), 3) == date(2024, 1, 16)

    def test_add_5_days(self):
        """Cộng 5 ngày giao dịch (1 tuần giao dịch)."""
        # 2024-01-08 (Mon) + 5 = 2024-01-15 (Mon tuần sau)
        assert add_trading_days(date(2024, 1, 8), 5) == date(2024, 1, 15)

    def test_add_days_over_holiday(self):
        """Cộng ngày giao dịch qua ngày lễ."""
        # 2024-04-29 (Mon) + 1 = skip 30/4, 1/5 → 2024-05-02 (Thu)
        assert add_trading_days(date(2024, 4, 29), 1) == date(2024, 5, 2)


# ==============================================================================
# Tests cho compute_buy_date
# ==============================================================================


class TestComputeBuyDate:
    """Tests cho hàm compute_buy_date()."""

    def test_analysis_monday_buy_tuesday(self):
        """Phân tích thứ 2 → mua thứ 3."""
        # T = 2024-01-08 (Mon) → T+1 = 2024-01-09 (Tue)
        assert compute_buy_date(date(2024, 1, 8)) == date(2024, 1, 9)

    def test_analysis_friday_buy_monday(self):
        """Phân tích thứ 6 → mua thứ 2 (skip weekend)."""
        # T = 2024-01-12 (Fri) → T+1 = 2024-01-15 (Mon)
        assert compute_buy_date(date(2024, 1, 12)) == date(2024, 1, 15)

    def test_analysis_before_holiday(self):
        """Phân tích trước ngày lễ → skip holiday."""
        # T = 2024-04-29 (Mon) → 30/4 lễ, 1/5 lễ → T+1 = 2024-05-02 (Thu)
        assert compute_buy_date(date(2024, 4, 29)) == date(2024, 5, 2)

    def test_analysis_thursday(self):
        """Phân tích thứ 5 → mua thứ 6."""
        # T = 2024-01-11 (Thu) → T+1 = 2024-01-12 (Fri)
        assert compute_buy_date(date(2024, 1, 11)) == date(2024, 1, 12)


# ==============================================================================
# Tests cho compute_earliest_sell_date
# ==============================================================================


class TestComputeEarliestSellDate:
    """Tests cho hàm compute_earliest_sell_date()."""

    def test_buy_monday_sell_thursday(self):
        """Mua thứ 2 → bán sớm nhất thứ 5 (3 ngày giao dịch)."""
        # Buy 2024-01-08 (Mon) + 3 trading days = 2024-01-11 (Thu)
        assert compute_earliest_sell_date(date(2024, 1, 8)) == date(2024, 1, 11)

    def test_buy_thursday_sell_tuesday(self):
        """Mua thứ 5 → bán sớm nhất thứ 3 tuần sau (skip weekend)."""
        # Buy 2024-01-11 (Thu) + 3 = Fri, Mon, Tue → 2024-01-16 (Tue)
        assert compute_earliest_sell_date(date(2024, 1, 11)) == date(2024, 1, 16)

    def test_buy_friday_sell_wednesday(self):
        """Mua thứ 6 → bán sớm nhất thứ 4 tuần sau."""
        # Buy 2024-01-12 (Fri) + 3 = Mon, Tue, Wed → 2024-01-17 (Wed)
        assert compute_earliest_sell_date(date(2024, 1, 12)) == date(2024, 1, 17)

    def test_buy_before_holiday_period(self):
        """Mua trước chuỗi ngày lễ → skip holidays."""
        # Buy 2024-04-29 (Mon) + 3 = skip 30/4, 1/5 → Thu, Fri, Mon
        # 2024-05-02 (Thu), 2024-05-03 (Fri), 2024-05-06 (Mon)
        assert compute_earliest_sell_date(date(2024, 4, 29)) == date(2024, 5, 6)

    def test_minimum_3_trading_days(self):
        """Luôn đảm bảo tối thiểu 3 ngày giao dịch."""
        buy = date(2024, 1, 15)  # Thứ 2
        earliest_sell = compute_earliest_sell_date(buy)
        # Đếm số ngày giao dịch giữa buy và earliest_sell
        count = 0
        current = buy
        while current < earliest_sell:
            current += __import__("datetime").timedelta(days=1)
            if is_trading_day(current):
                count += 1
        assert count == 3


# ==============================================================================
# Tests cho validate_lot_size
# ==============================================================================


class TestValidateLotSize:
    """Tests cho hàm validate_lot_size()."""

    def test_valid_100(self):
        """100 cổ phiếu hợp lệ."""
        assert validate_lot_size(100) is True

    def test_valid_1000(self):
        """1000 cổ phiếu hợp lệ."""
        assert validate_lot_size(1000) is True

    def test_valid_500(self):
        """500 cổ phiếu hợp lệ."""
        assert validate_lot_size(500) is True

    def test_invalid_50(self):
        """50 cổ phiếu không hợp lệ (không chia hết 100)."""
        assert validate_lot_size(50) is False

    def test_invalid_150(self):
        """150 cổ phiếu không hợp lệ."""
        assert validate_lot_size(150) is False

    def test_invalid_0(self):
        """0 cổ phiếu không hợp lệ."""
        assert validate_lot_size(0) is False

    def test_invalid_negative(self):
        """-100 cổ phiếu không hợp lệ."""
        assert validate_lot_size(-100) is False

    def test_invalid_1(self):
        """1 cổ phiếu không hợp lệ."""
        assert validate_lot_size(1) is False


# ==============================================================================
# Tests cho validate_price_limit
# ==============================================================================


class TestValidatePriceLimit:
    """Tests cho hàm validate_price_limit()."""

    def test_price_at_reference(self):
        """Giá bằng giá tham chiếu → hợp lệ."""
        assert validate_price_limit(100.0, 100.0) is True

    def test_price_at_ceiling(self):
        """Giá đúng trần (+7%) → hợp lệ."""
        assert validate_price_limit(107.0, 100.0) is True

    def test_price_at_floor(self):
        """Giá đúng sàn (-7%) → hợp lệ."""
        assert validate_price_limit(93.0, 100.0) is True

    def test_price_above_ceiling(self):
        """Giá vượt trần (+7.1%) → không hợp lệ."""
        assert validate_price_limit(107.1, 100.0) is False

    def test_price_below_floor(self):
        """Giá dưới sàn (-7.1%) → không hợp lệ."""
        assert validate_price_limit(92.9, 100.0) is False

    def test_price_within_range(self):
        """Giá trong khoảng ±7% → hợp lệ."""
        assert validate_price_limit(103.5, 100.0) is True

    def test_zero_reference_price(self):
        """Giá tham chiếu = 0 → không hợp lệ."""
        assert validate_price_limit(50.0, 0.0) is False

    def test_negative_price(self):
        """Giá âm → không hợp lệ."""
        assert validate_price_limit(-5.0, 100.0) is False

    def test_negative_reference(self):
        """Giá tham chiếu âm → không hợp lệ."""
        assert validate_price_limit(50.0, -100.0) is False

    def test_realistic_price(self):
        """Test với giá thực tế VNM ~80k."""
        ref = 80.0
        # +5% = 84 → hợp lệ
        assert validate_price_limit(84.0, ref) is True
        # +8% = 86.4 → không hợp lệ
        assert validate_price_limit(86.4, ref) is False


# ==============================================================================
# Tests cho is_within_settlement_period
# ==============================================================================


class TestIsWithinSettlementPeriod:
    """Tests cho hàm is_within_settlement_period()."""

    def test_sell_same_day_as_buy(self):
        """Bán cùng ngày mua → trong settlement period."""
        buy = date(2024, 1, 8)
        sell = date(2024, 1, 8)
        assert is_within_settlement_period(buy, sell) is True

    def test_sell_1_day_after_buy(self):
        """Bán 1 ngày sau → trong settlement period."""
        buy = date(2024, 1, 8)
        sell = date(2024, 1, 9)
        assert is_within_settlement_period(buy, sell) is True

    def test_sell_2_days_after_buy(self):
        """Bán 2 ngày giao dịch sau → vẫn trong settlement period."""
        buy = date(2024, 1, 8)  # Mon
        sell = date(2024, 1, 10)  # Wed (2 trading days after)
        assert is_within_settlement_period(buy, sell) is True

    def test_sell_at_earliest_date(self):
        """Bán đúng ngày sớm nhất → KHÔNG trong settlement period."""
        buy = date(2024, 1, 8)  # Mon
        # Earliest sell = Thu 2024-01-11 (3 trading days after Mon)
        sell = date(2024, 1, 11)
        assert is_within_settlement_period(buy, sell) is False

    def test_sell_after_earliest_date(self):
        """Bán sau ngày sớm nhất → KHÔNG trong settlement period."""
        buy = date(2024, 1, 8)
        sell = date(2024, 1, 15)
        assert is_within_settlement_period(buy, sell) is False


# ==============================================================================
# Tests cho enforce_vn_rules
# ==============================================================================


class TestEnforceVnRules:
    """Tests cho hàm enforce_vn_rules()."""

    def test_valid_trade_unchanged(self):
        """Trade hợp lệ không bị thay đổi."""
        trade = Trade(
            symbol="VNM",
            buy_date=date(2024, 1, 8),
            sell_date=date(2024, 1, 15),  # Sau settlement
            buy_price=80.0,
            sell_price=82.0,
            shares=200,
        )
        result = enforce_vn_rules(trade)
        assert result.shares == 200
        assert result.sell_date == date(2024, 1, 15)

    def test_sell_date_adjusted_if_too_early(self):
        """Sell date quá sớm → điều chỉnh sang earliest sell date."""
        trade = Trade(
            symbol="FPT",
            buy_date=date(2024, 1, 8),  # Mon
            sell_date=date(2024, 1, 9),  # Tue (quá sớm)
            buy_price=100.0,
            sell_price=103.0,
            shares=100,
        )
        result = enforce_vn_rules(trade)
        # Earliest sell = 2024-01-11 (Thu, 3 trading days after Mon)
        assert result.sell_date == date(2024, 1, 11)

    def test_shares_adjusted_to_lot_multiple(self):
        """Shares không chia hết 100 → làm tròn xuống."""
        trade = Trade(
            symbol="HPG",
            buy_date=date(2024, 1, 8),
            sell_date=date(2024, 1, 15),
            buy_price=25.0,
            sell_price=26.0,
            shares=350,  # Không chia hết 100
        )
        result = enforce_vn_rules(trade)
        assert result.shares == 300  # Tròn xuống bội 100

    def test_shares_less_than_100_set_to_100(self):
        """Shares < 100 → set tối thiểu 100."""
        trade = Trade(
            symbol="MBB",
            buy_date=date(2024, 1, 8),
            sell_date=date(2024, 1, 15),
            buy_price=20.0,
            sell_price=21.0,
            shares=50,
        )
        result = enforce_vn_rules(trade)
        assert result.shares == 100

    def test_pnl_recalculated(self):
        """PnL được tính lại sau khi điều chỉnh shares."""
        trade = Trade(
            symbol="VCB",
            buy_date=date(2024, 1, 8),
            sell_date=date(2024, 1, 15),
            buy_price=90.0,
            sell_price=95.0,
            shares=300,
        )
        result = enforce_vn_rules(trade)
        expected_pnl = (95.0 - 90.0) * 300
        assert result.pnl == pytest.approx(expected_pnl)
        assert result.pnl_pct == pytest.approx((95.0 - 90.0) / 90.0)

    def test_open_trade_no_sell_date(self):
        """Trade đang mở (sell_date=None) → không điều chỉnh sell_date."""
        trade = Trade(
            symbol="ACB",
            buy_date=date(2024, 1, 8),
            sell_date=None,
            buy_price=25.0,
            sell_price=None,
            shares=500,
        )
        result = enforce_vn_rules(trade)
        assert result.sell_date is None
        assert result.shares == 500

    def test_holding_days_recalculated(self):
        """Holding days được tính lại khi sell_date điều chỉnh."""
        trade = Trade(
            symbol="TCB",
            buy_date=date(2024, 1, 8),
            sell_date=date(2024, 1, 9),  # Quá sớm
            buy_price=30.0,
            sell_price=31.0,
            shares=200,
            holding_days=1,
        )
        result = enforce_vn_rules(trade)
        # sell_date điều chỉnh → 2024-01-11
        expected_days = (date(2024, 1, 11) - date(2024, 1, 8)).days
        assert result.holding_days == expected_days
