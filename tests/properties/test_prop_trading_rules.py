# -*- coding: utf-8 -*-
"""
Property-based tests cho Vietnamese trading rules.

Kiểm tra 3 properties:
- Property 11: Buy date is next trading day (recommend buy = T+1)
- Property 12: Vietnamese trading rules enforcement (sell ≥ T+3, price ±7%, lot 100)
- Property 13: No look-ahead bias in analysis (chỉ dùng data ≤ T)

# Feature: stock-trading-platform-refactor, Property 11: Buy date is next trading day
# Feature: stock-trading-platform-refactor, Property 12: Vietnamese trading rules enforcement
# Feature: stock-trading-platform-refactor, Property 13: No look-ahead bias in analysis
"""

from datetime import date, timedelta

import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.workers.vn_rules import (
    compute_buy_date,
    compute_earliest_sell_date,
    enforce_vn_rules,
    filter_data_for_analysis,
    is_trading_day,
    is_vn_holiday,
    validate_lot_size,
    validate_price_limit,
)
from config.vn_market_rules import LOT_SIZE, PRICE_LIMIT_PCT, SETTLEMENT_DAYS
from models.backtest_models import Trade


# ===========================================================================
# Strategies - Sinh dữ liệu ngẫu nhiên
# ===========================================================================

# Strategy sinh ngày trong khoảng 2020-2030, chỉ weekdays
weekday_date_strategy = st.dates(
    min_value=date(2020, 1, 1),
    max_value=date(2030, 12, 31),
).filter(lambda d: d.weekday() < 5)  # Chỉ Mon-Fri

# Strategy sinh giá ngẫu nhiên (10 - 500 nghìn VND)
price_strategy = st.floats(
    min_value=10.0,
    max_value=500.0,
    allow_nan=False,
    allow_infinity=False,
)

# Strategy sinh số lượng shares ngẫu nhiên (1 - 10000)
shares_strategy = st.integers(min_value=1, max_value=10000)

# Strategy sinh symbol ngẫu nhiên từ VN30
vn30_symbols = [
    "ACB", "BCM", "BID", "BVH", "CTG",
    "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW",
    "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB",
    "VIC", "VJC", "VNM", "VPB", "VRE",
]
symbol_strategy = st.sampled_from(vn30_symbols)


# ===========================================================================
# Property 11: Buy date is next trading day
# ===========================================================================


class TestBuyDateIsNextTradingDay:
    """
    Property tests đảm bảo compute_buy_date(T) trả về ngày giao dịch
    hợp lệ tiếp theo sau T.

    **Validates: Requirements 5.1**
    """

    @given(analysis_date=weekday_date_strategy)
    @settings(max_examples=200)
    def test_buy_date_is_after_analysis_date(self, analysis_date: date):
        """
        Property 11.1: Buy date luôn sau analysis date.

        Với bất kỳ ngày phân tích T, compute_buy_date(T) > T.

        **Validates: Requirements 5.1**
        """
        buy_date = compute_buy_date(analysis_date)
        assert buy_date > analysis_date, (
            f"Buy date {buy_date} không sau analysis date {analysis_date}"
        )

    @given(analysis_date=weekday_date_strategy)
    @settings(max_examples=200)
    def test_buy_date_is_valid_trading_day(self, analysis_date: date):
        """
        Property 11.2: Buy date luôn là ngày giao dịch hợp lệ.

        Ngày mua phải là weekday và không phải ngày lễ VN.

        **Validates: Requirements 5.1**
        """
        buy_date = compute_buy_date(analysis_date)
        assert is_trading_day(buy_date), (
            f"Buy date {buy_date} (weekday={buy_date.weekday()}) "
            f"không phải ngày giao dịch hợp lệ"
        )

    @given(analysis_date=weekday_date_strategy)
    @settings(max_examples=200)
    def test_no_trading_day_between_analysis_and_buy(self, analysis_date: date):
        """
        Property 11.3: Không có ngày giao dịch nào giữa T và buy_date.

        Buy date = T+1 trading day, nên không tồn tại ngày giao dịch D
        sao cho T < D < buy_date.

        **Validates: Requirements 5.1**
        """
        buy_date = compute_buy_date(analysis_date)

        # Kiểm tra không có trading day nào giữa analysis_date và buy_date
        current = analysis_date + timedelta(days=1)
        while current < buy_date:
            assert not is_trading_day(current), (
                f"Tồn tại trading day {current} giữa analysis {analysis_date} "
                f"và buy {buy_date}"
            )
            current += timedelta(days=1)


# ===========================================================================
# Property 12: Vietnamese trading rules enforcement
# ===========================================================================


class TestVietnameseTradingRulesEnforcement:
    """
    Property tests đảm bảo enforce_vn_rules áp dụng đúng luật TTCK VN:
    - sell_date >= buy_date + 3 trading days
    - shares chia hết cho 100
    - price trong biên độ ±7%

    **Validates: Requirements 5.2, 5.5, 7.5**
    """

    @given(
        symbol=symbol_strategy,
        buy_date=weekday_date_strategy,
        sell_offset=st.integers(min_value=0, max_value=30),
        buy_price=price_strategy,
        sell_price=price_strategy,
        shares=shares_strategy,
    )
    @settings(max_examples=200)
    def test_sell_date_respects_settlement_period(
        self,
        symbol: str,
        buy_date: date,
        sell_offset: int,
        buy_price: float,
        sell_price: float,
        shares: int,
    ):
        """
        Property 12.1: Sau enforce_vn_rules, sell_date >= buy_date + 3 trading days.

        Nếu sell_date quá sớm, hệ thống phải điều chỉnh sang ngày bán
        sớm nhất hợp lệ.

        **Validates: Requirements 5.2, 5.5**
        """
        # Bỏ qua ngày lễ VN cho buy_date
        assume(is_trading_day(buy_date))

        # Tạo sell_date = buy_date + sell_offset calendar days
        sell_date = buy_date + timedelta(days=sell_offset)

        trade = Trade(
            symbol=symbol,
            buy_date=buy_date,
            sell_date=sell_date,
            buy_price=buy_price,
            sell_price=sell_price,
            shares=shares,
        )

        # Enforce rules
        adjusted_trade = enforce_vn_rules(trade)

        # Kiểm tra sell_date >= earliest_sell_date (T+3 trading days)
        earliest_sell = compute_earliest_sell_date(buy_date)
        assert adjusted_trade.sell_date >= earliest_sell, (
            f"Sell date {adjusted_trade.sell_date} < earliest sell {earliest_sell} "
            f"(buy_date={buy_date}, settlement={SETTLEMENT_DAYS} trading days)"
        )

    @given(
        symbol=symbol_strategy,
        buy_date=weekday_date_strategy,
        buy_price=price_strategy,
        sell_price=price_strategy,
        shares=shares_strategy,
    )
    @settings(max_examples=200)
    def test_shares_always_multiple_of_lot_size(
        self,
        symbol: str,
        buy_date: date,
        buy_price: float,
        sell_price: float,
        shares: int,
    ):
        """
        Property 12.2: Sau enforce_vn_rules, shares luôn chia hết cho 100.

        Hệ thống làm tròn xuống bội 100 gần nhất (tối thiểu 1 lot = 100).

        **Validates: Requirements 7.5**
        """
        assume(is_trading_day(buy_date))

        sell_date = buy_date + timedelta(days=10)
        trade = Trade(
            symbol=symbol,
            buy_date=buy_date,
            sell_date=sell_date,
            buy_price=buy_price,
            sell_price=sell_price,
            shares=shares,
        )

        adjusted_trade = enforce_vn_rules(trade)

        # Shares phải chia hết cho LOT_SIZE (100)
        assert adjusted_trade.shares % LOT_SIZE == 0, (
            f"Shares {adjusted_trade.shares} không chia hết cho {LOT_SIZE} "
            f"(original shares={shares})"
        )
        # Shares phải > 0
        assert adjusted_trade.shares > 0, (
            f"Shares {adjusted_trade.shares} phải > 0"
        )

    @given(
        reference_price=price_strategy,
        price_offset_pct=st.floats(
            min_value=-15.0, max_value=15.0,
            allow_nan=False, allow_infinity=False,
        ),
    )
    @settings(max_examples=200)
    def test_price_limit_validates_within_7_percent(
        self,
        reference_price: float,
        price_offset_pct: float,
    ):
        """
        Property 12.3: validate_price_limit đúng biên độ ±7%.

        Giá trong [ref * 0.93, ref * 1.07] → valid.
        Giá ngoài khoảng → invalid.

        **Validates: Requirements 7.5**
        """
        assume(reference_price > 0)

        # Tính price dựa trên offset percentage
        price = reference_price * (1 + price_offset_pct / 100.0)
        assume(price > 0)

        is_valid = validate_price_limit(price, reference_price)

        if abs(price_offset_pct) <= PRICE_LIMIT_PCT:
            # Trong biên độ → phải valid
            assert is_valid, (
                f"Price {price} (offset {price_offset_pct}%) nên hợp lệ "
                f"với reference {reference_price} và limit ±{PRICE_LIMIT_PCT}%"
            )
        elif abs(price_offset_pct) > PRICE_LIMIT_PCT + 0.01:
            # Ngoài biên độ (thêm epsilon để tránh floating point) → phải invalid
            assert not is_valid, (
                f"Price {price} (offset {price_offset_pct}%) nên không hợp lệ "
                f"với reference {reference_price} và limit ±{PRICE_LIMIT_PCT}%"
            )


# ===========================================================================
# Property 13: No look-ahead bias in analysis
# ===========================================================================


class TestNoLookAheadBias:
    """
    Property tests đảm bảo filter_data_for_analysis chỉ trả về
    data có date <= analysis_date T. Không bao giờ chứa data tương lai.

    **Validates: Requirements 5.4**
    """

    @given(
        analysis_date=weekday_date_strategy,
        num_rows=st.integers(min_value=5, max_value=100),
        days_before=st.integers(min_value=1, max_value=60),
        days_after=st.integers(min_value=1, max_value=60),
    )
    @settings(max_examples=200)
    def test_filtered_data_contains_no_future_dates(
        self,
        analysis_date: date,
        num_rows: int,
        days_before: int,
        days_after: int,
    ):
        """
        Property 13.1: Sau filter, tất cả dates trong DataFrame <= analysis_date.

        Với data chứa cả quá khứ lẫn tương lai, filter_data_for_analysis
        phải loại bỏ tất cả rows có date > T.

        **Validates: Requirements 5.4**
        """
        # Tạo DataFrame với dates từ (T - days_before) đến (T + days_after)
        start_date = analysis_date - timedelta(days=days_before)
        all_dates = [start_date + timedelta(days=i)
                     for i in range(days_before + days_after + 1)]

        # Chỉ lấy đủ num_rows (hoặc ít hơn nếu range ngắn)
        dates_to_use = all_dates[:min(num_rows, len(all_dates))]

        df = pd.DataFrame({
            "time": [d.strftime("%Y-%m-%d") for d in dates_to_use],
            "open": range(len(dates_to_use)),
            "high": range(len(dates_to_use)),
            "low": range(len(dates_to_use)),
            "close": range(len(dates_to_use)),
            "volume": range(len(dates_to_use)),
        })

        # Filter data
        filtered = filter_data_for_analysis(df, analysis_date)

        # Assert: tất cả dates trong filtered <= analysis_date
        if not filtered.empty:
            filtered_dates = pd.to_datetime(filtered["time"])
            analysis_ts = pd.Timestamp(analysis_date)
            assert (filtered_dates <= analysis_ts).all(), (
                f"Filtered data chứa dates sau analysis_date {analysis_date}: "
                f"{filtered_dates[filtered_dates > analysis_ts].tolist()}"
            )

    @given(
        analysis_date=weekday_date_strategy,
        num_past_days=st.integers(min_value=1, max_value=90),
    )
    @settings(max_examples=200)
    def test_filtered_data_preserves_past_data(
        self,
        analysis_date: date,
        num_past_days: int,
    ):
        """
        Property 13.2: Filter giữ nguyên toàn bộ data có date <= T.

        Dữ liệu quá khứ và ngày T phải được giữ lại đầy đủ,
        không bị mất row nào.

        **Validates: Requirements 5.4**
        """
        # Tạo DataFrame chỉ với dates <= analysis_date
        past_dates = [analysis_date - timedelta(days=i)
                      for i in range(num_past_days)]
        past_dates.sort()

        df = pd.DataFrame({
            "time": [d.strftime("%Y-%m-%d") for d in past_dates],
            "open": range(len(past_dates)),
            "high": range(len(past_dates)),
            "low": range(len(past_dates)),
            "close": range(len(past_dates)),
            "volume": range(len(past_dates)),
        })

        # Filter data
        filtered = filter_data_for_analysis(df, analysis_date)

        # Phải giữ nguyên tất cả rows (vì tất cả <= T)
        assert len(filtered) == len(df), (
            f"Filter mất data: expected {len(df)} rows, got {len(filtered)}. "
            f"Analysis date: {analysis_date}"
        )

    @given(
        analysis_date=weekday_date_strategy,
        num_future_days=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=200)
    def test_only_future_data_is_removed(
        self,
        analysis_date: date,
        num_future_days: int,
    ):
        """
        Property 13.3: Chỉ data tương lai (> T) bị loại bỏ.

        Nếu DataFrame chỉ chứa future data (> analysis_date),
        kết quả filter phải rỗng.

        **Validates: Requirements 5.4**
        """
        # Tạo DataFrame chỉ với future dates (> analysis_date)
        future_dates = [analysis_date + timedelta(days=i + 1)
                        for i in range(num_future_days)]

        df = pd.DataFrame({
            "time": [d.strftime("%Y-%m-%d") for d in future_dates],
            "open": range(len(future_dates)),
            "high": range(len(future_dates)),
            "low": range(len(future_dates)),
            "close": range(len(future_dates)),
            "volume": range(len(future_dates)),
        })

        # Filter data
        filtered = filter_data_for_analysis(df, analysis_date)

        # Kết quả phải rỗng vì tất cả data đều > T
        assert len(filtered) == 0, (
            f"Filter không loại bỏ hết future data: "
            f"expected 0 rows, got {len(filtered)}"
        )
