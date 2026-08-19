# -*- coding: utf-8 -*-
"""
Property-based tests cho MistakeAnalyzer — incorrect prediction identification.

# Feature: stock-trading-platform-refactor, Property 5: Incorrect prediction identification

**Validates: Requirements 3.2, 3.3**

Property:
    Với bất kỳ tập hợp trades nào có known outcomes (pnl),
    identify_incorrect_predictions() phải trả về chính xác những trades
    có pnl < 0, loại trừ trades có pnl = None hoặc pnl >= 0.
"""

from datetime import date

from hypothesis import given, settings
from hypothesis import strategies as st

from engine.mistake_analyzer import identify_incorrect_predictions
from models.backtest_models import Trade


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Symbol: 2-5 ký tự uppercase
symbol_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu",)),
    min_size=2,
    max_size=5,
).filter(lambda s: len(s) >= 2)

# Date hợp lệ
date_strategy = st.dates(
    min_value=date(2020, 1, 1),
    max_value=date(2025, 12, 31),
)

# Giá: dương, hữu hạn
price_strategy = st.floats(
    min_value=1.0,
    max_value=500.0,
    allow_nan=False,
    allow_infinity=False,
)

# Shares: bội số 100
shares_strategy = st.integers(min_value=1, max_value=100).map(lambda x: x * 100)

# PnL: bao gồm None (vị thế đang mở), âm, zero, dương
pnl_strategy = st.one_of(
    st.none(),
    st.floats(min_value=-100000.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
)


@st.composite
def trade_strategy(draw):
    """
    Sinh Trade ngẫu nhiên với pnl có thể là None, âm, zero, hoặc dương.

    - pnl = None → vị thế đang mở (chưa đóng)
    - pnl < 0 → dự đoán sai (incorrect prediction)
    - pnl >= 0 → dự đoán đúng
    """
    symbol = draw(symbol_strategy)
    buy_date = draw(date_strategy)
    buy_price = draw(price_strategy)
    shares = draw(shares_strategy)
    pnl = draw(pnl_strategy)

    # Nếu pnl là None → vị thế đang mở, sell_date và sell_price = None
    if pnl is None:
        sell_date = None
        sell_price = None
        pnl_pct = None
        holding_days = draw(st.integers(min_value=0, max_value=60))
    else:
        sell_date = draw(date_strategy)
        sell_price = draw(price_strategy)
        pnl_pct = (pnl / (buy_price * shares)) * 100.0 if (buy_price * shares) > 0 else 0.0
        holding_days = draw(st.integers(min_value=1, max_value=60))

    return Trade(
        symbol=symbol,
        buy_date=buy_date,
        sell_date=sell_date,
        buy_price=buy_price,
        sell_price=sell_price,
        shares=shares,
        pnl=pnl,
        pnl_pct=pnl_pct,
        holding_days=holding_days,
    )


# Danh sách trades ngẫu nhiên
trade_list_strategy = st.lists(trade_strategy(), min_size=0, max_size=30)


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


class TestIncorrectPredictionIdentification:
    """Property 5: Incorrect prediction identification."""

    @given(trades=trade_list_strategy)
    @settings(max_examples=100)
    def test_identifies_exactly_trades_with_negative_pnl(self, trades: list):
        """
        Property: identify_incorrect_predictions trả về chính xác trades có pnl < 0.
        Loại trừ trades có pnl = None và pnl >= 0.

        # Feature: stock-trading-platform-refactor, Property 5: Incorrect prediction identification
        **Validates: Requirements 3.2, 3.3**
        """
        result = identify_incorrect_predictions(trades)

        # Tính expected: trades có pnl is not None AND pnl < 0
        expected = [t for t in trades if t.pnl is not None and t.pnl < 0]

        # Kiểm tra số lượng
        assert len(result) == len(expected), (
            f"Số lượng incorrect predictions sai: got {len(result)}, expected {len(expected)}"
        )

        # Kiểm tra tất cả result đều có pnl < 0
        for trade in result:
            assert trade.pnl is not None, (
                "Trade trong kết quả có pnl = None (vị thế đang mở không nên được tính)"
            )
            assert trade.pnl < 0, (
                f"Trade trong kết quả có pnl = {trade.pnl} >= 0 (không phải incorrect)"
            )

        # Kiểm tra không bỏ sót trade nào có pnl < 0
        result_set = set(id(t) for t in result)
        for trade in trades:
            if trade.pnl is not None and trade.pnl < 0:
                assert id(trade) in result_set, (
                    f"Trade với pnl={trade.pnl} bị bỏ sót khỏi kết quả"
                )

    @given(trades=trade_list_strategy)
    @settings(max_examples=50)
    def test_excludes_none_pnl_trades(self, trades: list):
        """
        Property: Trades có pnl = None (đang mở) KHÔNG xuất hiện trong kết quả.

        # Feature: stock-trading-platform-refactor, Property 5: Incorrect prediction identification
        **Validates: Requirements 3.2, 3.3**
        """
        result = identify_incorrect_predictions(trades)

        # Tất cả trades trong kết quả phải có pnl != None
        for trade in result:
            assert trade.pnl is not None, (
                "Trade đang mở (pnl=None) bị phân loại nhầm là incorrect prediction"
            )

    @given(trades=trade_list_strategy)
    @settings(max_examples=50)
    def test_excludes_non_negative_pnl_trades(self, trades: list):
        """
        Property: Trades có pnl >= 0 KHÔNG xuất hiện trong kết quả.

        # Feature: stock-trading-platform-refactor, Property 5: Incorrect prediction identification
        **Validates: Requirements 3.2, 3.3**
        """
        result = identify_incorrect_predictions(trades)

        # Tất cả trades trong kết quả phải có pnl < 0
        for trade in result:
            assert trade.pnl < 0, (
                f"Trade với pnl={trade.pnl} >= 0 bị phân loại nhầm là incorrect"
            )
