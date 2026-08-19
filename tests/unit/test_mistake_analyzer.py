"""
Unit tests cho engine/mistake_analyzer.py

Kiểm tra logic phân tích dự đoán sai, tạo hard examples,
tính mistake rate, và phân loại mistakes.

References: Requirements 3.2, 3.3
"""

from datetime import date

import pytest

from engine.mistake_analyzer import (
    calculate_mistake_rate,
    categorize_mistakes,
    generate_hard_examples,
    identify_incorrect_predictions,
)
from models.backtest_models import Trade


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def profitable_trade() -> Trade:
    """Trade có lợi nhuận (pnl > 0)."""
    return Trade(
        symbol="FPT",
        buy_date=date(2024, 1, 5),
        sell_date=date(2024, 1, 12),
        buy_price=80.0,
        sell_price=85.0,
        shares=100,
        pnl=500.0,
        pnl_pct=0.0625,
        holding_days=5,
    )


@pytest.fixture
def losing_trade() -> Trade:
    """Trade lỗ (pnl < 0)."""
    return Trade(
        symbol="HPG",
        buy_date=date(2024, 2, 1),
        sell_date=date(2024, 2, 8),
        buy_price=30.0,
        sell_price=27.0,
        shares=200,
        pnl=-600.0,
        pnl_pct=-0.10,
        holding_days=5,
    )


@pytest.fixture
def breakeven_trade() -> Trade:
    """Trade hòa vốn (pnl = 0)."""
    return Trade(
        symbol="VNM",
        buy_date=date(2024, 3, 1),
        sell_date=date(2024, 3, 8),
        buy_price=70.0,
        sell_price=70.0,
        shares=100,
        pnl=0.0,
        pnl_pct=0.0,
        holding_days=5,
    )


@pytest.fixture
def open_trade() -> Trade:
    """Trade đang mở (pnl = None)."""
    return Trade(
        symbol="VCB",
        buy_date=date(2024, 4, 1),
        sell_date=None,
        buy_price=90.0,
        sell_price=None,
        shares=100,
        pnl=None,
        pnl_pct=None,
        holding_days=3,
    )


@pytest.fixture
def short_hold_losing_trade() -> Trade:
    """Trade lỗ với holding_days <= 3 (wrong_sell)."""
    return Trade(
        symbol="MWG",
        buy_date=date(2024, 5, 1),
        sell_date=date(2024, 5, 4),
        buy_price=50.0,
        sell_price=47.0,
        shares=100,
        pnl=-300.0,
        pnl_pct=-0.06,
        holding_days=3,
    )


@pytest.fixture
def mixed_trades(
    profitable_trade: Trade,
    losing_trade: Trade,
    breakeven_trade: Trade,
    open_trade: Trade,
    short_hold_losing_trade: Trade,
) -> list:
    """Danh sách hỗn hợp đủ loại trades."""
    return [
        profitable_trade,
        losing_trade,
        breakeven_trade,
        open_trade,
        short_hold_losing_trade,
    ]


# ---------------------------------------------------------------------------
# Tests: identify_incorrect_predictions
# ---------------------------------------------------------------------------


class TestIdentifyIncorrectPredictions:
    """Tests cho identify_incorrect_predictions()."""

    def test_empty_list(self) -> None:
        """Danh sách rỗng trả về rỗng."""
        result = identify_incorrect_predictions([])
        assert result == []

    def test_only_profitable_trades(self, profitable_trade: Trade) -> None:
        """Tất cả trades có lời → không có incorrect."""
        result = identify_incorrect_predictions([profitable_trade])
        assert result == []

    def test_only_losing_trades(self, losing_trade: Trade) -> None:
        """Trade lỗ được nhận diện là incorrect."""
        result = identify_incorrect_predictions([losing_trade])
        assert result == [losing_trade]

    def test_open_trade_excluded(self, open_trade: Trade) -> None:
        """Trade đang mở (pnl=None) không được tính là incorrect."""
        result = identify_incorrect_predictions([open_trade])
        assert result == []

    def test_breakeven_not_incorrect(self, breakeven_trade: Trade) -> None:
        """Trade hòa vốn (pnl=0) không phải incorrect."""
        result = identify_incorrect_predictions([breakeven_trade])
        assert result == []

    def test_mixed_trades(self, mixed_trades: list) -> None:
        """Chỉ trades có pnl < 0 được trả về."""
        result = identify_incorrect_predictions(mixed_trades)
        # Chỉ losing_trade và short_hold_losing_trade có pnl < 0
        assert len(result) == 2
        for trade in result:
            assert trade.pnl is not None
            assert trade.pnl < 0

    def test_preserves_order(self) -> None:
        """Thứ tự trades được giữ nguyên."""
        trades = [
            Trade(symbol="A", buy_date=date(2024, 1, 1), sell_date=date(2024, 1, 5),
                  buy_price=10.0, sell_price=9.0, shares=100, pnl=-100.0, pnl_pct=-0.1),
            Trade(symbol="B", buy_date=date(2024, 2, 1), sell_date=date(2024, 2, 5),
                  buy_price=20.0, sell_price=18.0, shares=100, pnl=-200.0, pnl_pct=-0.1),
        ]
        result = identify_incorrect_predictions(trades)
        assert result[0].symbol == "A"
        assert result[1].symbol == "B"


# ---------------------------------------------------------------------------
# Tests: generate_hard_examples
# ---------------------------------------------------------------------------


class TestGenerateHardExamples:
    """Tests cho generate_hard_examples()."""

    def test_empty_list(self) -> None:
        """Danh sách rỗng trả về rỗng."""
        result = generate_hard_examples([])
        assert result == []

    def test_single_trade(self, losing_trade: Trade) -> None:
        """Một trade lỗ tạo đúng 1 hard example."""
        result = generate_hard_examples([losing_trade])
        assert len(result) == 1

        example = result[0]
        assert example["symbol"] == "HPG"
        assert example["buy_date"] == "2024-02-01"
        assert example["sell_date"] == "2024-02-08"
        assert example["buy_price"] == 30.0
        assert example["sell_price"] == 27.0
        assert example["pnl"] == -600.0
        assert example["pnl_pct"] == -0.10
        assert example["holding_days"] == 5
        assert example["outcome"] == "loss"

    def test_multiple_trades(
        self, losing_trade: Trade, short_hold_losing_trade: Trade
    ) -> None:
        """Nhiều trades tạo tương ứng nhiều hard examples."""
        result = generate_hard_examples([losing_trade, short_hold_losing_trade])
        assert len(result) == 2
        symbols = [ex["symbol"] for ex in result]
        assert "HPG" in symbols
        assert "MWG" in symbols

    def test_output_format_has_required_keys(self, losing_trade: Trade) -> None:
        """Hard example có đủ các keys cần thiết."""
        result = generate_hard_examples([losing_trade])
        expected_keys = {
            "symbol", "buy_date", "sell_date", "buy_price",
            "sell_price", "pnl", "pnl_pct", "holding_days", "outcome",
        }
        assert set(result[0].keys()) == expected_keys

    def test_dates_are_iso_format(self, losing_trade: Trade) -> None:
        """Dates được serialize dạng ISO format string."""
        result = generate_hard_examples([losing_trade])
        example = result[0]
        # Kiểm tra format ISO: YYYY-MM-DD
        assert isinstance(example["buy_date"], str)
        assert len(example["buy_date"]) == 10


# ---------------------------------------------------------------------------
# Tests: calculate_mistake_rate
# ---------------------------------------------------------------------------


class TestCalculateMistakeRate:
    """Tests cho calculate_mistake_rate()."""

    def test_empty_list(self) -> None:
        """Không có trades → 0.0."""
        result = calculate_mistake_rate([])
        assert result == 0.0

    def test_all_profitable(self, profitable_trade: Trade) -> None:
        """Tất cả trades lời → mistake rate = 0."""
        result = calculate_mistake_rate([profitable_trade, profitable_trade])
        assert result == 0.0

    def test_all_losing(self, losing_trade: Trade) -> None:
        """Tất cả trades lỗ → mistake rate = 100."""
        result = calculate_mistake_rate([losing_trade, losing_trade])
        assert result == 100.0

    def test_mixed_trades(self, mixed_trades: list) -> None:
        """Hỗn hợp trades: chỉ tính trên closed trades."""
        # mixed_trades: profitable, losing, breakeven, open, short_hold_losing
        # Closed trades (pnl not None): profitable(0+), losing(-), breakeven(0), short_hold(-)
        # → 4 closed, 2 incorrect → 50%
        result = calculate_mistake_rate(mixed_trades)
        assert result == 50.0

    def test_only_open_trades(self, open_trade: Trade) -> None:
        """Chỉ có trades đang mở → 0.0 (không có closed)."""
        result = calculate_mistake_rate([open_trade, open_trade])
        assert result == 0.0

    def test_result_range(self) -> None:
        """Kết quả luôn nằm trong [0.0, 100.0]."""
        trades = [
            Trade(symbol="X", buy_date=date(2024, 1, 1), sell_date=date(2024, 1, 5),
                  buy_price=10.0, sell_price=9.0, shares=100, pnl=-100.0),
            Trade(symbol="Y", buy_date=date(2024, 2, 1), sell_date=date(2024, 2, 5),
                  buy_price=10.0, sell_price=11.0, shares=100, pnl=100.0),
            Trade(symbol="Z", buy_date=date(2024, 3, 1), sell_date=None,
                  buy_price=10.0, sell_price=None, shares=100, pnl=None),
        ]
        result = calculate_mistake_rate(trades)
        assert 0.0 <= result <= 100.0


# ---------------------------------------------------------------------------
# Tests: categorize_mistakes
# ---------------------------------------------------------------------------


class TestCategorizeMistakes:
    """Tests cho categorize_mistakes()."""

    def test_empty_list(self) -> None:
        """Danh sách rỗng → tất cả categories rỗng."""
        result = categorize_mistakes([])
        assert result == {"wrong_buy": [], "wrong_sell": [], "timing_error": []}

    def test_wrong_sell_short_holding(self, short_hold_losing_trade: Trade) -> None:
        """Trade lỗ với holding_days <= 3 → wrong_sell."""
        result = categorize_mistakes([short_hold_losing_trade])
        assert len(result["wrong_sell"]) == 1
        assert len(result["wrong_buy"]) == 0
        assert len(result["timing_error"]) == 0

    def test_wrong_buy_normal_holding(self, losing_trade: Trade) -> None:
        """Trade lỗ với holding_days > 3, có cả buy/sell date → wrong_buy."""
        result = categorize_mistakes([losing_trade])
        assert len(result["wrong_buy"]) == 1
        assert len(result["wrong_sell"]) == 0
        assert len(result["timing_error"]) == 0

    def test_timing_error_no_sell_date(self) -> None:
        """Trade lỗ không có sell_date → timing_error."""
        trade = Trade(
            symbol="TCB",
            buy_date=date(2024, 6, 1),
            sell_date=None,
            buy_price=40.0,
            sell_price=None,
            shares=100,
            pnl=-200.0,
            pnl_pct=-0.05,
            holding_days=10,
        )
        result = categorize_mistakes([trade])
        assert len(result["timing_error"]) == 1
        assert len(result["wrong_buy"]) == 0
        assert len(result["wrong_sell"]) == 0

    def test_all_categories_populated(
        self, losing_trade: Trade, short_hold_losing_trade: Trade
    ) -> None:
        """Nhiều loại lỗi được phân loại đúng."""
        timing_trade = Trade(
            symbol="ACB",
            buy_date=date(2024, 7, 1),
            sell_date=None,
            buy_price=25.0,
            sell_price=None,
            shares=100,
            pnl=-150.0,
            pnl_pct=-0.06,
            holding_days=7,
        )
        result = categorize_mistakes([losing_trade, short_hold_losing_trade, timing_trade])
        assert len(result["wrong_buy"]) == 1
        assert len(result["wrong_sell"]) == 1
        assert len(result["timing_error"]) == 1

    def test_returns_all_required_keys(self) -> None:
        """Kết quả luôn có đủ 3 keys."""
        result = categorize_mistakes([])
        assert "wrong_buy" in result
        assert "wrong_sell" in result
        assert "timing_error" in result
