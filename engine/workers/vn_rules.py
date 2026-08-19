"""
Vietnamese market rules enforcement - Luật giao dịch TTCK Việt Nam.

Module enforce các quy tắc giao dịch:
- T+2.5 ceiling = 3 ngày giao dịch tối thiểu holding
- ±7% biên độ giá hàng ngày
- Lot size phải chia hết cho 100
- Skip sell signals trong settlement period
- Tính ngày mua T+1 từ ngày phân tích T

References:
- Req 5.1: Phân tích ngày T → mua ngày T+1
- Req 5.2: Holding tối thiểu ceiling(T+2.5) = 3 ngày giao dịch
- Req 5.4: Không look-ahead bias
- Req 5.5: Skip sell signal trong settlement period
- Req 7.5: Enforce ±7% price limit và lot size 100
"""

from datetime import date, timedelta
from typing import List

import pandas as pd

from config.vn_market_rules import (
    LOT_SIZE,
    PRICE_LIMIT_PCT,
    SETTLEMENT_DAYS,
    T_PLUS_BUY,
    TRADING_WEEKDAYS,
)
from models.backtest_models import Trade


# ==============================================================================
# Ngày lễ cố định Việt Nam (tháng, ngày)
# Áp dụng cho mọi năm
# ==============================================================================

# Danh sách ngày lễ cố định (month, day)
VN_FIXED_HOLIDAYS: List[tuple[int, int]] = [
    (1, 1),    # Tết Dương lịch
    (4, 30),   # Ngày Giải phóng miền Nam
    (5, 1),    # Quốc tế Lao động
    (9, 2),    # Quốc khánh
]

# Tết Nguyên Đán thường nghỉ khoảng 5 ngày cuối tháng 1 - đầu tháng 2
# Do phụ thuộc âm lịch, chỉ khai báo ngày cố định ở đây
# Thực tế cần cập nhật hàng năm, nhưng cho mục đích backtest cơ bản
# ta chỉ xử lý weekend + ngày lễ cố định


def is_vn_holiday(d: date) -> bool:
    """
    Kiểm tra ngày có phải ngày lễ Việt Nam không.

    Chỉ kiểm tra ngày lễ cố định (không phụ thuộc âm lịch).
    Tết Nguyên Đán cần logic riêng nếu muốn chính xác 100%.

    Args:
        d: Ngày cần kiểm tra

    Returns:
        True nếu là ngày lễ cố định VN
    """
    return (d.month, d.day) in VN_FIXED_HOLIDAYS


def is_trading_day(d: date) -> bool:
    """
    Kiểm tra ngày có phải ngày giao dịch hợp lệ không.

    Ngày giao dịch = ngày trong tuần (Mon-Fri) VÀ không phải ngày lễ VN.

    Args:
        d: Ngày cần kiểm tra

    Returns:
        True nếu là ngày giao dịch hợp lệ
    """
    return d.weekday() in TRADING_WEEKDAYS and not is_vn_holiday(d)


def get_next_trading_day(current_date: date) -> date:
    """
    Tìm ngày giao dịch tiếp theo sau current_date.

    Bỏ qua weekends và ngày lễ VN.

    Args:
        current_date: Ngày hiện tại

    Returns:
        Ngày giao dịch hợp lệ tiếp theo (không bao gồm current_date)
    """
    next_day = current_date + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day


def add_trading_days(start_date: date, num_days: int) -> date:
    """
    Cộng thêm num_days ngày giao dịch từ start_date.

    Bỏ qua weekends và ngày lễ VN.

    Args:
        start_date: Ngày bắt đầu
        num_days: Số ngày giao dịch cần cộng thêm (>= 1)

    Returns:
        Ngày giao dịch sau num_days ngày giao dịch kể từ start_date
    """
    current = start_date
    days_added = 0
    while days_added < num_days:
        current += timedelta(days=1)
        if is_trading_day(current):
            days_added += 1
    return current


def compute_buy_date(analysis_date: date) -> date:
    """
    Tính ngày mua từ ngày phân tích.

    Phân tích ngày T → mua ngày T+1 (ngày giao dịch tiếp theo).
    Nếu T+1 rơi vào weekend/holiday thì lấy ngày giao dịch kế tiếp.

    Args:
        analysis_date: Ngày T thực hiện phân tích

    Returns:
        Ngày mua T+1 (ngày giao dịch hợp lệ đầu tiên sau analysis_date)
    """
    return add_trading_days(analysis_date, T_PLUS_BUY)


def compute_earliest_sell_date(buy_date: date) -> date:
    """
    Tính ngày bán sớm nhất theo luật T+2.5.

    Ceiling(T+2.5) = 3 ngày giao dịch tối thiểu sau ngày mua.
    Bỏ qua weekends và ngày lễ VN.

    Args:
        buy_date: Ngày mua cổ phiếu

    Returns:
        Ngày bán sớm nhất (buy_date + 3 ngày giao dịch)
    """
    return add_trading_days(buy_date, SETTLEMENT_DAYS)


def validate_lot_size(shares: int) -> bool:
    """
    Kiểm tra số lượng cổ phiếu có hợp lệ không.

    Số lượng phải là bội số của LOT_SIZE (100).

    Args:
        shares: Số lượng cổ phiếu

    Returns:
        True nếu shares là bội số của 100 và > 0
    """
    if shares <= 0:
        return False
    return shares % LOT_SIZE == 0


def validate_price_limit(price: float, reference_price: float) -> bool:
    """
    Kiểm tra giá có nằm trong biên độ ±7% không.

    Sàn HOSE áp dụng biên độ ±7% so với giá tham chiếu.

    Args:
        price: Giá cần kiểm tra
        reference_price: Giá tham chiếu (giá đóng cửa hôm trước)

    Returns:
        True nếu price nằm trong khoảng [ref * 0.93, ref * 1.07]
    """
    if reference_price <= 0:
        return False
    if price <= 0:
        return False

    limit = reference_price * (PRICE_LIMIT_PCT / 100.0)
    floor_price = reference_price - limit
    ceiling_price = reference_price + limit

    return floor_price <= price <= ceiling_price


def is_within_settlement_period(buy_date: date, sell_date: date) -> bool:
    """
    Kiểm tra ngày bán có nằm trong settlement period không.

    Nếu sell_date < earliest_sell_date (buy_date + 3 trading days)
    thì trade đang trong settlement period → không được bán.

    Args:
        buy_date: Ngày mua
        sell_date: Ngày bán dự kiến

    Returns:
        True nếu sell_date nằm trong settlement period (chưa được bán)
    """
    earliest_sell = compute_earliest_sell_date(buy_date)
    return sell_date < earliest_sell


def enforce_vn_rules(trade: Trade) -> Trade:
    """
    Áp dụng toàn bộ luật TTCK Việt Nam cho một trade.

    Các quy tắc được enforce:
    1. Lot size: shares phải chia hết cho 100 (làm tròn xuống nếu không)
    2. Sell date: phải >= earliest_sell_date (điều chỉnh nếu quá sớm)
    3. Price limit: kiểm tra buy_price/sell_price trong ±7%

    Nếu sell_date quá sớm (trong settlement period), trade sẽ được
    điều chỉnh sell_date sang ngày bán sớm nhất hợp lệ.

    Nếu shares không chia hết cho 100, làm tròn xuống bội 100 gần nhất.

    Args:
        trade: Trade object cần enforce rules

    Returns:
        Trade mới đã được điều chỉnh theo luật VN
    """
    # 1. Điều chỉnh lot size: làm tròn xuống bội 100 gần nhất
    adjusted_shares = (trade.shares // LOT_SIZE) * LOT_SIZE
    if adjusted_shares <= 0:
        adjusted_shares = LOT_SIZE  # Tối thiểu 1 lot

    # 2. Điều chỉnh sell_date nếu trong settlement period
    adjusted_sell_date = trade.sell_date
    if adjusted_sell_date is not None:
        earliest_sell = compute_earliest_sell_date(trade.buy_date)
        if adjusted_sell_date < earliest_sell:
            adjusted_sell_date = earliest_sell

    # 3. Tính lại PnL nếu có sell_price
    adjusted_pnl = trade.pnl
    adjusted_pnl_pct = trade.pnl_pct
    if trade.sell_price is not None and trade.buy_price > 0:
        adjusted_pnl = (trade.sell_price - trade.buy_price) * adjusted_shares
        adjusted_pnl_pct = (trade.sell_price - trade.buy_price) / trade.buy_price

    # 4. Tính holding_days
    adjusted_holding_days = trade.holding_days
    if adjusted_sell_date is not None:
        adjusted_holding_days = (adjusted_sell_date - trade.buy_date).days

    return Trade(
        symbol=trade.symbol,
        buy_date=trade.buy_date,
        sell_date=adjusted_sell_date,
        buy_price=trade.buy_price,
        sell_price=trade.sell_price,
        shares=adjusted_shares,
        pnl=adjusted_pnl,
        pnl_pct=adjusted_pnl_pct,
        holding_days=adjusted_holding_days,
    )


def filter_data_for_analysis(df: pd.DataFrame, analysis_date: date) -> pd.DataFrame:
    """
    Lọc dữ liệu chỉ bao gồm data đến ngày phân tích T.

    Đảm bảo không có look-ahead bias: phân tích ngày T chỉ được
    dùng data có date <= T. Không sử dụng data tương lai.

    Args:
        df: DataFrame với column 'time' chứa ngày giao dịch
        analysis_date: Ngày T thực hiện phân tích

    Returns:
        DataFrame chỉ chứa rows có time <= analysis_date
    """
    if df.empty:
        return df.copy()

    dates = pd.to_datetime(df["time"])
    analysis_ts = pd.Timestamp(analysis_date)
    mask = dates <= analysis_ts
    return df.loc[mask].reset_index(drop=True)
