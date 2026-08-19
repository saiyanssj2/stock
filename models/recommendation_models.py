"""
Recommendation models - Định nghĩa các model liên quan đến khuyến nghị cổ phiếu.

Bao gồm:
- Action: Hành động khuyến nghị (BUY, HOLD, SELL)
- HoldingInfo: Thông tin một mã cổ phiếu đang nắm giữ
- Recommendation: Chi tiết khuyến nghị cho một mã cổ phiếu

References: Req 5.3, 6.2, 6.3
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional


class Action(Enum):
    """Hành động khuyến nghị cho một mã cổ phiếu."""

    BUY = "buy"    # Khuyến nghị mua
    HOLD = "hold"  # Giữ nguyên vị thế
    SELL = "sell"  # Khuyến nghị bán


@dataclass
class HoldingInfo:
    """
    Thông tin chi tiết về một mã cổ phiếu đang nắm giữ.

    Attributes:
        shares: Số lượng cổ phiếu đang giữ
        buy_price: Giá mua trung bình (đơn vị 1000 VND). None nếu không biết.
        buy_date: Ngày mua. None nếu không biết.
    """

    shares: int
    buy_price: Optional[float] = None
    buy_date: Optional[date] = None


@dataclass
class Recommendation:
    """
    Một khuyến nghị mã cổ phiếu từ RecommendationEngine.

    Được tạo sau khi model phân tích data ngày T,
    khuyến nghị mua ngày T+1, bán sớm nhất ngày T+3 (ceiling T+2.5).

    Attributes:
        symbol: Mã cổ phiếu (VD: "FPT", "VNM")
        action: Hành động khuyến nghị (BUY, HOLD, SELL)
        confidence: Độ tin cậy từ 0.0 đến 1.0
        position_score: Điểm vị thế từ -1.0 (strong sell) đến 1.0 (strong buy)
        expected_holding_days: Số ngày dự kiến giữ cổ phiếu
        earliest_sell_date: Ngày bán sớm nhất theo luật T+2.5
        analysis_date: Ngày T phân tích
        buy_date: Ngày T+1 mua
        model_version: Phiên bản model đã dùng để tạo khuyến nghị
        recommended_buy_price: Giá mua khuyến nghị (giá đóng cửa ngày T, đơn vị 1000 VND).
            None nếu không có data.
        recommended_shares: Số cổ phiếu nên mua/bán (bội số 100). None nếu chưa tính.
        recommended_sell_date: Ngày nên bán (buy_date + expected_holding_days ngày giao dịch).
            None nếu chưa tính.
    """

    symbol: str
    action: Action
    confidence: float
    position_score: float
    expected_holding_days: int
    earliest_sell_date: date
    analysis_date: date
    buy_date: date
    model_version: str
    recommended_buy_price: Optional[float] = None
    recommended_shares: Optional[int] = None  # Số CP nên mua/bán (bội số 100)
    recommended_sell_date: Optional[date] = None  # Ngày nên bán (buy_date + holding days)
