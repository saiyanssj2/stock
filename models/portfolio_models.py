# -*- coding: utf-8 -*-
"""
Portfolio models - Định nghĩa cấu trúc dữ liệu cho portfolio tracker.

Bao gồm:
- PortfolioHolding: Thông tin một vị thế cổ phiếu trong portfolio
- PortfolioState: Trạng thái toàn bộ portfolio (tiền mặt + các vị thế)
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List


@dataclass
class PortfolioHolding:
    """
    Một vị thế cổ phiếu trong portfolio.

    Attributes:
        symbol: Mã cổ phiếu (VD: "FPT", "VNM")
        shares: Số lượng cổ phiếu (bội số 100)
        buy_price: Giá mua trung bình (đơn vị 1000 VND)
        buy_date: Ngày mua vào
    """

    symbol: str
    shares: int
    buy_price: float  # đơn vị 1000 VND
    buy_date: date


@dataclass
class PortfolioState:
    """
    Trạng thái toàn bộ portfolio.

    Attributes:
        cash: Tiền mặt khả dụng (VND)
        holdings: Danh sách các vị thế đang nắm giữ
        initial_capital: Vốn ban đầu (VND)
        created_at: Thời điểm tạo portfolio
        last_updated: Thời điểm cập nhật gần nhất
    """

    cash: float  # VND
    holdings: List[PortfolioHolding] = field(default_factory=list)
    initial_capital: float = 1_000_000_000.0  # VND, mặc định 1 tỉ
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
