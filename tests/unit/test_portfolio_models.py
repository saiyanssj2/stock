# -*- coding: utf-8 -*-
"""
Tests cho models/portfolio_models.py
"""

import pytest
from datetime import date, datetime

from models.portfolio_models import PortfolioHolding, PortfolioState


class TestPortfolioHolding:
    """Tests cho PortfolioHolding dataclass."""

    def test_create_holding_happy_path(self) -> None:
        """Tạo holding với đầy đủ thông tin."""
        holding = PortfolioHolding(
            symbol="FPT",
            shares=1000,
            buy_price=76.5,
            buy_date=date(2025, 6, 10),
        )
        assert holding.symbol == "FPT"
        assert holding.shares == 1000
        assert holding.buy_price == 76.5
        assert holding.buy_date == date(2025, 6, 10)

    def test_create_holding_edge_case_zero_shares(self) -> None:
        """Holding với 0 shares vẫn tạo được (validation ở layer khác)."""
        holding = PortfolioHolding(
            symbol="VNM",
            shares=0,
            buy_price=50.0,
            buy_date=date(2025, 1, 1),
        )
        assert holding.shares == 0


class TestPortfolioState:
    """Tests cho PortfolioState dataclass."""

    def test_create_default_state(self) -> None:
        """Tạo state mặc định: 1 tỉ VND, rỗng holdings."""
        state = PortfolioState(cash=1_000_000_000.0)
        assert state.cash == 1_000_000_000.0
        assert state.holdings == []
        assert state.initial_capital == 1_000_000_000.0

    def test_create_state_with_holdings(self) -> None:
        """Tạo state với holdings có sẵn."""
        holdings = [
            PortfolioHolding("FPT", 1000, 76.5, date(2025, 6, 10)),
            PortfolioHolding("VNM", 500, 60.0, date(2025, 6, 8)),
        ]
        state = PortfolioState(
            cash=500_000_000.0,
            holdings=holdings,
            initial_capital=1_000_000_000.0,
        )
        assert len(state.holdings) == 2
        assert state.holdings[0].symbol == "FPT"
        assert state.holdings[1].symbol == "VNM"

    def test_timestamps_auto_generated(self) -> None:
        """created_at và last_updated tự động có giá trị."""
        state = PortfolioState(cash=1_000_000_000.0)
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.last_updated, datetime)
