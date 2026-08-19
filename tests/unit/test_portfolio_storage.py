# -*- coding: utf-8 -*-
"""
Tests cho engine/portfolio_storage.py
"""

import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path

import pytest

from engine.portfolio_storage import (
    _create_default_portfolio,
    _deserialize_portfolio,
    _serialize_portfolio,
    load_portfolio,
    save_portfolio,
)
from models.portfolio_models import PortfolioHolding, PortfolioState


@pytest.fixture
def tmp_portfolio_path(tmp_path: Path) -> str:
    """Tạo path tạm cho portfolio file."""
    return str(tmp_path / "portfolio.json")


@pytest.fixture
def sample_state() -> PortfolioState:
    """Tạo sample PortfolioState để test."""
    return PortfolioState(
        cash=500_000_000.0,
        holdings=[
            PortfolioHolding("FPT", 1000, 76.5, date(2025, 6, 10)),
            PortfolioHolding("VNM", 500, 60.0, date(2025, 6, 8)),
        ],
        initial_capital=1_000_000_000.0,
        created_at=datetime(2025, 1, 1, 8, 0, 0),
        last_updated=datetime(2025, 6, 15, 10, 30, 0),
    )


class TestSerialization:
    """Tests cho serialize/deserialize."""

    def test_serialize_roundtrip(self, sample_state: PortfolioState) -> None:
        """Serialize rồi deserialize phải trả về giá trị tương đương."""
        data = _serialize_portfolio(sample_state)
        restored = _deserialize_portfolio(data)

        assert restored.cash == sample_state.cash
        assert restored.initial_capital == sample_state.initial_capital
        assert len(restored.holdings) == 2
        assert restored.holdings[0].symbol == "FPT"
        assert restored.holdings[0].shares == 1000
        assert restored.holdings[0].buy_price == 76.5
        assert restored.holdings[0].buy_date == date(2025, 6, 10)

    def test_serialize_empty_holdings(self) -> None:
        """Serialize state không có holdings."""
        state = PortfolioState(cash=1_000_000_000.0)
        data = _serialize_portfolio(state)
        assert data["holdings"] == []
        assert data["cash"] == 1_000_000_000.0


class TestSaveLoad:
    """Tests cho save/load file operations."""

    def test_save_and_load(self, tmp_portfolio_path: str, sample_state: PortfolioState) -> None:
        """Save rồi load phải trả về giá trị tương đương."""
        save_portfolio(sample_state, tmp_portfolio_path)
        loaded = load_portfolio(tmp_portfolio_path)

        assert loaded is not None
        assert loaded.cash == sample_state.cash
        assert len(loaded.holdings) == 2

    def test_load_creates_default_when_no_file(self, tmp_portfolio_path: str) -> None:
        """Load khi file chưa tồn tại → tạo mặc định 1 tỉ."""
        assert not Path(tmp_portfolio_path).exists()
        loaded = load_portfolio(tmp_portfolio_path)

        assert loaded is not None
        assert loaded.cash == 1_000_000_000.0
        assert loaded.holdings == []
        assert Path(tmp_portfolio_path).exists()

    def test_load_returns_none_on_corrupt_file(self, tmp_portfolio_path: str) -> None:
        """Load file bị corrupt → trả về None."""
        Path(tmp_portfolio_path).parent.mkdir(parents=True, exist_ok=True)
        Path(tmp_portfolio_path).write_text("not valid json {{{", encoding="utf-8")
        loaded = load_portfolio(tmp_portfolio_path)
        assert loaded is None

    def test_atomic_write_no_corruption(self, tmp_portfolio_path: str, sample_state: PortfolioState) -> None:
        """Ghi atomic: file cuối cùng phải là JSON hợp lệ."""
        save_portfolio(sample_state, tmp_portfolio_path)
        content = Path(tmp_portfolio_path).read_text(encoding="utf-8")
        data = json.loads(content)
        assert "cash" in data
        assert "holdings" in data


class TestDefaultPortfolio:
    """Tests cho _create_default_portfolio."""

    def test_default_has_1_billion(self) -> None:
        """Portfolio mặc định = 1 tỉ VND."""
        state = _create_default_portfolio()
        assert state.cash == 1_000_000_000.0
        assert state.initial_capital == 1_000_000_000.0
        assert state.holdings == []
