# -*- coding: utf-8 -*-
"""
Portfolio Storage - Lưu/đọc trạng thái portfolio từ file JSON.

Đảm bảo ghi atomic (dùng temp file + os.replace) để tránh corrupt data
khi app crash giữa chừng.

File path: data/portfolio.json
"""

import json
import logging
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from models.portfolio_models import PortfolioHolding, PortfolioState

logger = logging.getLogger(__name__)

# Đường dẫn file portfolio mặc định
PORTFOLIO_FILE_PATH: str = "data/portfolio.json"

# Vốn ban đầu mặc định (1 tỉ VND)
DEFAULT_INITIAL_CAPITAL: float = 1_000_000_000.0


def _serialize_portfolio(state: PortfolioState) -> dict:
    """
    Chuyển PortfolioState thành dict để serialize JSON.

    Args:
        state: Trạng thái portfolio cần serialize

    Returns:
        Dict có thể json.dumps
    """
    return {
        "cash": state.cash,
        "initial_capital": state.initial_capital,
        "created_at": state.created_at.isoformat(),
        "last_updated": state.last_updated.isoformat(),
        "holdings": [
            {
                "symbol": h.symbol,
                "shares": h.shares,
                "buy_price": h.buy_price,
                "buy_date": h.buy_date.isoformat(),
            }
            for h in state.holdings
        ],
    }


def _deserialize_portfolio(data: dict) -> PortfolioState:
    """
    Chuyển dict từ JSON thành PortfolioState.

    Args:
        data: Dict đọc từ json.loads

    Returns:
        PortfolioState instance
    """
    holdings = [
        PortfolioHolding(
            symbol=h["symbol"],
            shares=h["shares"],
            buy_price=h["buy_price"],
            buy_date=date.fromisoformat(h["buy_date"]),
        )
        for h in data.get("holdings", [])
    ]

    return PortfolioState(
        cash=data["cash"],
        holdings=holdings,
        initial_capital=data.get("initial_capital", DEFAULT_INITIAL_CAPITAL),
        created_at=datetime.fromisoformat(data["created_at"]),
        last_updated=datetime.fromisoformat(data["last_updated"]),
    )


def save_portfolio(state: PortfolioState, file_path: str = PORTFOLIO_FILE_PATH) -> None:
    """
    Lưu portfolio state ra file JSON (atomic write).

    Sử dụng temp file + os.replace để đảm bảo không corrupt data
    nếu app crash giữa quá trình ghi.

    Args:
        state: Trạng thái portfolio cần lưu
        file_path: Đường dẫn file (mặc định data/portfolio.json)
    """
    # Cập nhật timestamp
    state.last_updated = datetime.now()

    # Đảm bảo thư mục tồn tại
    target_path = Path(file_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Ghi atomic: temp file → os.replace
    data = _serialize_portfolio(state)
    dir_path = str(target_path.parent)

    try:
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix="portfolio_",
            dir=dir_path,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        os.replace(tmp_path, str(target_path))
        logger.info(f"Portfolio đã lưu: {file_path}")
    except Exception as e:
        # Cleanup temp file nếu có lỗi
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.error(f"Lỗi lưu portfolio: {e}")
        raise


def load_portfolio(file_path: str = PORTFOLIO_FILE_PATH) -> Optional[PortfolioState]:
    """
    Đọc portfolio state từ file JSON.

    Nếu file không tồn tại → tạo mới với vốn mặc định 1 tỉ VND.
    Nếu file bị lỗi format → trả về None.

    Args:
        file_path: Đường dẫn file (mặc định data/portfolio.json)

    Returns:
        PortfolioState nếu đọc thành công, None nếu file corrupt
    """
    target_path = Path(file_path)

    if not target_path.exists():
        # Tạo portfolio mặc định
        default_state = _create_default_portfolio()
        save_portfolio(default_state, file_path)
        return default_state

    try:
        content = target_path.read_text(encoding="utf-8")
        data = json.loads(content)
        return _deserialize_portfolio(data)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.error(f"Lỗi đọc portfolio file: {e}")
        return None


def _create_default_portfolio() -> PortfolioState:
    """
    Tạo portfolio mặc định với 1 tỉ VND tiền mặt.

    Returns:
        PortfolioState mặc định
    """
    now = datetime.now()
    return PortfolioState(
        cash=DEFAULT_INITIAL_CAPITAL,
        holdings=[],
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        created_at=now,
        last_updated=now,
    )
