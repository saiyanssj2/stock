# -*- coding: utf-8 -*-
"""
Backtest Report - Lưu kết quả backtest chi tiết ra file CSV/JSON.

Mỗi lần auto-learner cycle hoặc manual backtest chạy xong,
lưu danh sách trades với đầy đủ thông tin:
- Mã, ngày mua, giá mua, khối lượng
- Ngày bán, giá bán, số ngày giữ
- Lãi/lỗ (VND và %)

Cũng lưu summary tổng hợp: vốn đầu, vốn cuối, tổng lãi, win rate, sharpe.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from models.backtest_models import AutoBacktestResult, BacktestResult, Trade

logger = logging.getLogger(__name__)

# Thư mục lưu reports
REPORTS_DIR = "data/engine/reports"

# Hệ số giá: CSV lưu đơn vị 1000 VND
PRICE_SCALE = 1000.0


def save_backtest_report(
    result: AutoBacktestResult,
    initial_capital: float = 100_000_000.0,
    report_dir: str = REPORTS_DIR,
) -> Optional[str]:
    """
    Lưu report chi tiết sau mỗi auto-backtest cycle.

    Tạo 2 file:
    - trades_cycle_{N}_{timestamp}.csv: Danh sách trades chi tiết
    - summary_cycle_{N}_{timestamp}.json: Tổng hợp metrics

    Args:
        result: AutoBacktestResult từ auto-learner cycle
        initial_capital: Vốn ban đầu (VND)
        report_dir: Thư mục lưu reports

    Returns:
        Đường dẫn file CSV trades (hoặc None nếu lỗi)
    """
    dir_path = Path(report_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cycle_num = result.cycle_number

    # === Lưu trades CSV ===
    csv_path = dir_path / f"trades_cycle_{cycle_num}_{timestamp}.csv"
    trades_data = _trades_to_records(result.trades)

    if trades_data:
        df = pd.DataFrame(trades_data)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"[BacktestReport] Saved {len(trades_data)} trades → {csv_path}")
    else:
        logger.warning("[BacktestReport] Không có trades để lưu")
        csv_path = None

    # === Lưu summary JSON ===
    summary_path = dir_path / f"summary_cycle_{cycle_num}_{timestamp}.json"
    summary = _build_summary(result, initial_capital)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"[BacktestReport] Summary → {summary_path}")
    return str(csv_path) if csv_path else None


def save_manual_backtest_report(
    result: BacktestResult,
    report_dir: str = REPORTS_DIR,
) -> Optional[str]:
    """
    Lưu report chi tiết cho manual backtest (1 symbol).

    Args:
        result: BacktestResult từ manual backtest
        report_dir: Thư mục lưu reports

    Returns:
        Đường dẫn file CSV trades
    """
    dir_path = Path(report_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Trades CSV
    csv_path = dir_path / f"trades_{result.symbol}_{timestamp}.csv"
    trades_data = _trades_to_records(result.trades)

    if trades_data:
        df = pd.DataFrame(trades_data)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        csv_path = None

    # Summary JSON
    summary_path = dir_path / f"summary_{result.symbol}_{timestamp}.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "symbol": result.symbol,
        "start_date": str(result.start_date),
        "end_date": str(result.end_date),
        "initial_capital_vnd": result.initial_capital,
        "final_capital_vnd": result.final_capital,
        "total_return_pct": round(result.total_return, 2),
        "profit_vnd": round(result.final_capital - result.initial_capital, 0),
        "sharpe_ratio": round(result.sharpe_ratio, 3),
        "win_rate_pct": round(result.win_rate, 1),
        "max_drawdown_pct": round(result.max_drawdown, 2),
        "total_trades": result.total_trades,
        "winning_trades": sum(1 for t in result.trades if t.pnl and t.pnl > 0),
        "losing_trades": sum(1 for t in result.trades if t.pnl and t.pnl < 0),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return str(csv_path) if csv_path else None


def _trades_to_records(trades: List[Trade]) -> List[dict]:
    """
    Chuyển danh sách Trade thành list of dict cho DataFrame.

    Mỗi record chứa đầy đủ thông tin giao dịch:
    - Mã, ngày mua/bán, giá mua/bán (VND), khối lượng
    - Số ngày giữ, lãi/lỗ VND, lãi/lỗ %
    """
    records = []
    for t in trades:
        buy_price_vnd = t.buy_price * PRICE_SCALE if t.buy_price else 0
        sell_price_vnd = t.sell_price * PRICE_SCALE if t.sell_price else 0
        cost = buy_price_vnd * t.shares
        revenue = sell_price_vnd * t.shares

        records.append({
            "ma_co_phieu": t.symbol,
            "ngay_mua": str(t.buy_date) if t.buy_date else "",
            "gia_mua_vnd": round(buy_price_vnd, 0),
            "khoi_luong": t.shares,
            "tong_tien_mua_vnd": round(cost, 0),
            "ngay_ban": str(t.sell_date) if t.sell_date else "(đang giữ)",
            "gia_ban_vnd": round(sell_price_vnd, 0) if t.sell_price else 0,
            "tong_tien_ban_vnd": round(revenue, 0) if t.sell_price else 0,
            "so_ngay_giu": t.holding_days,
            "lai_lo_vnd": round(t.pnl, 0) if t.pnl is not None else 0,
            "lai_lo_pct": round(t.pnl_pct * 100, 2) if t.pnl_pct is not None else 0,
            "ket_qua": "Lãi" if (t.pnl and t.pnl > 0) else "Lỗ" if (t.pnl and t.pnl < 0) else "Đang giữ",
        })

    return records


def _build_summary(result: AutoBacktestResult, initial_capital: float) -> dict:
    """Tạo summary dict từ AutoBacktestResult."""
    # Tính tổng PnL từ trades
    total_pnl = sum(t.pnl for t in result.trades if t.pnl is not None)
    winning = [t for t in result.trades if t.pnl is not None and t.pnl > 0]
    losing = [t for t in result.trades if t.pnl is not None and t.pnl < 0]

    return {
        "timestamp": datetime.now().isoformat(),
        "cycle_number": result.cycle_number,
        "symbols_tested": result.symbols_tested,
        "initial_capital_vnd": initial_capital,
        "total_pnl_vnd": round(total_pnl, 0),
        "total_return_pct": round(result.overall_return, 2),
        "sharpe_ratio": round(result.overall_sharpe, 3),
        "win_rate_pct": round(result.overall_win_rate, 1),
        "total_trades": len(result.trades),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "avg_holding_days": round(
            sum(t.holding_days for t in result.trades) / max(len(result.trades), 1), 1
        ),
        "best_trade_vnd": round(max((t.pnl for t in result.trades if t.pnl), default=0), 0),
        "worst_trade_vnd": round(min((t.pnl for t in result.trades if t.pnl), default=0), 0),
        "strategies_beaten": result.strategies_beaten,
        "duration_seconds": round(result.duration_seconds, 1),
    }
