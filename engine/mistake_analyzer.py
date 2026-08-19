"""
Mistake Analyzer — Phân tích dự đoán sai và tạo hard examples cho Auto-Learner.

Module phục vụ vòng lặp tự học: backtest → phân tích sai lầm → retrain.
Một trade được coi là "incorrect prediction" khi pnl < 0 (model dự đoán BUY/SELL
nhưng kết quả thực tế lại lỗ).

- Trade có pnl = None (vị thế đang mở) KHÔNG được phân loại là incorrect.
- Trade có pnl >= 0 là dự đoán đúng.

References: Requirements 3.2, 3.3
"""

from typing import Dict, List

from models.backtest_models import Trade


def identify_incorrect_predictions(trades: List[Trade]) -> List[Trade]:
    """Xác định các trades có dự đoán sai (pnl < 0).

    Một trade là incorrect nếu và chỉ nếu:
    - pnl is not None (đã đóng vị thế)
    - pnl < 0 (kết quả lỗ → model dự đoán sai)

    Trades với pnl = None (đang giữ) không bị tính là sai.
    Trades với pnl >= 0 là dự đoán đúng.

    Args:
        trades: Danh sách trades từ backtest.

    Returns:
        Danh sách trades có pnl < 0 (incorrect predictions).
    """
    return [
        trade for trade in trades
        if trade.pnl is not None and trade.pnl < 0
    ]


def generate_hard_examples(incorrect_trades: List[Trade]) -> List[dict]:
    """Tạo hard examples từ trades dự đoán sai cho training cycle tiếp theo.

    Mỗi hard example chứa thông tin symbol, khoảng ngày, và kết quả
    để model tập trung học lại các trường hợp đã sai.

    Args:
        incorrect_trades: Danh sách trades có pnl < 0.

    Returns:
        Danh sách dict chứa thông tin cho retraining.
    """
    hard_examples: List[dict] = []

    for trade in incorrect_trades:
        example: dict = {
            "symbol": trade.symbol,
            "buy_date": trade.buy_date.isoformat() if trade.buy_date else None,
            "sell_date": trade.sell_date.isoformat() if trade.sell_date else None,
            "buy_price": trade.buy_price,
            "sell_price": trade.sell_price,
            "pnl": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "holding_days": trade.holding_days,
            "outcome": "loss",
        }
        hard_examples.append(example)

    return hard_examples


def calculate_mistake_rate(trades: List[Trade]) -> float:
    """Tính tỷ lệ phần trăm trades sai trong tổng số trades đã đóng.

    Chỉ tính các trades đã đóng (pnl is not None).
    Trả về 0.0 nếu không có trades đã đóng.

    Args:
        trades: Danh sách trades từ backtest.

    Returns:
        Tỷ lệ phần trăm mistakes (0.0 - 100.0).
    """
    # Chỉ xét trades đã đóng (có pnl)
    closed_trades = [t for t in trades if t.pnl is not None]

    if not closed_trades:
        return 0.0

    incorrect_count = sum(1 for t in closed_trades if t.pnl < 0)
    return (incorrect_count / len(closed_trades)) * 100.0


def categorize_mistakes(incorrect_trades: List[Trade]) -> Dict[str, List[Trade]]:
    """Phân loại mistakes theo loại lỗi.

    Các loại:
    - "wrong_buy": Model khuyên mua nhưng giá giảm (pnl < 0, có buy_date và sell_date)
    - "wrong_sell": Model bán sớm, lỗ do timing (pnl < 0, holding_days <= 3)
    - "timing_error": Lỗi timing — giữ quá lâu hoặc vào sai thời điểm (còn lại)

    Args:
        incorrect_trades: Danh sách trades có pnl < 0.

    Returns:
        Dict phân loại mistakes theo category.
    """
    categories: Dict[str, List[Trade]] = {
        "wrong_buy": [],
        "wrong_sell": [],
        "timing_error": [],
    }

    for trade in incorrect_trades:
        if trade.holding_days <= 3 and trade.sell_date is not None:
            # Bán quá sớm — model sell signal sai timing
            categories["wrong_sell"].append(trade)
        elif trade.sell_date is not None and trade.buy_date is not None:
            # Model khuyên mua nhưng kết quả lỗ
            categories["wrong_buy"].append(trade)
        else:
            # Các trường hợp khác — lỗi timing chung
            categories["timing_error"].append(trade)

    return categories
