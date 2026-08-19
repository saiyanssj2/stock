"""
Backtest models - Định nghĩa các model liên quan đến backtest engine.

Bao gồm:
- Trade: Một giao dịch trong backtest
- ManualBacktestParams: Tham số cho backtest thủ công
- BacktestResult: Kết quả backtest manual
- AutoBacktestResult: Kết quả backtest tự động từ Auto_Learner

References: Req 7.1, 5.2, 5.5
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


@dataclass
class Trade:
    """
    Một giao dịch (mua-bán) trong backtest.

    Tuân thủ luật TTCK Việt Nam:
    - Lot size: bội số 100
    - Biên độ giá: ±7%
    - Settlement: T+2.5 (ceiling = 3 ngày giao dịch)

    Attributes:
        symbol: Mã cổ phiếu
        buy_date: Ngày mua
        sell_date: Ngày bán (None nếu đang giữ)
        buy_price: Giá mua
        sell_price: Giá bán (None nếu đang giữ)
        shares: Số lượng cổ phiếu (bội số 100)
        pnl: Lợi nhuận/lỗ của giao dịch (None nếu chưa đóng)
        pnl_pct: Phần trăm lợi nhuận/lỗ (None nếu chưa đóng)
        holding_days: Số ngày giữ
        confidence: Độ tin cậy của signal tại thời điểm mua (0-100%)
        cash_before: Tiền mặt trước khi mua (VND)
        cash_after: Tiền mặt sau khi mua (VND)
    """

    symbol: str
    buy_date: date
    sell_date: Optional[date]
    buy_price: float
    sell_price: Optional[float]
    shares: int
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    holding_days: int = 0
    confidence: Optional[float] = None
    cash_before: Optional[float] = None
    cash_after: Optional[float] = None


@dataclass
class ManualBacktestParams:
    """
    Tham số cho backtest thủ công do user chỉ định.

    User nhập symbol, khoảng thời gian, và vốn ban đầu.
    enforce_vn_rules bật mặc định để tuân thủ luật TTCK VN.

    Attributes:
        symbol: Mã cổ phiếu cần backtest
        start_date: Ngày bắt đầu backtest
        end_date: Ngày kết thúc backtest
        initial_capital: Vốn ban đầu (VND)
        enforce_vn_rules: Áp dụng luật T+2.5, ±7%, lot 100
    """

    symbol: str
    start_date: date
    end_date: date
    initial_capital: float
    enforce_vn_rules: bool = True


@dataclass
class BacktestResult:
    """
    Kết quả backtest manual.

    Bao gồm các metrics quan trọng và danh sách trades.

    Attributes:
        symbol: Mã cổ phiếu đã backtest
        start_date: Ngày bắt đầu
        end_date: Ngày kết thúc
        initial_capital: Vốn ban đầu
        final_capital: Vốn cuối cùng
        total_return: Tổng lợi nhuận (%)
        sharpe_ratio: Tỷ số Sharpe
        win_rate: Tỷ lệ thắng (%)
        max_drawdown: Drawdown tối đa (%)
        total_trades: Tổng số giao dịch
        trades: Danh sách giao dịch chi tiết
        equity_curve: Đường equity theo thời gian (list giá trị vốn)
    """

    symbol: str
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    total_return: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    total_trades: int
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


@dataclass
class AutoBacktestResult:
    """
    Kết quả backtest tự động từ Auto_Learner.

    So sánh AI strategy với các benchmark strategies.

    Attributes:
        cycle_number: Số thứ tự auto-learning cycle
        symbols_tested: Danh sách symbol đã backtest
        overall_sharpe: Sharpe ratio tổng hợp
        overall_win_rate: Tỷ lệ thắng tổng hợp (%)
        overall_return: Lợi nhuận tổng hợp (%)
        strategies_beaten: Số strategies đã beat (max 4)
        benchmark_results: Kết quả từng benchmark strategy
        trades: Tất cả trades trong auto-backtest
        duration_seconds: Thời gian chạy (giây)
    """

    cycle_number: int
    symbols_tested: List[str]
    overall_sharpe: float
    overall_win_rate: float
    overall_return: float
    strategies_beaten: int
    benchmark_results: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    duration_seconds: float = 0.0
