"""
Luật giao dịch chứng khoán Việt Nam (TTCK VN).

Định nghĩa các hằng số tuân thủ quy định giao dịch tại sàn HOSE/HNX.
Áp dụng cho BacktestEngine, RecommendationEngine, và AnalysisEngine.

References:
- Req 5.1: Phân tích ngày T → mua ngày T+1
- Req 5.2: Holding tối thiểu ceiling(T+2.5) = 3 ngày giao dịch
- Req 7.5: Enforce ±7% price limit và lot size 100
"""

# ==============================================================================
# Settlement Rules (Luật thanh toán)
# ==============================================================================

# Số ngày giao dịch chờ sau phân tích trước khi được mua
# Phân tích ngày T → chỉ khuyến nghị mua ngày T+1
T_PLUS_BUY: int = 1

# Số ngày giao dịch tối thiểu phải giữ cổ phiếu trước khi được bán
# Ceiling(T+2.5) = 3 ngày giao dịch (mua T+1, bán sớm nhất T+3)
SETTLEMENT_DAYS: int = 3

# ==============================================================================
# Price Limits (Biên độ giá)
# ==============================================================================

# Biên độ dao động giá tối đa trong ngày (%)
# Sàn HOSE: ±7% so với giá tham chiếu
PRICE_LIMIT_PCT: float = 7.0

# Biên độ sàn HNX (nếu cần mở rộng sau này)
# HNX_PRICE_LIMIT_PCT: float = 10.0

# ==============================================================================
# Lot Size (Đơn vị giao dịch)
# ==============================================================================

# Số lượng cổ phiếu tối thiểu mỗi lệnh (lô tròn)
# Giao dịch phải là bội số của LOT_SIZE
LOT_SIZE: int = 100

# ==============================================================================
# Trading Calendar (Lịch giao dịch)
# ==============================================================================

# Các ngày trong tuần có giao dịch (0=Monday, 4=Friday)
TRADING_WEEKDAYS: list[int] = [0, 1, 2, 3, 4]

# Giờ mở cửa phiên sáng
MARKET_OPEN_MORNING: str = "09:00"

# Giờ đóng cửa phiên sáng
MARKET_CLOSE_MORNING: str = "11:30"

# Giờ mở cửa phiên chiều
MARKET_OPEN_AFTERNOON: str = "13:00"

# Giờ đóng cửa phiên chiều (ATC xong lúc 14:45, nhưng 15:00 là end of day)
MARKET_CLOSE_AFTERNOON: str = "15:00"

# ==============================================================================
# VN30 Symbols (Danh sách mã VN30 chuẩn)
# ==============================================================================

# Danh sách 30 mã thuộc rổ VN30 (cập nhật định kỳ)
VN30_SYMBOLS: list[str] = [
    "ACB", "BCM", "BID", "BVH", "CTG",
    "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW",
    "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB",
    "VIC", "VJC", "VNM", "VPB", "VRE",
]
