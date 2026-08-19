"""
Cấu hình hệ thống trung tâm cho stock trading platform.

Định nghĩa các hằng số dùng chung giữa orchestrator, engine workers,
và UI layer. Các giá trị này được reference từ requirements:
- HEARTBEAT_INTERVAL: Req 1.7 (worker báo hiệu còn sống)
- POLL_INTERVAL: Req 2.2 (UI polling status)
- GPU_MAX_FRACTION: Req 1.4 (giới hạn VRAM per task)
- RATE_LIMIT: Req 8.6 (giới hạn API requests)
- CONFIDENCE_THRESHOLD: Req 6.4 (ngưỡng tin cậy recommend)
"""

# ==============================================================================
# Heartbeat & Monitoring
# ==============================================================================

# Chu kỳ heartbeat worker ghi timestamp vào status file (giây)
# Worker process ghi heartbeat mỗi 30s để UI biết task còn hoạt động
HEARTBEAT_INTERVAL: int = 30

# Chu kỳ UI polling đọc status files từ workers (giây)
# Dashboard cập nhật trạng thái training/backtest mỗi 10s
POLL_INTERVAL: int = 10

# Ngưỡng timeout heartbeat - task coi như frozen nếu vượt quá (giây)
# Nếu heartbeat_ts > HEARTBEAT_TIMEOUT_THRESHOLD → task bị frozen
HEARTBEAT_TIMEOUT_THRESHOLD: int = 60

# ==============================================================================
# Resource Management
# ==============================================================================

# Giới hạn tối đa phần trăm GPU memory mỗi task được sử dụng
# Ngăn một task chiếm hết VRAM, đảm bảo chạy song song ổn định
GPU_MAX_FRACTION: float = 0.7

# ==============================================================================
# API Rate Limiting
# ==============================================================================

# Số request tối đa mỗi phút khi fetch data từ API (vnstock)
# Token bucket: max 20 requests per 60-second sliding window
RATE_LIMIT: int = 20

# Khoảng thời gian tính rate limit (giây)
RATE_LIMIT_WINDOW: int = 60

# ==============================================================================
# Recommendation Engine
# ==============================================================================

# Ngưỡng confidence tối thiểu để hiển thị recommendation
# Nếu tất cả mã có confidence < threshold → "no strong recommendations"
CONFIDENCE_THRESHOLD: float = 0.5

# ==============================================================================
# Status Protocol
# ==============================================================================

# Đường dẫn thư mục lưu status files của workers
STATUS_DIR: str = "data/engine/status"

# Đường dẫn file watchlist chứa danh sách mã theo dõi
WATCHLIST_PATH: str = "data/code.txt"

# ==============================================================================
# Data Pipeline
# ==============================================================================

# Giờ auto-update data mặc định sau khi thị trường đóng cửa (HH:MM)
DEFAULT_UPDATE_TIME: str = "15:30"

# Số lần retry tối đa khi fetch data thất bại
MAX_RETRIES: int = 3

# Delay cơ sở cho exponential backoff (giây)
BASE_RETRY_DELAY: float = 1.0

# Delay tối đa cho retry (giây)
MAX_RETRY_DELAY: float = 60.0

# Hệ số nhân backoff
BACKOFF_FACTOR: float = 2.0
