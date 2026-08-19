"""
Training models - Định nghĩa các model liên quan đến training engine.

Bao gồm:
- TrainingPhase: Các phase training (PHASE_C, PHASE_B, PHASE_A)
- TrainingProgress: Chi tiết tiến độ training session
- SymbolTrainingStatus: Trạng thái training cho từng symbol
- SessionCheckpoint: Checkpoint để resume training sau crash

References: Req 2.3, 4.6, 4.9, 9.2
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class TrainingPhase(Enum):
    """
    Các phase training ML tự động.

    Phase C → Phase B → Phase A theo thứ tự nâng cao.
    Chuyển phase dựa trên performance criteria cụ thể.
    """

    PHASE_C = "phase_c"  # Supervised learning với simple labels từ price movement
    PHASE_B = "phase_b"  # Enhanced labels từ deep search validation
    PHASE_A = "phase_a"  # Self-play, model cải thiện bằng cách đấu với version cũ


@dataclass
class SymbolTrainingStatus:
    """
    Trạng thái training cho 1 symbol cụ thể.

    Được cập nhật real-time trong quá trình training session.

    Attributes:
        symbol: Mã cổ phiếu (VD: "FPT", "VNM")
        status: Trạng thái hiện tại (pending, training, completed, failed)
        epochs_completed: Số epoch đã hoàn thành
        current_loss: Loss hiện tại (None nếu chưa bắt đầu)
        duration_seconds: Thời gian training tính bằng giây (None nếu chưa xong)
    """

    symbol: str
    status: str
    epochs_completed: int
    current_loss: Optional[float] = None
    duration_seconds: Optional[float] = None


@dataclass
class TrainingProgress:
    """
    Chi tiết tiến độ training session.

    Được worker ghi vào status file mỗi 10s để Dashboard hiển thị.
    Bao gồm thông tin tổng quan và per-symbol status.

    Attributes:
        phase: Phase training hiện tại
        cycle_number: Số thứ tự training cycle
        current_symbol: Symbol đang được train
        symbols_completed: Số symbol đã hoàn thành
        symbols_total: Tổng số symbol cần train
        current_epoch: Epoch hiện tại trong symbol đang train
        total_epochs: Tổng số epoch cho symbol hiện tại
        current_loss: Giá trị loss hiện tại
        eta_seconds: Thời gian ước tính còn lại (giây)
        per_symbol_status: Trạng thái chi tiết từng symbol
    """

    phase: TrainingPhase
    cycle_number: int
    current_symbol: str
    symbols_completed: int
    symbols_total: int
    current_epoch: int
    total_epochs: int
    current_loss: float
    eta_seconds: float
    per_symbol_status: Dict[str, SymbolTrainingStatus] = field(default_factory=dict)


@dataclass
class SessionCheckpoint:
    """
    Checkpoint lưu trạng thái training session để resume sau crash.

    Được lưu atomic vào disk sau mỗi symbol hoàn thành.
    Khi resume, TrainingEngine đọc checkpoint và skip completed symbols.

    Attributes:
        session_id: ID phiên training
        phase: Phase training tại thời điểm checkpoint
        cycle_number: Số thứ tự cycle
        completed_symbols: Danh sách symbol đã train xong
        pending_symbols: Danh sách symbol chưa train
        current_symbol: Symbol đang train (nếu bị interrupt giữa chừng)
        current_epoch: Epoch hiện tại của symbol đang train
        total_epochs: Tổng epoch cho symbol hiện tại
        model_path: Đường dẫn file model checkpoint
        optimizer_state_path: Đường dẫn file optimizer state
        created_at: Thời điểm tạo checkpoint
        metadata: Dữ liệu bổ sung (hyperparams, config, etc.)
    """

    session_id: str
    phase: TrainingPhase
    cycle_number: int
    completed_symbols: List[str]
    pending_symbols: List[str]
    current_symbol: Optional[str] = None
    current_epoch: int = 0
    total_epochs: int = 0
    model_path: Optional[str] = None
    optimizer_state_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, str] = field(default_factory=dict)
