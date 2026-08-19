"""
Models package - Định nghĩa tất cả data models và enums cho stock trading platform.

Package này chứa các dataclass và enum dùng chung giữa orchestrator, engine workers,
và UI layer. Được thiết kế theo strict typing để đảm bảo type safety.
"""

from models.task_models import (
    TaskType,
    TaskState,
    TaskStatus,
    ResourceAllocation,
)
from models.training_models import (
    TrainingPhase,
    TrainingProgress,
    SymbolTrainingStatus,
    SessionCheckpoint,
)
from models.recommendation_models import (
    Action,
    Recommendation,
)
from models.backtest_models import (
    ManualBacktestParams,
    BacktestResult,
    AutoBacktestResult,
    Trade,
)
from models.data_models import (
    UpdateResult,
    AutoLearnerConfig,
    CycleResult,
    RetryPolicy,
)

__all__ = [
    # Task models
    "TaskType",
    "TaskState",
    "TaskStatus",
    "ResourceAllocation",
    # Training models
    "TrainingPhase",
    "TrainingProgress",
    "SymbolTrainingStatus",
    "SessionCheckpoint",
    # Recommendation models
    "Action",
    "Recommendation",
    # Backtest models
    "ManualBacktestParams",
    "BacktestResult",
    "AutoBacktestResult",
    "Trade",
    # Data models
    "UpdateResult",
    "AutoLearnerConfig",
    "CycleResult",
    "RetryPolicy",
]
