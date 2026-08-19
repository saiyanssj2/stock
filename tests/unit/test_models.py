"""
Unit tests cho models package.

Kiểm tra tất cả data models và enums hoạt động đúng:
- Enum values đúng
- Dataclass instantiation với required và optional fields
- Default values đúng
- Type safety cơ bản
"""

from datetime import date, datetime

import pytest

from models import (
    Action,
    AutoBacktestResult,
    AutoLearnerConfig,
    BacktestResult,
    CycleResult,
    ManualBacktestParams,
    Recommendation,
    ResourceAllocation,
    RetryPolicy,
    SessionCheckpoint,
    SymbolTrainingStatus,
    TaskState,
    TaskStatus,
    TaskType,
    Trade,
    TrainingPhase,
    TrainingProgress,
    UpdateResult,
)


# ==============================================================================
# Tests cho Enums
# ==============================================================================


class TestTaskType:
    """Kiểm tra TaskType enum."""

    def test_has_training(self) -> None:
        assert TaskType.TRAINING.value == "training"

    def test_has_backtest(self) -> None:
        assert TaskType.BACKTEST.value == "backtest"

    def test_has_analysis(self) -> None:
        assert TaskType.ANALYSIS.value == "analysis"

    def test_has_exactly_3_members(self) -> None:
        assert len(TaskType) == 3


class TestTaskState:
    """Kiểm tra TaskState enum."""

    def test_has_idle(self) -> None:
        assert TaskState.IDLE.value == "idle"

    def test_has_running(self) -> None:
        assert TaskState.RUNNING.value == "running"

    def test_has_paused(self) -> None:
        assert TaskState.PAUSED.value == "paused"

    def test_has_error(self) -> None:
        assert TaskState.ERROR.value == "error"

    def test_has_completed(self) -> None:
        assert TaskState.COMPLETED.value == "completed"

    def test_has_exactly_5_members(self) -> None:
        assert len(TaskState) == 5


class TestTrainingPhase:
    """Kiểm tra TrainingPhase enum."""

    def test_has_phase_c(self) -> None:
        assert TrainingPhase.PHASE_C.value == "phase_c"

    def test_has_phase_b(self) -> None:
        assert TrainingPhase.PHASE_B.value == "phase_b"

    def test_has_phase_a(self) -> None:
        assert TrainingPhase.PHASE_A.value == "phase_a"

    def test_has_exactly_3_members(self) -> None:
        assert len(TrainingPhase) == 3


class TestAction:
    """Kiểm tra Action enum."""

    def test_has_buy(self) -> None:
        assert Action.BUY.value == "buy"

    def test_has_hold(self) -> None:
        assert Action.HOLD.value == "hold"

    def test_has_sell(self) -> None:
        assert Action.SELL.value == "sell"

    def test_has_exactly_3_members(self) -> None:
        assert len(Action) == 3


# ==============================================================================
# Tests cho Task Models
# ==============================================================================


class TestTaskStatus:
    """Kiểm tra TaskStatus dataclass."""

    def test_required_fields(self) -> None:
        """Tạo TaskStatus với tất cả required fields."""
        now = datetime.now()
        status = TaskStatus(
            task_id="task-001",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=50.0,
            message="Training in progress",
            heartbeat_ts=now,
            started_at=now,
        )
        assert status.task_id == "task-001"
        assert status.task_type == TaskType.TRAINING
        assert status.state == TaskState.RUNNING
        assert status.progress_pct == 50.0
        assert status.message == "Training in progress"
        assert status.heartbeat_ts == now
        assert status.started_at == now

    def test_default_optional_fields(self) -> None:
        """Optional fields có default values đúng."""
        now = datetime.now()
        status = TaskStatus(
            task_id="task-002",
            task_type=TaskType.BACKTEST,
            state=TaskState.IDLE,
            progress_pct=0.0,
            message="Idle",
            heartbeat_ts=now,
            started_at=now,
        )
        assert status.error is None
        assert status.details == {}

    def test_with_error(self) -> None:
        """TaskStatus với error message."""
        now = datetime.now()
        status = TaskStatus(
            task_id="task-003",
            task_type=TaskType.ANALYSIS,
            state=TaskState.ERROR,
            progress_pct=30.0,
            message="Failed",
            heartbeat_ts=now,
            started_at=now,
            error="GPU OOM",
        )
        assert status.error == "GPU OOM"


class TestResourceAllocation:
    """Kiểm tra ResourceAllocation dataclass."""

    def test_can_start_true(self) -> None:
        """Allocation khi đủ resource."""
        alloc = ResourceAllocation(
            gpu_memory_fraction=0.6,
            cpu_threads=4,
            can_start=True,
        )
        assert alloc.gpu_memory_fraction == 0.6
        assert alloc.cpu_threads == 4
        assert alloc.can_start is True
        assert alloc.reason is None

    def test_can_start_false_with_reason(self) -> None:
        """Allocation khi không đủ resource."""
        alloc = ResourceAllocation(
            gpu_memory_fraction=0.0,
            cpu_threads=0,
            can_start=False,
            reason="GPU memory exhausted",
        )
        assert alloc.can_start is False
        assert alloc.reason == "GPU memory exhausted"


# ==============================================================================
# Tests cho Training Models
# ==============================================================================


class TestSymbolTrainingStatus:
    """Kiểm tra SymbolTrainingStatus dataclass."""

    def test_pending_status(self) -> None:
        """Symbol chưa bắt đầu train."""
        sts = SymbolTrainingStatus(
            symbol="FPT",
            status="pending",
            epochs_completed=0,
        )
        assert sts.symbol == "FPT"
        assert sts.status == "pending"
        assert sts.current_loss is None
        assert sts.duration_seconds is None

    def test_completed_status(self) -> None:
        """Symbol đã train xong."""
        sts = SymbolTrainingStatus(
            symbol="VNM",
            status="completed",
            epochs_completed=50,
            current_loss=0.02,
            duration_seconds=120.5,
        )
        assert sts.epochs_completed == 50
        assert sts.current_loss == 0.02
        assert sts.duration_seconds == 120.5


class TestTrainingProgress:
    """Kiểm tra TrainingProgress dataclass."""

    def test_basic_progress(self) -> None:
        """Tạo TrainingProgress với thông tin cơ bản."""
        progress = TrainingProgress(
            phase=TrainingPhase.PHASE_C,
            cycle_number=1,
            current_symbol="FPT",
            symbols_completed=5,
            symbols_total=30,
            current_epoch=10,
            total_epochs=50,
            current_loss=0.05,
            eta_seconds=3600.0,
        )
        assert progress.phase == TrainingPhase.PHASE_C
        assert progress.cycle_number == 1
        assert progress.symbols_completed == 5
        assert progress.symbols_total == 30
        assert progress.per_symbol_status == {}

    def test_with_per_symbol_status(self) -> None:
        """TrainingProgress với per-symbol status map."""
        symbol_status = SymbolTrainingStatus(
            symbol="FPT", status="completed", epochs_completed=50
        )
        progress = TrainingProgress(
            phase=TrainingPhase.PHASE_B,
            cycle_number=2,
            current_symbol="VNM",
            symbols_completed=1,
            symbols_total=30,
            current_epoch=5,
            total_epochs=50,
            current_loss=0.03,
            eta_seconds=7200.0,
            per_symbol_status={"FPT": symbol_status},
        )
        assert "FPT" in progress.per_symbol_status
        assert progress.per_symbol_status["FPT"].status == "completed"


class TestSessionCheckpoint:
    """Kiểm tra SessionCheckpoint dataclass."""

    def test_basic_checkpoint(self) -> None:
        """Tạo checkpoint cơ bản."""
        cp = SessionCheckpoint(
            session_id="sess-001",
            phase=TrainingPhase.PHASE_C,
            cycle_number=1,
            completed_symbols=["FPT", "VNM"],
            pending_symbols=["HPG", "MBB"],
        )
        assert cp.session_id == "sess-001"
        assert cp.completed_symbols == ["FPT", "VNM"]
        assert cp.pending_symbols == ["HPG", "MBB"]
        assert cp.current_symbol is None
        assert cp.model_path is None

    def test_checkpoint_with_full_state(self) -> None:
        """Checkpoint với đầy đủ thông tin resume."""
        cp = SessionCheckpoint(
            session_id="sess-002",
            phase=TrainingPhase.PHASE_B,
            cycle_number=3,
            completed_symbols=["FPT"],
            pending_symbols=["VNM", "HPG"],
            current_symbol="VNM",
            current_epoch=25,
            total_epochs=50,
            model_path="data/engine/models/model_v3.pt",
            optimizer_state_path="data/engine/models/optim_v3.pt",
            metadata={"learning_rate": "0.001"},
        )
        assert cp.current_symbol == "VNM"
        assert cp.current_epoch == 25
        assert cp.model_path == "data/engine/models/model_v3.pt"
        assert cp.metadata == {"learning_rate": "0.001"}


# ==============================================================================
# Tests cho Recommendation Models
# ==============================================================================


class TestRecommendation:
    """Kiểm tra Recommendation dataclass."""

    def test_buy_recommendation(self) -> None:
        """Khuyến nghị mua."""
        rec = Recommendation(
            symbol="FPT",
            action=Action.BUY,
            confidence=0.85,
            position_score=0.7,
            expected_holding_days=5,
            earliest_sell_date=date(2024, 1, 5),
            analysis_date=date(2024, 1, 1),
            buy_date=date(2024, 1, 2),
            model_version="v1.2.0",
        )
        assert rec.symbol == "FPT"
        assert rec.action == Action.BUY
        assert rec.confidence == 0.85
        assert rec.position_score == 0.7
        assert rec.earliest_sell_date == date(2024, 1, 5)

    def test_sell_recommendation(self) -> None:
        """Khuyến nghị bán."""
        rec = Recommendation(
            symbol="VNM",
            action=Action.SELL,
            confidence=0.6,
            position_score=-0.5,
            expected_holding_days=0,
            earliest_sell_date=date(2024, 1, 3),
            analysis_date=date(2024, 1, 1),
            buy_date=date(2024, 1, 2),
            model_version="v1.2.0",
        )
        assert rec.action == Action.SELL
        assert rec.position_score == -0.5


# ==============================================================================
# Tests cho Backtest Models
# ==============================================================================


class TestTrade:
    """Kiểm tra Trade dataclass."""

    def test_closed_trade(self) -> None:
        """Trade đã đóng (đã bán)."""
        trade = Trade(
            symbol="FPT",
            buy_date=date(2024, 1, 2),
            sell_date=date(2024, 1, 5),
            buy_price=100000.0,
            sell_price=105000.0,
            shares=100,
            pnl=500000.0,
            pnl_pct=5.0,
            holding_days=3,
        )
        assert trade.sell_date is not None
        assert trade.pnl == 500000.0
        assert trade.holding_days == 3

    def test_open_trade(self) -> None:
        """Trade đang mở (chưa bán)."""
        trade = Trade(
            symbol="VNM",
            buy_date=date(2024, 1, 2),
            sell_date=None,
            buy_price=80000.0,
            sell_price=None,
            shares=200,
        )
        assert trade.sell_date is None
        assert trade.sell_price is None
        assert trade.pnl is None


class TestManualBacktestParams:
    """Kiểm tra ManualBacktestParams dataclass."""

    def test_default_vn_rules(self) -> None:
        """Mặc định áp dụng luật VN."""
        params = ManualBacktestParams(
            symbol="FPT",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            initial_capital=100_000_000.0,
        )
        assert params.enforce_vn_rules is True

    def test_disable_vn_rules(self) -> None:
        """Tắt luật VN khi cần."""
        params = ManualBacktestParams(
            symbol="HPG",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=50_000_000.0,
            enforce_vn_rules=False,
        )
        assert params.enforce_vn_rules is False


class TestBacktestResult:
    """Kiểm tra BacktestResult dataclass."""

    def test_basic_result(self) -> None:
        """Kết quả backtest cơ bản."""
        result = BacktestResult(
            symbol="FPT",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            initial_capital=100_000_000.0,
            final_capital=115_000_000.0,
            total_return=15.0,
            sharpe_ratio=1.8,
            win_rate=65.0,
            max_drawdown=-8.5,
            total_trades=20,
        )
        assert result.total_return == 15.0
        assert result.trades == []
        assert result.equity_curve == []


class TestAutoBacktestResult:
    """Kiểm tra AutoBacktestResult dataclass."""

    def test_basic_auto_result(self) -> None:
        """Kết quả auto-backtest cơ bản."""
        result = AutoBacktestResult(
            cycle_number=5,
            symbols_tested=["FPT", "VNM", "HPG"],
            overall_sharpe=1.5,
            overall_win_rate=60.0,
            overall_return=12.0,
            strategies_beaten=3,
        )
        assert result.cycle_number == 5
        assert len(result.symbols_tested) == 3
        assert result.strategies_beaten == 3
        assert result.trades == []
        assert result.duration_seconds == 0.0


# ==============================================================================
# Tests cho Data Models
# ==============================================================================


class TestUpdateResult:
    """Kiểm tra UpdateResult dataclass."""

    def test_successful_update(self) -> None:
        """Update thành công toàn bộ."""
        result = UpdateResult(
            total_symbols=30,
            success_count=30,
            failed_symbols=[],
            errors={},
            duration_seconds=90.0,
        )
        assert result.success_count == result.total_symbols
        assert result.failed_symbols == []

    def test_partial_failure(self) -> None:
        """Update với một số symbol thất bại."""
        result = UpdateResult(
            total_symbols=30,
            success_count=28,
            failed_symbols=["ABC", "XYZ"],
            errors={"ABC": "timeout", "XYZ": "not found"},
            duration_seconds=120.5,
        )
        assert len(result.failed_symbols) == 2
        assert result.errors["ABC"] == "timeout"


class TestAutoLearnerConfig:
    """Kiểm tra AutoLearnerConfig dataclass."""

    def test_default_values(self) -> None:
        """Tất cả default values đúng theo design."""
        config = AutoLearnerConfig()
        assert config.cycle_interval_hours == 24.0
        assert config.min_improvement_cycles == 3
        assert config.phase_c_sharpe_threshold == 2
        assert config.phase_b_stable_cycles == 5
        assert config.data_update_time == "15:30"
        assert config.api_rate_limit == 20
        assert config.confidence_threshold == 0.5

    def test_custom_config(self) -> None:
        """Config tùy chỉnh."""
        config = AutoLearnerConfig(
            cycle_interval_hours=12.0,
            confidence_threshold=0.7,
        )
        assert config.cycle_interval_hours == 12.0
        assert config.confidence_threshold == 0.7


class TestCycleResult:
    """Kiểm tra CycleResult dataclass."""

    def test_basic_cycle(self) -> None:
        """Kết quả cycle cơ bản."""
        result = CycleResult(
            cycle_number=1,
            phase="phase_c",
            sharpe_ratio=1.5,
            win_rate=60.0,
            total_return=15.0,
            strategies_beaten=2,
            validation_loss=0.05,
            is_improving=True,
        )
        assert result.cycle_number == 1
        assert result.is_improving is True
        assert result.notes == ""

    def test_with_notes(self) -> None:
        """Cycle result với ghi chú."""
        result = CycleResult(
            cycle_number=5,
            phase="phase_b",
            sharpe_ratio=2.1,
            win_rate=70.0,
            total_return=25.0,
            strategies_beaten=4,
            validation_loss=0.03,
            is_improving=True,
            notes="phase transition triggered",
        )
        assert result.notes == "phase transition triggered"


class TestRetryPolicy:
    """Kiểm tra RetryPolicy dataclass."""

    def test_default_policy(self) -> None:
        """Default retry policy đúng theo design."""
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.base_delay_seconds == 1.0
        assert policy.max_delay_seconds == 60.0
        assert policy.backoff_factor == 2.0
        assert policy.retryable_errors == []

    def test_custom_policy(self) -> None:
        """Custom retry policy."""
        policy = RetryPolicy(
            max_retries=5,
            base_delay_seconds=2.0,
            retryable_errors=[ConnectionError, TimeoutError],
        )
        assert policy.max_retries == 5
        assert ConnectionError in policy.retryable_errors
