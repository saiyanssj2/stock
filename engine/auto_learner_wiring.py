# -*- coding: utf-8 -*-
"""
Auto-Learner Wiring — Kết nối end-to-end các component trong training cycle.

Module này đóng vai trò "glue code" kết nối:
- BacktestEngineWorker.run_auto() → chạy backtest tự động
- MistakeAnalyzer → phân tích trades sai, tạo hard examples
- PhaseTransition → kiểm tra & trigger chuyển phase
- CycleHistory → lưu kết quả cycle persistent
- AutoLearner → orchestrate toàn bộ flow

Flow hoàn chỉnh 1 cycle:
1. Run auto backtest trên tracked symbols
2. Analyze mistakes (incorrect predictions)
3. Generate hard examples từ mistakes
4. Incorporate manual backtest results (nếu có)
5. Retrain model với hard examples (STUB)
6. Check phase transition
7. Persist cycle result vào history

References: Req 3.1, 3.6, 7.3
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from engine.config import ModelConfig, TrainingConfig
from engine.cycle_history import (
    detect_improvement_trend,
    load_all_cycles,
    persist_cycle,
)
from engine.hard_example_trainer import retrain_from_hard_examples
from engine.mistake_analyzer import (
    generate_hard_examples,
    identify_incorrect_predictions,
)
from engine.workers.backtest_worker import BacktestEngineWorker
from engine.workers.phase_transition import check_phase_transition
from models.backtest_models import AutoBacktestResult, BacktestResult, Trade
from models.data_models import AutoLearnerConfig, CycleResult
from models.training_models import TrainingPhase

logger = logging.getLogger(__name__)

# Thư mục mặc định lưu cycle history
DEFAULT_HISTORY_DIR = "data/engine/history"


class AutoLearnerWiring:
    """
    Kết nối end-to-end các component trong Auto-Learning cycle.

    Thay vì để AutoLearner gọi trực tiếp các STUB methods,
    module này wire các component thực tế vào pipeline.

    Attributes:
        config: Cấu hình auto-learner
        backtest_worker: Worker thực hiện backtest
        current_phase: Phase training hiện tại
        cycle_count: Số cycle đã chạy
        manual_backtest_queue: Queue chứa manual backtest results chờ xử lý
        history_dir: Thư mục lưu persistent cycle history
    """

    def __init__(
        self,
        config: AutoLearnerConfig = AutoLearnerConfig(),
        backtest_worker: Optional[BacktestEngineWorker] = None,
        history_dir: str = DEFAULT_HISTORY_DIR,
        data_dir: Optional[Path] = None,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ) -> None:
        """
        Khởi tạo AutoLearnerWiring.

        Args:
            config: Cấu hình auto-learner
            backtest_worker: Instance BacktestEngineWorker (tạo mới nếu None)
            history_dir: Thư mục lưu cycle history
            data_dir: Thư mục chứa CSV data
            model_config: Cấu hình model architecture (None → dùng default)
            training_config: Cấu hình training pipeline (None → dùng default)
        """
        self.config = config
        self.backtest_worker = backtest_worker or BacktestEngineWorker(
            data_dir=data_dir
        )
        self.current_phase: TrainingPhase = TrainingPhase.PHASE_C
        self.cycle_count: int = 0
        self.manual_backtest_queue: List[BacktestResult] = []
        self.history_dir = history_dir
        self._last_cycle_time: Optional[datetime] = None

        # Lưu data_dir cho retrain (dùng default "data" nếu None)
        self._data_dir: str = str(data_dir) if data_dir else "data"

        # Config cho training
        self._model_config: ModelConfig = model_config or ModelConfig()
        self._training_config: TrainingConfig = training_config or TrainingConfig()

        # Load cycle history từ disk để restore state
        self._restore_state()

    def run_full_cycle(self, symbols: List[str]) -> CycleResult:
        """
        Chạy 1 auto-learning cycle hoàn chỉnh với real components.

        Flow:
        1. Run auto backtest → lấy trades + metrics
        2. Analyze mistakes → tìm incorrect predictions
        3. Generate hard examples → training signal
        4. Merge manual backtest trades vào analysis
        5. Retrain model (STUB — sẽ gọi TrainingEngine thật sau)
        6. Check phase transition
        7. Persist cycle result

        Args:
            symbols: Danh sách mã cổ phiếu cần backtest

        Returns:
            CycleResult chứa performance metrics của cycle
        """
        start_time = time.time()
        self.cycle_count += 1

        logger.info(
            f"[AutoLearner] Bắt đầu cycle #{self.cycle_count}, "
            f"phase={self.current_phase.value}, symbols={len(symbols)}"
        )

        # Bước 1: Chạy auto backtest thực tế
        auto_backtest_result = self._run_backtest(symbols)

        # Bước 2: Thu thập tất cả trades (auto + manual)
        all_trades = self._collect_all_trades(auto_backtest_result)

        # Bước 3: Phân tích mistakes qua MistakeAnalyzer
        incorrect_trades = identify_incorrect_predictions(all_trades)

        # Bước 4: Tạo hard examples
        hard_examples = generate_hard_examples(incorrect_trades)

        logger.info(
            f"[AutoLearner] Cycle #{self.cycle_count}: "
            f"total_trades={len(all_trades)}, mistakes={len(incorrect_trades)}, "
            f"hard_examples={len(hard_examples)}"
        )

        # Bước 5: Retrain model với hard examples (STUB)
        self._retrain_with_hard_examples(hard_examples)

        # Reload model cho backtest cycle tiếp theo
        self.backtest_worker.reload_model()

        # Bước 6: Tạo CycleResult
        duration = time.time() - start_time
        cycle_result = self._build_cycle_result(
            auto_backtest_result=auto_backtest_result,
            hard_examples_count=len(hard_examples),
            duration=duration,
        )

        # Bước 7: Check phase transition
        cycle_history = load_all_cycles(self.history_dir)
        cycle_history.append(cycle_result)

        new_phase = check_phase_transition(self.current_phase, cycle_history)
        if new_phase is not None:
            logger.info(
                f"[AutoLearner] Phase transition: "
                f"{self.current_phase.value} → {new_phase.value}"
            )
            self.current_phase = new_phase
            cycle_result.notes += f" | Phase transition → {new_phase.value}"

        # Bước 8: Persist cycle result
        persist_cycle(cycle_result, self.history_dir)

        # Bước 9: Lưu backtest report chi tiết (trades CSV + summary JSON)
        try:
            from engine.backtest_report import save_backtest_report
            save_backtest_report(auto_backtest_result)
        except Exception as e:
            logger.warning(f"[AutoLearner] Không lưu được backtest report: {e}")

        # Xóa manual backtest queue đã xử lý
        self.manual_backtest_queue.clear()
        self._last_cycle_time = datetime.now()

        logger.info(
            f"[AutoLearner] Cycle #{self.cycle_count} hoàn tất "
            f"trong {duration:.2f}s, phase={self.current_phase.value}"
        )

        return cycle_result

    def incorporate_manual_backtest(self, result: BacktestResult) -> None:
        """
        Thêm kết quả manual backtest vào queue chờ xử lý trong cycle tiếp theo.

        Manual backtest results sẽ được merge vào auto backtest trades
        khi run_full_cycle() được gọi, để model học thêm từ scenarios
        do user kiểm thử.

        Args:
            result: Kết quả backtest manual từ user
        """
        self.manual_backtest_queue.append(result)
        logger.info(
            f"[AutoLearner] Incorporated manual backtest: "
            f"symbol={result.symbol}, trades={result.total_trades}"
        )

    def get_current_phase(self) -> TrainingPhase:
        """Trả về phase training hiện tại."""
        return self.current_phase

    def get_cycle_count(self) -> int:
        """Trả về số cycle đã chạy."""
        return self.cycle_count

    def is_cycle_due(self) -> bool:
        """
        Kiểm tra đã đến lúc chạy cycle tiếp theo chưa.

        Returns:
            True nếu đã đến lúc chạy cycle mới
        """
        if self._last_cycle_time is None:
            return True
        elapsed = (datetime.now() - self._last_cycle_time).total_seconds()
        interval = self.config.cycle_interval_hours * 3600
        return elapsed >= interval

    def get_improvement_trend(self) -> bool:
        """
        Kiểm tra model có đang cải thiện liên tục không.

        Returns:
            True nếu 3+ cycles gần nhất có Sharpe tăng strictly
        """
        cycles = load_all_cycles(self.history_dir)
        return detect_improvement_trend(
            cycles, min_consecutive=self.config.min_improvement_cycles
        )

    # ==========================================================================
    # Private methods
    # ==========================================================================

    def _run_backtest(self, symbols: List[str]) -> AutoBacktestResult:
        """
        Chạy auto backtest thực tế qua BacktestEngineWorker.

        Args:
            symbols: Danh sách symbols cần backtest

        Returns:
            AutoBacktestResult với metrics và trades
        """
        return self.backtest_worker.run_auto(
            symbols=symbols, cycle_number=self.cycle_count
        )

    def _collect_all_trades(
        self, auto_result: AutoBacktestResult
    ) -> List[Trade]:
        """
        Thu thập trades từ auto backtest + manual backtest queue.

        Args:
            auto_result: Kết quả auto backtest

        Returns:
            Danh sách tất cả trades cần phân tích
        """
        all_trades: List[Trade] = list(auto_result.trades)

        # Merge trades từ manual backtest đã incorporate
        for manual_result in self.manual_backtest_queue:
            all_trades.extend(manual_result.trades)

        return all_trades

    def _retrain_with_hard_examples(self, hard_examples: List[dict]) -> None:
        """
        Retrain model với hard examples qua shared utility.

        Gọi retrain_from_hard_examples() để thực hiện real incremental training:
        load model → tạo training data từ hard examples → forward/backward/step → save.

        Giữ empty-list guard: skip training nếu hard_examples rỗng.
        Wrap trong try/except để không crash cycle khi training lỗi.

        Args:
            hard_examples: Danh sách hard examples cần học
        """
        if not hard_examples:
            logger.info(
                "[AutoLearner] Không có hard examples — skip retrain"
            )
            return

        logger.info(
            f"[AutoLearner] Bắt đầu retrain model với {len(hard_examples)} "
            f"hard examples"
        )

        try:
            checkpoint_dir = self._training_config.checkpoint_dir
            success = retrain_from_hard_examples(
                hard_examples=hard_examples,
                checkpoint_dir=checkpoint_dir,
                data_dir=self._data_dir,
                model_config=self._model_config,
                training_config=self._training_config,
            )

            if success:
                logger.info(
                    f"[AutoLearner] Retrain thành công với "
                    f"{len(hard_examples)} hard examples"
                )
            else:
                logger.warning(
                    "[AutoLearner] Retrain không thành công — "
                    "có thể do thiếu data hoặc checkpoint"
                )
        except Exception as e:
            logger.error(
                f"[AutoLearner] Lỗi khi retrain model: {e}",
                exc_info=True,
            )

    def _build_cycle_result(
        self,
        auto_backtest_result: AutoBacktestResult,
        hard_examples_count: int,
        duration: float,
    ) -> CycleResult:
        """
        Tạo CycleResult từ kết quả backtest.

        Args:
            auto_backtest_result: Kết quả auto backtest
            hard_examples_count: Số hard examples đã tạo
            duration: Thời gian chạy cycle (giây)

        Returns:
            CycleResult chứa metrics
        """
        # Kiểm tra improvement trend
        cycles = load_all_cycles(self.history_dir)
        is_improving = detect_improvement_trend(
            cycles, min_consecutive=self.config.min_improvement_cycles
        )

        return CycleResult(
            cycle_number=self.cycle_count,
            phase=self.current_phase.value,
            sharpe_ratio=auto_backtest_result.overall_sharpe,
            win_rate=auto_backtest_result.overall_win_rate,
            total_return=auto_backtest_result.overall_return,
            strategies_beaten=auto_backtest_result.strategies_beaten,
            validation_loss=0.0,  # Sẽ tính từ validation set thực tế sau
            is_improving=is_improving,
            timestamp=datetime.now(),
            duration_seconds=duration,
            notes=(
                f"Hard examples: {hard_examples_count}, "
                f"Symbols tested: {len(auto_backtest_result.symbols_tested)}"
            ),
        )

    def _restore_state(self) -> None:
        """
        Restore state từ persistent cycle history.

        Load cycle history để xác định cycle_count, current_phase,
        và _last_cycle_time từ cycle gần nhất.
        """
        cycles = load_all_cycles(self.history_dir)
        if cycles:
            latest = cycles[-1]
            self.cycle_count = latest.cycle_number

            # Restore phase từ cycle gần nhất
            try:
                self.current_phase = TrainingPhase(latest.phase)
            except ValueError:
                self.current_phase = TrainingPhase.PHASE_C

            # Restore last_cycle_time để is_cycle_due() hoạt động đúng
            if latest.timestamp is not None:
                self._last_cycle_time = latest.timestamp

            logger.info(
                f"[AutoLearner] Restored state: "
                f"cycle_count={self.cycle_count}, phase={self.current_phase.value}, "
                f"last_cycle_time={self._last_cycle_time}"
            )
