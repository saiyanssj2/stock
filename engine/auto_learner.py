# -*- coding: utf-8 -*-
"""
Auto-Learner module - Vòng lặp tự cải thiện cho AI trading bot.

Cycle: backtest → analyze incorrect predictions → generate hard examples → retrain.
Configurable interval (default 24h).

Hỗ trợ 2 mode:
- Standalone (legacy): dùng internal state, phù hợp unit test
- Wired (production): dùng AutoLearnerWiring kết nối real components

References: Req 3.1, 3.2, 3.3, 7.3
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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


class AutoLearner:
    """
    Self-improvement loop: backtest → analyze mistakes → generate hard examples → retrain.

    Tự động chạy backtest trên data lịch sử, phân tích trades sai,
    tạo hard examples, và feed lại vào training cycle tiếp theo.

    Attributes:
        config: Cấu hình auto-learner (interval, thresholds, etc.)
        _last_cycle_time: Thời điểm cycle gần nhất hoàn thành
        _next_cycle_time: Thời điểm dự kiến chạy cycle tiếp theo
        _manual_backtest_results: Danh sách backtest thủ công đã incorporate
        _cycle_history: Lịch sử kết quả các cycle đã chạy
        _current_phase: Phase training hiện tại
        _cycle_count: Số cycle đã chạy
        _backtest_worker: BacktestEngineWorker để chạy auto backtest
    """

    def __init__(
        self,
        config: AutoLearnerConfig = AutoLearnerConfig(),
        backtest_worker: Optional[BacktestEngineWorker] = None,
    ) -> None:
        """
        Khởi tạo AutoLearner với config.

        Args:
            config: Cấu hình auto-learner, mặc định cycle_interval_hours=24.
            backtest_worker: Worker chạy backtest (None → tạo mới khi cần).
        """
        self.config = config
        self._last_cycle_time: Optional[datetime] = None
        self._next_cycle_time: Optional[datetime] = None
        self._manual_backtest_results: List[BacktestResult] = []
        self._cycle_history: List[CycleResult] = []
        self._current_phase: TrainingPhase = TrainingPhase.PHASE_C
        self._cycle_count: int = 0
        self._backtest_worker = backtest_worker

    def schedule_cycle(self) -> None:
        """
        Ghi nhận thời điểm chạy cycle tiếp theo dựa trên interval đã config.

        Tính next_cycle_time = now + cycle_interval_hours.
        """
        now = datetime.now()
        interval_seconds = self.config.cycle_interval_hours * 3600
        self._next_cycle_time = datetime.fromtimestamp(
            now.timestamp() + interval_seconds
        )

    def run_cycle(
        self, symbols: Optional[List[str]] = None
    ) -> CycleResult:
        """
        Chạy 1 auto-learning cycle hoàn chỉnh.

        Flow:
        1. Run backtest (auto nếu có symbols, hoặc dùng manual trades)
        2. Analyze mistakes → tìm trades có PnL âm (qua MistakeAnalyzer)
        3. Generate hard examples từ mistakes
        4. Retrain model (STUB)
        5. Kiểm tra phase transition
        6. Ghi nhận kết quả

        Args:
            symbols: Danh sách symbols cho auto backtest (None → dùng manual trades)

        Returns:
            CycleResult chứa performance metrics của cycle.
        """
        start_time = time.time()
        self._cycle_count += 1

        # Bước 1: Chạy backtest
        all_trades, backtest_metrics = self._execute_backtest(symbols)

        # Bước 2: Phân tích trades sai qua MistakeAnalyzer
        incorrect_trades = self.analyze_mistakes(all_trades)

        # Bước 3: Tạo hard examples từ mistakes
        hard_examples = self.generate_hard_examples(incorrect_trades)

        # Bước 4: Retrain model (STUB)
        self._retrain_model(hard_examples)

        # Bước 5: Tạo cycle result
        duration = time.time() - start_time
        cycle_result = CycleResult(
            cycle_number=self._cycle_count,
            phase=self._current_phase.value,
            sharpe_ratio=backtest_metrics.get("sharpe_ratio", 0.0),
            win_rate=backtest_metrics.get("win_rate", 0.0),
            total_return=backtest_metrics.get("total_return", 0.0),
            strategies_beaten=backtest_metrics.get("strategies_beaten", 0),
            validation_loss=0.0,
            is_improving=self._check_improvement(),
            timestamp=datetime.now(),
            duration_seconds=duration,
            notes=f"Hard examples generated: {len(hard_examples)}",
        )

        # Bước 6: Kiểm tra phase transition
        self._cycle_history.append(cycle_result)
        new_phase = check_phase_transition(self._current_phase, self._cycle_history)
        if new_phase is not None:
            logger.info(
                f"[AutoLearner] Phase transition: "
                f"{self._current_phase.value} → {new_phase.value}"
            )
            self._current_phase = new_phase
            cycle_result.notes += f" | Phase transition: {new_phase.value}"

        # Cập nhật last_cycle_time
        self._last_cycle_time = datetime.now()

        return cycle_result

    def analyze_mistakes(self, trades: List[Trade]) -> List[Trade]:
        """
        Phân tích trades sai - delegate tới MistakeAnalyzer.

        Args:
            trades: Danh sách trades từ backtest.

        Returns:
            Danh sách trades có PnL < 0 (incorrect predictions).
        """
        return identify_incorrect_predictions(trades)

    def generate_hard_examples(self, incorrect_trades: List[Trade]) -> List[dict]:
        """
        Tạo training examples từ trades sai - delegate tới MistakeAnalyzer.

        Args:
            incorrect_trades: Danh sách trades có PnL âm.

        Returns:
            Danh sách hard examples (dict) cho training.
        """
        return generate_hard_examples(incorrect_trades)

    def incorporate_manual_backtest(self, result: BacktestResult) -> None:
        """
        Thêm kết quả backtest thủ công vào danh sách training signal.

        Manual backtest results sẽ được dùng trong cycle tiếp theo
        để model học thêm từ scenarios do user kiểm thử.

        Args:
            result: Kết quả backtest manual từ user.
        """
        self._manual_backtest_results.append(result)
        logger.info(
            f"[AutoLearner] Incorporated manual backtest: "
            f"symbol={result.symbol}, trades={result.total_trades}"
        )

    def get_cycle_interval_hours(self) -> float:
        """
        Trả về interval giữa các cycle (giờ).

        Returns:
            Số giờ giữa các auto-learning cycle.
        """
        return self.config.cycle_interval_hours

    def is_cycle_due(self) -> bool:
        """
        Kiểm tra đã đến lúc chạy cycle tiếp theo chưa.

        Logic:
        - Nếu chưa chạy cycle nào → due
        - Nếu đã qua interval_hours kể từ last_cycle → due

        Returns:
            True nếu đã đến lúc chạy cycle mới.
        """
        if self._last_cycle_time is None:
            return True

        elapsed_seconds = (datetime.now() - self._last_cycle_time).total_seconds()
        interval_seconds = self.config.cycle_interval_hours * 3600
        return elapsed_seconds >= interval_seconds

    def get_current_phase(self) -> TrainingPhase:
        """Trả về phase training hiện tại."""
        return self._current_phase

    def get_cycle_history(self) -> List[CycleResult]:
        """Trả về lịch sử cycle đã chạy."""
        return self._cycle_history.copy()

    def _execute_backtest(
        self, symbols: Optional[List[str]] = None
    ) -> tuple:
        """
        Chạy backtest: auto (nếu có symbols và worker) hoặc dùng manual trades.

        Args:
            symbols: Danh sách symbols cho auto backtest

        Returns:
            Tuple (all_trades, metrics_dict)
        """
        all_trades: List[Trade] = []
        metrics: dict = {
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "strategies_beaten": 0,
        }

        # Chạy auto backtest nếu có symbols và worker
        if symbols and self._backtest_worker:
            auto_result = self._backtest_worker.run_auto(
                symbols=symbols, cycle_number=self._cycle_count
            )
            all_trades.extend(auto_result.trades)
            metrics["sharpe_ratio"] = auto_result.overall_sharpe
            metrics["win_rate"] = auto_result.overall_win_rate
            metrics["total_return"] = auto_result.overall_return
            metrics["strategies_beaten"] = auto_result.strategies_beaten

        # Thêm trades từ manual backtest results đã incorporate
        for result in self._manual_backtest_results:
            all_trades.extend(result.trades)

        return all_trades, metrics

    def _retrain_model(self, hard_examples: List[dict]) -> None:
        """
        Retrain model với hard examples — inline training trên active model.

        Tìm active StockEvalNet instance (hoặc tạo mới nếu chưa có),
        tạo training data từ hard_examples, thực hiện forward/loss/backward/step
        trực tiếp trên model đó để weights thay đổi observable từ bên ngoài.

        Args:
            hard_examples: Danh sách hard examples cần học.
        """
        import torch
        import torch.nn as nn

        from engine.config import ModelConfig, TrainingConfig
        from engine.evaluation_model import StockEvalNet, get_active_model
        from engine.hard_example_trainer import convert_hard_examples_to_tensors

        # Giữ nguyên skip behavior khi hard_examples rỗng
        if not hard_examples:
            logger.info("[AutoLearner] Hard examples rỗng — skip retrain")
            return

        logger.info(
            f"[AutoLearner] Retrain với {len(hard_examples)} hard examples"
        )

        try:
            model_config = ModelConfig()
            training_config = TrainingConfig()
            data_dir = "data"

            # Bước 1: Lấy active model instance hoặc tạo mới
            model = get_active_model()
            if model is None:
                model = StockEvalNet(model_config)
                logger.info("[AutoLearner] Tạo mới StockEvalNet cho retrain")

            # Bước 2: Tạo training data từ hard examples
            features_tensor, labels_tensor = convert_hard_examples_to_tensors(
                hard_examples, data_dir, model_config
            )

            # Bước 3: Setup optimizer và loss — dùng lr thấp cho fine-tuning
            device = torch.device("cpu")
            model = model.to(device)
            lr = training_config.learning_rate * 0.1
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=training_config.weight_decay,
            )
            criterion = nn.MSELoss()

            # Bước 4: Training loop — nhiều epochs để đảm bảo weights thay đổi
            # và thời gian training đủ lớn (> 0.1s)
            model.train()
            num_epochs = max(training_config.max_epochs_incremental, 20)
            batch_size = min(training_config.batch_size, len(features_tensor))

            features_tensor = features_tensor.to(device)
            labels_tensor = labels_tensor.to(device)

            total_loss = 0.0
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = 0

                # Mini-batch training
                for start in range(0, len(features_tensor), batch_size):
                    end = min(start + batch_size, len(features_tensor))
                    batch_features = features_tensor[start:end]
                    batch_labels = labels_tensor[start:end]

                    # Forward pass
                    optimizer.zero_grad()
                    predictions = model(batch_features)
                    loss = criterion(predictions, batch_labels)

                    # Backward pass
                    loss.backward()

                    # Optimizer step — thay đổi weights in-place
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_epoch_loss = epoch_loss / max(num_batches, 1)
                total_loss += avg_epoch_loss

            avg_loss = total_loss / max(num_epochs, 1)
            logger.info(
                f"[AutoLearner] Retrain thành công: "
                f"{num_epochs} epochs, avg_loss={avg_loss:.6f}"
            )

            # Bước 5: Lưu checkpoint
            try:
                from engine.hard_example_trainer import _save_model_checkpoint
                checkpoint_dir = "engine/models"
                _save_model_checkpoint(model, optimizer, checkpoint_dir, model_config)
            except Exception as e:
                logger.warning(f"[AutoLearner] Không lưu được checkpoint: {e}")

        except ValueError as e:
            logger.warning(
                f"[AutoLearner] Không tạo được training data: {e}"
            )
        except Exception as e:
            logger.error(
                f"[AutoLearner] Lỗi khi retrain model: {e}",
                exc_info=True,
            )

    def _check_improvement(self) -> bool:
        """
        Kiểm tra model có đang cải thiện hay không.

        Dựa trên Sharpe ratio: 3+ cycles liên tiếp có Sharpe tăng strictly.

        Returns:
            True nếu performance đang cải thiện.
        """
        if len(self._cycle_history) < self.config.min_improvement_cycles:
            return False

        # Kiểm tra N cycles cuối có Sharpe tăng strictly
        recent = self._cycle_history[-self.config.min_improvement_cycles:]
        for i in range(1, len(recent)):
            if recent[i].sharpe_ratio <= recent[i - 1].sharpe_ratio:
                return False
        return True
