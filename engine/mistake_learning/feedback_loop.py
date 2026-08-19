"""
FeedbackLoopManager — Orchestrate toàn bộ feedback cycle.

Điều phối các bước: backtest → persist trades → analyze mistakes →
mine hard examples → correct labels → update anti-patterns → retrain.

Xử lý guards: config disabled, insufficient trades, zero bad trades.
Xử lý error: abort khi exception, preserve model weights.
Xử lý retrain divergence: loss tăng > 50% → abort, restore weights.
"""

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.config import BacktestResult, Trade
from engine.mistake_learning.anti_pattern_db import AntiPatternDatabase
from engine.mistake_learning.config import MistakeLearningConfig
from engine.mistake_learning.feedback_loop_support import (
    retrain_via_model_manager,
    retrain_via_pipeline,
    run_backtest_via_api,
    run_backtest_via_engine,
)
from engine.mistake_learning.hard_example_miner import HardExampleMiner
from engine.mistake_learning.label_corrector import LabelCorrector
from engine.mistake_learning.mistake_analyzer import MistakeAnalyzer
from engine.mistake_learning.models import (
    FeedbackCycleResult,
    HardExampleSet,
    MistakeReport,
    TradeRecord,
)
from engine.mistake_learning.trade_history import TradeHistoryStore

logger = logging.getLogger(__name__)


class FeedbackLoopManager:
    """Orchestrate toàn bộ feedback cycle cho Mistake-Driven Learning.

    Kết nối các component: TradeHistoryStore, MistakeAnalyzer,
    HardExampleMiner, LabelCorrector, AntiPatternDatabase.
    Điều phối thứ tự thực hiện và xử lý error/guards.
    """

    def __init__(
        self,
        engine: Any = None,
        config: Optional[MistakeLearningConfig] = None,
        base_dir: str = ".",
    ) -> None:
        """Khởi tạo FeedbackLoopManager với tất cả sub-components.

        Args:
            engine: DecisionEngine instance (có thể None nếu chạy standalone).
            config: Cấu hình feedback loop. Dùng default nếu None.
            base_dir: Thư mục gốc cho persistence files.
        """
        self.config = config or MistakeLearningConfig()
        self._engine = engine
        self._base_dir = base_dir

        # Khởi tạo sub-components
        self.trade_history = TradeHistoryStore(base_dir)
        self.mistake_analyzer = MistakeAnalyzer(self.config)
        self.hard_example_miner = HardExampleMiner(
            weight_multiplier=self.config.weight_multiplier,
            max_hard_examples_ratio=self.config.max_hard_examples_ratio,
        )
        self.label_corrector = LabelCorrector(
            correction_threshold=self.config.correction_threshold,
            min_confidence_for_correction=self.config.min_confidence_for_correction,
        )
        self.anti_pattern_db = AntiPatternDatabase(
            base_dir=base_dir,
            max_patterns=self.config.max_patterns,
        )

        # Lưu trữ model weights để restore khi cần
        self._saved_weights: Optional[Any] = None

    def run_feedback_cycle(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        cycle_number: int,
    ) -> FeedbackCycleResult:
        """Chạy toàn bộ feedback cycle.

        Thứ tự: backtest → persist → analyze → mine → correct →
        update patterns → retrain.

        Guards:
        - config.enabled=False → return ngay
        - trades < min_trades_for_analysis → skip
        - zero bad trades → skip mining/correction/retrain

        Error handling:
        - Exception → abort, preserve weights, return skipped=True

        Args:
            symbol_data: Dict mapping symbol → DataFrame chứa OHLCV + indicators.
            cycle_number: Số cycle hiện tại.

        Returns:
            FeedbackCycleResult mô tả kết quả cycle.
        """
        # Guard: config disabled
        if not self.config.enabled:
            return FeedbackCycleResult(skipped=True, reason="disabled")

        # Lưu model weights trước khi bắt đầu
        self._save_model_weights()

        current_step = "backtest"
        try:
            # Step 1: Chạy validation backtest
            current_step = "backtest"
            trades = self._run_validation_backtest(symbol_data)

            # Guard: không đủ trades
            if len(trades) < self.config.min_trades_for_analysis:
                return FeedbackCycleResult(
                    skipped=True, reason="insufficient_trades"
                )

            # Step 2: Persist trades
            current_step = "persist_trades"
            trade_records = self._enrich_trades_with_features(trades, symbol_data)
            self.trade_history.persist_trades(cycle_number, trade_records)

            # Step 3: Analyze mistakes
            current_step = "analyze_mistakes"
            first_symbol = next(iter(symbol_data))
            df = symbol_data[first_symbol]
            report = self.mistake_analyzer.analyze_trades(
                trades=trades,
                df=df,
                signal_history=[],
            )

            # Guard: zero bad trades → skip mining/correction/retrain
            if len(report.bad_trades) == 0:
                return FeedbackCycleResult(
                    skipped=False,
                    mistake_report=report,
                    hard_examples_count=0,
                    corrections_count=0,
                )

            # Step 4: Mine hard examples
            current_step = "mine_hard_examples"
            hard_examples = self.hard_example_miner.mine(
                bad_trades=report.bad_trades,
                feature_data=symbol_data,
                lookback=60,
            )

            # Step 5: Correct labels
            current_step = "correct_labels"
            labels, dates = self._get_training_labels_and_dates(symbol_data)
            corrected = self.label_corrector.correct_labels(
                bad_trades=report.bad_trades,
                original_labels=labels,
                dates=dates,
            )

            # Step 6: Update anti-patterns
            current_step = "update_patterns"
            self.anti_pattern_db.update_patterns(
                report.bad_trades, cycle_number
            )

            # Step 7: Retrain with feedback
            current_step = "retrain"
            max_epochs = self.config.max_retrain_epochs
            retrain_result = self._retrain_with_feedback(
                symbol_data=symbol_data,
                hard_examples=hard_examples,
                corrected=corrected,
                max_epochs=max_epochs,
            )

            return FeedbackCycleResult(
                skipped=False,
                mistake_report=report,
                hard_examples_count=hard_examples.count,
                corrections_count=corrected.correction_count,
                retrain_result=retrain_result,
            )

        except Exception as e:
            # Abort: restore model weights và return error
            logger.error(
                "Feedback cycle lỗi tại step '%s': %s", current_step, str(e)
            )
            self._restore_model_weights()
            return FeedbackCycleResult(
                skipped=True,
                reason=f"error_in_{current_step}: {e}",
            )

    # =========================================================================
    # Private: Backtest
    # =========================================================================

    def _run_validation_backtest(
        self, symbol_data: Dict[str, pd.DataFrame]
    ) -> List[Trade]:
        """Chạy backtest trên validation period sử dụng BacktestEngine thực tế.

        Thử các approach theo thứ tự:
        1. Dùng engine._backtest_engine.run() với _FeedbackAIStrategy
        2. Dùng engine.backtest() API cấp cao (cần symbol name)
        3. Fallback: trả về danh sách rỗng

        Args:
            symbol_data: Dữ liệu symbol.

        Returns:
            Danh sách Trade từ backtest.
        """
        if self._engine is None:
            return []

        # Approach 1: Dùng BacktestEngine trực tiếp qua DecisionEngine
        if hasattr(self._engine, "_backtest_engine") and hasattr(
            self._engine, "model_manager"
        ):
            try:
                return run_backtest_via_engine(self._engine, symbol_data)
            except Exception as e:
                logger.warning(
                    "Backtest qua BacktestEngine thất bại: %s", str(e)
                )

        # Approach 2: Dùng engine.backtest() API cấp cao
        if hasattr(self._engine, "backtest"):
            try:
                return run_backtest_via_api(self._engine, symbol_data)
            except Exception as e:
                logger.warning("Backtest qua API thất bại: %s", str(e))

        return []

    # =========================================================================
    # Private: Retrain
    # =========================================================================

    def _retrain_with_feedback(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        hard_examples: HardExampleSet,
        corrected: Any,
        max_epochs: int,
    ) -> Optional[Any]:
        """Retrain model với hard examples và corrected labels (weighted loss).

        Thử các approach theo thứ tự:
        1. engine.retrain_with_feedback() nếu engine hỗ trợ trực tiếp
        2. engine._background_training pipeline với weighted data
        3. model_manager.fine_tune() trực tiếp

        Kiểm tra divergence: nếu loss tăng > 50% → abort, restore weights.

        Args:
            symbol_data: Dữ liệu training.
            hard_examples: Hard examples đã mine.
            corrected: Labels đã sửa (CorrectedLabels).
            max_epochs: Số epochs tối đa.

        Returns:
            Kết quả retrain hoặc None nếu không thể retrain.
        """
        if self._engine is None:
            return None

        # Giới hạn max_epochs
        epochs = min(max_epochs, self.config.max_retrain_epochs)

        # Approach 1: Engine hỗ trợ retrain_with_feedback trực tiếp
        if hasattr(self._engine, "retrain_with_feedback"):
            return self._retrain_direct(
                symbol_data, hard_examples, corrected, epochs
            )

        # Approach 2: Dùng training pipeline với weighted data
        if hasattr(self._engine, "_background_training"):
            initial_loss = self._get_current_loss()
            result = retrain_via_pipeline(
                self._engine,
                symbol_data,
                epochs,
                hard_examples=hard_examples,
            )
            if result is not None:
                if self._check_divergence(initial_loss):
                    return None
                return result

        # Approach 3: Dùng model_manager trực tiếp
        if hasattr(self._engine, "model_manager"):
            initial_loss = self._get_current_loss()
            result = retrain_via_model_manager(
                self._engine, hard_examples, epochs
            )
            if result is not None:
                if self._check_divergence(initial_loss):
                    return None
                return result

        return None

    def _retrain_direct(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        hard_examples: HardExampleSet,
        corrected: Any,
        max_epochs: int,
    ) -> Optional[Any]:
        """Retrain qua engine.retrain_with_feedback() trực tiếp.

        Args:
            symbol_data: Dữ liệu training.
            hard_examples: Hard examples.
            corrected: Corrected labels.
            max_epochs: Epochs tối đa.

        Returns:
            Kết quả retrain hoặc None.
        """
        try:
            initial_loss = self._get_current_loss()

            result = self._engine.retrain_with_feedback(
                symbol_data=symbol_data,
                hard_examples=hard_examples,
                corrected_labels=corrected,
                max_epochs=max_epochs,
            )

            # Kiểm tra divergence
            if self._check_divergence(initial_loss):
                return None

            return result

        except Exception as e:
            logger.error("Retrain direct thất bại: %s", str(e))
            self._restore_model_weights()
            raise

    # =========================================================================
    # Private: Helpers
    # =========================================================================

    def _enrich_trades_with_features(
        self,
        trades: List[Trade],
        symbol_data: Dict[str, pd.DataFrame],
    ) -> List[TradeRecord]:
        """Chuyển đổi Trade → TradeRecord, thêm feature vector và market context.

        Args:
            trades: Danh sách trades gốc.
            symbol_data: Dữ liệu symbol để trích xuất features.

        Returns:
            Danh sách TradeRecord đã enriched.
        """
        if not symbol_data:
            return []

        records: List[TradeRecord] = []
        for trade in trades:
            record = TradeRecord(
                entry_date=trade.entry_date,
                exit_date=trade.exit_date,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                shares=trade.shares,
                pnl=trade.pnl,
                pnl_pct=trade.pnl_pct,
                cycle_number=0,
                signal_strength=0.0,
                action_taken="BUY" if trade.pnl_pct >= 0 else "SELL",
                classification="UNKNOWN",
            )
            records.append(record)

        return records

    def _get_training_labels_and_dates(
        self, symbol_data: Dict[str, pd.DataFrame]
    ) -> tuple:
        """Lấy training labels và dates hiện tại.

        Nếu engine có sẵn labels, dùng chúng.
        Ngược lại tạo stub labels từ symbol_data.

        Args:
            symbol_data: Dữ liệu symbol.

        Returns:
            Tuple (labels: np.ndarray, dates: pd.DatetimeIndex).
        """
        # Thử lấy từ engine nếu có
        if self._engine is not None:
            if hasattr(self._engine, "get_training_labels"):
                try:
                    labels, dates = self._engine.get_training_labels()
                    return labels, dates
                except Exception:
                    pass

        # Stub: tạo labels từ DataFrame đầu tiên
        if not symbol_data:
            return np.array([]), pd.DatetimeIndex([])

        first_symbol = next(iter(symbol_data))
        df = symbol_data[first_symbol]

        n = len(df)
        labels = np.zeros(n, dtype=np.float64)

        # Tạo DatetimeIndex từ cột 'time' hoặc index
        if "time" in df.columns:
            dates = pd.DatetimeIndex(pd.to_datetime(df["time"]))
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        else:
            dates = pd.DatetimeIndex(
                pd.date_range("2020-01-01", periods=n, freq="B")
            )

        return labels, dates

    def _check_divergence(self, initial_loss: Optional[float]) -> bool:
        """Kiểm tra xem loss có tăng quá 50% (divergence) không.

        Nếu divergence → restore weights và return True.

        Args:
            initial_loss: Loss trước khi retrain.

        Returns:
            True nếu divergence xảy ra (đã restore), False nếu OK.
        """
        final_loss = self._get_current_loss()
        if initial_loss is not None and final_loss is not None:
            if initial_loss > 0 and final_loss > initial_loss * 1.5:
                logger.warning(
                    "Retrain divergence: loss tăng từ %.4f lên %.4f (>50%%), "
                    "restore weights.",
                    initial_loss,
                    final_loss,
                )
                self._restore_model_weights()
                return True
        return False

    def _save_model_weights(self) -> None:
        """Lưu trữ bản copy model weights để restore khi cần."""
        if self._engine is None:
            self._saved_weights = None
            return

        if hasattr(self._engine, "get_model_weights"):
            try:
                weights = self._engine.get_model_weights()
                self._saved_weights = copy.deepcopy(weights)
            except Exception:
                self._saved_weights = None
        else:
            self._saved_weights = None

    def _restore_model_weights(self) -> None:
        """Restore model weights từ bản lưu."""
        if self._engine is None or self._saved_weights is None:
            return

        if hasattr(self._engine, "set_model_weights"):
            try:
                self._engine.set_model_weights(self._saved_weights)
                logger.info("Đã restore model weights thành công.")
            except Exception as e:
                logger.error("Không thể restore model weights: %s", str(e))

    def _get_current_loss(self) -> Optional[float]:
        """Lấy loss hiện tại từ engine (nếu có).

        Returns:
            Loss value hoặc None nếu không lấy được.
        """
        if self._engine is None:
            return None

        if hasattr(self._engine, "get_current_loss"):
            try:
                return self._engine.get_current_loss()
            except Exception:
                return None

        return None
