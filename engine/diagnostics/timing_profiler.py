# -*- coding: utf-8 -*-
"""
Timing Profiler — Đo wall-clock time cho training/backtest phases.

Component này đo lường thời gian thực tế cho từng phase trong pipeline,
phát hiện training suspiciously fast và stub functions.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""

import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

from engine.diagnostics.models import (
    BacktestTimingReport,
    EpochValidation,
    PhaseTimingResult,
    TimingProfileReport,
)

# Torch operations cần kiểm tra để xác định stub
TORCH_OPERATIONS = [
    "torch.tensor",
    "torch.zeros",
    "torch.ones",
    "loss.backward()",
    "optimizer.step()",
    "model.forward(",
    ".backward(",
    ".step(",
]

# Pattern riêng cho model call — cần regex đơn giản để tránh false positive
# với tên hàm chứa "model" (ví dụ: _retrain_model)
MODEL_CALL_PATTERN = re.compile(r"(?<![_\w])model\s*\(")


class TimingProfiler:
    """Đo wall-clock time cho training/backtest phases.

    Cung cấp context manager để đo thời gian từng phase,
    validation cho epochs, và phát hiện stub functions.

    Attributes:
        MIN_SECONDS_PER_SYMBOL: Ngưỡng tối thiểu giây/symbol cho training.
        BACKTEST_BENCHMARK_MIN_SECONDS: Ngưỡng benchmark cho backtest (mode auto).
        BACKTEST_FAST_THRESHOLD_SECONDS: Ngưỡng fast cho backtest.
        BACKTEST_BENCHMARK_MIN_SESSIONS: Số sessions tối thiểu cho benchmark warning.
        BACKTEST_FAST_MIN_SESSIONS: Số sessions tối thiểu cho fast warning.
    """

    # Cấu hình thresholds — Training
    MIN_SECONDS_PER_SYMBOL: float = 0.9

    # Cấu hình thresholds — Backtest (định nghĩa ở class level, dùng trong task 3.2)
    BACKTEST_BENCHMARK_MIN_SECONDS: float = 0.3
    BACKTEST_FAST_THRESHOLD_SECONDS: float = 0.1
    BACKTEST_BENCHMARK_MIN_SESSIONS: int = 250
    BACKTEST_FAST_MIN_SESSIONS: int = 500

    def __init__(
        self, output_path: str = "data/engine/diagnostics/timing_profile.json"
    ):
        """Khởi tạo TimingProfiler.

        Args:
            output_path: Đường dẫn file output cho timing report.
        """
        self._output_path = output_path
        self._session_id = str(uuid.uuid4())
        self._phase_results: List[PhaseTimingResult] = []
        self._epoch_validations: List[EpochValidation] = []
        self._warnings: List[str] = []
        self._is_stub: bool = False
        self._symbol_count: int = 0

    @contextmanager
    def profile_phase(self, phase_name: str):
        """Context manager đo thời gian một phase.

        Đo wall-clock time (ms) và lưu kết quả vào danh sách phases.

        Args:
            phase_name: Tên phase (data_loading, feature_extraction, etc.).

        Yields:
            None — block bên trong context sẽ được đo thời gian.
        """
        started_at = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()

        yield

        end_time = time.perf_counter()
        ended_at = datetime.now(timezone.utc).isoformat()
        duration_ms = (end_time - start_time) * 1000.0

        result = PhaseTimingResult(
            phase_name=phase_name,
            duration_ms=duration_ms,
            started_at=started_at,
            ended_at=ended_at,
        )
        self._phase_results.append(result)

    def validate_epoch(
        self,
        epoch_number: int,
        has_forward: bool,
        has_backward: bool,
        has_step: bool,
        duration_ms: float,
    ) -> EpochValidation:
        """Validate một epoch có đủ operations hay không.

        Epoch hợp lệ khi có đủ forward pass, backward pass, và optimizer step.
        Thiếu backward hoặc step → epoch invalid (Req 2.2).

        Args:
            epoch_number: Số thứ tự epoch.
            has_forward: Có forward pass hay không.
            has_backward: Có backward pass (loss.backward()) hay không.
            has_step: Có optimizer.step() hay không.
            duration_ms: Thời gian epoch tính bằng milliseconds.

        Returns:
            EpochValidation chứa kết quả validation.
        """
        is_valid = has_forward and has_backward and has_step

        validation = EpochValidation(
            epoch_number=epoch_number,
            has_forward_pass=has_forward,
            has_backward_pass=has_backward,
            has_optimizer_step=has_step,
            is_valid=is_valid,
            duration_ms=duration_ms,
        )
        self._epoch_validations.append(validation)
        return validation

    def check_training_duration(
        self, total_duration_seconds: float, symbol_count: int
    ) -> Optional[str]:
        """Kiểm tra training duration có suspiciously fast không.

        Ngưỡng: duration < symbol_count * MIN_SECONDS_PER_SYMBOL.
        Mặc định: 60s cho 65 symbols (65 * 0.9 = 58.5 ≈ 60).

        Args:
            total_duration_seconds: Tổng thời gian training (giây).
            symbol_count: Số lượng symbols được train.

        Returns:
            Chuỗi warning "SUSPICIOUSLY_FAST" nếu quá nhanh, None nếu bình thường.
        """
        self._symbol_count = symbol_count
        min_expected_seconds = symbol_count * self.MIN_SECONDS_PER_SYMBOL

        if total_duration_seconds < min_expected_seconds:
            warning = "SUSPICIOUSLY_FAST"
            if warning not in self._warnings:
                self._warnings.append(warning)
            return warning

        return None

    def check_stub_function(self, function_source: str) -> bool:
        """Kiểm tra function có phải stub (không có torch ops).

        Kiểm tra source code có chứa bất kỳ torch operation nào
        trong danh sách TORCH_OPERATIONS không. Nếu không có → stub.

        Args:
            function_source: Source code của function cần kiểm tra.

        Returns:
            True nếu function là stub (không có torch ops), False nếu có.
        """
        # Kiểm tra substring operations
        for op in TORCH_OPERATIONS:
            if op in function_source:
                self._is_stub = False
                return False

        # Kiểm tra model call pattern (model( nhưng không phải _model( hay def model()
        if MODEL_CALL_PATTERN.search(function_source):
            self._is_stub = False
            return False

        self._is_stub = True
        return True

    def generate_training_report(self) -> TimingProfileReport:
        """Tạo báo cáo training timing.

        Tổng hợp kết quả từ tất cả phases đã đo, epoch validations,
        và xuất JSON report vào output_path.

        Returns:
            TimingProfileReport chứa toàn bộ thông tin training session.
        """
        # Tính per_phase_duration_ms
        per_phase_duration_ms: Dict[str, float] = {}
        for phase in self._phase_results:
            per_phase_duration_ms[phase.phase_name] = phase.duration_ms

        # Tính total_duration_ms
        total_duration_ms = sum(per_phase_duration_ms.values())

        # Đếm valid epochs
        valid_epochs = [e for e in self._epoch_validations if e.is_valid]
        epoch_count = len(valid_epochs)

        # Xác định is_real_computation (Req 2.6)
        is_real_computation = epoch_count > 0

        # Nếu epoch_count = 0 → warning NO_EPOCHS_EXECUTED
        if epoch_count == 0:
            if "NO_EPOCHS_EXECUTED" not in self._warnings:
                self._warnings.append("NO_EPOCHS_EXECUTED")

        # Tạo report
        report = TimingProfileReport(
            session_id=self._session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol_count=self._symbol_count,
            per_phase_duration_ms=per_phase_duration_ms,
            total_duration_ms=total_duration_ms,
            epoch_count=epoch_count,
            is_stub=self._is_stub,
            is_real_computation=is_real_computation,
            warnings=list(self._warnings),
        )

        # Ghi report ra file
        self._write_report(report)

        return report

    def check_backtest_symbol(
        self,
        symbol: str,
        sessions_count: int,
        total_time_seconds: float,
        processed_days: int,
        total_days: int,
        mode: str = "manual",
    ) -> List[str]:
        """Kiểm tra backtest timing cho một symbol và trả về danh sách warnings.

        Áp dụng các ngưỡng:
        - BENCHMARK_BELOW_EXPECTED: mode="auto" AND sessions >= 250 AND time < 0.3s
        - BACKTEST_TOO_FAST: sessions >= 500 AND time < 0.1s
        - INCOMPLETE_ITERATION: processed_days < total_days

        Args:
            symbol: Mã chứng khoán.
            sessions_count: Số phiên giao dịch trong dataset.
            total_time_seconds: Tổng thời gian backtest cho symbol (giây).
            processed_days: Số ngày đã xử lý.
            total_days: Tổng số ngày trong dataset.
            mode: Chế độ chạy ("auto" từ AutoLearner, "manual" từ user).

        Returns:
            Danh sách mã cảnh báo áp dụng cho symbol này.
        """
        warnings: List[str] = []

        # Req 3.2: BENCHMARK_BELOW_EXPECTED
        if (
            mode == "auto"
            and sessions_count >= self.BACKTEST_BENCHMARK_MIN_SESSIONS
            and total_time_seconds < self.BACKTEST_BENCHMARK_MIN_SECONDS
        ):
            warnings.append("BENCHMARK_BELOW_EXPECTED")

        # Req 3.3: BACKTEST_TOO_FAST
        if (
            sessions_count >= self.BACKTEST_FAST_MIN_SESSIONS
            and total_time_seconds < self.BACKTEST_FAST_THRESHOLD_SECONDS
        ):
            warnings.append("BACKTEST_TOO_FAST")

        # Req 3.4: INCOMPLETE_ITERATION
        if processed_days < total_days:
            warnings.append("INCOMPLETE_ITERATION")

        return warnings

    def generate_backtest_report(
        self, symbol_results: List[BacktestTimingReport], mode: str = "manual"
    ) -> Dict:
        """Tạo báo cáo backtest timing với per-symbol breakdown.

        Kiểm tra từng symbol theo benchmark thresholds và tổng hợp kết quả
        thành structured report. Ghi output JSON vào output_path.

        Args:
            symbol_results: Danh sách BacktestTimingReport cho từng symbol.
            mode: Chế độ chạy ("auto" từ AutoLearner, "manual" từ user).

        Returns:
            Dict chứa toàn bộ backtest timing report (session_id, timestamp,
            symbols, total_time_ms, total_symbols, warnings).
        """
        all_warnings: List[str] = []
        symbols_output: List[Dict] = []

        for result in symbol_results:
            # Tính total_time_seconds từ total_time_ms
            total_time_seconds = result.total_time_ms / 1000.0

            # Kiểm tra warnings cho symbol này
            symbol_warnings = self.check_backtest_symbol(
                symbol=result.symbol,
                sessions_count=result.sessions_count,
                total_time_seconds=total_time_seconds,
                processed_days=result.processed_days,
                total_days=result.total_days,
                mode=mode,
            )

            # Cập nhật warnings vào result
            result.warnings = symbol_warnings

            # Tổng hợp warnings ở report level (không trùng lặp)
            for w in symbol_warnings:
                if w not in all_warnings:
                    all_warnings.append(w)

            symbols_output.append(
                {
                    "symbol": result.symbol,
                    "sessions_count": result.sessions_count,
                    "total_time_ms": result.total_time_ms,
                    "strategy_signal_generation_ms": result.strategy_signal_generation_ms,
                    "trade_execution_ms": result.trade_execution_ms,
                    "metrics_computation_ms": result.metrics_computation_ms,
                    "processed_days": result.processed_days,
                    "total_days": result.total_days,
                    "warnings": symbol_warnings,
                }
            )

        # Tính tổng thời gian
        total_time_ms = sum(r.total_time_ms for r in symbol_results)

        report_dict = {
            "session_id": self._session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols": symbols_output,
            "total_time_ms": total_time_ms,
            "total_symbols": len(symbol_results),
            "warnings": all_warnings,
        }

        # Ghi report ra file
        self._write_backtest_report(report_dict)

        return report_dict

    def _write_backtest_report(self, report_dict: Dict) -> None:
        """Ghi backtest report ra file JSON.

        Tự động tạo directory nếu chưa tồn tại.

        Args:
            report_dict: Dict chứa backtest timing report.
        """
        output_dir = os.path.dirname(self._output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Ghi vào file riêng cho backtest (cùng directory)
        backtest_path = self._output_path.replace(
            "timing_profile.json", "backtest_timing.json"
        )
        with open(backtest_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

    def _write_report(self, report: TimingProfileReport) -> None:
        """Ghi report ra file JSON.

        Tự động tạo directory nếu chưa tồn tại.

        Args:
            report: TimingProfileReport cần ghi.
        """
        output_dir = os.path.dirname(self._output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        report_dict = {
            "session_id": report.session_id,
            "timestamp": report.timestamp,
            "symbol_count": report.symbol_count,
            "per_phase_duration_ms": report.per_phase_duration_ms,
            "total_duration_ms": report.total_duration_ms,
            "epoch_count": report.epoch_count,
            "is_stub": report.is_stub,
            "is_real_computation": report.is_real_computation,
            "warnings": report.warnings,
        }

        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
