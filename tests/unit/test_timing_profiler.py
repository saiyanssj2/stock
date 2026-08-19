# -*- coding: utf-8 -*-
"""
Unit tests cho engine/diagnostics/timing_profiler.py — Training profiling.

Kiểm tra TimingProfiler:
- profile_phase(): context manager đo wall-clock time
- validate_epoch(): validation forward/backward/step
- check_training_duration(): phát hiện suspiciously fast
- check_stub_function(): phát hiện stub (không có torch ops)
- generate_training_report(): xuất JSON report

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""

import json
import os
import tempfile
import time

import pytest

from engine.diagnostics.models import (
    EpochValidation,
    PhaseTimingResult,
    TimingProfileReport,
)
from engine.diagnostics.timing_profiler import TORCH_OPERATIONS, TimingProfiler


class TestProfilePhase:
    """Test context manager profile_phase() — Req 2.1."""

    def test_profile_phase_records_duration(self):
        """Phase duration được ghi lại chính xác (ms)."""
        profiler = TimingProfiler()

        with profiler.profile_phase("data_loading"):
            time.sleep(0.01)  # 10ms

        assert len(profiler._phase_results) == 1
        result = profiler._phase_results[0]
        assert result.phase_name == "data_loading"
        # Ít nhất 10ms
        assert result.duration_ms >= 10.0
        assert result.started_at != ""
        assert result.ended_at != ""

    def test_profile_multiple_phases(self):
        """Nhiều phases được ghi lại đúng thứ tự."""
        profiler = TimingProfiler()
        phases = [
            "data_loading",
            "feature_extraction",
            "label_generation",
            "model_training",
            "checkpoint_saving",
        ]

        for phase in phases:
            with profiler.profile_phase(phase):
                pass

        assert len(profiler._phase_results) == 5
        for i, phase in enumerate(phases):
            assert profiler._phase_results[i].phase_name == phase

    def test_profile_phase_zero_duration(self):
        """Phase rất nhanh vẫn được ghi duration >= 0."""
        profiler = TimingProfiler()

        with profiler.profile_phase("fast_phase"):
            pass

        assert profiler._phase_results[0].duration_ms >= 0.0


class TestValidateEpoch:
    """Test validate_epoch() — Req 2.2."""

    def test_valid_epoch_all_operations(self):
        """Epoch có đủ forward/backward/step → is_valid=True."""
        profiler = TimingProfiler()
        result = profiler.validate_epoch(
            epoch_number=1,
            has_forward=True,
            has_backward=True,
            has_step=True,
            duration_ms=100.0,
        )

        assert result.is_valid is True
        assert result.epoch_number == 1
        assert result.has_forward_pass is True
        assert result.has_backward_pass is True
        assert result.has_optimizer_step is True
        assert result.duration_ms == 100.0

    def test_invalid_epoch_missing_backward(self):
        """Epoch thiếu backward → is_valid=False."""
        profiler = TimingProfiler()
        result = profiler.validate_epoch(
            epoch_number=1,
            has_forward=True,
            has_backward=False,
            has_step=True,
            duration_ms=50.0,
        )

        assert result.is_valid is False

    def test_invalid_epoch_missing_step(self):
        """Epoch thiếu optimizer.step() → is_valid=False."""
        profiler = TimingProfiler()
        result = profiler.validate_epoch(
            epoch_number=1,
            has_forward=True,
            has_backward=True,
            has_step=False,
            duration_ms=50.0,
        )

        assert result.is_valid is False

    def test_invalid_epoch_missing_forward(self):
        """Epoch thiếu forward pass → is_valid=False."""
        profiler = TimingProfiler()
        result = profiler.validate_epoch(
            epoch_number=1,
            has_forward=False,
            has_backward=True,
            has_step=True,
            duration_ms=50.0,
        )

        assert result.is_valid is False

    def test_invalid_epoch_all_missing(self):
        """Epoch không có operation nào → is_valid=False."""
        profiler = TimingProfiler()
        result = profiler.validate_epoch(
            epoch_number=1,
            has_forward=False,
            has_backward=False,
            has_step=False,
            duration_ms=0.0,
        )

        assert result.is_valid is False

    def test_multiple_epochs_tracked(self):
        """Nhiều epochs được lưu trong danh sách."""
        profiler = TimingProfiler()
        profiler.validate_epoch(1, True, True, True, 100.0)
        profiler.validate_epoch(2, True, True, True, 110.0)
        profiler.validate_epoch(3, True, False, True, 50.0)

        assert len(profiler._epoch_validations) == 3


class TestCheckTrainingDuration:
    """Test check_training_duration() — Req 2.3."""

    def test_fast_training_triggers_warning(self):
        """Duration < symbol_count * 0.9s → SUSPICIOUSLY_FAST."""
        profiler = TimingProfiler()
        # 65 symbols * 0.9 = 58.5s, actual = 7s → warning
        result = profiler.check_training_duration(7.0, 65)

        assert result == "SUSPICIOUSLY_FAST"
        assert "SUSPICIOUSLY_FAST" in profiler._warnings

    def test_normal_training_no_warning(self):
        """Duration >= symbol_count * 0.9s → None."""
        profiler = TimingProfiler()
        # 65 symbols * 0.9 = 58.5s, actual = 60s → no warning
        result = profiler.check_training_duration(60.0, 65)

        assert result is None
        assert "SUSPICIOUSLY_FAST" not in profiler._warnings

    def test_exact_threshold_no_warning(self):
        """Duration == threshold exactly → no warning."""
        profiler = TimingProfiler()
        # 10 symbols * 0.9 = 9.0s, actual = 9.0s → no warning
        result = profiler.check_training_duration(9.0, 10)

        assert result is None

    def test_just_below_threshold_triggers_warning(self):
        """Duration slightly below threshold → warning."""
        profiler = TimingProfiler()
        # 10 symbols * 0.9 = 9.0s, actual = 8.99s → warning
        result = profiler.check_training_duration(8.99, 10)

        assert result == "SUSPICIOUSLY_FAST"

    def test_zero_symbols(self):
        """0 symbols → threshold = 0, duration 0 → no warning."""
        profiler = TimingProfiler()
        result = profiler.check_training_duration(0.0, 0)

        assert result is None

    def test_warning_not_duplicated(self):
        """Gọi nhiều lần không duplicate warning."""
        profiler = TimingProfiler()
        profiler.check_training_duration(1.0, 65)
        profiler.check_training_duration(2.0, 65)

        assert profiler._warnings.count("SUSPICIOUSLY_FAST") == 1


class TestCheckStubFunction:
    """Test check_stub_function() — Req 2.4."""

    def test_stub_function_no_torch_ops(self):
        """Function không có torch operations → is_stub=True."""
        profiler = TimingProfiler()
        source = '''
def _retrain_model(self, symbols):
    """Retrain model cho danh sách symbols."""
    logger.info(f"Retraining {len(symbols)} symbols")
    return True
'''
        assert profiler.check_stub_function(source) is True
        assert profiler._is_stub is True

    def test_real_function_with_tensor_creation(self):
        """Function có torch.tensor → is_stub=False."""
        profiler = TimingProfiler()
        source = '''
def train_model(self, data):
    x = torch.tensor(data)
    output = model(x)
    return output
'''
        assert profiler.check_stub_function(source) is False
        assert profiler._is_stub is False

    def test_real_function_with_backward(self):
        """Function có loss.backward() → is_stub=False."""
        profiler = TimingProfiler()
        source = '''
def train_step(self, batch):
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()
'''
        assert profiler.check_stub_function(source) is False

    def test_real_function_with_optimizer_step(self):
        """Function có optimizer.step() → is_stub=False."""
        profiler = TimingProfiler()
        source = '''
def optimize(self):
    optimizer.step()
'''
        assert profiler.check_stub_function(source) is False

    def test_real_function_with_model_forward(self):
        """Function có model.forward( → is_stub=False."""
        profiler = TimingProfiler()
        source = '''
def predict(self, x):
    return model.forward(x)
'''
        assert profiler.check_stub_function(source) is False

    def test_real_function_with_model_call(self):
        """Function có model( → is_stub=False."""
        profiler = TimingProfiler()
        source = '''
def predict(self, x):
    return model(x)
'''
        assert profiler.check_stub_function(source) is False

    def test_real_function_with_torch_zeros(self):
        """Function có torch.zeros → is_stub=False."""
        profiler = TimingProfiler()
        source = '''
def init_weights(self):
    w = torch.zeros(10, 10)
    return w
'''
        assert profiler.check_stub_function(source) is False

    def test_real_function_with_torch_ones(self):
        """Function có torch.ones → is_stub=False."""
        profiler = TimingProfiler()
        source = '''
def init_bias(self):
    b = torch.ones(10)
    return b
'''
        assert profiler.check_stub_function(source) is False

    def test_empty_function_is_stub(self):
        """Function rỗng (chỉ có pass) → is_stub=True."""
        profiler = TimingProfiler()
        source = '''
def placeholder(self):
    pass
'''
        assert profiler.check_stub_function(source) is True


class TestGenerateTrainingReport:
    """Test generate_training_report() — Req 2.5, 2.6."""

    def test_report_with_valid_epochs(self):
        """Report có valid epochs → is_real_computation=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "timing_profile.json")
            profiler = TimingProfiler(output_path=output_path)
            profiler._symbol_count = 10

            # Thêm phase results
            with profiler.profile_phase("model_training"):
                pass

            # Thêm valid epoch
            profiler.validate_epoch(1, True, True, True, 100.0)

            report = profiler.generate_training_report()

            assert report.session_id != ""
            assert report.timestamp != ""
            assert report.symbol_count == 10
            assert "model_training" in report.per_phase_duration_ms
            assert report.total_duration_ms >= 0.0
            assert report.epoch_count == 1
            assert report.is_real_computation is True
            assert "NO_EPOCHS_EXECUTED" not in report.warnings

    def test_report_zero_epochs_warning(self):
        """epoch_count=0 → is_real_computation=False, warning NO_EPOCHS_EXECUTED."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "timing_profile.json")
            profiler = TimingProfiler(output_path=output_path)
            profiler._symbol_count = 65

            report = profiler.generate_training_report()

            assert report.epoch_count == 0
            assert report.is_real_computation is False
            assert "NO_EPOCHS_EXECUTED" in report.warnings

    def test_report_invalid_epochs_not_counted(self):
        """Invalid epochs không được đếm vào epoch_count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "timing_profile.json")
            profiler = TimingProfiler(output_path=output_path)

            # 1 valid, 2 invalid
            profiler.validate_epoch(1, True, True, True, 100.0)
            profiler.validate_epoch(2, True, False, True, 50.0)
            profiler.validate_epoch(3, True, True, False, 50.0)

            report = profiler.generate_training_report()

            assert report.epoch_count == 1
            assert report.is_real_computation is True

    def test_report_json_output_file(self):
        """Report được ghi ra JSON file đúng schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "timing_profile.json")
            profiler = TimingProfiler(output_path=output_path)
            profiler._symbol_count = 5

            with profiler.profile_phase("data_loading"):
                pass
            profiler.validate_epoch(1, True, True, True, 50.0)

            profiler.generate_training_report()

            # Verify file tồn tại và có đúng schema
            assert os.path.exists(output_path)
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            required_fields = [
                "session_id",
                "timestamp",
                "symbol_count",
                "per_phase_duration_ms",
                "total_duration_ms",
                "epoch_count",
                "is_stub",
                "is_real_computation",
                "warnings",
            ]
            for field_name in required_fields:
                assert field_name in data, f"Missing field: {field_name}"

            assert isinstance(data["session_id"], str)
            assert isinstance(data["timestamp"], str)
            assert isinstance(data["symbol_count"], int)
            assert isinstance(data["per_phase_duration_ms"], dict)
            assert isinstance(data["total_duration_ms"], (int, float))
            assert isinstance(data["epoch_count"], int)
            assert isinstance(data["is_stub"], bool)
            assert isinstance(data["is_real_computation"], bool)
            assert isinstance(data["warnings"], list)

    def test_report_creates_output_directory(self):
        """Tự động tạo directory nếu chưa tồn tại."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "nested", "dir", "timing.json")
            profiler = TimingProfiler(output_path=output_path)

            profiler.generate_training_report()

            assert os.path.exists(output_path)

    def test_report_includes_all_warnings(self):
        """Report bao gồm tất cả warnings đã thu thập."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "timing_profile.json")
            profiler = TimingProfiler(output_path=output_path)
            profiler._symbol_count = 65

            # Trigger SUSPICIOUSLY_FAST
            profiler.check_training_duration(5.0, 65)

            report = profiler.generate_training_report()

            assert "SUSPICIOUSLY_FAST" in report.warnings
            assert "NO_EPOCHS_EXECUTED" in report.warnings


class TestTimingProfilerConstants:
    """Test class-level constants."""

    def test_min_seconds_per_symbol(self):
        """MIN_SECONDS_PER_SYMBOL = 0.9."""
        assert TimingProfiler.MIN_SECONDS_PER_SYMBOL == 0.9

    def test_backtest_constants_defined(self):
        """Backtest constants được định nghĩa ở class level."""
        assert TimingProfiler.BACKTEST_BENCHMARK_MIN_SECONDS == 0.3
        assert TimingProfiler.BACKTEST_FAST_THRESHOLD_SECONDS == 0.1
        assert TimingProfiler.BACKTEST_BENCHMARK_MIN_SESSIONS == 250
        assert TimingProfiler.BACKTEST_FAST_MIN_SESSIONS == 500


class TestCheckBacktestSymbol:
    """Test check_backtest_symbol() — Req 3.2, 3.3, 3.4."""

    def test_benchmark_below_expected_auto_mode(self):
        """mode=auto, sessions>=250, time<0.3s → BENCHMARK_BELOW_EXPECTED."""
        profiler = TimingProfiler()
        warnings = profiler.check_backtest_symbol(
            symbol="VNM",
            sessions_count=300,
            total_time_seconds=0.2,
            processed_days=300,
            total_days=300,
            mode="auto",
        )

        assert "BENCHMARK_BELOW_EXPECTED" in warnings

    def test_benchmark_not_triggered_manual_mode(self):
        """mode=manual → BENCHMARK_BELOW_EXPECTED không được trigger."""
        profiler = TimingProfiler()
        warnings = profiler.check_backtest_symbol(
            symbol="VNM",
            sessions_count=300,
            total_time_seconds=0.2,
            processed_days=300,
            total_days=300,
            mode="manual",
        )

        assert "BENCHMARK_BELOW_EXPECTED" not in warnings

    def test_benchmark_not_triggered_below_session_threshold(self):
        """sessions < 250 → BENCHMARK_BELOW_EXPECTED không được trigger."""
        profiler = TimingProfiler()
        warnings = profiler.check_backtest_symbol(
            symbol="VNM",
            sessions_count=200,
            total_time_seconds=0.1,
            processed_days=200,
            total_days=200,
            mode="auto",
        )

        assert "BENCHMARK_BELOW_EXPECTED" not in warnings

    def test_benchmark_not_triggered_above_time_threshold(self):
        """time >= 0.3s → BENCHMARK_BELOW_EXPECTED không được trigger."""
        profiler = TimingProfiler()
        warnings = profiler.check_backtest_symbol(
            symbol="VNM",
            sessions_count=300,
            total_time_seconds=0.3,
            processed_days=300,
            total_days=300,
            mode="auto",
        )

        assert "BENCHMARK_BELOW_EXPECTED" not in warnings

    def test_backtest_too_fast(self):
        """sessions>=500, time<0.1s → BACKTEST_TOO_FAST."""
        profiler = TimingProfiler()
        warnings = profiler.check_backtest_symbol(
            symbol="VNM",
            sessions_count=500,
            total_time_seconds=0.05,
            processed_days=500,
            total_days=500,
            mode="manual",
        )

        assert "BACKTEST_TOO_FAST" in warnings

    def test_backtest_too_fast_not_triggered_below_sessions(self):
        """sessions < 500 → BACKTEST_TOO_FAST không được trigger."""
        profiler = TimingProfiler()
        warnings = profiler.check_backtest_symbol(
            symbol="VNM",
            sessions_count=400,
            total_time_seconds=0.05,
            processed_days=400,
            total_days=400,
            mode="manual",
        )

        assert "BACKTEST_TOO_FAST" not in warnings

    def test_backtest_too_fast_not_triggered_above_time(self):
        """time >= 0.1s → BACKTEST_TOO_FAST không được trigger."""
        profiler = TimingProfiler()
        warnings = profiler.check_backtest_symbol(
            symbol="VNM",
            sessions_count=600,
            total_time_seconds=0.1,
            processed_days=600,
            total_days=600,
            mode="manual",
        )

        assert "BACKTEST_TOO_FAST" not in warnings

    def test_incomplete_iteration(self):
        """processed_days < total_days → INCOMPLETE_ITERATION."""
        profiler = TimingProfiler()
        warnings = profiler.check_backtest_symbol(
            symbol="VNM",
            sessions_count=300,
            total_time_seconds=1.0,
            processed_days=250,
            total_days=300,
            mode="manual",
        )

        assert "INCOMPLETE_ITERATION" in warnings

    def test_complete_iteration_no_warning(self):
        """processed_days == total_days → không có INCOMPLETE_ITERATION."""
        profiler = TimingProfiler()
        warnings = profiler.check_backtest_symbol(
            symbol="VNM",
            sessions_count=300,
            total_time_seconds=1.0,
            processed_days=300,
            total_days=300,
            mode="manual",
        )

        assert "INCOMPLETE_ITERATION" not in warnings

    def test_multiple_warnings_combined(self):
        """Nhiều điều kiện thỏa mãn → nhiều warnings."""
        profiler = TimingProfiler()
        # mode=auto, sessions=600>=250, time=0.05<0.3 → BENCHMARK_BELOW_EXPECTED
        # sessions=600>=500, time=0.05<0.1 → BACKTEST_TOO_FAST
        # processed_days=500 < total_days=600 → INCOMPLETE_ITERATION
        warnings = profiler.check_backtest_symbol(
            symbol="VNM",
            sessions_count=600,
            total_time_seconds=0.05,
            processed_days=500,
            total_days=600,
            mode="auto",
        )

        assert "BENCHMARK_BELOW_EXPECTED" in warnings
        assert "BACKTEST_TOO_FAST" in warnings
        assert "INCOMPLETE_ITERATION" in warnings

    def test_no_warnings_normal_case(self):
        """Trường hợp bình thường → không có warnings."""
        profiler = TimingProfiler()
        warnings = profiler.check_backtest_symbol(
            symbol="VNM",
            sessions_count=300,
            total_time_seconds=1.5,
            processed_days=300,
            total_days=300,
            mode="manual",
        )

        assert warnings == []


class TestGenerateBacktestReport:
    """Test generate_backtest_report() — Req 3.1, 3.2, 3.3, 3.4."""

    def test_report_structure(self):
        """Report có đúng cấu trúc: session_id, timestamp, symbols, etc."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "timing_profile.json")
            profiler = TimingProfiler(output_path=output_path)

            from engine.diagnostics.models import BacktestTimingReport

            results = [
                BacktestTimingReport(
                    symbol="VNM",
                    sessions_count=500,
                    total_time_ms=1500.0,
                    strategy_signal_generation_ms=800.0,
                    trade_execution_ms=500.0,
                    metrics_computation_ms=200.0,
                    processed_days=500,
                    total_days=500,
                )
            ]

            report = profiler.generate_backtest_report(results, mode="manual")

            assert "session_id" in report
            assert "timestamp" in report
            assert "symbols" in report
            assert "total_time_ms" in report
            assert "total_symbols" in report
            assert "warnings" in report
            assert report["total_symbols"] == 1
            assert report["total_time_ms"] == 1500.0

    def test_report_per_symbol_breakdown(self):
        """Mỗi symbol có đầy đủ breakdown phases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "timing_profile.json")
            profiler = TimingProfiler(output_path=output_path)

            from engine.diagnostics.models import BacktestTimingReport

            results = [
                BacktestTimingReport(
                    symbol="VNM",
                    sessions_count=300,
                    total_time_ms=1000.0,
                    strategy_signal_generation_ms=500.0,
                    trade_execution_ms=300.0,
                    metrics_computation_ms=200.0,
                    processed_days=300,
                    total_days=300,
                )
            ]

            report = profiler.generate_backtest_report(results, mode="manual")
            sym = report["symbols"][0]

            assert sym["symbol"] == "VNM"
            assert sym["sessions_count"] == 300
            assert sym["total_time_ms"] == 1000.0
            assert sym["strategy_signal_generation_ms"] == 500.0
            assert sym["trade_execution_ms"] == 300.0
            assert sym["metrics_computation_ms"] == 200.0
            assert sym["processed_days"] == 300
            assert sym["total_days"] == 300

    def test_report_applies_warnings(self):
        """Report áp dụng đúng warnings cho symbols có vấn đề."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "timing_profile.json")
            profiler = TimingProfiler(output_path=output_path)

            from engine.diagnostics.models import BacktestTimingReport

            results = [
                BacktestTimingReport(
                    symbol="VNM",
                    sessions_count=600,
                    total_time_ms=50.0,  # 0.05s < 0.1s → BACKTEST_TOO_FAST
                    strategy_signal_generation_ms=30.0,
                    trade_execution_ms=15.0,
                    metrics_computation_ms=5.0,
                    processed_days=500,
                    total_days=600,  # incomplete
                ),
            ]

            report = profiler.generate_backtest_report(results, mode="auto")

            # Kiểm tra symbol warnings
            sym = report["symbols"][0]
            assert "BENCHMARK_BELOW_EXPECTED" in sym["warnings"]
            assert "BACKTEST_TOO_FAST" in sym["warnings"]
            assert "INCOMPLETE_ITERATION" in sym["warnings"]

            # Kiểm tra report-level warnings
            assert "BENCHMARK_BELOW_EXPECTED" in report["warnings"]
            assert "BACKTEST_TOO_FAST" in report["warnings"]
            assert "INCOMPLETE_ITERATION" in report["warnings"]

    def test_report_multiple_symbols(self):
        """Report tổng hợp đúng nhiều symbols."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "timing_profile.json")
            profiler = TimingProfiler(output_path=output_path)

            from engine.diagnostics.models import BacktestTimingReport

            results = [
                BacktestTimingReport(
                    symbol="VNM",
                    sessions_count=300,
                    total_time_ms=1000.0,
                    strategy_signal_generation_ms=500.0,
                    trade_execution_ms=300.0,
                    metrics_computation_ms=200.0,
                    processed_days=300,
                    total_days=300,
                ),
                BacktestTimingReport(
                    symbol="FPT",
                    sessions_count=400,
                    total_time_ms=2000.0,
                    strategy_signal_generation_ms=1000.0,
                    trade_execution_ms=600.0,
                    metrics_computation_ms=400.0,
                    processed_days=400,
                    total_days=400,
                ),
            ]

            report = profiler.generate_backtest_report(results, mode="manual")

            assert report["total_symbols"] == 2
            assert report["total_time_ms"] == 3000.0
            assert len(report["symbols"]) == 2

    def test_report_writes_json_file(self):
        """Report được ghi ra file JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "timing_profile.json")
            profiler = TimingProfiler(output_path=output_path)

            from engine.diagnostics.models import BacktestTimingReport

            results = [
                BacktestTimingReport(
                    symbol="VNM",
                    sessions_count=100,
                    total_time_ms=500.0,
                    strategy_signal_generation_ms=250.0,
                    trade_execution_ms=150.0,
                    metrics_computation_ms=100.0,
                    processed_days=100,
                    total_days=100,
                )
            ]

            profiler.generate_backtest_report(results)

            backtest_path = os.path.join(tmpdir, "backtest_timing.json")
            assert os.path.exists(backtest_path)

            with open(backtest_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "session_id" in data
            assert "symbols" in data
            assert len(data["symbols"]) == 1

    def test_report_empty_results(self):
        """Report với 0 symbols vẫn trả về đúng cấu trúc."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "timing_profile.json")
            profiler = TimingProfiler(output_path=output_path)

            report = profiler.generate_backtest_report([], mode="manual")

            assert report["total_symbols"] == 0
            assert report["total_time_ms"] == 0.0
            assert report["symbols"] == []
            assert report["warnings"] == []
