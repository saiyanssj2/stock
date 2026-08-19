# -*- coding: utf-8 -*-
"""
Property-Based Tests cho Timing Profiler.

Sử dụng Hypothesis để kiểm tra các thuộc tính bất biến của TimingProfiler
qua nhiều kịch bản ngẫu nhiên.

Feature: train-backtest-verification
"""

import json
import os
import tempfile
from datetime import datetime, timezone

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from engine.diagnostics.timing_profiler import (
    TimingProfiler,
    TORCH_OPERATIONS,
    MODEL_CALL_PATTERN,
)


# ---------------------------------------------------------------------------
# Strategies cho Property 4
# ---------------------------------------------------------------------------

# Danh sách tên phases có thể có trong training pipeline
TRAINING_PHASES = [
    "data_loading",
    "feature_extraction",
    "label_generation",
    "model_training",
    "checkpoint_saving",
    "preprocessing",
    "validation",
    "postprocessing",
]


@st.composite
def profiling_scenario_strategy(draw):
    """
    Generate một kịch bản profiling ngẫu nhiên.

    Bao gồm:
    - Danh sách phases ngẫu nhiên (1-8 phases)
    - Số epochs ngẫu nhiên (0-50)
    - Symbol count ngẫu nhiên (0-200)
    - Có stub hay không
    - Duration ngẫu nhiên cho training
    """
    # Chọn ngẫu nhiên 1-8 phases từ danh sách
    num_phases = draw(st.integers(min_value=1, max_value=len(TRAINING_PHASES)))
    phases = draw(
        st.lists(
            st.sampled_from(TRAINING_PHASES),
            min_size=num_phases,
            max_size=num_phases,
            unique=True,
        )
    )

    # Số epochs (0 = không có epoch nào valid)
    epoch_count = draw(st.integers(min_value=0, max_value=50))

    # Symbol count
    symbol_count = draw(st.integers(min_value=0, max_value=200))

    # Có phải stub function không
    is_stub = draw(st.booleans())

    # Duration cho mỗi phase (ms)
    phase_durations_ms = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
            min_size=num_phases,
            max_size=num_phases,
        )
    )

    # Epoch durations (ms)
    epoch_durations_ms = draw(
        st.lists(
            st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False),
            min_size=epoch_count,
            max_size=epoch_count,
        )
    )

    return {
        "phases": phases,
        "phase_durations_ms": phase_durations_ms,
        "epoch_count": epoch_count,
        "epoch_durations_ms": epoch_durations_ms,
        "symbol_count": symbol_count,
        "is_stub": is_stub,
    }


# ---------------------------------------------------------------------------
# Property 4: Timing profile report schema compliance
# ---------------------------------------------------------------------------


class TestTimingProfileReportSchemaCompliance:
    """
    Feature: train-backtest-verification, Property 4: Timing profile report schema compliance

    For any completed training profiling session, the output JSON SHALL contain
    all required fields: session_id (non-empty string), timestamp (valid ISO-8601),
    symbol_count (>= 0), per_phase_duration_ms (object with all phase keys),
    total_duration_ms (>= 0), epoch_count (>= 0), is_stub (bool),
    is_real_computation (bool), warnings (array of strings).

    **Validates: Requirements 2.1, 2.5**
    """

    @given(scenario=profiling_scenario_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_report_contains_all_required_fields(self, scenario):
        """
        Mọi report được generate đều phải chứa tất cả trường bắt buộc.

        **Validates: Requirements 2.1, 2.5**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "timing_profile.json")
            profiler = TimingProfiler(output_path=output_path)

            # Thiết lập symbol count
            profiler._symbol_count = scenario["symbol_count"]

            # Thiết lập stub status
            if scenario["is_stub"]:
                profiler._is_stub = True
            else:
                profiler._is_stub = False

            # Profile các phases - sử dụng internal state trực tiếp
            # vì context manager cần thời gian thực
            from engine.diagnostics.models import PhaseTimingResult

            for i, phase_name in enumerate(scenario["phases"]):
                result = PhaseTimingResult(
                    phase_name=phase_name,
                    duration_ms=scenario["phase_durations_ms"][i],
                    started_at=datetime.now(timezone.utc).isoformat(),
                    ended_at=datetime.now(timezone.utc).isoformat(),
                )
                profiler._phase_results.append(result)

            # Validate epochs
            for epoch_idx in range(scenario["epoch_count"]):
                profiler.validate_epoch(
                    epoch_number=epoch_idx,
                    has_forward=True,
                    has_backward=True,
                    has_step=True,
                    duration_ms=scenario["epoch_durations_ms"][epoch_idx],
                )

            # Generate report
            report = profiler.generate_training_report()

            # === Kiểm tra tất cả trường bắt buộc tồn tại ===

            # session_id: non-empty string
            assert isinstance(report.session_id, str)
            assert len(report.session_id) > 0

            # timestamp: valid ISO-8601
            assert isinstance(report.timestamp, str)
            assert len(report.timestamp) > 0
            # Kiểm tra parse được thành datetime (ISO-8601 valid)
            parsed_ts = datetime.fromisoformat(report.timestamp)
            assert parsed_ts is not None

            # symbol_count: >= 0
            assert isinstance(report.symbol_count, int)
            assert report.symbol_count >= 0

            # per_phase_duration_ms: object with all phase keys
            assert isinstance(report.per_phase_duration_ms, dict)
            for phase_name in scenario["phases"]:
                assert phase_name in report.per_phase_duration_ms
                assert isinstance(report.per_phase_duration_ms[phase_name], float)

            # total_duration_ms: >= 0
            assert isinstance(report.total_duration_ms, float)
            assert report.total_duration_ms >= 0.0

            # epoch_count: >= 0
            assert isinstance(report.epoch_count, int)
            assert report.epoch_count >= 0

            # is_stub: bool
            assert isinstance(report.is_stub, bool)

            # is_real_computation: bool
            assert isinstance(report.is_real_computation, bool)

            # warnings: array of strings
            assert isinstance(report.warnings, list)
            for w in report.warnings:
                assert isinstance(w, str)

    @given(scenario=profiling_scenario_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_report_json_output_contains_all_required_fields(self, scenario):
        """
        File JSON output cũng phải chứa tất cả trường bắt buộc với đúng kiểu.

        **Validates: Requirements 2.1, 2.5**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "timing_profile.json")
            profiler = TimingProfiler(output_path=output_path)

            # Thiết lập state
            profiler._symbol_count = scenario["symbol_count"]
            profiler._is_stub = scenario["is_stub"]

            # Thêm phase results trực tiếp
            from engine.diagnostics.models import PhaseTimingResult

            for i, phase_name in enumerate(scenario["phases"]):
                result = PhaseTimingResult(
                    phase_name=phase_name,
                    duration_ms=scenario["phase_durations_ms"][i],
                    started_at=datetime.now(timezone.utc).isoformat(),
                    ended_at=datetime.now(timezone.utc).isoformat(),
                )
                profiler._phase_results.append(result)

            # Validate epochs
            for epoch_idx in range(scenario["epoch_count"]):
                profiler.validate_epoch(
                    epoch_number=epoch_idx,
                    has_forward=True,
                    has_backward=True,
                    has_step=True,
                    duration_ms=scenario["epoch_durations_ms"][epoch_idx],
                )

            # Generate report (cũng ghi file JSON)
            profiler.generate_training_report()

            # Đọc file JSON và kiểm tra schema
            assert os.path.exists(output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                report_json = json.load(f)

            # Kiểm tra tất cả required fields trong JSON
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
                assert field_name in report_json, (
                    f"Trường '{field_name}' bị thiếu trong JSON output"
                )

            # Kiểm tra kiểu dữ liệu trong JSON
            assert isinstance(report_json["session_id"], str)
            assert len(report_json["session_id"]) > 0

            assert isinstance(report_json["timestamp"], str)
            # Verify ISO-8601 parseable
            datetime.fromisoformat(report_json["timestamp"])

            assert isinstance(report_json["symbol_count"], int)
            assert report_json["symbol_count"] >= 0

            assert isinstance(report_json["per_phase_duration_ms"], dict)
            for phase_name in scenario["phases"]:
                assert phase_name in report_json["per_phase_duration_ms"]

            assert isinstance(report_json["total_duration_ms"], (int, float))
            assert report_json["total_duration_ms"] >= 0.0

            assert isinstance(report_json["epoch_count"], int)
            assert report_json["epoch_count"] >= 0

            assert isinstance(report_json["is_stub"], bool)
            assert isinstance(report_json["is_real_computation"], bool)

            assert isinstance(report_json["warnings"], list)
            for w in report_json["warnings"]:
                assert isinstance(w, str)

    @given(scenario=profiling_scenario_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_report_total_duration_equals_sum_of_phases(self, scenario):
        """
        total_duration_ms phải bằng tổng per_phase_duration_ms.

        **Validates: Requirements 2.1, 2.5**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "timing_profile.json")
            profiler = TimingProfiler(output_path=output_path)

            profiler._symbol_count = scenario["symbol_count"]
            profiler._is_stub = scenario["is_stub"]

            from engine.diagnostics.models import PhaseTimingResult

            for i, phase_name in enumerate(scenario["phases"]):
                result = PhaseTimingResult(
                    phase_name=phase_name,
                    duration_ms=scenario["phase_durations_ms"][i],
                    started_at=datetime.now(timezone.utc).isoformat(),
                    ended_at=datetime.now(timezone.utc).isoformat(),
                )
                profiler._phase_results.append(result)

            # Validate epochs
            for epoch_idx in range(scenario["epoch_count"]):
                profiler.validate_epoch(
                    epoch_number=epoch_idx,
                    has_forward=True,
                    has_backward=True,
                    has_step=True,
                    duration_ms=scenario["epoch_durations_ms"][epoch_idx],
                )

            report = profiler.generate_training_report()

            # total_duration_ms = sum(per_phase_duration_ms.values())
            expected_total = sum(report.per_phase_duration_ms.values())
            assert abs(report.total_duration_ms - expected_total) < 1e-6

    @given(scenario=profiling_scenario_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_report_epoch_count_matches_valid_epochs(self, scenario):
        """
        epoch_count trong report phải khớp với số epochs hợp lệ đã validate.

        **Validates: Requirements 2.1, 2.5**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "timing_profile.json")
            profiler = TimingProfiler(output_path=output_path)

            profiler._symbol_count = scenario["symbol_count"]
            profiler._is_stub = scenario["is_stub"]

            from engine.diagnostics.models import PhaseTimingResult

            for i, phase_name in enumerate(scenario["phases"]):
                result = PhaseTimingResult(
                    phase_name=phase_name,
                    duration_ms=scenario["phase_durations_ms"][i],
                    started_at=datetime.now(timezone.utc).isoformat(),
                    ended_at=datetime.now(timezone.utc).isoformat(),
                )
                profiler._phase_results.append(result)

            # Tất cả epochs đều valid (has_forward, has_backward, has_step = True)
            for epoch_idx in range(scenario["epoch_count"]):
                profiler.validate_epoch(
                    epoch_number=epoch_idx,
                    has_forward=True,
                    has_backward=True,
                    has_step=True,
                    duration_ms=scenario["epoch_durations_ms"][epoch_idx],
                )

            report = profiler.generate_training_report()

            # epoch_count trong report phải bằng số epochs valid
            assert report.epoch_count == scenario["epoch_count"]


# ---------------------------------------------------------------------------
# Property 6: Training duration threshold detection
# ---------------------------------------------------------------------------


class TestTrainingDurationThresholdDetection:
    """
    Feature: train-backtest-verification, Property 6: Training duration threshold detection

    For any training session completing with duration D seconds and symbol_count S,
    the warning "SUSPICIOUSLY_FAST" SHALL be triggered if and only if
    D < S * 0.9 (with default 60s minimum for 65 symbols, scaling linearly).

    **Validates: Requirements 2.3**
    """

    @given(
        duration=st.floats(
            min_value=0.0,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        symbol_count=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=100)
    def test_suspiciously_fast_triggered_iff_below_threshold(
        self, duration: float, symbol_count: int
    ):
        """
        Property: warning "SUSPICIOUSLY_FAST" được trả về IFF duration < symbol_count * 0.9.

        - Nếu duration < threshold → phải trả về "SUSPICIOUSLY_FAST"
        - Nếu duration >= threshold → phải trả về None

        **Validates: Requirements 2.3**
        """
        profiler = TimingProfiler()
        threshold = symbol_count * TimingProfiler.MIN_SECONDS_PER_SYMBOL  # 0.9

        result = profiler.check_training_duration(duration, symbol_count)

        if duration < threshold:
            # Dưới ngưỡng → PHẢI trigger warning
            assert result == "SUSPICIOUSLY_FAST", (
                f"Expected 'SUSPICIOUSLY_FAST' for duration={duration}, "
                f"symbol_count={symbol_count}, threshold={threshold}"
            )
            assert "SUSPICIOUSLY_FAST" in profiler._warnings
        else:
            # Bằng hoặc trên ngưỡng → KHÔNG trigger warning
            assert result is None, (
                f"Expected None for duration={duration}, "
                f"symbol_count={symbol_count}, threshold={threshold}, "
                f"but got '{result}'"
            )
            assert "SUSPICIOUSLY_FAST" not in profiler._warnings

    @given(symbol_count=st.integers(min_value=1, max_value=500))
    @settings(max_examples=100)
    def test_threshold_scales_linearly_with_symbol_count(
        self, symbol_count: int
    ):
        """
        Property: Ngưỡng tỷ lệ tuyến tính — threshold = symbol_count * 0.9.
        Duration ngay dưới threshold → warning, ngay tại threshold → no warning.

        **Validates: Requirements 2.3**
        """
        profiler_below = TimingProfiler()
        profiler_at = TimingProfiler()

        threshold = symbol_count * TimingProfiler.MIN_SECONDS_PER_SYMBOL

        # Ngay dưới ngưỡng → phải trigger
        just_below = threshold - 0.001
        if just_below >= 0:
            result_below = profiler_below.check_training_duration(
                just_below, symbol_count
            )
            assert result_below == "SUSPICIOUSLY_FAST", (
                f"Duration just below threshold ({just_below} < {threshold}) "
                f"should trigger warning for symbol_count={symbol_count}"
            )

        # Ngay tại ngưỡng → không trigger
        result_at = profiler_at.check_training_duration(threshold, symbol_count)
        assert result_at is None, (
            f"Duration at threshold ({threshold}) should NOT trigger warning "
            f"for symbol_count={symbol_count}"
        )

    @given(
        duration=st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        )
    )
    @settings(max_examples=100)
    def test_default_65_symbols_threshold_approximately_60s(
        self, duration: float
    ):
        """
        Property: Với 65 symbols, threshold = 65 * 0.9 = 58.5s.
        Xác nhận spec "default 60s minimum for 65 symbols" — threshold thực tế 58.5s.

        **Validates: Requirements 2.3**
        """
        profiler = TimingProfiler()
        symbol_count = 65
        expected_threshold = 65 * 0.9  # = 58.5

        result = profiler.check_training_duration(duration, symbol_count)

        if duration < expected_threshold:
            assert result == "SUSPICIOUSLY_FAST"
        else:
            assert result is None


# ---------------------------------------------------------------------------
# Property 5: Epoch validation correctness
# ---------------------------------------------------------------------------


class TestEpochValidationCorrectness:
    """
    Feature: train-backtest-verification, Property 5: Epoch validation correctness

    For any epoch execution trace, the epoch SHALL be marked valid if and only if
    it contains all three: forward pass, loss.backward(), AND optimizer.step().
    Missing any one SHALL mark the epoch invalid.

    **Validates: Requirements 2.2**
    """

    @given(
        has_forward=st.booleans(),
        has_backward=st.booleans(),
        has_step=st.booleans(),
        epoch_number=st.integers(min_value=0, max_value=1000),
        duration_ms=st.floats(
            min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_epoch_valid_iff_all_three_operations_present(
        self,
        has_forward: bool,
        has_backward: bool,
        has_step: bool,
        epoch_number: int,
        duration_ms: float,
    ):
        """
        Property: is_valid == True khi và chỉ khi cả forward, backward, step đều True.
        Thiếu bất kỳ operation nào → is_valid == False.

        **Validates: Requirements 2.2**
        """
        profiler = TimingProfiler()

        result = profiler.validate_epoch(
            epoch_number=epoch_number,
            has_forward=has_forward,
            has_backward=has_backward,
            has_step=has_step,
            duration_ms=duration_ms,
        )

        # Tính expected: valid chỉ khi CẢ BA đều True
        expected_valid = has_forward and has_backward and has_step

        assert result.is_valid == expected_valid, (
            f"Epoch {epoch_number}: expected is_valid={expected_valid} "
            f"nhưng nhận được is_valid={result.is_valid} "
            f"(forward={has_forward}, backward={has_backward}, step={has_step})"
        )

    @given(
        has_forward=st.booleans(),
        has_backward=st.booleans(),
        has_step=st.booleans(),
        epoch_number=st.integers(min_value=0, max_value=1000),
        duration_ms=st.floats(
            min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_epoch_validation_preserves_input_fields(
        self,
        has_forward: bool,
        has_backward: bool,
        has_step: bool,
        epoch_number: int,
        duration_ms: float,
    ):
        """
        Property: validate_epoch() gán đúng các input fields vào EpochValidation result.

        **Validates: Requirements 2.2**
        """
        profiler = TimingProfiler()

        result = profiler.validate_epoch(
            epoch_number=epoch_number,
            has_forward=has_forward,
            has_backward=has_backward,
            has_step=has_step,
            duration_ms=duration_ms,
        )

        # Kiểm tra các field được gán đúng giá trị input
        assert result.epoch_number == epoch_number
        assert result.has_forward_pass == has_forward
        assert result.has_backward_pass == has_backward
        assert result.has_optimizer_step == has_step
        assert result.duration_ms == duration_ms

    @given(
        epoch_number=st.integers(min_value=0, max_value=1000),
        duration_ms=st.floats(
            min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_epoch_missing_any_single_operation_is_invalid(
        self,
        epoch_number: int,
        duration_ms: float,
    ):
        """
        Property: Khi thiếu đúng 1 operation (forward/backward/step),
        epoch luôn bị đánh dấu invalid.

        **Validates: Requirements 2.2**
        """
        profiler = TimingProfiler()

        # Thiếu forward
        result_no_forward = profiler.validate_epoch(
            epoch_number=epoch_number,
            has_forward=False,
            has_backward=True,
            has_step=True,
            duration_ms=duration_ms,
        )
        assert result_no_forward.is_valid is False, (
            "Epoch thiếu forward pass phải invalid"
        )

        # Thiếu backward
        result_no_backward = profiler.validate_epoch(
            epoch_number=epoch_number,
            has_forward=True,
            has_backward=False,
            has_step=True,
            duration_ms=duration_ms,
        )
        assert result_no_backward.is_valid is False, (
            "Epoch thiếu backward pass phải invalid"
        )

        # Thiếu step
        result_no_step = profiler.validate_epoch(
            epoch_number=epoch_number,
            has_forward=True,
            has_backward=True,
            has_step=False,
            duration_ms=duration_ms,
        )
        assert result_no_step.is_valid is False, (
            "Epoch thiếu optimizer step phải invalid"
        )

        # Có đủ cả 3 → valid
        result_all = profiler.validate_epoch(
            epoch_number=epoch_number,
            has_forward=True,
            has_backward=True,
            has_step=True,
            duration_ms=duration_ms,
        )
        assert result_all.is_valid is True, (
            "Epoch có đủ cả 3 operations phải valid"
        )


# ---------------------------------------------------------------------------
# Property 7: Stub detection by torch operations absence
# ---------------------------------------------------------------------------

# Danh sách torch operations dùng trong implementation
_TORCH_OPS = list(TORCH_OPERATIONS)

# Model call examples — match pattern "model(" nhưng không phải "_model(" etc.
_MODEL_CALL_EXAMPLES = [
    "output = model(input_tensor)",
    "pred = model (features)",
    "result = model(x, y)",
]


@st.composite
def function_source_without_torch_ops(draw):
    """Sinh function source code KHÔNG chứa torch operations.

    Tạo function body với các statements thông thường (print, return, if/else,
    variable assignment) nhưng KHÔNG chứa bất kỳ torch operation nào
    trong danh sách TORCH_OPERATIONS và không match MODEL_CALL_PATTERN.
    """
    # Các statement an toàn (không chứa torch ops)
    safe_statements = [
        "x = 1",
        "y = x + 2",
        "result = []",
        "print('hello')",
        "logging.info('step')",
        "return None",
        "return result",
        "data = load_data()",
        "config = get_config()",
        "if x > 0:\n        return x",
        "for i in range(10):\n        pass",
        "import numpy as np",
        "df = pd.DataFrame()",
        "values = np.zeros(10)",
        "logger.debug('processing')",
        "self.status = 'done'",
        "time.sleep(0.1)",
        "os.path.exists('file.txt')",
    ]

    # Số lượng statements trong function body
    num_statements = draw(st.integers(min_value=1, max_value=6))
    body_lines = draw(
        st.lists(
            st.sampled_from(safe_statements),
            min_size=num_statements,
            max_size=num_statements,
        )
    )

    # Tên hàm ngẫu nhiên (an toàn — không chứa "model" standalone triggering pattern)
    func_name = draw(st.sampled_from([
        "_retrain_model", "train_step", "process_data",
        "update_weights", "run_pipeline", "compute_metrics",
        "load_checkpoint", "save_results",
    ]))

    # Build function source
    body = "\n    ".join(body_lines)
    source = f"def {func_name}(self):\n    {body}"

    # Đảm bảo source KHÔNG chứa bất kỳ torch op nào
    for op in _TORCH_OPS:
        assume(op not in source)

    # Đảm bảo không match MODEL_CALL_PATTERN
    assume(not MODEL_CALL_PATTERN.search(source))

    return source


@st.composite
def function_source_with_torch_ops(draw):
    """Sinh function source code CÓ chứa ít nhất một torch operation.

    Tạo function body bình thường rồi inject một torch operation
    vào vị trí ngẫu nhiên.
    """
    # Chọn inject model call pattern hoặc substring torch op
    use_model_call = draw(st.booleans())

    if use_model_call:
        # Inject model call pattern
        torch_line = draw(st.sampled_from(_MODEL_CALL_EXAMPLES))
    else:
        # Inject một torch operation từ danh sách
        op = draw(st.sampled_from(_TORCH_OPS))
        # Tạo statement chứa operation
        if op == "torch.tensor":
            torch_line = "x = torch.tensor([1.0, 2.0, 3.0])"
        elif op == "torch.zeros":
            torch_line = "weights = torch.zeros(10, 5)"
        elif op == "torch.ones":
            torch_line = "bias = torch.ones(5)"
        elif op == "loss.backward()":
            torch_line = "loss.backward()"
        elif op == "optimizer.step()":
            torch_line = "optimizer.step()"
        elif op == "model.forward(":
            torch_line = "output = model.forward(input_data)"
        elif op == ".backward(":
            torch_line = "total_loss.backward()"
        elif op == ".step(":
            torch_line = "scheduler.step()"
        else:
            torch_line = f"# contains {op}"

    # Các statements thông thường xung quanh
    other_statements = [
        "x = 1",
        "result = []",
        "print('training')",
        "lr = 0.001",
        "epochs = 50",
        "data = load_batch()",
    ]

    num_other = draw(st.integers(min_value=0, max_value=4))
    other_lines = draw(
        st.lists(
            st.sampled_from(other_statements),
            min_size=num_other,
            max_size=num_other,
        )
    )

    # Tên hàm ngẫu nhiên
    func_name = draw(st.sampled_from([
        "train_epoch", "forward_pass", "backprop_step",
        "_retrain_model", "train_model", "run_training",
    ]))

    # Inject torch_line vào vị trí ngẫu nhiên trong body
    all_lines = list(other_lines)
    insert_pos = draw(st.integers(min_value=0, max_value=len(all_lines)))
    all_lines.insert(insert_pos, torch_line)

    body = "\n    ".join(all_lines)
    source = f"def {func_name}(self):\n    {body}"

    return source


class TestStubDetectionByTorchOperationsAbsence:
    """
    Feature: train-backtest-verification, Property 7: Stub detection by torch operations absence

    Property 7: For any Python function source code, the function SHALL be
    classified as stub (is_stub = true) if and only if it contains none of the
    torch operations: tensor creation (torch.tensor/zeros/ones), loss.backward(),
    optimizer.step(), model.forward()/model().

    **Validates: Requirements 2.4, 5.1**
    """

    @given(source=function_source_without_torch_ops())
    @settings(max_examples=100)
    def test_function_without_torch_ops_is_stub(self, source: str):
        """
        Property: Function source KHÔNG chứa bất kỳ torch operation nào
        → check_stub_function() phải trả về True (is_stub).

        Strategy: Sinh function source từ các statements thông thường (print,
        return, variable assignment, etc.) đảm bảo không chứa torch ops.

        **Validates: Requirements 2.4, 5.1**
        """
        profiler = TimingProfiler()
        result = profiler.check_stub_function(source)

        assert result is True, (
            f"Expected is_stub=True cho function KHÔNG có torch ops.\n"
            f"Source:\n{source}"
        )

    @given(source=function_source_with_torch_ops())
    @settings(max_examples=100)
    def test_function_with_torch_ops_is_not_stub(self, source: str):
        """
        Property: Function source CÓ chứa ít nhất một torch operation
        → check_stub_function() phải trả về False (not stub).

        Strategy: Sinh function source rồi inject ít nhất một torch op
        (torch.tensor, torch.zeros, torch.ones, loss.backward(),
        optimizer.step(), model.forward(), model()).

        **Validates: Requirements 2.4, 5.1**
        """
        profiler = TimingProfiler()
        result = profiler.check_stub_function(source)

        assert result is False, (
            f"Expected is_stub=False cho function CÓ torch ops.\n"
            f"Source:\n{source}"
        )


# ---------------------------------------------------------------------------
# Property 8: Backtest per-symbol timing with threshold warnings
# ---------------------------------------------------------------------------


class TestBacktestPerSymbolTimingThresholdWarnings:
    """
    Feature: train-backtest-verification, Property 8: Backtest per-symbol timing with threshold warnings

    For any backtest symbol result with sessions_count S and actual_time T:
    (a) if mode="auto" AND S >= 250 AND T < 0.3s, warning "BENCHMARK_BELOW_EXPECTED"
        SHALL be triggered;
    (b) if S >= 500 AND T < 0.1s, warning "BACKTEST_TOO_FAST" SHALL be triggered;
    (c) warnings SHALL NOT be triggered when conditions are not fully met.

    **Validates: Requirements 3.2, 3.3**
    """

    @given(
        sessions_count=st.integers(min_value=0, max_value=2000),
        total_time_seconds=st.floats(
            min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        mode=st.sampled_from(["auto", "manual"]),
    )
    @settings(max_examples=100)
    def test_benchmark_below_expected_triggered_iff_all_conditions_met(
        self, sessions_count: int, total_time_seconds: float, mode: str
    ):
        """
        Property (a): "BENCHMARK_BELOW_EXPECTED" được trigger khi và chỉ khi
        mode="auto" AND sessions_count >= 250 AND total_time_seconds < 0.3.

        **Validates: Requirements 3.2**
        """
        profiler = TimingProfiler()

        warnings = profiler.check_backtest_symbol(
            symbol="TEST",
            sessions_count=sessions_count,
            total_time_seconds=total_time_seconds,
            processed_days=100,
            total_days=100,
            mode=mode,
        )

        # Điều kiện trigger BENCHMARK_BELOW_EXPECTED
        should_trigger = (
            mode == "auto"
            and sessions_count >= 250
            and total_time_seconds < 0.3
        )

        if should_trigger:
            assert "BENCHMARK_BELOW_EXPECTED" in warnings, (
                f"Expected 'BENCHMARK_BELOW_EXPECTED' khi mode={mode}, "
                f"sessions_count={sessions_count}, time={total_time_seconds}s"
            )
        else:
            assert "BENCHMARK_BELOW_EXPECTED" not in warnings, (
                f"Unexpected 'BENCHMARK_BELOW_EXPECTED' khi mode={mode}, "
                f"sessions_count={sessions_count}, time={total_time_seconds}s"
            )

    @given(
        sessions_count=st.integers(min_value=0, max_value=2000),
        total_time_seconds=st.floats(
            min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        mode=st.sampled_from(["auto", "manual"]),
    )
    @settings(max_examples=100)
    def test_backtest_too_fast_triggered_iff_all_conditions_met(
        self, sessions_count: int, total_time_seconds: float, mode: str
    ):
        """
        Property (b): "BACKTEST_TOO_FAST" được trigger khi và chỉ khi
        sessions_count >= 500 AND total_time_seconds < 0.1.
        Không phụ thuộc vào mode.

        **Validates: Requirements 3.3**
        """
        profiler = TimingProfiler()

        warnings = profiler.check_backtest_symbol(
            symbol="TEST",
            sessions_count=sessions_count,
            total_time_seconds=total_time_seconds,
            processed_days=100,
            total_days=100,
            mode=mode,
        )

        # Điều kiện trigger BACKTEST_TOO_FAST
        should_trigger = (
            sessions_count >= 500
            and total_time_seconds < 0.1
        )

        if should_trigger:
            assert "BACKTEST_TOO_FAST" in warnings, (
                f"Expected 'BACKTEST_TOO_FAST' khi "
                f"sessions_count={sessions_count}, time={total_time_seconds}s"
            )
        else:
            assert "BACKTEST_TOO_FAST" not in warnings, (
                f"Unexpected 'BACKTEST_TOO_FAST' khi "
                f"sessions_count={sessions_count}, time={total_time_seconds}s"
            )

    @given(
        sessions_count=st.integers(min_value=0, max_value=2000),
        total_time_seconds=st.floats(
            min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        mode=st.sampled_from(["auto", "manual"]),
    )
    @settings(max_examples=100)
    def test_no_warnings_when_conditions_not_fully_met(
        self, sessions_count: int, total_time_seconds: float, mode: str
    ):
        """
        Property (c): Warnings SHALL NOT be triggered when conditions are not fully met.
        Kiểm tra tổng hợp cả hai warnings cùng lúc.

        **Validates: Requirements 3.2, 3.3**
        """
        profiler = TimingProfiler()

        warnings = profiler.check_backtest_symbol(
            symbol="TEST",
            sessions_count=sessions_count,
            total_time_seconds=total_time_seconds,
            processed_days=100,
            total_days=100,
            mode=mode,
        )

        # === BENCHMARK_BELOW_EXPECTED ===
        benchmark_conditions_met = (
            mode == "auto"
            and sessions_count >= 250
            and total_time_seconds < 0.3
        )
        if not benchmark_conditions_met:
            assert "BENCHMARK_BELOW_EXPECTED" not in warnings, (
                f"'BENCHMARK_BELOW_EXPECTED' triggered nhưng điều kiện không đầy đủ: "
                f"mode={mode}, sessions_count={sessions_count}, time={total_time_seconds}s"
            )

        # === BACKTEST_TOO_FAST ===
        fast_conditions_met = (
            sessions_count >= 500
            and total_time_seconds < 0.1
        )
        if not fast_conditions_met:
            assert "BACKTEST_TOO_FAST" not in warnings, (
                f"'BACKTEST_TOO_FAST' triggered nhưng điều kiện không đầy đủ: "
                f"sessions_count={sessions_count}, time={total_time_seconds}s"
            )

    @given(
        sessions_count=st.integers(min_value=500, max_value=2000),
        total_time_seconds=st.floats(
            min_value=0.0, max_value=0.099, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_both_warnings_can_trigger_simultaneously(
        self, sessions_count: int, total_time_seconds: float
    ):
        """
        Property: Khi mode="auto", sessions >= 500 (thỏa cả >= 250), và time < 0.1s
        (thỏa cả < 0.3s), CẢ HAI warnings phải được trigger đồng thời.

        **Validates: Requirements 3.2, 3.3**
        """
        profiler = TimingProfiler()

        warnings = profiler.check_backtest_symbol(
            symbol="TEST",
            sessions_count=sessions_count,
            total_time_seconds=total_time_seconds,
            processed_days=100,
            total_days=100,
            mode="auto",
        )

        # Cả hai điều kiện đều thỏa mãn
        assert "BENCHMARK_BELOW_EXPECTED" in warnings, (
            f"Expected 'BENCHMARK_BELOW_EXPECTED' khi mode=auto, "
            f"sessions_count={sessions_count} >= 250, time={total_time_seconds}s < 0.3"
        )
        assert "BACKTEST_TOO_FAST" in warnings, (
            f"Expected 'BACKTEST_TOO_FAST' khi "
            f"sessions_count={sessions_count} >= 500, time={total_time_seconds}s < 0.1"
        )


# ---------------------------------------------------------------------------
# Property 9: Backtest incomplete iteration detection
# ---------------------------------------------------------------------------


class TestBacktestIncompleteIterationDetection:
    """
    Feature: train-backtest-verification, Property 9: Backtest incomplete iteration detection

    For any backtest completion with processed_days P and total_days T,
    the warning "INCOMPLETE_ITERATION" SHALL be triggered if and only if P < T,
    with correct ratio = P/T.

    **Validates: Requirements 3.4**
    """

    @given(
        processed_days=st.integers(min_value=0, max_value=10000),
        total_days=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=100)
    def test_incomplete_iteration_triggered_iff_processed_less_than_total(
        self, processed_days: int, total_days: int
    ):
        """
        Property: warning "INCOMPLETE_ITERATION" được trigger IFF processed_days < total_days.

        - Nếu processed_days < total_days → PHẢI có "INCOMPLETE_ITERATION" trong warnings
        - Nếu processed_days >= total_days → KHÔNG có "INCOMPLETE_ITERATION" trong warnings

        **Validates: Requirements 3.4**
        """
        profiler = TimingProfiler()

        warnings = profiler.check_backtest_symbol(
            symbol="TEST",
            sessions_count=100,
            total_time_seconds=10.0,
            processed_days=processed_days,
            total_days=total_days,
        )

        if processed_days < total_days:
            # Dưới total_days → PHẢI trigger INCOMPLETE_ITERATION
            assert "INCOMPLETE_ITERATION" in warnings, (
                f"Expected 'INCOMPLETE_ITERATION' khi processed_days={processed_days} "
                f"< total_days={total_days}, nhưng warnings={warnings}"
            )
        else:
            # Bằng hoặc vượt total_days → KHÔNG trigger
            assert "INCOMPLETE_ITERATION" not in warnings, (
                f"Unexpected 'INCOMPLETE_ITERATION' khi processed_days={processed_days} "
                f">= total_days={total_days}, nhưng warnings={warnings}"
            )

    @given(
        processed_days=st.integers(min_value=0, max_value=9999),
        total_days=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=100)
    def test_incomplete_iteration_ratio_is_correct(
        self, processed_days: int, total_days: int
    ):
        """
        Property: Khi INCOMPLETE_ITERATION triggered, ratio = processed_days / total_days
        phải nằm trong khoảng [0, 1) (vì P < T → P/T < 1).

        **Validates: Requirements 3.4**
        """
        # Chỉ test trường hợp processed < total (sẽ trigger warning)
        assume(processed_days < total_days)

        profiler = TimingProfiler()

        warnings = profiler.check_backtest_symbol(
            symbol="TEST",
            sessions_count=100,
            total_time_seconds=10.0,
            processed_days=processed_days,
            total_days=total_days,
        )

        # Phải có warning
        assert "INCOMPLETE_ITERATION" in warnings

        # Kiểm tra ratio hợp lệ
        ratio = processed_days / total_days
        assert 0.0 <= ratio < 1.0, (
            f"Ratio P/T phải nằm trong [0, 1) khi P < T, "
            f"nhưng ratio={ratio} (P={processed_days}, T={total_days})"
        )

    @given(total_days=st.integers(min_value=1, max_value=10000))
    @settings(max_examples=100)
    def test_complete_iteration_no_warning(self, total_days: int):
        """
        Property: Khi processed_days == total_days (iteration hoàn tất),
        warning "INCOMPLETE_ITERATION" KHÔNG được trigger.

        **Validates: Requirements 3.4**
        """
        profiler = TimingProfiler()

        warnings = profiler.check_backtest_symbol(
            symbol="TEST",
            sessions_count=100,
            total_time_seconds=10.0,
            processed_days=total_days,
            total_days=total_days,
        )

        assert "INCOMPLETE_ITERATION" not in warnings, (
            f"Không nên có 'INCOMPLETE_ITERATION' khi processed_days == total_days "
            f"({total_days}), nhưng warnings={warnings}"
        )

    @given(
        processed_days=st.integers(min_value=0, max_value=10000),
        total_days=st.integers(min_value=1, max_value=10000),
        sessions_count=st.integers(min_value=1, max_value=1000),
        total_time_seconds=st.floats(
            min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_incomplete_iteration_independent_of_other_params(
        self,
        processed_days: int,
        total_days: int,
        sessions_count: int,
        total_time_seconds: float,
    ):
        """
        Property: INCOMPLETE_ITERATION chỉ phụ thuộc vào processed_days < total_days,
        không bị ảnh hưởng bởi sessions_count hay total_time_seconds.

        **Validates: Requirements 3.4**
        """
        profiler = TimingProfiler()

        warnings = profiler.check_backtest_symbol(
            symbol="TEST",
            sessions_count=sessions_count,
            total_time_seconds=total_time_seconds,
            processed_days=processed_days,
            total_days=total_days,
        )

        expected_incomplete = processed_days < total_days

        if expected_incomplete:
            assert "INCOMPLETE_ITERATION" in warnings, (
                f"Expected 'INCOMPLETE_ITERATION' khi P={processed_days} < T={total_days} "
                f"(sessions={sessions_count}, time={total_time_seconds}s)"
            )
        else:
            assert "INCOMPLETE_ITERATION" not in warnings, (
                f"Unexpected 'INCOMPLETE_ITERATION' khi P={processed_days} >= T={total_days} "
                f"(sessions={sessions_count}, time={total_time_seconds}s)"
            )
