# -*- coding: utf-8 -*-
"""
Property-based tests cho DataPipeline - data validation, error isolation,
rate limiting, và ETA calculation.

# Feature: stock-trading-platform-refactor, Properties 17-20: DataPipeline

**Validates: Requirements 8.2, 8.4, 8.6, 8.8**

Properties:
17. Data validation completeness — valid iff DataFrame has all required columns
18. Error isolation in batch operations — K failures → (N-K) successes, no abort
19. Rate limiter correctness — max 20 requests per sliding 60s window
20. ETA calculation accuracy — ETA = ceil(N / R) minutes
"""

import math
from unittest.mock import patch

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from config.settings import RATE_LIMIT
from engine.data_pipeline import DataPipeline, REQUIRED_COLUMNS
from engine.rate_limiter import TokenBucketRateLimiter


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Tất cả 6 cột bắt buộc
ALL_REQUIRED_COLUMNS = list(REQUIRED_COLUMNS)

# Strategy sinh tập con ngẫu nhiên của required columns
subset_columns_strategy = st.lists(
    st.sampled_from(ALL_REQUIRED_COLUMNS),
    min_size=0,
    max_size=len(ALL_REQUIRED_COLUMNS),
    unique=True,
)

# Strategy sinh thêm cột phụ không bắt buộc
extra_columns_strategy = st.lists(
    st.text(
        alphabet=st.characters(whitelist_categories=("L",)),
        min_size=3,
        max_size=10,
    ),
    min_size=0,
    max_size=5,
    unique=True,
)

# Strategy cho số lượng symbols (dùng cho Property 18, 20)
num_symbols_strategy = st.integers(min_value=1, max_value=50)

# Strategy cho số symbols thất bại (K < N)
# Sẽ dùng @st.composite để đảm bảo K < N


@st.composite
def dataframe_with_columns_strategy(draw, columns):
    """
    Sinh DataFrame với danh sách cột cho trước và ít nhất 1 row.

    Parameters
    ----------
    columns : list[str]
        Danh sách tên cột cần có trong DataFrame.

    Returns
    -------
    pd.DataFrame với các cột đã chỉ định và 1-10 rows dữ liệu ngẫu nhiên.
    """
    num_rows = draw(st.integers(min_value=1, max_value=10))
    data = {}
    for col in columns:
        data[col] = draw(
            st.lists(
                st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
                min_size=num_rows,
                max_size=num_rows,
            )
        )
    return pd.DataFrame(data)


@st.composite
def batch_failure_strategy(draw):
    """
    Sinh danh sách N symbols và chọn K symbols sẽ fail (K < N).

    Returns
    -------
    dict với keys: symbols, failed_indices, N, K
    """
    n = draw(st.integers(min_value=2, max_value=30))
    symbols = [f"SYM{i:03d}" for i in range(n)]
    k = draw(st.integers(min_value=0, max_value=n - 1))
    # Chọn K indices ngẫu nhiên sẽ fail
    failed_indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=n - 1),
            min_size=k,
            max_size=k,
            unique=True,
        )
    )
    return {
        "symbols": symbols,
        "failed_indices": set(failed_indices),
        "N": n,
        "K": len(failed_indices),
    }


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestDataValidationCompleteness:
    """Property 17: Data validation completeness."""

    @given(
        extra_cols=extra_columns_strategy,
        num_rows=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_valid_when_all_required_columns_present(self, extra_cols, num_rows):
        """
        Property: validate_data trả về True khi DataFrame có đầy đủ 6 cột bắt buộc,
        bất kể có thêm cột phụ nào.

        # Feature: stock-trading-platform-refactor, Property 17: Data validation completeness
        **Validates: Requirements 8.2**
        """
        pipeline = DataPipeline()

        # Tạo DataFrame với tất cả required columns + extra columns
        # Lọc extra_cols để không trùng tên required columns
        filtered_extras = [c for c in extra_cols if c.lower() not in ALL_REQUIRED_COLUMNS]
        all_cols = ALL_REQUIRED_COLUMNS + filtered_extras

        data = {col: list(range(1, num_rows + 1)) for col in all_cols}
        df = pd.DataFrame(data)

        result = pipeline.validate_data(df)

        assert result is True, (
            f"DataFrame có đầy đủ required columns {ALL_REQUIRED_COLUMNS} "
            f"nhưng validate_data trả về False. Columns: {list(df.columns)}"
        )

    @given(
        columns_subset=subset_columns_strategy,
        num_rows=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_invalid_when_missing_required_columns(self, columns_subset, num_rows):
        """
        Property: validate_data trả về False khi DataFrame thiếu bất kỳ cột bắt buộc nào.

        # Feature: stock-trading-platform-refactor, Property 17: Data validation completeness
        **Validates: Requirements 8.2**
        """
        pipeline = DataPipeline()

        # Chỉ test khi thiếu ít nhất 1 cột
        if set(columns_subset) == set(ALL_REQUIRED_COLUMNS):
            return  # Bỏ qua case đầy đủ, đã test ở trên

        data = {col: list(range(1, num_rows + 1)) for col in columns_subset}
        df = pd.DataFrame(data)

        result = pipeline.validate_data(df)

        missing = set(ALL_REQUIRED_COLUMNS) - set(columns_subset)
        assert result is False, (
            f"DataFrame thiếu columns {missing} "
            f"nhưng validate_data trả về True. Columns: {list(df.columns)}"
        )

    @given(
        extra_cols=extra_columns_strategy,
        num_rows=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50)
    def test_valid_case_insensitive_columns(self, extra_cols, num_rows):
        """
        Property: validate_data chấp nhận cột viết hoa/thường lẫn lộn
        (lowercase comparison).

        # Feature: stock-trading-platform-refactor, Property 17: Data validation completeness
        **Validates: Requirements 8.2**
        """
        pipeline = DataPipeline()

        # Tạo DataFrame với columns viết hoa (Title case)
        upper_cols = [col.upper() for col in ALL_REQUIRED_COLUMNS]
        filtered_extras = [c for c in extra_cols if c.lower() not in ALL_REQUIRED_COLUMNS]
        all_cols = upper_cols + filtered_extras

        data = {col: list(range(1, num_rows + 1)) for col in all_cols}
        df = pd.DataFrame(data)

        result = pipeline.validate_data(df)

        assert result is True, (
            f"DataFrame có columns viết hoa {upper_cols} nhưng validate_data trả về False. "
            f"validate_data phải case-insensitive."
        )


class TestErrorIsolationBatchOperations:
    """Property 18: Error isolation in batch operations."""

    @given(data=batch_failure_strategy())
    @settings(max_examples=100)
    def test_k_failures_produce_n_minus_k_successes(self, data):
        """
        Property: Khi K symbols fail trong batch N symbols,
        kết quả phải có success_count == N-K và len(failed_symbols) == K.
        Batch KHÔNG bị abort.

        # Feature: stock-trading-platform-refactor, Property 18: Error isolation in batch operations
        **Validates: Requirements 8.4**
        """
        symbols = data["symbols"]
        failed_indices = data["failed_indices"]
        n = data["N"]
        k = data["K"]

        pipeline = DataPipeline()

        # Mock get_tracked_symbols để trả về symbols đã định
        # Mock update_symbol: fail cho indices trong failed_indices, success cho còn lại
        def mock_update_symbol(symbol):
            idx = symbols.index(symbol)
            if idx in failed_indices:
                return False
            return True

        with patch.object(pipeline, "get_tracked_symbols", return_value=symbols):
            with patch.object(pipeline, "update_symbol", side_effect=mock_update_symbol):
                result = pipeline.update_all()

        # Kiểm tra error isolation: batch hoàn thành, không abort
        assert result.total_symbols == n, (
            f"total_symbols phải == {n}, got {result.total_symbols}"
        )
        assert result.success_count == n - k, (
            f"success_count phải == {n - k}, got {result.success_count}"
        )
        assert len(result.failed_symbols) == k, (
            f"len(failed_symbols) phải == {k}, got {len(result.failed_symbols)}"
        )

    @given(data=batch_failure_strategy())
    @settings(max_examples=50)
    def test_batch_never_raises_exception(self, data):
        """
        Property: Batch operation KHÔNG raise exception dù có symbol fail.

        # Feature: stock-trading-platform-refactor, Property 18: Error isolation in batch operations
        **Validates: Requirements 8.4**
        """
        symbols = data["symbols"]
        failed_indices = data["failed_indices"]

        pipeline = DataPipeline()

        # Mock update_symbol: raise Exception cho failed symbols
        def mock_update_symbol(symbol):
            idx = symbols.index(symbol)
            if idx in failed_indices:
                raise RuntimeError(f"Network error for {symbol}")
            return True

        with patch.object(pipeline, "get_tracked_symbols", return_value=symbols):
            with patch.object(pipeline, "update_symbol", side_effect=mock_update_symbol):
                # Không được raise exception
                try:
                    result = pipeline.update_all()
                except Exception as e:
                    raise AssertionError(
                        f"update_all() raise exception '{e}' thay vì isolate error. "
                        f"Batch phải tiếp tục chạy khi symbol fail."
                    )

        # Kết quả vẫn phải có đầy đủ thông tin
        assert result.total_symbols == len(symbols)


class TestRateLimiterCorrectness:
    """Property 19: Rate limiter correctness."""

    @given(
        max_tokens=st.integers(min_value=1, max_value=50),
        extra_requests=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=100)
    def test_max_requests_within_window(self, max_tokens, extra_requests):
        """
        Property: Sau khi acquire đúng max_tokens lần, lần acquire tiếp theo
        trả về False (bị rate limit).

        # Feature: stock-trading-platform-refactor, Property 19: Rate limiter correctness
        **Validates: Requirements 8.6**
        """
        limiter = TokenBucketRateLimiter(max_tokens=max_tokens, window_seconds=60.0)

        # Fill hết tokens
        for i in range(max_tokens):
            result = limiter.acquire()
            assert result is True, (
                f"acquire() lần {i+1}/{max_tokens} phải trả về True, got False"
            )

        # Lần tiếp theo phải bị reject
        for _ in range(extra_requests):
            result = limiter.acquire()
            assert result is False, (
                f"Sau {max_tokens} requests trong window, "
                f"acquire() phải trả về False nhưng got True"
            )

    @given(max_tokens=st.integers(min_value=1, max_value=50))
    @settings(max_examples=50)
    def test_remaining_tokens_decreases(self, max_tokens):
        """
        Property: Mỗi lần acquire thành công, remaining_tokens giảm đi 1.

        # Feature: stock-trading-platform-refactor, Property 19: Rate limiter correctness
        **Validates: Requirements 8.6**
        """
        limiter = TokenBucketRateLimiter(max_tokens=max_tokens, window_seconds=60.0)

        initial_remaining = limiter.get_remaining_tokens()
        assert initial_remaining == max_tokens

        for i in range(max_tokens):
            limiter.acquire()
            remaining = limiter.get_remaining_tokens()
            assert remaining == max_tokens - (i + 1), (
                f"Sau {i+1} acquire(), remaining phải == {max_tokens - (i+1)}, "
                f"got {remaining}"
            )

    @given(max_tokens=st.integers(min_value=1, max_value=30))
    @settings(max_examples=50)
    def test_default_rate_limit_20_per_60s(self, max_tokens):
        """
        Property: TokenBucketRateLimiter với default config cho phép tối đa
        RATE_LIMIT requests trong 60s window.

        # Feature: stock-trading-platform-refactor, Property 19: Rate limiter correctness
        **Validates: Requirements 8.6**
        """
        # Dùng default config từ settings
        limiter = TokenBucketRateLimiter()

        # Acquire RATE_LIMIT lần phải thành công
        successes = 0
        for _ in range(RATE_LIMIT):
            if limiter.acquire():
                successes += 1

        assert successes == RATE_LIMIT, (
            f"Default rate limiter phải cho phép {RATE_LIMIT} requests, "
            f"nhưng chỉ thành công {successes}"
        )

        # Request tiếp theo phải bị reject
        assert limiter.acquire() is False, (
            f"Sau {RATE_LIMIT} requests, acquire() phải trả về False"
        )


class TestETACalculationAccuracy:
    """Property 20: ETA calculation accuracy."""

    @given(num_symbols=st.integers(min_value=1, max_value=1000))
    @settings(max_examples=100)
    def test_eta_equals_ceil_n_over_r(self, num_symbols):
        """
        Property: estimate_update_duration(N) == ceil(N / RATE_LIMIT) phút
        với default RATE_LIMIT=20.

        # Feature: stock-trading-platform-refactor, Property 20: ETA calculation accuracy
        **Validates: Requirements 8.8**
        """
        pipeline = DataPipeline()

        result = pipeline.estimate_update_duration(num_symbols)
        expected = float(math.ceil(num_symbols / RATE_LIMIT))

        assert result == expected, (
            f"estimate_update_duration({num_symbols}) == {result}, "
            f"expected ceil({num_symbols}/{RATE_LIMIT}) == {expected}"
        )

    @given(
        num_symbols=st.integers(min_value=1, max_value=1000),
        rate_limit=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100)
    def test_rate_limiter_eta_with_custom_rate(self, num_symbols, rate_limit):
        """
        Property: TokenBucketRateLimiter.estimate_duration(N) == ceil(N / max_tokens)
        cho bất kỳ max_tokens > 0.

        # Feature: stock-trading-platform-refactor, Property 20: ETA calculation accuracy
        **Validates: Requirements 8.8**
        """
        limiter = TokenBucketRateLimiter(max_tokens=rate_limit)

        result = limiter.estimate_duration(num_symbols)
        expected = float(math.ceil(num_symbols / rate_limit))

        assert result == expected, (
            f"estimate_duration({num_symbols}) với max_tokens={rate_limit} "
            f"== {result}, expected {expected}"
        )

    def test_eta_zero_for_zero_symbols(self):
        """
        Edge case: estimate_update_duration(0) == 0.0.

        # Feature: stock-trading-platform-refactor, Property 20: ETA calculation accuracy
        **Validates: Requirements 8.8**
        """
        pipeline = DataPipeline()
        assert pipeline.estimate_update_duration(0) == 0.0

        limiter = TokenBucketRateLimiter()
        assert limiter.estimate_duration(0) == 0.0

    @given(num_symbols=st.integers(min_value=-100, max_value=-1))
    @settings(max_examples=20)
    def test_eta_zero_for_negative_symbols(self, num_symbols):
        """
        Edge case: estimate_update_duration(negative) == 0.0.

        # Feature: stock-trading-platform-refactor, Property 20: ETA calculation accuracy
        **Validates: Requirements 8.8**
        """
        pipeline = DataPipeline()
        assert pipeline.estimate_update_duration(num_symbols) == 0.0

        limiter = TokenBucketRateLimiter()
        assert limiter.estimate_duration(num_symbols) == 0.0
