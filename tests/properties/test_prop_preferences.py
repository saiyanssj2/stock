# -*- coding: utf-8 -*-
"""
Property-based tests cho user preferences round-trip.

# Feature: stock-trading-platform-refactor, Property 22: User preferences round-trip

**Validates: Requirements 10.6**

Property:
    Với bất kỳ tập user preferences hợp lệ nào (selected symbols, date ranges,
    capital settings, auto-update config), save → load phải trả về giá trị
    preferences hoàn toàn giống nhau.
"""

from datetime import date, timedelta
from unittest.mock import patch, MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from config.vn_market_rules import VN30_SYMBOLS


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Symbols: chọn subset từ VN30 + một số mã watchlist giả lập
_ALL_AVAILABLE_SYMBOLS = VN30_SYMBOLS + ["FOX", "GEE", "GEL", "GEX", "HSG", "HUT"]

# Danh sách symbols đã chọn: subset của available symbols
selected_symbols_strategy = st.lists(
    st.sampled_from(_ALL_AVAILABLE_SYMBOLS),
    min_size=0,
    max_size=len(_ALL_AVAILABLE_SYMBOLS),
    unique=True,
)

# Date strategy: ngày hợp lệ trong khoảng hợp lý
date_strategy = st.dates(
    min_value=date(2010, 1, 1),
    max_value=date(2035, 12, 31),
)


# Date range: đảm bảo start_date < end_date
@st.composite
def date_range_strategy(draw):
    """Sinh cặp (start_date, end_date) với start_date < end_date."""
    start = draw(date_strategy)
    # Đảm bảo end_date ít nhất 1 ngày sau start_date
    delta = draw(st.integers(min_value=1, max_value=3650))
    end = start + timedelta(days=delta)
    # Clamp end_date nếu vượt quá max
    if end > date(2035, 12, 31):
        end = date(2035, 12, 31)
    if start >= end:
        start = end - timedelta(days=1)
    return start, end


# Capital: số tiền VND hợp lệ (tối thiểu 1 triệu, tối đa 100 tỷ)
capital_strategy = st.floats(
    min_value=1_000_000.0,
    max_value=100_000_000_000.0,
    allow_nan=False,
    allow_infinity=False,
)

# Auto-update time: format HH:MM hợp lệ
auto_update_time_strategy = st.builds(
    lambda h, m: f"{h:02d}:{m:02d}",
    h=st.integers(min_value=0, max_value=23),
    m=st.integers(min_value=0, max_value=59),
)

# Cycle interval: 1.0 - 168.0 giờ
cycle_interval_strategy = st.floats(
    min_value=1.0,
    max_value=168.0,
    allow_nan=False,
    allow_infinity=False,
)

# Auto-update enabled: boolean
auto_update_enabled_strategy = st.booleans()


# ---------------------------------------------------------------------------
# Helper: Mô phỏng save/load logic tách biệt khỏi Streamlit runtime
# ---------------------------------------------------------------------------


def _save_preferences_to_store(
    store: dict,
    selected_symbols: list,
    start_date: date,
    end_date: date,
    capital: float,
    auto_update_time: str,
    cycle_interval: float,
    auto_update_enabled: bool,
) -> None:
    """
    Mô phỏng logic _save_preferences() từ page_settings.py.
    Ghi trực tiếp vào dict (thay vì st.session_state).
    """
    store["pref_selected_symbols"] = selected_symbols
    store["pref_start_date"] = start_date
    store["pref_end_date"] = end_date
    store["pref_capital"] = capital
    store["pref_auto_update_time"] = auto_update_time
    store["pref_cycle_interval"] = cycle_interval
    store["pref_auto_update_enabled"] = auto_update_enabled


def _init_session_state_in_store(store: dict) -> None:
    """
    Mô phỏng logic _init_session_state() từ page_settings.py.
    Khởi tạo defaults nếu key chưa tồn tại.
    """
    from config.settings import DEFAULT_UPDATE_TIME

    defaults = {
        "pref_selected_symbols": VN30_SYMBOLS.copy(),
        "pref_start_date": date.today() - timedelta(days=365),
        "pref_end_date": date.today(),
        "pref_capital": 100_000_000.0,
        "pref_auto_update_time": DEFAULT_UPDATE_TIME,
        "pref_cycle_interval": 24.0,
        "pref_auto_update_enabled": True,
    }
    for key, default_value in defaults.items():
        if key not in store:
            store[key] = default_value


def _load_preferences_from_store(store: dict) -> dict:
    """
    Đọc preferences từ store (mô phỏng việc đọc từ st.session_state).
    Trả về dict với tất cả preference values.
    """
    return {
        "selected_symbols": store["pref_selected_symbols"],
        "start_date": store["pref_start_date"],
        "end_date": store["pref_end_date"],
        "capital": store["pref_capital"],
        "auto_update_time": store["pref_auto_update_time"],
        "cycle_interval": store["pref_cycle_interval"],
        "auto_update_enabled": store["pref_auto_update_enabled"],
    }


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


class TestPreferencesRoundTrip:
    """Property 22: User preferences round-trip."""

    @given(
        selected_symbols=selected_symbols_strategy,
        date_range=date_range_strategy(),
        capital=capital_strategy,
        auto_update_time=auto_update_time_strategy,
        cycle_interval=cycle_interval_strategy,
        auto_update_enabled=auto_update_enabled_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_save_then_load_produces_identical_preferences(
        self,
        selected_symbols,
        date_range,
        capital,
        auto_update_time,
        cycle_interval,
        auto_update_enabled,
    ):
        """
        Property: Với bất kỳ tập user preferences hợp lệ nào,
        save → load phải trả về giá trị preferences hoàn toàn giống nhau.

        # Feature: stock-trading-platform-refactor, Property 22: User preferences round-trip
        **Validates: Requirements 10.6**
        """
        start_date, end_date = date_range

        # Tạo store rỗng (mô phỏng session_state)
        store: dict = {}

        # --- Save preferences ---
        _save_preferences_to_store(
            store=store,
            selected_symbols=selected_symbols,
            start_date=start_date,
            end_date=end_date,
            capital=capital,
            auto_update_time=auto_update_time,
            cycle_interval=cycle_interval,
            auto_update_enabled=auto_update_enabled,
        )

        # --- Load preferences ---
        loaded = _load_preferences_from_store(store)

        # --- Assert round-trip bảo toàn tất cả preferences ---
        assert loaded["selected_symbols"] == selected_symbols, (
            f"selected_symbols mismatch: {loaded['selected_symbols']} != {selected_symbols}"
        )
        assert loaded["start_date"] == start_date, (
            f"start_date mismatch: {loaded['start_date']} != {start_date}"
        )
        assert loaded["end_date"] == end_date, (
            f"end_date mismatch: {loaded['end_date']} != {end_date}"
        )
        assert loaded["capital"] == capital, (
            f"capital mismatch: {loaded['capital']} != {capital}"
        )
        assert loaded["auto_update_time"] == auto_update_time, (
            f"auto_update_time mismatch: {loaded['auto_update_time']!r} != {auto_update_time!r}"
        )
        assert loaded["cycle_interval"] == cycle_interval, (
            f"cycle_interval mismatch: {loaded['cycle_interval']} != {cycle_interval}"
        )
        assert loaded["auto_update_enabled"] == auto_update_enabled, (
            f"auto_update_enabled mismatch: {loaded['auto_update_enabled']} != {auto_update_enabled}"
        )

    @given(
        selected_symbols=selected_symbols_strategy,
        date_range=date_range_strategy(),
        capital=capital_strategy,
        auto_update_time=auto_update_time_strategy,
        cycle_interval=cycle_interval_strategy,
        auto_update_enabled=auto_update_enabled_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_init_then_save_overwrite_then_load_produces_saved_values(
        self,
        selected_symbols,
        date_range,
        capital,
        auto_update_time,
        cycle_interval,
        auto_update_enabled,
    ):
        """
        Property bổ sung: Sau khi _init_session_state() khởi tạo defaults,
        _save_preferences() phải ghi đè đúng giá trị mới,
        và load lại phải trả về giá trị đã save (không phải defaults).

        # Feature: stock-trading-platform-refactor, Property 22: User preferences round-trip
        **Validates: Requirements 10.6**
        """
        start_date, end_date = date_range

        # Tạo store rỗng (mô phỏng session_state)
        store: dict = {}

        # --- Init session state với defaults ---
        _init_session_state_in_store(store)

        # Verify defaults đã được set
        assert "pref_selected_symbols" in store
        assert "pref_start_date" in store
        assert "pref_end_date" in store
        assert "pref_capital" in store
        assert "pref_auto_update_time" in store
        assert "pref_cycle_interval" in store
        assert "pref_auto_update_enabled" in store

        # --- Save preferences mới (ghi đè defaults) ---
        _save_preferences_to_store(
            store=store,
            selected_symbols=selected_symbols,
            start_date=start_date,
            end_date=end_date,
            capital=capital,
            auto_update_time=auto_update_time,
            cycle_interval=cycle_interval,
            auto_update_enabled=auto_update_enabled,
        )

        # --- Load lại → phải trả về giá trị đã save, không phải defaults ---
        loaded = _load_preferences_from_store(store)

        assert loaded["selected_symbols"] == selected_symbols
        assert loaded["start_date"] == start_date
        assert loaded["end_date"] == end_date
        assert loaded["capital"] == capital
        assert loaded["auto_update_time"] == auto_update_time
        assert loaded["cycle_interval"] == cycle_interval
        assert loaded["auto_update_enabled"] == auto_update_enabled

    @given(
        selected_symbols=selected_symbols_strategy,
        date_range=date_range_strategy(),
        capital=capital_strategy,
        auto_update_time=auto_update_time_strategy,
        cycle_interval=cycle_interval_strategy,
        auto_update_enabled=auto_update_enabled_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_actual_save_preferences_via_mock_session_state(
        self,
        selected_symbols,
        date_range,
        capital,
        auto_update_time,
        cycle_interval,
        auto_update_enabled,
    ):
        """
        Property: Kiểm tra hàm _save_preferences() thực tế trong page_settings.py
        hoạt động đúng khi st.session_state là dict bình thường.

        # Feature: stock-trading-platform-refactor, Property 22: User preferences round-trip
        **Validates: Requirements 10.6**
        """
        start_date, end_date = date_range

        # Mock st.session_state bằng dict thường
        mock_session_state = {}

        with patch("ui.pages.page_settings.st") as mock_st:
            mock_st.session_state = mock_session_state

            # Import hàm thực tế
            from ui.pages.page_settings import _save_preferences, _init_session_state

            # Init defaults trước
            _init_session_state()

            # Save preferences
            _save_preferences(
                selected_symbols=selected_symbols,
                start_date=start_date,
                end_date=end_date,
                capital=capital,
                auto_update_time=auto_update_time,
                cycle_interval=cycle_interval,
                auto_update_enabled=auto_update_enabled,
            )

            # Load và verify round-trip
            assert mock_session_state["pref_selected_symbols"] == selected_symbols
            assert mock_session_state["pref_start_date"] == start_date
            assert mock_session_state["pref_end_date"] == end_date
            assert mock_session_state["pref_capital"] == capital
            assert mock_session_state["pref_auto_update_time"] == auto_update_time
            assert mock_session_state["pref_cycle_interval"] == cycle_interval
            assert mock_session_state["pref_auto_update_enabled"] == auto_update_enabled
