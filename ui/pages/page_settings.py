"""
Settings page - user preferences và auto-update config.

Cho phép người dùng cấu hình:
- Symbols selection (VN30 + watchlist)
- Date ranges cho analysis/backtest
- Capital settings (vốn đầu tư)
- Auto-update time config
- Cycle interval cho Auto-Learner

Preferences được persist qua st.session_state để giữ nguyên khi navigate giữa pages.

Requirements: 10.5 (group related controls), 10.6 (persist preferences)
"""

import logging
from datetime import date, timedelta
from typing import List

import streamlit as st

from config.preferences import load_preferences_from_file, save_preferences_to_file
from config.settings import DEFAULT_UPDATE_TIME
from config.vn_market_rules import VN30_SYMBOLS

logger = logging.getLogger(__name__)


# ==============================================================================
# Default values
# ==============================================================================

# Vốn mặc định (VND)
DEFAULT_CAPITAL: float = 100_000_000.0

# Cycle interval mặc định (giờ)
DEFAULT_CYCLE_INTERVAL: float = 24.0

# Ngày bắt đầu mặc định: 1 năm trước
DEFAULT_START_DATE: date = date.today() - timedelta(days=365)

# Ngày kết thúc mặc định: hôm nay
DEFAULT_END_DATE: date = date.today()


# ==============================================================================
# Session state helpers
# ==============================================================================


def _get_watchlist_symbols() -> List[str]:
    """Đọc danh sách mã từ watchlist file (data/code.txt)."""
    try:
        with open("data/code.txt", "r", encoding="utf-8") as f:
            content = f.read()
        # Parse danh sách mã, bỏ khoảng trắng và dòng trống
        symbols = [
            s.strip()
            for s in content.replace("\n", ",").split(",")
            if s.strip()
        ]
        return sorted(set(symbols))
    except FileNotFoundError:
        return []


def _get_all_available_symbols() -> List[str]:
    """Lấy toàn bộ symbols có sẵn: VN30 + watchlist."""
    watchlist = _get_watchlist_symbols()
    all_symbols = sorted(set(VN30_SYMBOLS + watchlist))
    return all_symbols


def _init_session_state() -> None:
    """Khởi tạo session state với giá trị từ disk file (nếu có), fallback về defaults.

    Thứ tự ưu tiên:
    1. Giá trị đã có trong session_state (từ widget interaction trước đó) → giữ nguyên
    2. Giá trị từ preferences file trên disk → hydrate vào session_state
    3. Hardcoded defaults → fallback cuối cùng
    """
    # Hardcoded defaults (fallback cuối cùng)
    defaults = {
        "pref_selected_symbols": VN30_SYMBOLS.copy(),
        "pref_start_date": DEFAULT_START_DATE,
        "pref_end_date": DEFAULT_END_DATE,
        "pref_capital": DEFAULT_CAPITAL,
        "pref_auto_update_time": DEFAULT_UPDATE_TIME,
        "pref_cycle_interval": DEFAULT_CYCLE_INTERVAL,
        "pref_auto_update_enabled": True,
    }

    # Thử load từ disk file trước
    loaded = load_preferences_from_file()

    if loaded is not None:
        # Mapping: preferences file keys → session_state keys
        file_to_state = {
            "selected_symbols": "pref_selected_symbols",
            "capital": "pref_capital",
            "auto_update_time": "pref_auto_update_time",
            "cycle_interval": "pref_cycle_interval",
            "auto_update_enabled": "pref_auto_update_enabled",
            "start_date": "pref_start_date",
            "end_date": "pref_end_date",
        }
        # Keys cần coerce sang float để khớp kiểu với Streamlit widgets
        _float_keys = {"capital", "cycle_interval"}
        # Keys cần convert từ string ISO → date object cho st.date_input
        _date_keys = {"start_date", "end_date"}

        for file_key, state_key in file_to_state.items():
            if file_key in loaded and state_key not in st.session_state:
                value = loaded[file_key]
                # Đảm bảo numeric values luôn là float (Streamlit yêu cầu consistent types)
                if file_key in _float_keys and isinstance(value, (int, float)):
                    value = float(value)
                # Convert date string → date object (Streamlit widget yêu cầu)
                if file_key in _date_keys and isinstance(value, str):
                    try:
                        value = date.fromisoformat(value)
                    except (ValueError, TypeError):
                        # Nếu parse fail → dùng default
                        continue
                st.session_state[state_key] = value

    # Fallback: set defaults cho các key chưa có trong session_state
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def _save_preferences(
    selected_symbols: List[str],
    start_date: date,
    end_date: date,
    capital: float,
    auto_update_time: str,
    cycle_interval: float,
    auto_update_enabled: bool,
) -> None:
    """Lưu preferences vào session state và persist ra disk."""
    # Nếu chưa chọn symbols nào → fallback VN30
    if not selected_symbols:
        selected_symbols = VN30_SYMBOLS.copy()

    # Ghi vào session_state (giữ nguyên logic cũ)
    st.session_state["pref_selected_symbols"] = selected_symbols
    st.session_state["pref_start_date"] = start_date
    st.session_state["pref_end_date"] = end_date
    st.session_state["pref_capital"] = capital
    st.session_state["pref_auto_update_time"] = auto_update_time
    st.session_state["pref_cycle_interval"] = cycle_interval
    st.session_state["pref_auto_update_enabled"] = auto_update_enabled

    # Persist ra disk để không mất khi browser refresh/close
    prefs_dict = {
        "selected_symbols": selected_symbols,
        "capital": float(capital),
        "auto_update_time": auto_update_time,
        "cycle_interval": float(cycle_interval),
        "auto_update_enabled": auto_update_enabled,
        "start_date": str(start_date),
        "end_date": str(end_date),
    }

    # Debug log ra file
    from datetime import datetime
    from pathlib import Path
    _debug_log = Path("pipeline_debug.log")
    with open(_debug_log, "a", encoding="utf-8") as _f:
        _f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [SETTINGS] Saving preferences:\n")
        _f.write(f"  selected_symbols: {len(selected_symbols)} mã = {selected_symbols[:5]}...\n")
        _f.write(f"  capital={capital}, auto_update_time={auto_update_time}\n")
        _f.write(f"  cycle_interval={cycle_interval}, auto_update_enabled={auto_update_enabled}\n")
        _f.write(f"  start_date={start_date}, end_date={end_date}\n")

    try:
        save_preferences_to_file(prefs_dict)
        logger.info("Preferences đã được persist ra disk thành công.")
        with open(_debug_log, "a", encoding="utf-8") as _f:
            _f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [SETTINGS] SAVE OK\n")
    except (ValueError, IOError, OSError) as e:
        # Không crash UI nếu disk write fail — session_state vẫn hoạt động
        logger.error("Không thể persist preferences ra disk: %s", e)
        with open(_debug_log, "a", encoding="utf-8") as _f:
            _f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [SETTINGS] SAVE ERROR: {e}\n")


# ==============================================================================
# UI Sections
# ==============================================================================


def _render_symbols_section(available_symbols: List[str]) -> List[str]:
    """Render phần chọn symbols."""
    st.subheader("📋 Symbols Selection")
    st.caption("Chọn các mã cổ phiếu để theo dõi, phân tích và backtest.")

    # Quick select buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Chọn tất cả", key="btn_select_all"):
            st.session_state["widget_selected_symbols"] = available_symbols.copy()
            st.rerun()
    with col2:
        if st.button("Chỉ VN30", key="btn_select_vn30"):
            st.session_state["widget_selected_symbols"] = VN30_SYMBOLS.copy()
            st.rerun()
    with col3:
        if st.button("Bỏ chọn tất cả", key="btn_deselect_all"):
            st.session_state["widget_selected_symbols"] = []
            st.rerun()

    # Multiselect cho symbols — dùng default chỉ khi widget key chưa tồn tại
    if "widget_selected_symbols" not in st.session_state:
        st.session_state["widget_selected_symbols"] = st.session_state.get(
            "pref_selected_symbols", VN30_SYMBOLS.copy()
        )

    selected = st.multiselect(
        "Danh sách mã theo dõi",
        options=available_symbols,
        key="widget_selected_symbols",
        help="Chọn các mã cổ phiếu từ VN30 và watchlist để phân tích.",
    )

    # Hiển thị thống kê
    st.info(f"Đã chọn: **{len(selected)}** / {len(available_symbols)} mã")

    return selected


def _render_date_range_section() -> tuple:
    """Render phần cấu hình date range."""
    st.subheader("📅 Date Range")
    st.caption("Khoảng thời gian mặc định cho analysis và backtest.")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Ngày bắt đầu",
            value=st.session_state["pref_start_date"],
            key="widget_start_date",
            help="Ngày bắt đầu phân tích dữ liệu lịch sử.",
        )
    with col2:
        end_date = st.date_input(
            "Ngày kết thúc",
            value=st.session_state["pref_end_date"],
            key="widget_end_date",
            help="Ngày kết thúc phân tích (mặc định: hôm nay).",
        )

    # Validate date range
    if start_date >= end_date:
        st.warning("⚠️ Ngày bắt đầu phải trước ngày kết thúc.")

    return start_date, end_date


def _render_capital_section() -> float:
    """Render phần cấu hình vốn đầu tư."""
    st.subheader("💰 Capital Settings")
    st.caption("Vốn đầu tư ban đầu cho backtest và mô phỏng giao dịch.")

    capital = st.number_input(
        "Vốn đầu tư (VND)",
        min_value=1_000_000.0,
        max_value=100_000_000_000.0,
        value=st.session_state["pref_capital"],
        step=10_000_000.0,
        format="%.0f",
        key="widget_capital",
        help="Vốn ban đầu dùng cho backtest. Tối thiểu 1 triệu VND.",
    )

    # Hiển thị format dễ đọc
    formatted = f"{capital:,.0f} VND"
    st.caption(f"💵 {formatted}")

    return capital


def _render_auto_update_section() -> tuple:
    """Render phần cấu hình auto-update."""
    st.subheader("🔄 Auto-Update Config")
    st.caption("Cấu hình tự động cập nhật dữ liệu thị trường sau giờ đóng cửa.")

    # Toggle bật/tắt auto-update
    auto_update_enabled = st.toggle(
        "Bật auto-update",
        value=st.session_state["pref_auto_update_enabled"],
        key="widget_auto_update_enabled",
        help="Tự động fetch dữ liệu mới sau giờ đóng cửa.",
    )

    # Parse thời gian hiện tại từ session state
    current_time = st.session_state["pref_auto_update_time"]
    try:
        hour, minute = map(int, current_time.split(":"))
    except (ValueError, AttributeError):
        hour, minute = 15, 30

    col1, col2 = st.columns(2)
    with col1:
        update_hour = st.number_input(
            "Giờ update (0-23)",
            min_value=0,
            max_value=23,
            value=hour,
            key="widget_update_hour",
            help="Giờ tự động update data (sau market close 15:00).",
            disabled=not auto_update_enabled,
        )
    with col2:
        update_minute = st.number_input(
            "Phút (0-59)",
            min_value=0,
            max_value=59,
            value=minute,
            key="widget_update_minute",
            help="Phút tự động update data.",
            disabled=not auto_update_enabled,
        )

    auto_update_time = f"{update_hour:02d}:{update_minute:02d}"

    if auto_update_enabled:
        st.caption(f"⏰ Auto-update sẽ chạy lúc **{auto_update_time}** mỗi ngày giao dịch.")
    else:
        st.caption("Auto-update đã tắt. Bạn cần update thủ công.")

    return auto_update_time, auto_update_enabled


def _render_cycle_interval_section() -> float:
    """Render phần cấu hình cycle interval cho Auto-Learner."""
    st.subheader("🔁 Auto-Learner Cycle Interval")
    st.caption("Khoảng cách giữa các vòng lặp tự động: train → backtest → phân tích → retrain.")

    cycle_interval = st.number_input(
        "Cycle interval (giờ)",
        min_value=1.0,
        max_value=168.0,
        value=st.session_state["pref_cycle_interval"],
        step=1.0,
        format="%.1f",
        key="widget_cycle_interval",
        help="Khoảng cách giữa các auto-learning cycles. Mặc định 24 giờ.",
    )

    # Hiển thị mô tả dễ hiểu
    if cycle_interval < 24:
        st.caption(f"🔁 Chạy mỗi **{cycle_interval:.1f} giờ** ({cycle_interval * 60:.0f} phút)")
    elif cycle_interval == 24:
        st.caption("🔁 Chạy **1 lần/ngày**")
    else:
        days = cycle_interval / 24
        st.caption(f"🔁 Chạy mỗi **{days:.1f} ngày** ({cycle_interval:.0f} giờ)")

    return cycle_interval


# ==============================================================================
# Main page function
# ==============================================================================


def page_settings() -> None:
    """Render trang Settings chính."""
    st.title("⚙️ Settings")
    st.markdown("Cấu hình preferences cho hệ thống giao dịch. "
                "Các thay đổi được lưu tự động trong session.")

    # Khởi tạo session state
    _init_session_state()

    # Lấy danh sách symbols có sẵn
    available_symbols = _get_all_available_symbols()

    # --- Symbols Selection ---
    selected_symbols = _render_symbols_section(available_symbols)

    st.divider()

    # --- Date Range ---
    start_date, end_date = _render_date_range_section()

    st.divider()

    # --- Capital Settings ---
    capital = _render_capital_section()

    st.divider()

    # --- Auto-Update Config ---
    auto_update_time, auto_update_enabled = _render_auto_update_section()

    st.divider()

    # --- Cycle Interval ---
    cycle_interval = _render_cycle_interval_section()

    st.divider()

    # --- Save button ---
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("💾 Lưu cài đặt", type="primary", key="btn_save_settings"):
            _save_preferences(
                selected_symbols=selected_symbols,
                start_date=start_date,
                end_date=end_date,
                capital=capital,
                auto_update_time=auto_update_time,
                cycle_interval=cycle_interval,
                auto_update_enabled=auto_update_enabled,
            )
            st.success("✅ Đã lưu cài đặt thành công!")

    # --- Hiển thị tóm tắt preferences hiện tại ---
    with st.expander("📊 Tóm tắt cài đặt hiện tại", expanded=False):
        st.json({
            "selected_symbols_count": len(st.session_state["pref_selected_symbols"]),
            "start_date": str(st.session_state["pref_start_date"]),
            "end_date": str(st.session_state["pref_end_date"]),
            "capital_vnd": st.session_state["pref_capital"],
            "auto_update_enabled": st.session_state["pref_auto_update_enabled"],
            "auto_update_time": st.session_state["pref_auto_update_time"],
            "cycle_interval_hours": st.session_state["pref_cycle_interval"],
        })


# Streamlit page entry point — chạy khi được load bởi st.navigation
import streamlit.runtime.scriptrunner as _sr

try:
    _ctx = _sr.get_script_run_ctx()
except Exception:
    _ctx = None

if _ctx is not None:
    page_settings()
