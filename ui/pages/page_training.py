# -*- coding: utf-8 -*-
"""
Training page — Giao diện điều khiển pipeline training.

2 nút: Start (chạy update data + walk-forward cycle) và Stop.
Hiển thị trạng thái + log realtime.
"""

import json
import threading
from datetime import datetime
from pathlib import Path

import streamlit as st

from engine.pipeline_worker import (
    DEBUG_LOG,
    STATUS_FILE,
    is_running,
    request_stop,
    run_pipeline,
    force_reset_status,
)


def _read_status() -> dict:
    """Đọc trạng thái pipeline từ file."""
    if not STATUS_FILE.exists():
        return {"state": "idle", "message": "Chưa chạy"}
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"state": "idle", "message": "Chưa chạy"}


def _read_log_tail(lines: int = 30) -> str:
    """Đọc N dòng cuối của log file."""
    if not DEBUG_LOG.exists():
        return "(Chưa có log)"
    try:
        all_lines = DEBUG_LOG.read_text(encoding="utf-8").splitlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return "\n".join(tail)
    except Exception:
        return "(Lỗi đọc log)"


def _start_pipeline():
    """Chạy pipeline trong background thread."""
    if is_running():
        return
    thread = threading.Thread(target=run_pipeline, daemon=True, name="pipeline-worker")
    thread.start()


def page_training() -> None:
    """Trang Training — điều khiển pipeline."""
    st.title("🧠 Training Pipeline")

    # Đọc trạng thái
    status = _read_status()
    state = status.get("state", "idle")
    message = status.get("message", "")
    step = status.get("step", "")

    # === Hiển thị trạng thái ===
    if state == "running":
        st.success(f"🔄 **Đang chạy:** {message}", icon="⏳")
    elif state == "done":
        st.info(f"✅ **Hoàn tất:** {message}", icon="✔️")
    else:
        st.warning("⏸️ Chưa chạy pipeline")

    # === 3 nút: Start / Stop / Stop Immediate ===
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        start_disabled = (state == "running")
        if st.button("▶️ Start", type="primary", disabled=start_disabled, use_container_width=True):
            _start_pipeline()
            st.rerun()

    with col2:
        stop_disabled = (state != "running")
        if st.button("⏹️ Stop", type="secondary", disabled=stop_disabled, use_container_width=True,
                     help="Dừng sau khi hoàn thành bước hiện tại"):
            request_stop()
            st.toast("🛑 Đã gửi yêu cầu dừng. Pipeline sẽ dừng sau bước hiện tại.")

    with col3:
        stop_now_disabled = (state != "running")
        if st.button("⚡ Stop Now", type="secondary", disabled=stop_now_disabled, use_container_width=True,
                     help="Dừng ngay lập tức (dùng khi bị treo hoặc đã tắt app)"):
            force_reset_status()
            st.toast("⚡ Đã dừng ngay. Có thể Start lại.")
            st.rerun()

    # === Log realtime ===
    st.divider()
    st.subheader("📋 Pipeline Log")

    log_lines = st.slider("Số dòng hiển thị", min_value=10, max_value=100, value=30, step=10)
    log_content = _read_log_tail(log_lines)
    st.code(log_content, language="log")

    # Auto-refresh khi đang chạy
    if state == "running":
        @st.fragment(run_every=5)
        def _refresh():
            pass
        _refresh()


# Entry point cho st.Page
page_training()
