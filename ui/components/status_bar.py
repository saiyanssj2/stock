# -*- coding: utf-8 -*-
"""
StatusBar component - Hiển thị compact status cho tất cả active tasks.

Luôn visible ở top mọi page (kể cả khi không có task active).
Poll read_all_statuses() mỗi 10s (Streamlit auto-rerun handles refresh cycle).

Color coding:
- RUNNING  = 🟢
- PAUSED   = 🟡
- ERROR    = 🔴
- COMPLETED = ✅
- IDLE     = ⚪

References: Req 2.2, 10.3
"""

from typing import Dict

import streamlit as st

from models.task_models import TaskState, TaskStatus
from orchestrator.status_protocol import read_all_statuses

# Mapping TaskState → icon hiển thị
_STATE_ICONS: Dict[TaskState, str] = {
    TaskState.RUNNING: "🟢",
    TaskState.PAUSED: "🟡",
    TaskState.ERROR: "🔴",
    TaskState.COMPLETED: "✅",
    TaskState.IDLE: "⚪",
}

# Mapping TaskState → label tiếng Việt
_STATE_LABELS: Dict[TaskState, str] = {
    TaskState.RUNNING: "Đang chạy",
    TaskState.PAUSED: "Tạm dừng",
    TaskState.ERROR: "Lỗi",
    TaskState.COMPLETED: "Hoàn thành",
    TaskState.IDLE: "Chờ",
}


def _render_task_row(status: TaskStatus) -> None:
    """
    Render một dòng compact cho một task trong status bar.

    Hiển thị: icon + task_type + state label + progress bar.

    Parameters
    ----------
    status : TaskStatus
        Trạng thái task cần hiển thị.
    """
    icon = _STATE_ICONS.get(status.state, "⚪")
    label = _STATE_LABELS.get(status.state, status.state.value)
    task_type_display = status.task_type.value.capitalize()

    # Layout compact: 4 cột - icon+type | state | progress | message
    col_type, col_state, col_progress, col_msg = st.columns([2, 1.5, 2, 3])

    with col_type:
        st.caption(f"{icon} **{task_type_display}**")

    with col_state:
        st.caption(label)

    with col_progress:
        # Hiển thị progress bar + percentage
        progress_value = max(0.0, min(100.0, status.progress_pct)) / 100.0
        st.progress(progress_value, text=f"{status.progress_pct:.0f}%")

    with col_msg:
        st.caption(status.message or "")


def render_status_bar() -> None:
    """
    Render compact status bar ở top page, hiển thị tất cả active tasks.

    - Poll read_all_statuses() để lấy danh sách task hiện tại
    - Lọc chỉ hiển thị tasks có state RUNNING hoặc PAUSED (active)
    - Hiển thị COMPLETED/ERROR tasks trong 30s rồi tự ẩn
    - Khi không có active task: hiển thị placeholder
    - Streamlit auto-rerun mỗi khi user tương tác sẽ refresh status

    References: Req 2.2, 10.3
    """
    from datetime import datetime, timedelta

    with st.container():
        # Đọc tất cả status files
        all_statuses = read_all_statuses()

        # Lọc active tasks (RUNNING hoặc PAUSED)
        active_statuses = {
            task_id: status
            for task_id, status in all_statuses.items()
            if status.state in (TaskState.RUNNING, TaskState.PAUSED)
        }

        # Hiển thị ERROR tasks (luôn hiện cho đến khi user xóa)
        error_statuses = {
            task_id: status
            for task_id, status in all_statuses.items()
            if status.state == TaskState.ERROR
        }

        if not active_statuses and not error_statuses:
            # Placeholder khi không có task active
            st.caption("⚪ Không có task đang chạy")
        else:
            # Render active tasks
            for status in active_statuses.values():
                _render_task_row(status)
            # Render error tasks
            for status in error_statuses.values():
                _render_task_row(status)

        # Divider nhỏ phân cách status bar với content chính
        st.divider()
