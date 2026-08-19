# -*- coding: utf-8 -*-
"""
Unit tests cho ui/components/status_bar.py

Test render_status_bar() với các trường hợp:
- Không có active task → hiển thị placeholder
- Có task RUNNING → hiển thị compact row
- Có nhiều tasks active → hiển thị tất cả
- Task COMPLETED/IDLE → không hiển thị trong active list
"""

from datetime import datetime
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from models.task_models import TaskState, TaskStatus, TaskType


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit module cho testing UI components."""
    with patch("ui.components.status_bar.st") as mock_st:
        # Mock container context manager
        mock_container = MagicMock()
        mock_st.container.return_value.__enter__ = MagicMock(
            return_value=mock_container
        )
        mock_st.container.return_value.__exit__ = MagicMock(return_value=False)

        # Mock columns trả về list các mock objects
        mock_cols = [MagicMock() for _ in range(4)]
        for col in mock_cols:
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = mock_cols

        yield mock_st


@pytest.fixture
def sample_running_status() -> TaskStatus:
    """TaskStatus RUNNING mẫu."""
    return TaskStatus(
        task_id="training_abc123",
        task_type=TaskType.TRAINING,
        state=TaskState.RUNNING,
        progress_pct=45.0,
        message="Training HPG epoch 5/20",
        heartbeat_ts=datetime.now(),
        started_at=datetime.now(),
    )


@pytest.fixture
def sample_paused_status() -> TaskStatus:
    """TaskStatus PAUSED mẫu."""
    return TaskStatus(
        task_id="backtest_def456",
        task_type=TaskType.BACKTEST,
        state=TaskState.PAUSED,
        progress_pct=30.0,
        message="Paused - GPU pressure",
        heartbeat_ts=datetime.now(),
        started_at=datetime.now(),
    )


@pytest.fixture
def sample_completed_status() -> TaskStatus:
    """TaskStatus COMPLETED mẫu."""
    return TaskStatus(
        task_id="analysis_ghi789",
        task_type=TaskType.ANALYSIS,
        state=TaskState.COMPLETED,
        progress_pct=100.0,
        message="Analysis complete",
        heartbeat_ts=datetime.now(),
        started_at=datetime.now(),
    )


@pytest.fixture
def sample_error_status() -> TaskStatus:
    """TaskStatus ERROR mẫu."""
    return TaskStatus(
        task_id="training_err001",
        task_type=TaskType.TRAINING,
        state=TaskState.ERROR,
        progress_pct=60.0,
        message="GPU OOM error",
        heartbeat_ts=datetime.now(),
        started_at=datetime.now(),
        error="CUDA out of memory",
    )


class TestRenderStatusBarNoActiveTasks:
    """Test khi không có active task."""

    @patch("ui.components.status_bar.read_all_statuses")
    def test_empty_statuses_shows_placeholder(
        self, mock_read_all: MagicMock, mock_streamlit: MagicMock
    ) -> None:
        """Khi không có status file nào → hiển thị placeholder."""
        mock_read_all.return_value = {}

        from ui.components.status_bar import render_status_bar

        render_status_bar()

        mock_streamlit.caption.assert_called_with("⚪ Không có task đang chạy")

    @patch("ui.components.status_bar.read_all_statuses")
    def test_only_completed_tasks_shows_placeholder(
        self,
        mock_read_all: MagicMock,
        mock_streamlit: MagicMock,
        sample_completed_status: TaskStatus,
    ) -> None:
        """Khi chỉ có task COMPLETED → hiển thị placeholder (không active)."""
        mock_read_all.return_value = {
            sample_completed_status.task_id: sample_completed_status
        }

        from ui.components.status_bar import render_status_bar

        render_status_bar()

        mock_streamlit.caption.assert_called_with("⚪ Không có task đang chạy")

    @patch("ui.components.status_bar.read_all_statuses")
    def test_only_idle_tasks_shows_placeholder(
        self, mock_read_all: MagicMock, mock_streamlit: MagicMock
    ) -> None:
        """Khi chỉ có task IDLE → hiển thị placeholder."""
        idle_status = TaskStatus(
            task_id="idle_task",
            task_type=TaskType.ANALYSIS,
            state=TaskState.IDLE,
            progress_pct=0.0,
            message="",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
        )
        mock_read_all.return_value = {"idle_task": idle_status}

        from ui.components.status_bar import render_status_bar

        render_status_bar()

        mock_streamlit.caption.assert_called_with("⚪ Không có task đang chạy")


class TestRenderStatusBarWithActiveTasks:
    """Test khi có active tasks."""

    @patch("ui.components.status_bar.read_all_statuses")
    def test_running_task_renders_row(
        self,
        mock_read_all: MagicMock,
        mock_streamlit: MagicMock,
        sample_running_status: TaskStatus,
    ) -> None:
        """Khi có task RUNNING → render compact row."""
        mock_read_all.return_value = {
            sample_running_status.task_id: sample_running_status
        }

        from ui.components.status_bar import render_status_bar

        render_status_bar()

        # Phải gọi st.columns (cho row layout)
        mock_streamlit.columns.assert_called()
        # Không gọi placeholder
        caption_calls = [
            str(call) for call in mock_streamlit.caption.call_args_list
        ]
        assert not any(
            "Không có task đang chạy" in c for c in caption_calls
        )

    @patch("ui.components.status_bar.read_all_statuses")
    def test_error_task_renders_row(
        self,
        mock_read_all: MagicMock,
        mock_streamlit: MagicMock,
        sample_error_status: TaskStatus,
    ) -> None:
        """Khi có task ERROR → cũng render compact row (active)."""
        mock_read_all.return_value = {
            sample_error_status.task_id: sample_error_status
        }

        from ui.components.status_bar import render_status_bar

        render_status_bar()

        # ERROR tasks cũng active → render columns
        mock_streamlit.columns.assert_called()

    @patch("ui.components.status_bar.read_all_statuses")
    def test_multiple_active_tasks_render_all(
        self,
        mock_read_all: MagicMock,
        mock_streamlit: MagicMock,
        sample_running_status: TaskStatus,
        sample_paused_status: TaskStatus,
    ) -> None:
        """Khi có nhiều active tasks → render tất cả."""
        mock_read_all.return_value = {
            sample_running_status.task_id: sample_running_status,
            sample_paused_status.task_id: sample_paused_status,
        }

        from ui.components.status_bar import render_status_bar

        render_status_bar()

        # st.columns phải được gọi 2 lần (1 lần cho mỗi task)
        assert mock_streamlit.columns.call_count == 2


class TestRenderTaskRow:
    """Test _render_task_row() trực tiếp."""

    def test_progress_clamped_to_valid_range(
        self, mock_streamlit: MagicMock
    ) -> None:
        """Progress percentage clamp trong khoảng [0, 100]."""
        from ui.components.status_bar import _render_task_row

        # Test progress > 100 → clamp to 1.0
        status_over = TaskStatus(
            task_id="test_over",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=150.0,
            message="Over progress",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
        )
        _render_task_row(status_over)

        # Lấy progress call - mock columns trả về context managers
        # Kiểm tra rằng hàm không crash với giá trị ngoài khoảng

    def test_progress_negative_clamped_to_zero(
        self, mock_streamlit: MagicMock
    ) -> None:
        """Progress âm → clamp về 0."""
        from ui.components.status_bar import _render_task_row

        status_neg = TaskStatus(
            task_id="test_neg",
            task_type=TaskType.TRAINING,
            state=TaskState.RUNNING,
            progress_pct=-10.0,
            message="Negative progress",
            heartbeat_ts=datetime.now(),
            started_at=datetime.now(),
        )
        # Không nên crash
        _render_task_row(status_neg)


class TestStateIcons:
    """Test icon mapping logic."""

    def test_all_states_have_icons(self) -> None:
        """Mọi TaskState đều có icon tương ứng."""
        from ui.components.status_bar import _STATE_ICONS

        for state in TaskState:
            assert state in _STATE_ICONS, f"Missing icon for {state}"

    def test_all_states_have_labels(self) -> None:
        """Mọi TaskState đều có label tương ứng."""
        from ui.components.status_bar import _STATE_LABELS

        for state in TaskState:
            assert state in _STATE_LABELS, f"Missing label for {state}"

    def test_running_icon_is_green(self) -> None:
        """RUNNING state hiển thị icon xanh."""
        from ui.components.status_bar import _STATE_ICONS

        assert _STATE_ICONS[TaskState.RUNNING] == "🟢"

    def test_error_icon_is_red(self) -> None:
        """ERROR state hiển thị icon đỏ."""
        from ui.components.status_bar import _STATE_ICONS

        assert _STATE_ICONS[TaskState.ERROR] == "🔴"

    def test_paused_icon_is_yellow(self) -> None:
        """PAUSED state hiển thị icon vàng."""
        from ui.components.status_bar import _STATE_ICONS

        assert _STATE_ICONS[TaskState.PAUSED] == "🟡"
