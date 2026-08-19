# -*- coding: utf-8 -*-
"""
Unit tests cho engine/data_update_wiring.py

Kiểm tra:
1. run_data_update_with_refresh: data update → save timestamp → refresh recommendations
2. wire_scheduler_to_recommendation: scheduler callback → save + refresh
3. save/load_last_update_timestamp: round-trip persistence
4. Partial failure vẫn trigger refresh (Req 8.3)

References: Req 8.3, 8.5, 6.7
"""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from engine.data_update_wiring import (
    LAST_UPDATE_STATUS_PATH,
    load_last_update_timestamp,
    run_data_update_with_refresh,
    save_last_update_timestamp,
    wire_scheduler_to_recommendation,
)
from models.data_models import UpdateResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_status_path(tmp_path):
    """Tạo đường dẫn tạm cho file status."""
    return str(tmp_path / "last_data_update.json")


@pytest.fixture
def success_result() -> UpdateResult:
    """UpdateResult thành công hoàn toàn."""
    return UpdateResult(
        total_symbols=30,
        success_count=30,
        failed_symbols=[],
        errors={},
        duration_seconds=45.2,
    )


@pytest.fixture
def partial_failure_result() -> UpdateResult:
    """UpdateResult với một số symbols thất bại."""
    return UpdateResult(
        total_symbols=30,
        success_count=27,
        failed_symbols=["ABC", "XYZ", "FOO"],
        errors={
            "ABC": "Connection timeout",
            "XYZ": "Invalid data format",
            "FOO": "Symbol not found",
        },
        duration_seconds=120.5,
    )


@pytest.fixture
def mock_pipeline():
    """Mock DataPipeline."""
    pipeline = MagicMock()
    pipeline.update_all.return_value = UpdateResult(
        total_symbols=30,
        success_count=28,
        failed_symbols=["ERR1", "ERR2"],
        errors={"ERR1": "timeout", "ERR2": "not found"},
        duration_seconds=60.0,
    )
    return pipeline


@pytest.fixture
def mock_recommendation_engine():
    """Mock RecommendationEngine."""
    engine = MagicMock()
    engine.refresh.return_value = None
    return engine


@pytest.fixture
def mock_scheduler():
    """Mock DataScheduler."""
    scheduler = MagicMock()
    return scheduler


# =============================================================================
# Tests: save/load_last_update_timestamp
# =============================================================================


class TestSaveLoadTimestamp:
    """Tests cho save/load round-trip."""

    def test_save_and_load_success(self, tmp_status_path, success_result):
        """Lưu rồi load lại phải giữ nguyên thông tin."""
        save_last_update_timestamp(success_result, path=tmp_status_path)

        loaded = load_last_update_timestamp(path=tmp_status_path)

        assert loaded is not None
        assert loaded["total_symbols"] == 30
        assert loaded["success_count"] == 30
        assert loaded["failed_count"] == 0
        assert loaded["failed_symbols"] == []
        assert loaded["duration_seconds"] == 45.2
        assert "timestamp" in loaded

    def test_save_and_load_partial_failure(self, tmp_status_path, partial_failure_result):
        """Lưu với partial failure phải giữ thông tin lỗi."""
        save_last_update_timestamp(partial_failure_result, path=tmp_status_path)

        loaded = load_last_update_timestamp(path=tmp_status_path)

        assert loaded is not None
        assert loaded["total_symbols"] == 30
        assert loaded["success_count"] == 27
        assert loaded["failed_count"] == 3
        assert loaded["failed_symbols"] == ["ABC", "XYZ", "FOO"]

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Load file chưa tồn tại → None."""
        result = load_last_update_timestamp(path=str(tmp_path / "nonexist.json"))
        assert result is None

    def test_load_corrupt_file_returns_none(self, tmp_status_path):
        """Load file corrupt → None, không crash."""
        # Tạo thư mục cha
        os.makedirs(os.path.dirname(tmp_status_path), exist_ok=True)
        with open(tmp_status_path, "w") as f:
            f.write("not valid json {{{")

        result = load_last_update_timestamp(path=tmp_status_path)
        assert result is None

    def test_timestamp_is_recent(self, tmp_status_path, success_result):
        """Timestamp lưu phải gần thời điểm hiện tại."""
        before = datetime.now()
        save_last_update_timestamp(success_result, path=tmp_status_path)

        loaded = load_last_update_timestamp(path=tmp_status_path)
        ts = datetime.fromisoformat(loaded["timestamp"])

        assert ts >= before
        # Chênh lệch không quá 5 giây
        diff = (datetime.now() - ts).total_seconds()
        assert diff < 5.0

    def test_save_creates_directory(self, tmp_path):
        """Save tạo thư mục nếu chưa tồn tại."""
        path = str(tmp_path / "nested" / "dir" / "status.json")
        result = UpdateResult(
            total_symbols=5,
            success_count=5,
            failed_symbols=[],
            errors={},
            duration_seconds=10.0,
        )

        save_last_update_timestamp(result, path=path)
        assert os.path.exists(path)


# =============================================================================
# Tests: run_data_update_with_refresh
# =============================================================================


class TestRunDataUpdateWithRefresh:
    """Tests cho flow: update → save → refresh."""

    def test_calls_pipeline_update(self, mock_pipeline, mock_recommendation_engine, tmp_status_path):
        """Phải gọi pipeline.update_all()."""
        run_data_update_with_refresh(
            mock_pipeline, mock_recommendation_engine, status_path=tmp_status_path
        )

        mock_pipeline.update_all.assert_called_once()

    def test_calls_recommendation_refresh(self, mock_pipeline, mock_recommendation_engine, tmp_status_path):
        """Phải gọi recommendation_engine.refresh() sau update."""
        run_data_update_with_refresh(
            mock_pipeline, mock_recommendation_engine, status_path=tmp_status_path
        )

        mock_recommendation_engine.refresh.assert_called_once()

    def test_saves_timestamp(self, mock_pipeline, mock_recommendation_engine, tmp_status_path):
        """Phải lưu timestamp file sau update."""
        run_data_update_with_refresh(
            mock_pipeline, mock_recommendation_engine, status_path=tmp_status_path
        )

        assert os.path.exists(tmp_status_path)
        loaded = load_last_update_timestamp(path=tmp_status_path)
        assert loaded is not None
        assert loaded["total_symbols"] == 30

    def test_partial_failure_still_refreshes(self, mock_recommendation_engine, tmp_status_path):
        """Req 8.3: Partial failure vẫn trigger refresh."""
        # Pipeline trả về kết quả với failures
        pipeline = MagicMock()
        pipeline.update_all.return_value = UpdateResult(
            total_symbols=30,
            success_count=20,
            failed_symbols=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            errors={s: "error" for s in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]},
            duration_seconds=100.0,
        )

        run_data_update_with_refresh(
            pipeline, mock_recommendation_engine, status_path=tmp_status_path
        )

        # refresh() vẫn được gọi dù có 10 symbols fail
        mock_recommendation_engine.refresh.assert_called_once()

    def test_all_failures_still_refreshes(self, mock_recommendation_engine, tmp_status_path):
        """Req 8.3: Kể cả ALL symbols fail, refresh vẫn chạy."""
        pipeline = MagicMock()
        pipeline.update_all.return_value = UpdateResult(
            total_symbols=5,
            success_count=0,
            failed_symbols=["A", "B", "C", "D", "E"],
            errors={s: "error" for s in ["A", "B", "C", "D", "E"]},
            duration_seconds=30.0,
        )

        run_data_update_with_refresh(
            pipeline, mock_recommendation_engine, status_path=tmp_status_path
        )

        # Refresh vẫn chạy → recommendations dùng data cũ còn valid
        mock_recommendation_engine.refresh.assert_called_once()

    def test_returns_update_result(self, mock_pipeline, mock_recommendation_engine, tmp_status_path):
        """Phải trả về UpdateResult từ pipeline."""
        result = run_data_update_with_refresh(
            mock_pipeline, mock_recommendation_engine, status_path=tmp_status_path
        )

        assert isinstance(result, UpdateResult)
        assert result.total_symbols == 30


# =============================================================================
# Tests: wire_scheduler_to_recommendation
# =============================================================================


class TestWireSchedulerToRecommendation:
    """Tests cho wiring scheduler callback."""

    def test_sets_callback_on_scheduler(self, mock_scheduler, mock_recommendation_engine, tmp_status_path):
        """Wire phải gọi scheduler.set_on_update_complete() với callback."""
        wire_scheduler_to_recommendation(
            mock_scheduler, mock_recommendation_engine, status_path=tmp_status_path
        )

        mock_scheduler.set_on_update_complete.assert_called_once()

    def test_callback_refreshes_recommendations(self, mock_recommendation_engine, tmp_status_path):
        """Callback khi triggered phải refresh recommendations."""
        scheduler = MagicMock()
        # Capture callback
        captured_callback = None

        def capture(cb):
            nonlocal captured_callback
            captured_callback = cb

        scheduler.set_on_update_complete.side_effect = capture

        wire_scheduler_to_recommendation(
            scheduler, mock_recommendation_engine, status_path=tmp_status_path
        )

        assert captured_callback is not None

        # Simulate update complete
        update_result = UpdateResult(
            total_symbols=10,
            success_count=10,
            failed_symbols=[],
            errors={},
            duration_seconds=20.0,
        )
        captured_callback(update_result)

        # Phải refresh recommendations
        mock_recommendation_engine.refresh.assert_called_once()

    def test_callback_saves_timestamp(self, mock_recommendation_engine, tmp_status_path):
        """Callback khi triggered phải lưu timestamp."""
        scheduler = MagicMock()
        captured_callback = None

        def capture(cb):
            nonlocal captured_callback
            captured_callback = cb

        scheduler.set_on_update_complete.side_effect = capture

        wire_scheduler_to_recommendation(
            scheduler, mock_recommendation_engine, status_path=tmp_status_path
        )

        # Trigger callback
        update_result = UpdateResult(
            total_symbols=15,
            success_count=13,
            failed_symbols=["XX", "YY"],
            errors={"XX": "e1", "YY": "e2"},
            duration_seconds=50.0,
        )
        captured_callback(update_result)

        # Kiểm tra timestamp đã lưu
        loaded = load_last_update_timestamp(path=tmp_status_path)
        assert loaded is not None
        assert loaded["success_count"] == 13
        assert loaded["failed_count"] == 2

    def test_callback_with_partial_failure_still_refreshes(self, mock_recommendation_engine, tmp_status_path):
        """Callback với partial failure vẫn phải refresh (Req 8.3)."""
        scheduler = MagicMock()
        captured_callback = None

        def capture(cb):
            nonlocal captured_callback
            captured_callback = cb

        scheduler.set_on_update_complete.side_effect = capture

        wire_scheduler_to_recommendation(
            scheduler, mock_recommendation_engine, status_path=tmp_status_path
        )

        # Trigger với partial failure
        update_result = UpdateResult(
            total_symbols=10,
            success_count=7,
            failed_symbols=["A", "B", "C"],
            errors={"A": "e", "B": "e", "C": "e"},
            duration_seconds=30.0,
        )
        captured_callback(update_result)

        # Refresh vẫn phải chạy
        mock_recommendation_engine.refresh.assert_called_once()
