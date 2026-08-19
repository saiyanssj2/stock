"""
Unit tests cho DataScheduler module.

Kiểm tra logic lên lịch auto-update, callback mechanism,
và validation thời gian.

References: Req 8.3, 8.5, 8.7
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from engine.data_scheduler import DataScheduler
from engine.data_pipeline import DataPipeline
from models.data_models import UpdateResult


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Tạo mock DataPipeline."""
    pipeline = MagicMock(spec=DataPipeline)
    pipeline.update_all.return_value = UpdateResult(
        total_symbols=30,
        success_count=28,
        failed_symbols=["ABC", "XYZ"],
        errors={"ABC": "network error", "XYZ": "missing data"},
        duration_seconds=5.0,
    )
    return pipeline


@pytest.fixture
def scheduler(mock_pipeline: MagicMock) -> DataScheduler:
    """Tạo DataScheduler instance với mock pipeline."""
    return DataScheduler(data_pipeline=mock_pipeline, update_time="15:30")


class TestDataSchedulerInit:
    """Test khởi tạo DataScheduler."""

    def test_default_update_time(self, mock_pipeline: MagicMock) -> None:
        """Khởi tạo với default time từ settings."""
        scheduler = DataScheduler(data_pipeline=mock_pipeline)
        assert scheduler.get_next_update_time() == "15:30"

    def test_custom_update_time(self, mock_pipeline: MagicMock) -> None:
        """Khởi tạo với custom update time."""
        scheduler = DataScheduler(data_pipeline=mock_pipeline, update_time="16:00")
        assert scheduler.get_next_update_time() == "16:00"

    def test_invalid_time_format_raises(self, mock_pipeline: MagicMock) -> None:
        """Hour ngoài phạm vi phải raise ValueError."""
        with pytest.raises(ValueError, match="Invalid hour"):
            DataScheduler(data_pipeline=mock_pipeline, update_time="25:00")

    def test_invalid_time_no_colon_raises(self, mock_pipeline: MagicMock) -> None:
        """Thiếu dấu : phải raise ValueError."""
        with pytest.raises(ValueError, match="Invalid time format"):
            DataScheduler(data_pipeline=mock_pipeline, update_time="1530")

    def test_invalid_hour_raises(self, mock_pipeline: MagicMock) -> None:
        """Hour > 23 phải raise ValueError."""
        with pytest.raises(ValueError, match="Invalid hour"):
            DataScheduler(data_pipeline=mock_pipeline, update_time="24:00")

    def test_invalid_minute_raises(self, mock_pipeline: MagicMock) -> None:
        """Minute > 59 phải raise ValueError."""
        with pytest.raises(ValueError, match="Invalid minute"):
            DataScheduler(data_pipeline=mock_pipeline, update_time="15:60")


class TestShouldUpdateNow:
    """Test logic kiểm tra thời điểm update."""

    def test_exact_match_should_update(self, scheduler: DataScheduler) -> None:
        """Đúng giờ scheduled → should update."""
        mock_now = datetime(2024, 1, 15, 15, 30, 0)
        with patch("engine.data_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            assert scheduler.should_update_now() is True

    def test_one_minute_before_should_update(self, scheduler: DataScheduler) -> None:
        """1 phút trước scheduled time → should update."""
        mock_now = datetime(2024, 1, 15, 15, 29, 0)
        with patch("engine.data_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            assert scheduler.should_update_now() is True

    def test_one_minute_after_should_update(self, scheduler: DataScheduler) -> None:
        """1 phút sau scheduled time → should update."""
        mock_now = datetime(2024, 1, 15, 15, 31, 0)
        with patch("engine.data_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            assert scheduler.should_update_now() is True

    def test_two_minutes_before_should_not_update(
        self, scheduler: DataScheduler
    ) -> None:
        """2 phút trước scheduled time → should NOT update."""
        mock_now = datetime(2024, 1, 15, 15, 28, 0)
        with patch("engine.data_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            assert scheduler.should_update_now() is False

    def test_two_minutes_after_should_not_update(
        self, scheduler: DataScheduler
    ) -> None:
        """2 phút sau scheduled time → should NOT update."""
        mock_now = datetime(2024, 1, 15, 15, 32, 0)
        with patch("engine.data_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            assert scheduler.should_update_now() is False

    def test_already_updated_today_should_not_update(
        self, scheduler: DataScheduler
    ) -> None:
        """Đã update hôm nay → should NOT update lại."""
        scheduler._last_update_date = "2024-01-15"
        mock_now = datetime(2024, 1, 15, 15, 30, 0)
        with patch("engine.data_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            assert scheduler.should_update_now() is False

    def test_updated_yesterday_should_update_today(
        self, scheduler: DataScheduler
    ) -> None:
        """Update hôm qua → hôm nay vẫn chạy."""
        scheduler._last_update_date = "2024-01-14"
        mock_now = datetime(2024, 1, 15, 15, 30, 0)
        with patch("engine.data_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            assert scheduler.should_update_now() is True


class TestRunUpdate:
    """Test thực hiện data update."""

    def test_run_update_calls_pipeline(
        self, scheduler: DataScheduler, mock_pipeline: MagicMock
    ) -> None:
        """run_update phải gọi data_pipeline.update_all()."""
        scheduler.run_update()
        mock_pipeline.update_all.assert_called_once()

    def test_run_update_returns_result(
        self, scheduler: DataScheduler
    ) -> None:
        """run_update phải trả về UpdateResult."""
        result = scheduler.run_update()
        assert isinstance(result, UpdateResult)
        assert result.total_symbols == 30
        assert result.success_count == 28

    def test_run_update_marks_today(self, scheduler: DataScheduler) -> None:
        """Sau run_update, _last_update_date phải là hôm nay."""
        scheduler.run_update()
        today = datetime.now().strftime("%Y-%m-%d")
        assert scheduler._last_update_date == today

    def test_run_update_triggers_callback(
        self, scheduler: DataScheduler
    ) -> None:
        """run_update phải trigger on_update_complete callback."""
        callback = MagicMock()
        scheduler.set_on_update_complete(callback)
        result = scheduler.run_update()
        callback.assert_called_once_with(result)

    def test_run_update_no_callback_ok(
        self, scheduler: DataScheduler
    ) -> None:
        """run_update không có callback phải chạy bình thường."""
        # Không set callback → không lỗi
        result = scheduler.run_update()
        assert result is not None


class TestAutoUpdateAtTime:
    """Test auto_update_at_time integration."""

    def test_auto_update_when_time_matches(
        self, scheduler: DataScheduler, mock_pipeline: MagicMock
    ) -> None:
        """Đúng giờ → auto_update_at_time trả về UpdateResult."""
        mock_now = datetime(2024, 1, 15, 15, 30, 0)
        with patch("engine.data_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            result = scheduler.auto_update_at_time()
            assert result is not None
            assert result.total_symbols == 30
            mock_pipeline.update_all.assert_called_once()

    def test_auto_update_when_not_time(
        self, scheduler: DataScheduler, mock_pipeline: MagicMock
    ) -> None:
        """Chưa đến giờ → auto_update_at_time trả về None."""
        mock_now = datetime(2024, 1, 15, 10, 0, 0)
        with patch("engine.data_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            result = scheduler.auto_update_at_time()
            assert result is None
            mock_pipeline.update_all.assert_not_called()


class TestScheduleUpdate:
    """Test schedule_update method."""

    def test_schedule_update_resets_last_date(
        self, scheduler: DataScheduler
    ) -> None:
        """schedule_update phải reset _last_update_date."""
        scheduler._last_update_date = "2024-01-15"
        scheduler.schedule_update()
        assert scheduler._last_update_date is None

    def test_schedule_update_with_new_time(
        self, scheduler: DataScheduler
    ) -> None:
        """schedule_update với time mới phải cập nhật scheduled time."""
        scheduler.schedule_update("16:45")
        assert scheduler.get_next_update_time() == "16:45"

    def test_schedule_update_invalid_time_raises(
        self, scheduler: DataScheduler
    ) -> None:
        """schedule_update với time sai format phải raise ValueError."""
        with pytest.raises(ValueError):
            scheduler.schedule_update("invalid")


class TestSetUpdateTime:
    """Test set_update_time validation."""

    def test_valid_times(self, scheduler: DataScheduler) -> None:
        """Các format hợp lệ phải được accept."""
        valid_times = ["00:00", "09:30", "15:30", "23:59", "12:00"]
        for t in valid_times:
            scheduler.set_update_time(t)
            assert scheduler.get_next_update_time() == t

    def test_invalid_format_no_colon(self, scheduler: DataScheduler) -> None:
        """Thiếu dấu : phải raise."""
        with pytest.raises(ValueError, match="Invalid time format"):
            scheduler.set_update_time("1530")

    def test_invalid_format_single_digit(self, scheduler: DataScheduler) -> None:
        """Single digit hour phải raise."""
        with pytest.raises(ValueError, match="Invalid time format"):
            scheduler.set_update_time("9:30")

    def test_invalid_format_extra_chars(self, scheduler: DataScheduler) -> None:
        """Extra characters phải raise."""
        with pytest.raises(ValueError, match="Invalid time format"):
            scheduler.set_update_time("15:30:00")

    def test_empty_string_raises(self, scheduler: DataScheduler) -> None:
        """Empty string phải raise."""
        with pytest.raises(ValueError, match="Invalid time format"):
            scheduler.set_update_time("")


class TestCallback:
    """Test callback mechanism."""

    def test_set_callback(self, scheduler: DataScheduler) -> None:
        """set_on_update_complete phải lưu callback."""
        callback = MagicMock()
        scheduler.set_on_update_complete(callback)
        assert scheduler._on_update_complete is callback

    def test_callback_receives_update_result(
        self, scheduler: DataScheduler
    ) -> None:
        """Callback nhận đúng UpdateResult từ pipeline."""
        received_results = []

        def capture_result(result: UpdateResult) -> None:
            received_results.append(result)

        scheduler.set_on_update_complete(capture_result)
        scheduler.run_update()

        assert len(received_results) == 1
        assert received_results[0].total_symbols == 30
        assert received_results[0].success_count == 28

    def test_replace_callback(self, scheduler: DataScheduler) -> None:
        """Có thể thay thế callback bằng callback mới."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        scheduler.set_on_update_complete(callback1)
        scheduler.set_on_update_complete(callback2)
        scheduler.run_update()

        callback1.assert_not_called()
        callback2.assert_called_once()


class TestGetNextUpdateTime:
    """Test get_next_update_time."""

    def test_returns_current_scheduled_time(
        self, scheduler: DataScheduler
    ) -> None:
        """Trả về đúng update time đã set."""
        assert scheduler.get_next_update_time() == "15:30"

    def test_returns_updated_time(self, scheduler: DataScheduler) -> None:
        """Trả về time mới sau khi thay đổi."""
        scheduler.set_update_time("16:00")
        assert scheduler.get_next_update_time() == "16:00"
