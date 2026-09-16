"""
Unit tests cho DataPipeline.

Kiểm tra các chức năng: get_tracked_symbols, validate_data,
update_symbol, update_all, estimate_update_duration.
"""

import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from engine.data_pipeline import DataPipeline, REQUIRED_COLUMNS
from models.data_models import UpdateResult


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Tạo thư mục data tạm với file CSV hợp lệ."""
    # Tạo CSV hợp lệ cho FPT
    df = pd.DataFrame({
        "time": ["2024-01-01", "2024-01-02"],
        "open": [100.0, 101.0],
        "high": [105.0, 106.0],
        "low": [99.0, 100.0],
        "close": [103.0, 104.0],
        "volume": [1000, 2000],
    })
    df.to_csv(tmp_path / "FPT.csv", index=False)
    df.to_csv(tmp_path / "VNM.csv", index=False)
    return str(tmp_path)


@pytest.fixture
def pipeline(tmp_data_dir):
    """Tạo DataPipeline instance với thư mục data tạm."""
    return DataPipeline(data_dir=tmp_data_dir)


class TestGetTrackedSymbols:
    """Test get_tracked_symbols() đọc VN30 + watchlist."""

    def test_returns_vn30_when_no_watchlist(self, tmp_path) -> None:
        """Trả về VN30 symbols khi watchlist file không tồn tại."""
        with patch("engine.data_pipeline.WATCHLIST_PATH", str(tmp_path / "nonexistent.txt")):
            dp = DataPipeline(data_dir=str(tmp_path))
            symbols = dp.get_tracked_symbols()
            # Phải chứa tất cả VN30
            from config.vn_market_rules import VN30_SYMBOLS
            for s in VN30_SYMBOLS:
                assert s in symbols

    def test_includes_watchlist_symbols(self, tmp_path) -> None:
        """Kết hợp VN30 với symbols từ watchlist file."""
        watchlist = tmp_path / "code.txt"
        watchlist.write_text("FOX, GEL, KSF\n")
        with patch("engine.data_pipeline.WATCHLIST_PATH", str(watchlist)):
            dp = DataPipeline(data_dir=str(tmp_path))
            symbols = dp.get_tracked_symbols()
            assert "FOX" in symbols
            assert "GEL" in symbols
            assert "KSF" in symbols

    def test_deduplicates_symbols(self, tmp_path) -> None:
        """Loại bỏ symbol trùng lặp giữa VN30 và watchlist."""
        watchlist = tmp_path / "code.txt"
        # FPT đã có trong VN30
        watchlist.write_text("FPT, FPT, ACB\n")
        with patch("engine.data_pipeline.WATCHLIST_PATH", str(watchlist)):
            dp = DataPipeline(data_dir=str(tmp_path))
            symbols = dp.get_tracked_symbols()
            # Không có duplicate
            assert len(symbols) == len(set(symbols))

    def test_excludes_market_indices(self, tmp_path) -> None:
        """Loại trừ VNINDEX, VN30, HNX, UPCOM khỏi kết quả."""
        watchlist = tmp_path / "code.txt"
        watchlist.write_text("VNINDEX, VN30, HNX, UPCOM, FPT\n")
        with patch("engine.data_pipeline.WATCHLIST_PATH", str(watchlist)):
            dp = DataPipeline(data_dir=str(tmp_path))
            symbols = dp.get_tracked_symbols()
            assert "VNINDEX" not in symbols
            assert "HNX" not in symbols
            assert "UPCOM" not in symbols

    def test_sorted_alphabetically(self, tmp_path) -> None:
        """Kết quả được sắp xếp theo alphabet."""
        watchlist = tmp_path / "code.txt"
        watchlist.write_text("ZZZ, AAA, MMM\n")
        with patch("engine.data_pipeline.WATCHLIST_PATH", str(watchlist)):
            dp = DataPipeline(data_dir=str(tmp_path))
            symbols = dp.get_tracked_symbols()
            assert symbols == sorted(symbols)


class TestValidateData:
    """Test validate_data() kiểm tra required OHLCV columns."""

    def test_valid_dataframe(self, pipeline) -> None:
        """DataFrame có đủ required columns → True."""
        df = pd.DataFrame({
            "time": ["2024-01-01"],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "volume": [1000],
        })
        assert pipeline.validate_data(df) is True

    def test_valid_with_extra_columns(self, pipeline) -> None:
        """DataFrame có extra columns vẫn valid."""
        df = pd.DataFrame({
            "time": ["2024-01-01"],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "volume": [1000],
            "EMA_9": [101.0],
        })
        assert pipeline.validate_data(df) is True

    def test_missing_column(self, pipeline) -> None:
        """DataFrame thiếu column → False."""
        df = pd.DataFrame({
            "time": ["2024-01-01"],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            # Thiếu close và volume
        })
        assert pipeline.validate_data(df) is False

    def test_empty_dataframe(self, pipeline) -> None:
        """DataFrame rỗng → False."""
        df = pd.DataFrame()
        assert pipeline.validate_data(df) is False

    def test_none_dataframe(self, pipeline) -> None:
        """None input → False."""
        assert pipeline.validate_data(None) is False

    def test_case_insensitive_columns(self, pipeline) -> None:
        """Column names case-insensitive."""
        df = pd.DataFrame({
            "Time": ["2024-01-01"],
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [103.0],
            "Volume": [1000],
        })
        assert pipeline.validate_data(df) is True


class TestUpdateSymbol:
    """Test update_symbol() kiểm tra CSV tồn tại và valid."""

    def test_valid_csv_returns_true(self, pipeline, tmp_data_dir) -> None:
        """Symbol có CSV hợp lệ → True."""
        assert pipeline.update_symbol("FPT") is True

    def test_missing_csv_returns_false(self, pipeline) -> None:
        """Symbol không có file CSV → False."""
        assert pipeline.update_symbol("NONEXIST") is False

    def test_invalid_csv_returns_false(self, pipeline, tmp_data_dir) -> None:
        """CSV không có required columns → False."""
        # Tạo CSV thiếu columns
        invalid_path = os.path.join(tmp_data_dir, "BAD.csv")
        pd.DataFrame({"col1": [1], "col2": [2]}).to_csv(invalid_path, index=False)
        assert pipeline.update_symbol("BAD") is False

    def test_empty_csv_returns_false(self, pipeline, tmp_data_dir) -> None:
        """CSV rỗng → False."""
        empty_path = os.path.join(tmp_data_dir, "EMPTY.csv")
        with open(empty_path, "w") as f:
            f.write("")
        assert pipeline.update_symbol("EMPTY") is False


class TestUpdateAll:
    """Test update_all() batch processing với fail-fast mode."""

    def test_returns_update_result(self, pipeline) -> None:
        """Trả về UpdateResult với thông tin batch."""
        result = pipeline.update_all()
        assert result.total_symbols > 0
        assert result.success_count + len(result.failed_symbols) <= result.total_symbols
        assert result.duration_seconds >= 0.0

    def test_fail_fast_stops_on_first_failure(self, tmp_path) -> None:
        """Khi một symbol fail, batch dừng ngay (fail-fast mode)."""
        # Tạo 1 CSV invalid (AAA - sẽ fail trước theo alphabet) và 1 valid (ZZZ)
        valid_df = pd.DataFrame({
            "time": ["2024-01-01"],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "volume": [1000],
        })
        pd.DataFrame({"x": [1]}).to_csv(tmp_path / "AAA.csv", index=False)  # Invalid - fail đầu tiên
        valid_df.to_csv(tmp_path / "ZZZ.csv", index=False)  # Valid - không được xử lý

        # Chỉ track AAA và ZZZ
        with patch("engine.data_pipeline.VN30_SYMBOLS", ["AAA", "ZZZ"]):
            with patch("engine.data_pipeline.WATCHLIST_PATH", str(tmp_path / "none.txt")):
                dp = DataPipeline(data_dir=str(tmp_path))
                result = dp.update_all()
                # AAA fail → dừng ngay, ZZZ không được xử lý
                assert result.success_count == 0
                assert "AAA" in result.failed_symbols
                # Chỉ có 1 failed symbol vì dừng ngay
                assert len(result.failed_symbols) == 1

    def test_all_success_when_all_valid(self, tmp_path) -> None:
        """Khi tất cả symbols hợp lệ, batch hoàn thành."""
        valid_df = pd.DataFrame({
            "time": ["2024-01-01"],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "volume": [1000],
        })
        valid_df.to_csv(tmp_path / "AAA.csv", index=False)
        valid_df.to_csv(tmp_path / "BBB.csv", index=False)

        with patch("engine.data_pipeline.VN30_SYMBOLS", ["AAA", "BBB"]):
            with patch("engine.data_pipeline.WATCHLIST_PATH", str(tmp_path / "none.txt")):
                dp = DataPipeline(data_dir=str(tmp_path))
                result = dp.update_all()
                assert result.success_count == 2
                assert len(result.failed_symbols) == 0

    def test_callback_called_for_each_symbol(self, tmp_path) -> None:
        """Callback được gọi cho mỗi symbol với progress percent."""
        valid_df = pd.DataFrame({
            "time": ["2024-01-01"],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "volume": [1000],
        })
        valid_df.to_csv(tmp_path / "AAA.csv", index=False)
        valid_df.to_csv(tmp_path / "BBB.csv", index=False)

        with patch("engine.data_pipeline.VN30_SYMBOLS", ["AAA", "BBB"]):
            with patch("engine.data_pipeline.WATCHLIST_PATH", str(tmp_path / "none.txt")):
                dp = DataPipeline(data_dir=str(tmp_path))

                callback_calls = []
                def mock_callback(symbol: str, pct: float) -> None:
                    callback_calls.append((symbol, pct))

                dp.update_all(callback=mock_callback)
                assert len(callback_calls) == 2
                # Progress phải tăng dần
                assert callback_calls[0][1] == 50.0
                assert callback_calls[1][1] == 100.0

    def test_no_callback_still_works(self, pipeline) -> None:
        """update_all() hoạt động bình thường khi không có callback."""
        result = pipeline.update_all(callback=None)
        assert isinstance(result, UpdateResult)


class TestEstimateUpdateDuration:
    """Test estimate_update_duration() tính thời gian ước tính."""

    def test_exact_division(self, pipeline) -> None:
        """20 symbols / 20 per minute = 1 phút."""
        assert pipeline.estimate_update_duration(20) == 1.0

    def test_ceil_division(self, pipeline) -> None:
        """21 symbols / 20 per minute = ceil = 2 phút."""
        assert pipeline.estimate_update_duration(21) == 2.0

    def test_zero_symbols(self, pipeline) -> None:
        """0 symbols → 0 phút."""
        assert pipeline.estimate_update_duration(0) == 0.0

    def test_negative_symbols(self, pipeline) -> None:
        """Số âm → 0 phút."""
        assert pipeline.estimate_update_duration(-5) == 0.0

    def test_large_number(self, pipeline) -> None:
        """100 symbols / 20 per minute = 5 phút."""
        assert pipeline.estimate_update_duration(100) == 5.0
