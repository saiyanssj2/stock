# -*- coding: utf-8 -*-
"""
Integration test: Data Update Flow end-to-end.

Kiểm tra luồng data update hoàn chỉnh:
- fetch → validate → refresh recommendations

Sử dụng tmp_path cho file I/O, mock cho API calls bên ngoài.

Requirements: 8.1 (data update mechanism)
"""

import json
import os
from datetime import date
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from engine.data_pipeline import DataPipeline
from engine.data_update_wiring import (
    load_last_update_timestamp,
    run_data_update_with_refresh,
    save_last_update_timestamp,
    wire_scheduler_to_recommendation,
)
from engine.recommendation_engine import RecommendationEngine
from models.data_models import UpdateResult


def _create_valid_csv(path: Path, symbol: str) -> None:
    """Tạo file CSV hợp lệ với đầy đủ cột OHLCV."""
    df = pd.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "open": [80.0, 81.0, 82.0],
            "high": [82.0, 83.0, 84.0],
            "low": [79.0, 80.0, 81.0],
            "close": [81.0, 82.0, 83.0],
            "volume": [1000000, 1100000, 1200000],
        }
    )
    csv_path = path / f"{symbol}.csv"
    df.to_csv(csv_path, index=False)


def _create_invalid_csv(path: Path, symbol: str) -> None:
    """Tạo file CSV thiếu cột (invalid)."""
    df = pd.DataFrame(
        {
            "time": ["2024-01-02"],
            "open": [80.0],
            # Thiếu high, low, close, volume
        }
    )
    csv_path = path / f"{symbol}.csv"
    df.to_csv(csv_path, index=False)


class TestDataUpdateFlowEndToEnd:
    """Test luồng data update: fetch → validate → refresh recommendations."""

    def test_update_all_with_valid_data(self, tmp_path: Path) -> None:
        """update_all() thành công khi tất cả CSV files hợp lệ."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Tạo CSV hợp lệ cho 3 symbols
        symbols = ["FPT", "VNM", "HPG"]
        for sym in symbols:
            _create_valid_csv(data_dir, sym)

        # Tạo watchlist file
        code_file = data_dir / "code.txt"
        code_file.write_text("FPT,VNM,HPG", encoding="utf-8")

        # Patch settings để dùng tmp_path
        with patch("engine.data_pipeline.WATCHLIST_PATH", str(code_file)):
            with patch("engine.data_pipeline.VN30_SYMBOLS", []):
                pipeline = DataPipeline(data_dir=str(data_dir))
                result = pipeline.update_all()

        assert result.total_symbols == 3
        assert result.success_count == 3
        assert len(result.failed_symbols) == 0

    def test_update_all_partial_failure_isolation(self, tmp_path: Path) -> None:
        """Partial failure: symbol fail không abort batch."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # 2 valid + 1 invalid CSV + 1 missing file
        _create_valid_csv(data_dir, "FPT")
        _create_valid_csv(data_dir, "VNM")
        _create_invalid_csv(data_dir, "BAD")
        # "XYZ" không có CSV file → sẽ fail

        code_file = data_dir / "code.txt"
        code_file.write_text("FPT,VNM,BAD,XYZ", encoding="utf-8")

        with patch("engine.data_pipeline.WATCHLIST_PATH", str(code_file)):
            with patch("engine.data_pipeline.VN30_SYMBOLS", []):
                pipeline = DataPipeline(data_dir=str(data_dir))
                result = pipeline.update_all()

        # 2 thành công, 2 thất bại
        assert result.total_symbols == 4
        assert result.success_count == 2
        assert len(result.failed_symbols) == 2
        assert "BAD" in result.failed_symbols
        assert "XYZ" in result.failed_symbols

    def test_data_validation_requires_all_columns(self, tmp_path: Path) -> None:
        """validate_data() kiểm tra đầy đủ cột OHLCV."""
        pipeline = DataPipeline(data_dir=str(tmp_path))

        # DataFrame hợp lệ
        valid_df = pd.DataFrame(
            {
                "time": ["2024-01-02"],
                "open": [80.0],
                "high": [82.0],
                "low": [79.0],
                "close": [81.0],
                "volume": [1000000],
            }
        )
        assert pipeline.validate_data(valid_df) is True

        # DataFrame thiếu cột
        invalid_df = pd.DataFrame(
            {
                "time": ["2024-01-02"],
                "open": [80.0],
                "close": [81.0],
            }
        )
        assert pipeline.validate_data(invalid_df) is False

        # DataFrame rỗng
        empty_df = pd.DataFrame()
        assert pipeline.validate_data(empty_df) is False

    def test_data_update_triggers_recommendation_refresh(
        self, tmp_path: Path
    ) -> None:
        """Sau data update → RecommendationEngine.refresh() được gọi."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        status_path = str(tmp_path / "last_update.json")

        # Tạo valid CSV
        _create_valid_csv(data_dir, "FPT")
        code_file = data_dir / "code.txt"
        code_file.write_text("FPT", encoding="utf-8")

        with patch("engine.data_pipeline.WATCHLIST_PATH", str(code_file)):
            with patch("engine.data_pipeline.VN30_SYMBOLS", []):
                pipeline = DataPipeline(data_dir=str(data_dir))

                # Mock recommendation engine
                mock_rec_engine = MagicMock(spec=RecommendationEngine)

                # Chạy update + refresh
                result = run_data_update_with_refresh(
                    pipeline=pipeline,
                    recommendation_engine=mock_rec_engine,
                    status_path=status_path,
                )

        # Verify update thành công
        assert result.success_count >= 1

        # Verify recommendation refresh được gọi
        mock_rec_engine.refresh.assert_called_once()

    def test_data_update_saves_timestamp(self, tmp_path: Path) -> None:
        """Sau data update → timestamp được lưu ra file JSON."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        status_path = str(tmp_path / "status" / "last_update.json")

        _create_valid_csv(data_dir, "FPT")
        code_file = data_dir / "code.txt"
        code_file.write_text("FPT", encoding="utf-8")

        with patch("engine.data_pipeline.WATCHLIST_PATH", str(code_file)):
            with patch("engine.data_pipeline.VN30_SYMBOLS", []):
                pipeline = DataPipeline(data_dir=str(data_dir))
                mock_rec_engine = MagicMock(spec=RecommendationEngine)

                run_data_update_with_refresh(
                    pipeline=pipeline,
                    recommendation_engine=mock_rec_engine,
                    status_path=status_path,
                )

        # Verify timestamp file tồn tại
        assert os.path.exists(status_path)

        # Load và verify nội dung
        info = load_last_update_timestamp(path=status_path)
        assert info is not None
        assert "timestamp" in info
        assert info["total_symbols"] == 1
        assert info["success_count"] == 1

    def test_partial_failure_still_triggers_refresh(self, tmp_path: Path) -> None:
        """Partial failure trong data fetch vẫn trigger recommendation refresh."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        status_path = str(tmp_path / "last_update.json")

        # 1 valid + 1 missing file → partial failure
        _create_valid_csv(data_dir, "FPT")
        # "XYZ" không có CSV file → sẽ fail
        code_file = data_dir / "code.txt"
        code_file.write_text("FPT,XYZ", encoding="utf-8")

        with patch("engine.data_pipeline.WATCHLIST_PATH", str(code_file)):
            with patch("engine.data_pipeline.VN30_SYMBOLS", []):
                pipeline = DataPipeline(data_dir=str(data_dir))
                mock_rec_engine = MagicMock(spec=RecommendationEngine)

                result = run_data_update_with_refresh(
                    pipeline=pipeline,
                    recommendation_engine=mock_rec_engine,
                    status_path=status_path,
                )

        # Có failure nhưng refresh vẫn gọi (Req 8.3)
        assert len(result.failed_symbols) > 0
        mock_rec_engine.refresh.assert_called_once()

    def test_recommendation_refresh_uses_new_data(self, tmp_path: Path) -> None:
        """Recommendation engine dùng data mới sau refresh."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Tạo CSV data
        _create_valid_csv(data_dir, "FPT")
        _create_valid_csv(data_dir, "VNM")
        code_file = data_dir / "code.txt"
        code_file.write_text("FPT,VNM", encoding="utf-8")

        with patch("engine.data_pipeline.WATCHLIST_PATH", str(code_file)):
            with patch("engine.data_pipeline.VN30_SYMBOLS", []):
                pipeline = DataPipeline(data_dir=str(data_dir))

                # Dùng real recommendation engine (stub mode)
                rec_engine = RecommendationEngine(model_path=None)

                # Initial scan
                initial_recs = rec_engine.scan_symbols()

                # Simulate data update + refresh
                status_path = str(tmp_path / "last_update.json")

                with patch(
                    "engine.recommendation_engine.WATCHLIST_PATH",
                    str(code_file),
                ):
                    with patch(
                        "engine.recommendation_engine.VN30_SYMBOLS", []
                    ):
                        result = run_data_update_with_refresh(
                            pipeline=pipeline,
                            recommendation_engine=rec_engine,
                            status_path=status_path,
                        )

                # Verify refresh đã tạo recommendations mới
                refreshed_recs = rec_engine.get_recommendations()
                assert len(refreshed_recs) > 0

    def test_wire_scheduler_callback(self, tmp_path: Path) -> None:
        """wire_scheduler_to_recommendation thiết lập callback đúng."""
        status_path = str(tmp_path / "last_update.json")

        # Mock scheduler
        mock_scheduler = MagicMock()
        mock_rec_engine = MagicMock(spec=RecommendationEngine)

        # Wire callback
        wire_scheduler_to_recommendation(
            scheduler=mock_scheduler,
            recommendation_engine=mock_rec_engine,
            status_path=status_path,
        )

        # Verify scheduler.set_on_update_complete được gọi
        mock_scheduler.set_on_update_complete.assert_called_once()

        # Lấy callback đã register
        callback = mock_scheduler.set_on_update_complete.call_args[0][0]

        # Gọi callback simulate update hoàn tất
        mock_result = UpdateResult(
            total_symbols=5,
            success_count=4,
            failed_symbols=["BAD"],
            errors={"BAD": "fetch failed"},
            duration_seconds=30.0,
        )
        callback(mock_result)

        # Verify timestamp saved
        info = load_last_update_timestamp(path=status_path)
        assert info is not None
        assert info["total_symbols"] == 5
        assert info["success_count"] == 4

        # Verify recommendation engine refresh
        mock_rec_engine.refresh.assert_called_once()

    def test_save_load_timestamp_roundtrip(self, tmp_path: Path) -> None:
        """save/load timestamp JSON round-trip hoạt động đúng."""
        status_path = str(tmp_path / "status" / "update.json")

        result = UpdateResult(
            total_symbols=10,
            success_count=8,
            failed_symbols=["X", "Y"],
            errors={"X": "timeout", "Y": "invalid data"},
            duration_seconds=45.5,
        )

        # Save
        save_last_update_timestamp(result, path=status_path)

        # Load
        loaded = load_last_update_timestamp(path=status_path)
        assert loaded is not None
        assert loaded["total_symbols"] == 10
        assert loaded["success_count"] == 8
        assert loaded["failed_count"] == 2
        assert loaded["failed_symbols"] == ["X", "Y"]
        assert loaded["duration_seconds"] == 45.5
        assert "timestamp" in loaded

    def test_load_nonexistent_timestamp_returns_none(self, tmp_path: Path) -> None:
        """load_last_update_timestamp() trả về None khi file chưa tồn tại."""
        path = str(tmp_path / "nonexistent" / "file.json")
        result = load_last_update_timestamp(path=path)
        assert result is None

    def test_update_callback_receives_progress(self, tmp_path: Path) -> None:
        """update_all() gọi callback với progress percentage đúng."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        _create_valid_csv(data_dir, "A")
        _create_valid_csv(data_dir, "B")
        _create_valid_csv(data_dir, "C")

        code_file = data_dir / "code.txt"
        code_file.write_text("A,B,C", encoding="utf-8")

        progress_calls: list = []

        def on_progress(symbol: str, pct: float) -> None:
            progress_calls.append((symbol, pct))

        with patch("engine.data_pipeline.WATCHLIST_PATH", str(code_file)):
            with patch("engine.data_pipeline.VN30_SYMBOLS", []):
                pipeline = DataPipeline(data_dir=str(data_dir))
                pipeline.update_all(callback=on_progress)

        # 3 symbols → 3 callback calls
        assert len(progress_calls) == 3

        # Progress phải kết thúc ở 100%
        last_pct = progress_calls[-1][1]
        assert abs(last_pct - 100.0) < 0.01

        # Progress phải tăng monotonic
        pcts = [p[1] for p in progress_calls]
        for i in range(1, len(pcts)):
            assert pcts[i] >= pcts[i - 1]

    def test_estimate_update_duration(self, tmp_path: Path) -> None:
        """estimate_update_duration tính đúng ETA dựa trên rate limit."""
        pipeline = DataPipeline(data_dir=str(tmp_path))

        # 20 symbols / 20 req/min = 1 phút
        assert pipeline.estimate_update_duration(20) == 1.0

        # 21 symbols → ceil(21/20) = 2 phút
        assert pipeline.estimate_update_duration(21) == 2.0

        # 0 symbols → 0 phút
        assert pipeline.estimate_update_duration(0) == 0.0

        # 1 symbol → 1 phút
        assert pipeline.estimate_update_duration(1) == 1.0
