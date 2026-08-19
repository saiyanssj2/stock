"""
Unit tests cho RecommendationEngine.

Kiểm tra các chức năng chính:
- scan_symbols(): scan VN30 + watchlist, generate recommendations
- refresh(): re-generate recommendations (hot-swap support)
- get_recommendations(): return cached recommendations
- has_strong_recommendations(): check confidence >= threshold
- get_tracked_symbols(): read VN30 + watchlist, deduplicate
- hot_swap_model(): swap model và regenerate
"""

import os
import tempfile
from datetime import date
from unittest.mock import patch

import pytest

from config.settings import CONFIDENCE_THRESHOLD, WATCHLIST_PATH
from config.vn_market_rules import VN30_SYMBOLS
from engine.recommendation_engine import RecommendationEngine
from models.recommendation_models import Action, Recommendation


class TestGetTrackedSymbols:
    """Test get_tracked_symbols() - đọc VN30 + watchlist."""

    def test_includes_all_vn30_symbols(self) -> None:
        """VN30 symbols luôn có trong tracked list."""
        engine = RecommendationEngine()
        tracked = engine.get_tracked_symbols()
        for symbol in VN30_SYMBOLS:
            assert symbol in tracked

    def test_includes_watchlist_symbols(self) -> None:
        """Symbols từ watchlist file cũng được include."""
        engine = RecommendationEngine()
        tracked = engine.get_tracked_symbols()
        # watchlist chứa FRT, PNJ, etc. mà không thuộc VN30
        assert "FRT" in tracked
        assert "PNJ" in tracked

    def test_no_duplicates(self) -> None:
        """Không có symbol trùng lặp trong tracked list."""
        engine = RecommendationEngine()
        tracked = engine.get_tracked_symbols()
        assert len(tracked) == len(set(tracked))

    def test_sorted_alphabetically(self) -> None:
        """Tracked list được sort theo alphabet."""
        engine = RecommendationEngine()
        tracked = engine.get_tracked_symbols()
        assert tracked == sorted(tracked)

    def test_handles_missing_watchlist_gracefully(self) -> None:
        """Khi file watchlist không tồn tại → chỉ dùng VN30."""
        with patch(
            "engine.recommendation_engine.WATCHLIST_PATH",
            "nonexistent/path/watchlist.txt",
        ):
            engine = RecommendationEngine()
            tracked = engine.get_tracked_symbols()
            # Phải có đúng VN30 symbols
            assert set(tracked) == set(VN30_SYMBOLS)


class TestScanSymbols:
    """Test scan_symbols() - generate recommendations."""

    def test_returns_recommendations_for_all_tracked(self) -> None:
        """Mỗi tracked symbol phải có đúng 1 recommendation."""
        engine = RecommendationEngine()
        recs = engine.scan_symbols()
        tracked = engine.get_tracked_symbols()
        rec_symbols = [r.symbol for r in recs]
        assert set(rec_symbols) == set(tracked)

    def test_sorted_by_confidence_descending(self) -> None:
        """Kết quả phải sort theo confidence giảm dần."""
        engine = RecommendationEngine()
        recs = engine.scan_symbols()
        for i in range(len(recs) - 1):
            assert recs[i].confidence >= recs[i + 1].confidence

    def test_confidence_bounded_0_1(self) -> None:
        """Confidence score nằm trong [0.0, 1.0]."""
        engine = RecommendationEngine()
        recs = engine.scan_symbols()
        for rec in recs:
            assert 0.0 <= rec.confidence <= 1.0

    def test_position_score_bounded(self) -> None:
        """Position score nằm trong [-1.0, 1.0]."""
        engine = RecommendationEngine()
        recs = engine.scan_symbols()
        for rec in recs:
            assert -1.0 <= rec.position_score <= 1.0

    def test_action_is_valid_enum(self) -> None:
        """Action phải là một trong BUY, HOLD, SELL."""
        engine = RecommendationEngine()
        recs = engine.scan_symbols()
        for rec in recs:
            assert rec.action in (Action.BUY, Action.HOLD, Action.SELL)

    def test_analysis_date_is_today(self) -> None:
        """Analysis date phải là ngày hôm nay."""
        engine = RecommendationEngine()
        recs = engine.scan_symbols()
        today = date.today()
        for rec in recs:
            assert rec.analysis_date == today

    def test_buy_date_after_analysis_date(self) -> None:
        """Buy date (T+1) phải sau analysis date (T)."""
        engine = RecommendationEngine()
        recs = engine.scan_symbols()
        for rec in recs:
            assert rec.buy_date > rec.analysis_date

    def test_earliest_sell_date_after_buy_date(self) -> None:
        """Earliest sell date phải sau buy date (T+2.5 rule → 3 trading days)."""
        engine = RecommendationEngine()
        recs = engine.scan_symbols()
        for rec in recs:
            assert rec.earliest_sell_date > rec.buy_date

    def test_model_version_set(self) -> None:
        """Model version phải được set (không rỗng)."""
        engine = RecommendationEngine()
        recs = engine.scan_symbols()
        for rec in recs:
            assert rec.model_version
            assert len(rec.model_version) > 0

    def test_caches_results(self) -> None:
        """Sau scan, get_recommendations() trả về kết quả đã cache."""
        engine = RecommendationEngine()
        recs = engine.scan_symbols()
        cached = engine.get_recommendations()
        assert recs == cached


class TestGetRecommendations:
    """Test get_recommendations() - cached recommendations."""

    def test_returns_empty_before_scan(self) -> None:
        """Trước khi scan, trả về list rỗng."""
        engine = RecommendationEngine()
        assert engine.get_recommendations() == []

    def test_returns_cached_after_scan(self) -> None:
        """Sau scan, trả về đúng kết quả đã cache."""
        engine = RecommendationEngine()
        engine.scan_symbols()
        recs = engine.get_recommendations()
        assert len(recs) > 0


class TestHasStrongRecommendations:
    """Test has_strong_recommendations() - check confidence threshold."""

    def test_returns_false_before_scan(self) -> None:
        """Trước khi scan, không có strong recommendations."""
        engine = RecommendationEngine()
        assert engine.has_strong_recommendations() is False

    def test_detects_strong_recommendations(self) -> None:
        """Khi có ít nhất 1 rec >= threshold → True."""
        engine = RecommendationEngine()
        engine.scan_symbols()
        # Với stub model và nhiều symbols, chắc chắn có rec >= 0.5
        assert engine.has_strong_recommendations() is True

    def test_no_strong_when_all_below_threshold(self) -> None:
        """Khi tất cả confidence < threshold → False."""
        engine = RecommendationEngine()
        # Manually set low-confidence recommendations
        engine._recommendations = [
            Recommendation(
                symbol="TEST",
                action=Action.HOLD,
                confidence=0.3,
                position_score=0.0,
                expected_holding_days=5,
                earliest_sell_date=date(2024, 1, 10),
                analysis_date=date(2024, 1, 5),
                buy_date=date(2024, 1, 6),
                model_version="test-v1",
            ),
            Recommendation(
                symbol="TEST2",
                action=Action.HOLD,
                confidence=0.1,
                position_score=-0.1,
                expected_holding_days=5,
                earliest_sell_date=date(2024, 1, 10),
                analysis_date=date(2024, 1, 5),
                buy_date=date(2024, 1, 6),
                model_version="test-v1",
            ),
        ]
        assert engine.has_strong_recommendations() is False


class TestRefresh:
    """Test refresh() - re-generate recommendations."""

    def test_refresh_updates_recommendations(self) -> None:
        """Refresh scan lại symbols và cập nhật cache."""
        engine = RecommendationEngine()
        engine.scan_symbols()
        first_recs = engine.get_recommendations()
        engine.refresh()
        refreshed_recs = engine.get_recommendations()
        # Vẫn có recommendations sau refresh
        assert len(refreshed_recs) > 0
        # Số lượng recommendations bằng nhau
        assert len(refreshed_recs) == len(first_recs)

    def test_refresh_keeps_sorted(self) -> None:
        """Sau refresh, danh sách vẫn sorted by confidence descending."""
        engine = RecommendationEngine()
        engine.refresh()
        recs = engine.get_recommendations()
        for i in range(len(recs) - 1):
            assert recs[i].confidence >= recs[i + 1].confidence


class TestHotSwapModel:
    """Test hot_swap_model() - swap model và regenerate."""

    def test_hot_swap_updates_model_version(self) -> None:
        """Hot swap sang model mới → model_version thay đổi."""
        engine = RecommendationEngine()
        old_version = engine._model_version

        # Tạo temp file giả làm model
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"fake model data")
            model_path = f.name

        try:
            engine.hot_swap_model(model_path)
            assert engine._model_version != old_version
        finally:
            os.unlink(model_path)

    def test_hot_swap_regenerates_recommendations(self) -> None:
        """Hot swap → recommendations được regenerate."""
        engine = RecommendationEngine()
        engine.scan_symbols()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"new model data")
            model_path = f.name

        try:
            engine.hot_swap_model(model_path)
            recs = engine.get_recommendations()
            # Vẫn có recommendations
            assert len(recs) > 0
            # Model version trong recs phải khớp version mới
            for rec in recs:
                assert rec.model_version == engine._model_version
        finally:
            os.unlink(model_path)


class TestDetermineAction:
    """Test _determine_action() - logic xác định hành động."""

    def test_buy_when_positive_score_high(self) -> None:
        """position_score > 0.3 → BUY."""
        engine = RecommendationEngine()
        assert engine._determine_action(0.5) == Action.BUY
        assert engine._determine_action(1.0) == Action.BUY

    def test_sell_when_negative_score_high(self) -> None:
        """position_score < -0.3 → SELL."""
        engine = RecommendationEngine()
        assert engine._determine_action(-0.5) == Action.SELL
        assert engine._determine_action(-1.0) == Action.SELL

    def test_hold_when_score_neutral(self) -> None:
        """position_score trong [-0.3, 0.3] → HOLD."""
        engine = RecommendationEngine()
        assert engine._determine_action(0.0) == Action.HOLD
        assert engine._determine_action(0.2) == Action.HOLD
        assert engine._determine_action(-0.2) == Action.HOLD

    def test_boundary_values(self) -> None:
        """Boundary: 0.3 → HOLD (not > 0.3), -0.3 → HOLD (not < -0.3)."""
        engine = RecommendationEngine()
        assert engine._determine_action(0.3) == Action.HOLD
        assert engine._determine_action(-0.3) == Action.HOLD


class TestModelVersion:
    """Test model version tracking."""

    def test_default_stub_version(self) -> None:
        """Khi không có model path → dùng stub version."""
        engine = RecommendationEngine()
        assert engine._model_version == "stub-v1"

    def test_version_from_file(self) -> None:
        """Khi có model file → version dựa trên file hash."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"model content")
            model_path = f.name

        try:
            engine = RecommendationEngine(model_path=model_path)
            assert engine._model_version != "stub-v1"
            assert len(engine._model_version) == 8  # MD5 truncated 8 chars
        finally:
            os.unlink(model_path)

    def test_version_changes_when_file_changes(self) -> None:
        """Model version thay đổi khi nội dung file thay đổi."""
        with tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False, mode="wb"
        ) as f:
            f.write(b"version 1")
            model_path = f.name

        try:
            engine = RecommendationEngine(model_path=model_path)
            v1 = engine._model_version

            # Ghi nội dung mới vào file
            with open(model_path, "wb") as f:
                f.write(b"version 2 with different content")

            engine.refresh()
            v2 = engine._model_version
            assert v1 != v2
        finally:
            os.unlink(model_path)
