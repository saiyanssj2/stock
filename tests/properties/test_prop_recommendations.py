# -*- coding: utf-8 -*-
"""
Property-based tests cho RecommendationEngine.

Kiểm tra 4 properties:
- Property 4: Recommendations sorted descending by confidence
- Property 14: Symbol filtering — output symbols ∈ VN30 ∪ watchlist
- Property 15: Confidence score bounded — confidence ∈ [0.0, 1.0]
- Property 16: No strong recommendations threshold — all < 0.5 → "no strong recommendations"

# Feature: stock-trading-platform-refactor, Property 4: Recommendations sorted by strength
# Feature: stock-trading-platform-refactor, Property 14: Symbol filtering
# Feature: stock-trading-platform-refactor, Property 15: Confidence score bounded
# Feature: stock-trading-platform-refactor, Property 16: No strong recommendations threshold
"""

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from config.vn_market_rules import VN30_SYMBOLS
from config.settings import CONFIDENCE_THRESHOLD
from engine.recommendation_engine import RecommendationEngine
from models.recommendation_models import Action, Recommendation


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy cho watchlist symbols — mã cổ phiếu hợp lệ (3-4 ký tự alpha uppercase)
watchlist_symbol_strategy = st.text(
    alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    min_size=3,
    max_size=4,
).filter(lambda s: s.isalpha())

# Strategy sinh danh sách watchlist symbols (có thể rỗng hoặc có giá trị)
watchlist_strategy = st.lists(
    watchlist_symbol_strategy,
    min_size=0,
    max_size=10,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_with_watchlist(watchlist: list[str], fn):
    """
    Tạo temp file chứa watchlist, patch WATCHLIST_PATH, và gọi fn(engine).

    Dùng tempfile thay vì tmp_path fixture để tương thích với Hypothesis @given.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(",".join(watchlist))
        tmp_path = f.name

    try:
        with patch("engine.recommendation_engine.WATCHLIST_PATH", tmp_path):
            engine = RecommendationEngine(model_path=None)
            return fn(engine)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestRecommendationsSortedByStrength:
    """
    Property 4: Recommendations sorted by strength.

    Với bất kỳ tập symbols nào, scan_symbols() luôn trả về danh sách
    sorted descending by confidence.

    **Validates: Requirements 2.4**
    """

    @given(watchlist=watchlist_strategy)
    @settings(max_examples=100)
    def test_scan_symbols_returns_sorted_descending(self, watchlist):
        """
        Property 4: scan_symbols() luôn trả về danh sách sorted descending
        by confidence cho bất kỳ tổ hợp watchlist nào.

        **Validates: Requirements 2.4**
        """
        def check(engine):
            recommendations = engine.scan_symbols()
            # Danh sách phải sorted descending by confidence
            if len(recommendations) > 1:
                for i in range(len(recommendations) - 1):
                    assert recommendations[i].confidence >= recommendations[i + 1].confidence, (
                        f"Recommendations không sorted descending: "
                        f"index {i} confidence={recommendations[i].confidence} < "
                        f"index {i+1} confidence={recommendations[i+1].confidence}"
                    )

        _run_with_watchlist(watchlist, check)

    @given(watchlist=watchlist_strategy)
    @settings(max_examples=100)
    def test_get_recommendations_after_scan_is_sorted(self, watchlist):
        """
        Property 4 (cached): get_recommendations() sau scan cũng sorted descending.

        **Validates: Requirements 2.4**
        """
        def check(engine):
            engine.scan_symbols()
            recommendations = engine.get_recommendations()
            if len(recommendations) > 1:
                for i in range(len(recommendations) - 1):
                    assert recommendations[i].confidence >= recommendations[i + 1].confidence, (
                        f"Cached recommendations không sorted descending: "
                        f"index {i}={recommendations[i].confidence} < "
                        f"index {i+1}={recommendations[i+1].confidence}"
                    )

        _run_with_watchlist(watchlist, check)


class TestSymbolFiltering:
    """
    Property 14: Symbol filtering.

    Mọi symbol xuất hiện trong output phải thuộc VN30 ∪ watchlist.

    **Validates: Requirements 6.1**
    """

    @given(watchlist=watchlist_strategy)
    @settings(max_examples=100)
    def test_all_symbols_in_vn30_union_watchlist(self, watchlist):
        """
        Property 14: Mọi symbol trong recommendations phải thuộc VN30 ∪ watchlist.

        **Validates: Requirements 6.1**
        """
        # Tập hợp symbols hợp lệ = VN30 ∪ watchlist
        valid_symbols = set(VN30_SYMBOLS) | set(watchlist)

        def check(engine):
            recommendations = engine.scan_symbols()
            for rec in recommendations:
                assert rec.symbol in valid_symbols, (
                    f"Symbol '{rec.symbol}' không thuộc VN30 ∪ watchlist. "
                    f"VN30 có {len(VN30_SYMBOLS)} symbols, "
                    f"watchlist có {len(watchlist)} symbols."
                )

        _run_with_watchlist(watchlist, check)

    @given(watchlist=watchlist_strategy)
    @settings(max_examples=100)
    def test_no_symbols_outside_tracked_set(self, watchlist):
        """
        Property 14 (negative): Không có symbol nào ngoài tracked set.

        **Validates: Requirements 6.1**
        """
        def check(engine):
            tracked = set(engine.get_tracked_symbols())
            recommendations = engine.scan_symbols()
            output_symbols = {rec.symbol for rec in recommendations}
            assert output_symbols <= tracked, (
                f"Symbols ngoài tracked set: {output_symbols - tracked}"
            )

        _run_with_watchlist(watchlist, check)


class TestConfidenceScoreBounded:
    """
    Property 15: Confidence score bounded.

    Mọi confidence value phải nằm trong [0.0, 1.0].

    **Validates: Requirements 6.2**
    """

    @given(watchlist=watchlist_strategy)
    @settings(max_examples=100)
    def test_all_confidence_scores_in_valid_range(self, watchlist):
        """
        Property 15: confidence ∈ [0.0, 1.0] cho mọi recommendation.

        **Validates: Requirements 6.2**
        """
        def check(engine):
            recommendations = engine.scan_symbols()
            for rec in recommendations:
                assert 0.0 <= rec.confidence <= 1.0, (
                    f"Confidence {rec.confidence} cho symbol '{rec.symbol}' "
                    f"ngoài phạm vi [0.0, 1.0]"
                )

        _run_with_watchlist(watchlist, check)

    @given(
        watchlist=watchlist_strategy,
        model_seed=st.text(min_size=1, max_size=20),
    )
    @settings(max_examples=100)
    def test_confidence_bounded_with_different_model_versions(
        self, watchlist, model_seed
    ):
        """
        Property 15 (model variation): Confidence vẫn bounded khi model version thay đổi.

        **Validates: Requirements 6.2**
        """
        def check(engine):
            # Thay đổi model version để sinh confidence khác
            engine._model_version = model_seed
            recommendations = engine.scan_symbols()
            for rec in recommendations:
                assert 0.0 <= rec.confidence <= 1.0, (
                    f"Confidence {rec.confidence} ngoài [0.0, 1.0] "
                    f"với model_version='{model_seed}', symbol='{rec.symbol}'"
                )

        _run_with_watchlist(watchlist, check)


class TestNoStrongRecommendationsThreshold:
    """
    Property 16: No strong recommendations threshold.

    Khi tất cả confidence < 0.5, has_strong_recommendations() trả về False.

    **Validates: Requirements 6.4**
    """

    @given(
        confidences=st.lists(
            st.floats(min_value=0.0, max_value=0.499),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=100)
    def test_all_below_threshold_returns_no_strong(self, confidences):
        """
        Property 16: Khi tất cả confidence < 0.5, has_strong_recommendations() = False.

        **Validates: Requirements 6.4**
        """
        # Arrange - Tạo engine và manually set recommendations với confidence thấp
        engine = RecommendationEngine(model_path=None)

        fake_recs = []
        for i, conf in enumerate(confidences):
            rec = Recommendation(
                symbol=f"SYM{i}",
                action=Action.HOLD,
                confidence=conf,
                position_score=0.0,
                expected_holding_days=5,
                earliest_sell_date=date(2024, 1, 10),
                analysis_date=date(2024, 1, 5),
                buy_date=date(2024, 1, 6),
                model_version="test-v1",
            )
            fake_recs.append(rec)

        engine._recommendations = fake_recs

        # Act & Assert
        assert engine.has_strong_recommendations() is False, (
            f"has_strong_recommendations() trả về True nhưng tất cả "
            f"confidence < {CONFIDENCE_THRESHOLD}: "
            f"{[r.confidence for r in fake_recs]}"
        )

    @given(
        low_confidences=st.lists(
            st.floats(min_value=0.0, max_value=0.499),
            min_size=0,
            max_size=10,
        ),
        high_confidence=st.floats(min_value=0.5, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_at_least_one_above_threshold_returns_strong(
        self, low_confidences, high_confidence
    ):
        """
        Property 16 (converse): Khi có ít nhất 1 confidence >= 0.5,
        has_strong_recommendations() = True.

        **Validates: Requirements 6.4**
        """
        # Arrange
        engine = RecommendationEngine(model_path=None)

        fake_recs = []
        for i, conf in enumerate(low_confidences):
            rec = Recommendation(
                symbol=f"LOW{i}",
                action=Action.HOLD,
                confidence=conf,
                position_score=0.0,
                expected_holding_days=5,
                earliest_sell_date=date(2024, 1, 10),
                analysis_date=date(2024, 1, 5),
                buy_date=date(2024, 1, 6),
                model_version="test-v1",
            )
            fake_recs.append(rec)

        # Thêm 1 recommendation confidence cao (>= threshold)
        high_rec = Recommendation(
            symbol="HIGH0",
            action=Action.BUY,
            confidence=high_confidence,
            position_score=0.8,
            expected_holding_days=3,
            earliest_sell_date=date(2024, 1, 10),
            analysis_date=date(2024, 1, 5),
            buy_date=date(2024, 1, 6),
            model_version="test-v1",
        )
        fake_recs.append(high_rec)

        engine._recommendations = fake_recs

        # Act & Assert
        assert engine.has_strong_recommendations() is True, (
            f"has_strong_recommendations() trả về False nhưng có "
            f"confidence={high_confidence} >= {CONFIDENCE_THRESHOLD}"
        )

    def test_empty_recommendations_returns_no_strong(self):
        """
        Property 16 (edge case): Khi không có recommendation nào,
        has_strong_recommendations() = False.

        **Validates: Requirements 6.4**
        """
        # Arrange
        engine = RecommendationEngine(model_path=None)
        engine._recommendations = []

        # Act & Assert
        assert engine.has_strong_recommendations() is False, (
            "has_strong_recommendations() trả về True với danh sách rỗng"
        )
