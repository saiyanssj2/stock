"""
Unit tests for DecisionReportGenerator.

Tests cover:
- Confidence computation from scenario agreement
- Low-confidence HOLD override (confidence < 0.3)
- Top 3 scenario extraction with action sequences and scores
- Top 5 indicator contributions sorted by absolute contribution
- Structural completeness of the DecisionReport output
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from engine.config import (
    Action,
    DecisionReport,
    EngineConfig,
    IndicatorContribution,
    ScenarioResult,
    SearchConfig,
)
from engine.market_state import MarketState, NUM_INDICATORS, INDICATOR_COLUMNS
from engine.search_module import (
    DecisionReportGenerator,
    SearchNode,
    SearchResult,
)


# ==============================================================================
# Fixtures
# ==============================================================================


def _make_market_state(
    symbol: str = "FPT",
    lookback: int = 60,
    close_price: float = 100.0,
) -> MarketState:
    """Create a minimal valid MarketState for testing."""
    ohlcv = np.zeros((lookback, 5), dtype=np.float64)
    ohlcv[:, 0] = close_price * 0.99  # open
    ohlcv[:, 1] = close_price * 1.01  # high
    ohlcv[:, 2] = close_price * 0.98  # low
    ohlcv[:, 3] = close_price  # close
    ohlcv[:, 4] = 1_000_000  # volume

    # Generate indicators with some variation for contribution analysis
    rng = np.random.default_rng(42)
    indicators = rng.random((lookback, NUM_INDICATORS)) * 100

    return MarketState(
        symbol=symbol,
        timestamp=pd.Timestamp("2024-01-15"),
        ohlcv=ohlcv,
        indicators=indicators,
        lookback=lookback,
    )


def _make_search_result(
    best_action: Action = Action.BUY,
    best_score: float = 0.6,
    action_scores: dict = None,
    root_children: list = None,
    nodes_evaluated: int = 100,
) -> SearchResult:
    """Create a SearchResult with configurable parameters."""
    if action_scores is None:
        action_scores = {
            Action.BUY: 0.6,
            Action.HOLD: 0.3,
            Action.SELL: -0.2,
        }
    if root_children is None:
        state = _make_market_state()
        root_children = [
            SearchNode(
                state=state,
                action=Action.BUY,
                score=0.6,
                depth=0,
                children=[
                    SearchNode(state=state, score=0.7, depth=1),
                    SearchNode(state=state, score=0.5, depth=1),
                ],
            ),
            SearchNode(
                state=state,
                action=Action.HOLD,
                score=0.3,
                depth=0,
                children=[
                    SearchNode(state=state, score=0.3, depth=1),
                    SearchNode(state=state, score=0.2, depth=1),
                ],
            ),
            SearchNode(
                state=state,
                action=Action.SELL,
                score=-0.2,
                depth=0,
                children=[
                    SearchNode(state=state, score=-0.1, depth=1),
                    SearchNode(state=state, score=-0.3, depth=1),
                ],
            ),
        ]

    return SearchResult(
        best_action=best_action,
        best_score=best_score,
        action_scores=action_scores,
        nodes_evaluated=nodes_evaluated,
        depth_reached=3,
        timed_out=False,
        root_children=root_children,
    )


# ==============================================================================
# DecisionReportGenerator Tests
# ==============================================================================


class TestDecisionReportGeneration:
    """Tests for the main generate() method."""

    def test_generates_valid_report(self):
        """Generator produces a well-formed DecisionReport."""
        state = _make_market_state()
        search_result = _make_search_result()
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        assert isinstance(report, DecisionReport)
        assert report.symbol == "FPT"
        assert report.recommended_action in [Action.BUY, Action.HOLD, Action.SELL]
        assert 0.0 <= report.confidence <= 1.0
        assert -1.0 <= report.position_score <= 1.0
        assert len(report.top_scenarios) <= 3
        assert len(report.top_indicators) <= 5

    def test_position_score_from_search_result(self):
        """Position score is taken from the search result's best_score."""
        state = _make_market_state()
        search_result = _make_search_result(best_score=0.75)
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        assert report.position_score == 0.75

    def test_symbol_from_market_state(self):
        """Report symbol comes from the MarketState."""
        state = _make_market_state(symbol="VNM")
        search_result = _make_search_result()
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        assert report.symbol == "VNM"


class TestConfidenceComputation:
    """Tests for confidence scoring."""

    def test_confidence_in_valid_range(self):
        """Confidence is always between 0.0 and 1.0."""
        state = _make_market_state()
        search_result = _make_search_result()
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        assert 0.0 <= report.confidence <= 1.0

    def test_high_confidence_when_scores_agree(self):
        """When all action scores are similar, confidence is relatively high."""
        state = _make_market_state()
        # All actions have very similar scores
        search_result = _make_search_result(
            action_scores={
                Action.BUY: 0.50,
                Action.HOLD: 0.49,
                Action.SELL: 0.48,
            },
            root_children=[
                SearchNode(
                    state=state,
                    action=Action.BUY,
                    score=0.50,
                    depth=0,
                    children=[
                        SearchNode(state=state, score=0.50, depth=1),
                        SearchNode(state=state, score=0.49, depth=1),
                    ],
                ),
                SearchNode(
                    state=state,
                    action=Action.HOLD,
                    score=0.49,
                    depth=0,
                    children=[
                        SearchNode(state=state, score=0.49, depth=1),
                        SearchNode(state=state, score=0.48, depth=1),
                    ],
                ),
                SearchNode(
                    state=state,
                    action=Action.SELL,
                    score=0.48,
                    depth=0,
                    children=[
                        SearchNode(state=state, score=0.48, depth=1),
                        SearchNode(state=state, score=0.47, depth=1),
                    ],
                ),
            ],
        )
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        # Scores are tightly clustered → high confidence
        assert report.confidence > 0.7

    def test_low_confidence_when_scores_diverge(self):
        """When action scores span a wide range, confidence is lower."""
        state = _make_market_state()
        # Large spread in action scores AND child scores
        search_result = _make_search_result(
            action_scores={
                Action.BUY: 0.9,
                Action.HOLD: 0.0,
                Action.SELL: -0.9,
            },
            root_children=[
                SearchNode(
                    state=state,
                    action=Action.BUY,
                    score=0.9,
                    depth=0,
                    children=[
                        SearchNode(state=state, score=0.9, depth=1),
                        SearchNode(state=state, score=-0.8, depth=1),
                    ],
                ),
                SearchNode(
                    state=state,
                    action=Action.HOLD,
                    score=0.0,
                    depth=0,
                    children=[
                        SearchNode(state=state, score=0.5, depth=1),
                        SearchNode(state=state, score=-0.5, depth=1),
                    ],
                ),
                SearchNode(
                    state=state,
                    action=Action.SELL,
                    score=-0.9,
                    depth=0,
                    children=[
                        SearchNode(state=state, score=0.7, depth=1),
                        SearchNode(state=state, score=-0.9, depth=1),
                    ],
                ),
            ],
        )
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        # Large score spread → lower confidence
        assert report.confidence < 0.5

    def test_confidence_zero_when_no_action_scores(self):
        """Confidence is 0.0 when there are no action scores."""
        state = _make_market_state()
        search_result = _make_search_result(
            action_scores={},
            root_children=[],
        )
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        assert report.confidence == 0.0

    def test_confidence_with_single_action(self):
        """Confidence is 0.5 when only one action was evaluated."""
        state = _make_market_state()
        search_result = _make_search_result(
            action_scores={Action.HOLD: 0.5},
            root_children=[
                SearchNode(state=state, action=Action.HOLD, score=0.5, depth=0),
            ],
        )
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        assert report.confidence == 0.5


class TestLowConfidenceHoldOverride:
    """Tests for the low-confidence HOLD override (req 5.6)."""

    def test_hold_override_when_confidence_below_threshold(self):
        """Action is overridden to HOLD when confidence < 0.3."""
        state = _make_market_state()
        # Create a scenario where confidence will be very low
        # Wide spread in scores + high variance in children
        search_result = _make_search_result(
            best_action=Action.BUY,
            best_score=0.9,
            action_scores={
                Action.BUY: 0.9,
                Action.HOLD: 0.0,
                Action.SELL: -0.9,
            },
            root_children=[
                SearchNode(
                    state=state,
                    action=Action.BUY,
                    score=0.9,
                    depth=0,
                    children=[
                        SearchNode(state=state, score=1.0, depth=1),
                        SearchNode(state=state, score=-1.0, depth=1),
                    ],
                ),
                SearchNode(
                    state=state,
                    action=Action.HOLD,
                    score=0.0,
                    depth=0,
                    children=[
                        SearchNode(state=state, score=0.9, depth=1),
                        SearchNode(state=state, score=-0.9, depth=1),
                    ],
                ),
                SearchNode(
                    state=state,
                    action=Action.SELL,
                    score=-0.9,
                    depth=0,
                    children=[
                        SearchNode(state=state, score=0.8, depth=1),
                        SearchNode(state=state, score=-1.0, depth=1),
                    ],
                ),
            ],
        )

        # Use a high threshold to guarantee the override triggers
        engine_config = EngineConfig(confidence_hold_threshold=0.99)
        generator = DecisionReportGenerator(engine_config=engine_config)

        report = generator.generate(search_result, state)

        # Since threshold is 0.99, any confidence < 0.99 → HOLD
        assert report.recommended_action == Action.HOLD

    def test_no_override_when_confidence_above_threshold(self):
        """Action is preserved when confidence >= 0.3."""
        state = _make_market_state()
        # Tight scores → high confidence
        search_result = _make_search_result(
            best_action=Action.BUY,
            best_score=0.5,
            action_scores={
                Action.BUY: 0.50,
                Action.HOLD: 0.49,
                Action.SELL: 0.48,
            },
            root_children=[
                SearchNode(
                    state=state,
                    action=Action.BUY,
                    score=0.50,
                    depth=0,
                    children=[
                        SearchNode(state=state, score=0.50, depth=1),
                        SearchNode(state=state, score=0.50, depth=1),
                    ],
                ),
                SearchNode(
                    state=state,
                    action=Action.HOLD,
                    score=0.49,
                    depth=0,
                    children=[
                        SearchNode(state=state, score=0.49, depth=1),
                        SearchNode(state=state, score=0.49, depth=1),
                    ],
                ),
                SearchNode(
                    state=state,
                    action=Action.SELL,
                    score=0.48,
                    depth=0,
                    children=[
                        SearchNode(state=state, score=0.48, depth=1),
                        SearchNode(state=state, score=0.48, depth=1),
                    ],
                ),
            ],
        )
        # Use a low threshold (default 0.3)
        generator = DecisionReportGenerator(
            engine_config=EngineConfig(confidence_hold_threshold=0.3)
        )

        report = generator.generate(search_result, state)

        # High confidence → BUY is preserved
        assert report.recommended_action == Action.BUY

    def test_hold_override_uses_config_threshold(self):
        """The HOLD override uses the configured threshold, not hardcoded 0.3."""
        state = _make_market_state()
        search_result = _make_search_result(
            best_action=Action.SELL,
            best_score=-0.5,
            action_scores={
                Action.BUY: -0.5,
                Action.HOLD: -0.5,
                Action.SELL: -0.5,
            },
            root_children=[
                SearchNode(
                    state=state,
                    action=Action.BUY,
                    score=-0.5,
                    depth=0,
                    children=[SearchNode(state=state, score=-0.5, depth=1)],
                ),
                SearchNode(
                    state=state,
                    action=Action.HOLD,
                    score=-0.5,
                    depth=0,
                    children=[SearchNode(state=state, score=-0.5, depth=1)],
                ),
                SearchNode(
                    state=state,
                    action=Action.SELL,
                    score=-0.5,
                    depth=0,
                    children=[SearchNode(state=state, score=-0.5, depth=1)],
                ),
            ],
        )

        # Very low threshold: even low confidence should pass
        generator = DecisionReportGenerator(
            engine_config=EngineConfig(confidence_hold_threshold=0.01)
        )
        report = generator.generate(search_result, state)
        # Confidence should be high here (all scores same) → original action preserved
        assert report.recommended_action == Action.SELL


class TestTopScenarios:
    """Tests for top scenario extraction."""

    def test_returns_up_to_3_scenarios(self):
        """Report includes at most 3 top scenarios."""
        state = _make_market_state()
        search_result = _make_search_result()
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        assert len(report.top_scenarios) <= 3

    def test_scenarios_have_required_fields(self):
        """Each scenario has action_sequence, leaf_score, and description."""
        state = _make_market_state()
        search_result = _make_search_result()
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        for scenario in report.top_scenarios:
            assert isinstance(scenario, ScenarioResult)
            assert len(scenario.action_sequence) >= 1
            assert all(
                isinstance(a, Action) for a in scenario.action_sequence
            )
            assert isinstance(scenario.leaf_score, float)
            assert isinstance(scenario.description, str)
            assert len(scenario.description) > 0

    def test_scenarios_sorted_by_absolute_score(self):
        """Scenarios are sorted by absolute leaf score, descending."""
        state = _make_market_state()
        search_result = _make_search_result(
            root_children=[
                SearchNode(state=state, action=Action.BUY, score=0.2, depth=0),
                SearchNode(state=state, action=Action.HOLD, score=-0.8, depth=0),
                SearchNode(state=state, action=Action.SELL, score=0.5, depth=0),
            ],
        )
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        # Should be sorted by |score|: 0.8, 0.5, 0.2
        scores = [abs(s.leaf_score) for s in report.top_scenarios]
        assert scores == sorted(scores, reverse=True)

    def test_fewer_than_3_scenarios_when_fewer_available(self):
        """If fewer than 3 root children, report includes all available."""
        state = _make_market_state()
        search_result = _make_search_result(
            root_children=[
                SearchNode(state=state, action=Action.BUY, score=0.5, depth=0),
            ],
        )
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        assert len(report.top_scenarios) == 1

    def test_empty_root_children_returns_empty_scenarios(self):
        """No scenarios when root_children is empty."""
        state = _make_market_state()
        search_result = _make_search_result(root_children=[])
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        assert report.top_scenarios == []

    def test_scenario_description_contains_action(self):
        """Scenario description mentions the action."""
        state = _make_market_state()
        search_result = _make_search_result(
            root_children=[
                SearchNode(state=state, action=Action.BUY, score=0.7, depth=0),
            ],
        )
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        assert "BUY" in report.top_scenarios[0].description


class TestIndicatorContributions:
    """Tests for indicator contribution computation."""

    def test_returns_up_to_5_indicators(self):
        """Report includes at most 5 top indicators."""
        state = _make_market_state()
        search_result = _make_search_result()
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        assert len(report.top_indicators) <= 5

    def test_indicators_have_required_fields(self):
        """Each indicator contribution has name, value, and contribution."""
        state = _make_market_state()
        search_result = _make_search_result()
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        for ind in report.top_indicators:
            assert isinstance(ind, IndicatorContribution)
            assert isinstance(ind.name, str)
            assert len(ind.name) > 0
            assert isinstance(ind.value, float)
            assert isinstance(ind.contribution, float)
            assert ind.contribution >= 0.0  # Absolute contribution

    def test_indicators_sorted_by_absolute_contribution(self):
        """Indicators are sorted by absolute contribution, descending."""
        state = _make_market_state()
        search_result = _make_search_result()
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        contributions = [ind.contribution for ind in report.top_indicators]
        assert contributions == sorted(contributions, reverse=True)

    def test_indicator_names_from_known_list(self):
        """Indicator names come from the INDICATOR_COLUMNS list."""
        state = _make_market_state()
        search_result = _make_search_result()
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        for ind in report.top_indicators:
            assert ind.name in INDICATOR_COLUMNS

    def test_no_indicators_when_empty(self):
        """Returns empty list when state has no indicators."""
        state = MarketState(
            symbol="FPT",
            timestamp=pd.Timestamp("2024-01-15"),
            ohlcv=np.ones((60, 5)),
            indicators=np.zeros((60, 0)),
            lookback=60,
        )
        search_result = _make_search_result()
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        assert report.top_indicators == []

    def test_handles_nan_indicators_gracefully(self):
        """NaN indicator values are excluded from contribution ranking."""
        lookback = 60
        ohlcv = np.ones((lookback, 5)) * 100.0
        indicators = np.full((lookback, NUM_INDICATORS), np.nan)
        # Set only a few indicators with valid values
        indicators[:, 0] = np.linspace(90, 110, lookback)  # EMA_9
        indicators[:, 1] = np.linspace(85, 115, lookback)  # EMA_20

        state = MarketState(
            symbol="FPT",
            timestamp=pd.Timestamp("2024-01-15"),
            ohlcv=ohlcv,
            indicators=indicators,
            lookback=lookback,
        )
        search_result = _make_search_result()
        generator = DecisionReportGenerator()

        report = generator.generate(search_result, state)

        # Should only include non-NaN indicators
        for ind in report.top_indicators:
            assert not np.isnan(ind.value)
            assert not np.isnan(ind.contribution)


class TestCustomConfiguration:
    """Tests for configurable parameters."""

    def test_custom_top_scenarios_count(self):
        """SearchConfig.top_scenarios_report controls scenario count."""
        state = _make_market_state()
        search_result = _make_search_result()
        config = SearchConfig(top_scenarios_report=2)
        generator = DecisionReportGenerator(config=config)

        report = generator.generate(search_result, state)

        assert len(report.top_scenarios) <= 2

    def test_custom_top_indicators_count(self):
        """SearchConfig.top_indicators_report controls indicator count."""
        state = _make_market_state()
        search_result = _make_search_result()
        config = SearchConfig(top_indicators_report=3)
        generator = DecisionReportGenerator(config=config)

        report = generator.generate(search_result, state)

        assert len(report.top_indicators) <= 3
