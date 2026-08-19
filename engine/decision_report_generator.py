"""
Decision Report Generator: builds structured DecisionReport from search results.

Computes confidence from scenario agreement, applies low-confidence HOLD override,
extracts top scenarios, and determines indicator contributions.
"""

import logging
import warnings
from typing import List, Optional

import numpy as np

from engine.config import (
    Action,
    DecisionReport,
    EngineConfig,
    IndicatorContribution,
    ScenarioResult,
    SearchConfig,
)
from engine.market_state import INDICATOR_COLUMNS, MarketState

logger = logging.getLogger(__name__)


class DecisionReportGenerator:
    """Generates a DecisionReport from a SearchResult and MarketState.

    Computes confidence from scenario agreement, applies low-confidence
    HOLD override, extracts top scenarios, and determines indicator
    contributions.

    Parameters
    ----------
    config : SearchConfig, optional
        Search configuration for report parameters (top_scenarios_report,
        top_indicators_report).
    engine_config : EngineConfig, optional
        Engine configuration for confidence threshold.
    """

    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        engine_config: Optional[EngineConfig] = None,
    ):
        self._config = config or SearchConfig()
        self._engine_config = engine_config or EngineConfig()

    def generate(
        self,
        search_result,
        state: MarketState,
    ) -> DecisionReport:
        """Generate a DecisionReport from a SearchResult and current MarketState.

        Steps:
        1. Compute confidence from scenario agreement level.
        2. Apply low-confidence HOLD override if confidence < threshold.
        3. Extract top 3 scenarios from root_children.
        4. Compute top 5 indicator contributions.

        Parameters
        ----------
        search_result : SearchResult
            Result from SearchModule.search().
        state : MarketState
            Current market state that was analyzed.

        Returns
        -------
        DecisionReport
            Complete decision report with all fields populated.
        """
        # Step 1: Compute confidence from scenario agreement
        confidence = self._compute_confidence(search_result)

        # Step 2: Determine recommended action (with HOLD override)
        recommended_action = search_result.best_action
        if confidence < self._engine_config.confidence_hold_threshold:
            recommended_action = Action.HOLD

        # Step 3: Extract top scenarios
        top_scenarios = self._extract_top_scenarios(search_result)

        # Step 4: Compute indicator contributions
        top_indicators = self._compute_indicator_contributions(state)

        return DecisionReport(
            symbol=state.symbol,
            recommended_action=recommended_action,
            confidence=confidence,
            position_score=search_result.best_score,
            top_scenarios=top_scenarios,
            top_indicators=top_indicators,
        )

    def _compute_confidence(self, search_result) -> float:
        """Compute confidence from how much scenarios agree.

        Confidence measures the agreement level among root-level action scores.
        High confidence means one action clearly dominates; low confidence
        means actions have similar scores (ambiguous situation).

        The confidence is computed as:
            confidence = 1.0 - (score_range / max_possible_range)

        Where score_range is the difference between the highest and lowest
        action scores, and max_possible_range is 2.0 (from -1.0 to +1.0).

        If scores are all the same (perfect agreement on ambiguity), confidence
        is high (1.0). If scores span the full range, confidence is low (0.0).

        Actually, a better interpretation for "scenario agreement" is:
        - If all scenarios (within each action) produce similar scores,
          confidence is HIGH (we're sure about the evaluation).
        - If scenarios produce widely different scores, confidence is LOW
          (the outcome is uncertain).

        We compute it as: 1.0 - normalized_variance_of_child_scores.

        Parameters
        ----------
        search_result : SearchResult
            The search result with action_scores and root_children.

        Returns
        -------
        float
            Confidence in [0.0, 1.0].
        """
        action_scores = search_result.action_scores

        if not action_scores:
            return 0.0

        scores = list(action_scores.values())

        if len(scores) <= 1:
            # Only one action evaluated - low confidence
            return 0.5

        # Compute the range of action scores
        score_max = max(scores)
        score_min = min(scores)
        score_range = score_max - score_min

        # Max possible range for scores in [-1, 1] is 2.0
        max_possible_range = 2.0

        # Also consider child score variance for deeper confidence analysis
        child_scores = []
        for root_child in search_result.root_children:
            for child in root_child.children:
                child_scores.append(child.score)

        if child_scores:
            # Compute variance of all explored child scores
            child_arr = np.array(child_scores)
            child_std = float(np.std(child_arr))
            # Normalize: max std for [-1, 1] is 1.0
            normalized_std = min(child_std, 1.0)
        else:
            # No child scores available, use action score spread only
            normalized_std = score_range / max_possible_range

        # Confidence combines both:
        # - Action score spread (how clearly one action dominates)
        # - Child score variance (how uncertain the scenarios are)
        # Lower spread AND lower variance → higher confidence
        action_agreement = 1.0 - (score_range / max_possible_range)
        scenario_agreement = 1.0 - normalized_std

        # Weight: scenario agreement is more important as it measures
        # the actual uncertainty in outcomes
        confidence = 0.4 * action_agreement + 0.6 * scenario_agreement

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def _extract_top_scenarios(self, search_result) -> List[ScenarioResult]:
        """Extract top N scenarios from root children, sorted by score.

        Each scenario represents an action node from the root level with
        its best evaluated child path.

        Parameters
        ----------
        search_result : SearchResult
            The search result containing root_children.

        Returns
        -------
        List[ScenarioResult]
            Top scenarios (up to top_scenarios_report, default 3).
        """
        top_n = self._config.top_scenarios_report  # Default 3
        root_children = search_result.root_children

        if not root_children:
            return []

        # Build scenario results from root children
        scenario_results = []
        for root_child in root_children:
            if root_child.action is None:
                continue

            # Build action sequence: the root action + best child path
            action_sequence = [root_child.action]

            # Find the best child's action sequence (if any)
            best_child_score = root_child.score
            if root_child.children:
                # Sort children by score (descending for the best outcome)
                sorted_children = sorted(
                    root_child.children, key=lambda c: c.score, reverse=True
                )
                best_child = sorted_children[0]
                best_child_score = best_child.score
                # If the child has an action, include it
                if best_child.action is not None:
                    action_sequence.append(best_child.action)

            # Generate description
            description = self._describe_scenario(
                action_sequence, root_child.score
            )

            scenario_results.append(
                ScenarioResult(
                    action_sequence=action_sequence,
                    leaf_score=root_child.score,
                    description=description,
                )
            )

        # Sort by absolute leaf_score (most impactful scenarios first)
        scenario_results.sort(key=lambda s: abs(s.leaf_score), reverse=True)

        # Return top N (or all if fewer available)
        return scenario_results[:top_n]

    def _describe_scenario(
        self, action_sequence: List[Action], score: float
    ) -> str:
        """Generate a human-readable description for a scenario.

        Parameters
        ----------
        action_sequence : List[Action]
            Sequence of actions in this scenario path.
        score : float
            The evaluated score for this scenario.

        Returns
        -------
        str
            Description string.
        """
        action_names = " → ".join(a.value for a in action_sequence)

        if score > 0.5:
            outlook = "strongly favorable"
        elif score > 0.2:
            outlook = "moderately favorable"
        elif score > -0.2:
            outlook = "neutral"
        elif score > -0.5:
            outlook = "moderately unfavorable"
        else:
            outlook = "strongly unfavorable"

        return f"{action_names}: {outlook} (score: {score:.3f})"

    def _compute_indicator_contributions(
        self, state: MarketState
    ) -> List[IndicatorContribution]:
        """Compute top N indicator contributions to the position score.

        Contributions are estimated by analyzing which indicators deviate
        most from their neutral values (midpoint of their typical range).
        Indicators with larger absolute deviations from neutral contribute
        more to the score differentiation.

        For a simple but effective approach, we use the most recent indicator
        values and compute their z-score-like deviation from the lookback mean,
        then rank by absolute deviation.

        Parameters
        ----------
        state : MarketState
            Current market state with indicator values.

        Returns
        -------
        List[IndicatorContribution]
            Top 5 indicators sorted by absolute contribution (descending).
        """
        top_n = self._config.top_indicators_report  # Default 5

        if state.indicators.shape[1] == 0:
            return []

        # Get the most recent indicator values
        current_values = state.indicators[-1]  # Shape: (num_indicators,)

        # Compute mean and std over the lookback window for each indicator
        # This gives us a sense of how "unusual" the current value is
        # Suppress warnings for all-NaN slices (expected for some indicators)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            indicator_means = np.nanmean(state.indicators, axis=0)
            indicator_stds = np.nanstd(state.indicators, axis=0)

        # Compute contributions (z-score-like deviation)
        contributions = []
        for i in range(min(len(INDICATOR_COLUMNS), state.indicators.shape[1])):
            name = INDICATOR_COLUMNS[i]
            value = current_values[i]

            if np.isnan(value):
                continue

            mean_val = indicator_means[i]
            std_val = indicator_stds[i]

            # Compute contribution as deviation from mean (normalized by std)
            if np.isnan(mean_val) or np.isnan(std_val) or std_val < 1e-10:
                # If std is near zero, contribution is minimal
                contribution = 0.0
            else:
                contribution = (value - mean_val) / std_val

            contributions.append(
                IndicatorContribution(
                    name=name,
                    value=float(value),
                    contribution=float(abs(contribution)),
                )
            )

        # Sort by absolute contribution (descending)
        contributions.sort(key=lambda ic: ic.contribution, reverse=True)

        # Return top N
        return contributions[:top_n]
