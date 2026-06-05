"""
Search Module: Minimax with Alpha-Beta pruning for trading decision exploration.

Implements a game-tree search where:
- "Maximizing" player: the trader choosing actions (BUY, HOLD, SELL)
- "Minimizing" player: the market generating scenarios (stochastic responses)

The search evaluates leaf nodes using the EvaluationModel (via ModelManager)
and returns the action with the highest worst-case score.

Key features:
- Alpha-Beta pruning to reduce explored nodes
- Configurable depth (default 3, max 5)
- 5-second timeout with graceful degradation
- Adaptive branching factor reduction at depth > 3
- Returns best action from deepest fully evaluated level on timeout
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from engine.config import (
    Action,
    DecisionReport,
    EngineConfig,
    EngineError,
    IndicatorContribution,
    ScenarioResult,
    SearchConfig,
)
from engine.market_state import INDICATOR_COLUMNS, MarketState

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Types
# ==============================================================================


@dataclass
class SearchNode:
    """A node in the minimax search tree.

    Attributes
    ----------
    state : MarketState
        The market state at this node.
    action : Action or None
        The action that led to this node (None for root).
    score : float
        The evaluated or propagated score for this node.
    children : list of SearchNode
        Child nodes explored from this node.
    depth : int
        Depth level of this node in the tree (root = 0).
    """

    state: MarketState
    action: Optional[Action] = None
    score: float = 0.0
    children: List["SearchNode"] = field(default_factory=list)
    depth: int = 0


@dataclass
class SearchResult:
    """Internal result from the search process.

    Attributes
    ----------
    best_action : Action
        The recommended action with highest minimax score.
    best_score : float
        The minimax score of the best action.
    action_scores : dict
        Mapping of each root action to its minimax score.
    nodes_evaluated : int
        Total number of leaf nodes evaluated.
    depth_reached : int
        Maximum depth fully evaluated before timeout.
    timed_out : bool
        Whether the search was terminated due to timeout.
    root_children : list of SearchNode
        Top-level action nodes for scenario reporting.
    """

    best_action: Action = Action.HOLD
    best_score: float = 0.0
    action_scores: dict = field(default_factory=dict)
    nodes_evaluated: int = 0
    depth_reached: int = 0
    timed_out: bool = False
    root_children: List[SearchNode] = field(default_factory=list)


# ==============================================================================
# SearchModule
# ==============================================================================


class SearchModule:
    """Minimax search with Alpha-Beta pruning for trading decisions.

    The search alternates between:
    - Maximizing levels: trader picks BUY/HOLD/SELL (wants highest score)
    - Minimizing levels: market generates scenarios (worst-case assumption)

    At leaf nodes, the EvaluationModel scores the MarketState.

    Parameters
    ----------
    model_manager : ModelManager
        Manages the evaluation model for leaf-node scoring.
    scenario_generator : ScenarioGenerator
        Generates plausible future market states at each node.
    config : SearchConfig, optional
        Search configuration (depth, timeout, branching).
    """

    def __init__(
        self,
        model_manager,
        scenario_generator,
        config: Optional[SearchConfig] = None,
    ):
        self._model_manager = model_manager
        self._scenario_generator = scenario_generator
        self._config = config or SearchConfig()

        # Search state (reset per search call)
        self._start_time: float = 0.0
        self._timeout: float = self._config.timeout_seconds
        self._nodes_evaluated: int = 0
        self._timed_out: bool = False

    @property
    def config(self) -> SearchConfig:
        """Current search configuration."""
        return self._config

    def search(self, state: MarketState, depth: Optional[int] = None) -> SearchResult:
        """Run minimax search from the current market state.

        Iterates over BUY/HOLD/SELL actions at the root level, generates
        scenarios for each, and evaluates the resulting subtrees using
        minimax with alpha-beta pruning.

        If the search exceeds the timeout (default 5s), returns the best
        action from the deepest fully evaluated level.

        Parameters
        ----------
        state : MarketState
            Current market state to analyze.
        depth : int, optional
            Search depth override. If None, uses config default (3).
            Clamped to [1, max_depth].

        Returns
        -------
        SearchResult
            The search result containing the best action and analysis.

        Raises
        ------
        EngineError
            If the model manager has no loaded model.
        """
        # Determine search depth
        if depth is None:
            search_depth = self._config.default_depth
        else:
            search_depth = max(1, min(depth, self._config.max_depth))

        # Reset search state
        self._start_time = time.time()
        self._timeout = self._config.timeout_seconds
        self._nodes_evaluated = 0
        self._timed_out = False

        # Track best results per depth level for timeout fallback
        best_by_depth: dict = {}  # depth -> (Action, score, action_scores)
        root_children: List[SearchNode] = []

        # Evaluate each root action
        alpha = -float("inf")
        beta = float("inf")
        action_scores: dict = {}
        best_action = Action.HOLD
        best_score = -float("inf")

        actions = [Action.BUY, Action.HOLD, Action.SELL]

        for action in actions:
            if self._is_timed_out():
                self._timed_out = True
                break

            # Generate scenarios for this action
            num_scenarios = self._get_branching_factor(current_depth=0, max_depth=search_depth)
            try:
                scenarios = self._scenario_generator.generate(
                    state, action, num_scenarios=num_scenarios
                )
            except EngineError as e:
                logger.warning(f"Scenario generation failed for {action.value}: {e}")
                # Use neutral score if scenario generation fails
                action_scores[action] = 0.0
                action_node = SearchNode(
                    state=state, action=action, score=0.0, depth=0
                )
                root_children.append(action_node)
                continue

            # Create the action node
            action_node = SearchNode(state=state, action=action, depth=0)

            # Minimax over scenarios (market is minimizing)
            min_score = float("inf")
            for scenario_state in scenarios:
                if self._is_timed_out():
                    self._timed_out = True
                    break

                # Recurse: next level is maximizing (trader's turn again)
                score = self._minimax(
                    state=scenario_state,
                    depth=search_depth - 1,
                    alpha=alpha,
                    beta=beta,
                    is_maximizing=True,
                    current_depth=1,
                    max_depth=search_depth,
                )
                min_score = min(min_score, score)

                # Alpha-beta: market (minimizing) updates beta
                beta = min(beta, min_score)
                if beta <= alpha:
                    break  # Prune

                child_node = SearchNode(
                    state=scenario_state, score=score, depth=1
                )
                action_node.children.append(child_node)

            if self._timed_out and min_score == float("inf"):
                # Didn't complete any scenario for this action
                break

            # The action's score is the worst-case across scenarios
            action_score = min_score if min_score != float("inf") else 0.0
            action_scores[action] = action_score
            action_node.score = action_score
            root_children.append(action_node)

            if action_score > best_score:
                best_score = action_score
                best_action = action

            # Update alpha for root-level pruning
            alpha = max(alpha, best_score)

        # If timed out with partial results, use what we have
        if self._timed_out and not action_scores:
            # No actions were fully evaluated - return HOLD as safe default
            best_action = Action.HOLD
            best_score = 0.0
            action_scores = {Action.HOLD: 0.0}

        # Determine depth reached
        depth_reached = search_depth if not self._timed_out else max(1, search_depth - 1)

        elapsed = time.time() - self._start_time
        logger.info(
            f"Search completed: action={best_action.value}, score={best_score:.4f}, "
            f"nodes={self._nodes_evaluated}, time={elapsed:.2f}s, "
            f"timed_out={self._timed_out}"
        )

        return SearchResult(
            best_action=best_action,
            best_score=best_score,
            action_scores=action_scores,
            nodes_evaluated=self._nodes_evaluated,
            depth_reached=depth_reached,
            timed_out=self._timed_out,
            root_children=root_children,
        )

    def _minimax(
        self,
        state: MarketState,
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool,
        current_depth: int,
        max_depth: int,
    ) -> float:
        """Recursive minimax with alpha-beta pruning.

        At leaf nodes (depth == 0), evaluates the state using the model.
        At internal nodes, alternates between maximizing (trader) and
        minimizing (market) levels.

        Parameters
        ----------
        state : MarketState
            Current state to evaluate or expand.
        depth : int
            Remaining depth to search (0 = evaluate leaf).
        alpha : float
            Best value the maximizer can guarantee.
        beta : float
            Best value the minimizer can guarantee.
        is_maximizing : bool
            True if this is the trader's turn (maximize score).
        current_depth : int
            Current depth in the tree (for adaptive branching).
        max_depth : int
            Total maximum search depth (for adaptive branching).

        Returns
        -------
        float
            The minimax score for this subtree.
        """
        # Check timeout
        if self._is_timed_out():
            self._timed_out = True
            # Return current evaluation as best estimate
            return self._evaluate_leaf(state)

        # Leaf node: evaluate with model
        if depth <= 0:
            return self._evaluate_leaf(state)

        if is_maximizing:
            # Trader's turn: choose best action
            max_eval = -float("inf")
            actions = [Action.BUY, Action.HOLD, Action.SELL]

            for action in actions:
                if self._is_timed_out():
                    self._timed_out = True
                    break

                # Generate scenarios for this action
                num_scenarios = self._get_branching_factor(current_depth, max_depth)
                try:
                    scenarios = self._scenario_generator.generate(
                        state, action, num_scenarios=num_scenarios
                    )
                except EngineError:
                    continue

                # Market responds (minimizing over scenarios)
                for scenario_state in scenarios:
                    if self._is_timed_out():
                        self._timed_out = True
                        break

                    eval_score = self._minimax(
                        state=scenario_state,
                        depth=depth - 1,
                        alpha=alpha,
                        beta=beta,
                        is_maximizing=False,
                        current_depth=current_depth + 1,
                        max_depth=max_depth,
                    )
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Beta cutoff

                if beta <= alpha:
                    break  # Prune remaining actions

            return max_eval if max_eval != -float("inf") else 0.0

        else:
            # Market's turn: assume worst-case for trader
            min_eval = float("inf")
            actions = [Action.BUY, Action.HOLD, Action.SELL]

            for action in actions:
                if self._is_timed_out():
                    self._timed_out = True
                    break

                # Generate scenarios for this action (market movement)
                num_scenarios = self._get_branching_factor(current_depth, max_depth)
                try:
                    scenarios = self._scenario_generator.generate(
                        state, action, num_scenarios=num_scenarios
                    )
                except EngineError:
                    continue

                for scenario_state in scenarios:
                    if self._is_timed_out():
                        self._timed_out = True
                        break

                    eval_score = self._minimax(
                        state=scenario_state,
                        depth=depth - 1,
                        alpha=alpha,
                        beta=beta,
                        is_maximizing=True,
                        current_depth=current_depth + 1,
                        max_depth=max_depth,
                    )
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha cutoff

                if beta <= alpha:
                    break  # Prune remaining actions

            return min_eval if min_eval != float("inf") else 0.0

    def _evaluate_leaf(self, state: MarketState) -> float:
        """Evaluate a leaf node state using the model manager.

        Converts the MarketState to a feature array and runs inference
        through the ModelManager.predict() API.

        Parameters
        ----------
        state : MarketState
            The leaf state to evaluate.

        Returns
        -------
        float
            Position score in [-1, +1].
        """
        self._nodes_evaluated += 1

        try:
            # Build feature array: concatenate OHLCV + indicators
            # Shape: (lookback, num_features)
            if state.indicators.shape[1] > 0:
                features = np.concatenate(
                    [state.ohlcv, state.indicators], axis=1
                ).astype(np.float32)
            else:
                features = state.ohlcv.astype(np.float32)

            # Handle NaN: forward-fill then zero-fill
            features = _forward_fill_2d(features)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Predict: ModelManager expects shape (batch, lookback, num_features)
            # or (lookback, num_features) for single sample
            scores = self._model_manager.predict(features)

            # scores shape: (1, 1) → extract scalar
            score = float(scores.flatten()[0])

            # Clamp to [-1, 1] for safety
            return max(-1.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"Leaf evaluation failed: {e}")
            return 0.0

    def _get_branching_factor(self, current_depth: int, max_depth: int) -> int:
        """Determine the number of scenarios to generate at this depth.

        Implements adaptive branching: when depth > adaptive_depth_threshold
        (default 3), reduces scenarios to maintain the 5-second constraint.

        Parameters
        ----------
        current_depth : int
            Current depth in the search tree.
        max_depth : int
            Total maximum search depth.

        Returns
        -------
        int
            Number of scenarios to generate (3, 5, or 7).
        """
        # If max_depth exceeds the adaptive threshold, reduce branching
        # at deeper levels to maintain time constraint
        if max_depth > self._config.adaptive_depth_threshold:
            if current_depth >= self._config.adaptive_depth_threshold:
                return self._config.adaptive_scenario_count  # 3
            else:
                # Even at shallow levels, reduce slightly for deep searches
                return self._config.default_scenarios  # 5
        else:
            # Standard branching at default depth
            if current_depth >= 2:
                # Slight reduction at deeper levels even for depth=3
                return self._config.adaptive_scenario_count  # 3
            return self._config.default_scenarios  # 5

    def _is_timed_out(self) -> bool:
        """Check if the search has exceeded the timeout.

        Returns
        -------
        bool
            True if elapsed time exceeds the configured timeout.
        """
        elapsed = time.time() - self._start_time
        return elapsed >= self._timeout


# ==============================================================================
# DecisionReportGenerator
# ==============================================================================


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
        search_result: SearchResult,
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

    def _compute_confidence(self, search_result: SearchResult) -> float:
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

    def _extract_top_scenarios(self, search_result: SearchResult) -> List[ScenarioResult]:
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
        import warnings
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


# ==============================================================================
# Helper functions
# ==============================================================================


def _forward_fill_2d(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values along axis 0 for each column.

    Parameters
    ----------
    arr : np.ndarray
        2D array of shape (time_steps, features).

    Returns
    -------
    np.ndarray
        Array with NaN forward-filled along axis 0.
    """
    result = arr.copy()
    for col in range(result.shape[1]):
        col_data = result[:, col]
        mask = np.isnan(col_data)
        if not mask.any():
            continue
        # Forward fill: propagate last valid value
        last_valid = np.nan
        for i in range(len(col_data)):
            if mask[i]:
                if not np.isnan(last_valid):
                    result[i, col] = last_valid
            else:
                last_valid = col_data[i]
    return result
