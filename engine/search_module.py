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
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from engine.config import (
    Action,
    ConfigError,
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
# Extended Search Data Types (imported from engine.extended_search)
# ==============================================================================

from engine.extended_search import (  # noqa: E402
    ExtendedDecisionReport,
    ExtendedSearchConfig,
    ExtendedSearchProgress,
)


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

        # Extended search state
        self._cancel_requested: bool = False

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
        from engine.search_module_impl import minimax, get_branching_factor, is_timed_out

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

        Delegates to the implementation in search_module_impl.

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
        from engine.search_module_impl import minimax
        return minimax(
            self, state, depth, alpha, beta, is_maximizing, current_depth, max_depth
        )

    def _evaluate_leaf(self, state: MarketState) -> float:
        """Evaluate a leaf node state using the model manager.

        Delegates to the implementation in search_module_impl.

        Parameters
        ----------
        state : MarketState
            The leaf state to evaluate.

        Returns
        -------
        float
            Position score in [-1, +1].
        """
        from engine.search_module_impl import evaluate_leaf
        return evaluate_leaf(self, state)

    def _get_branching_factor(self, current_depth: int, max_depth: int) -> int:
        """Determine the number of scenarios to generate at this depth.

        Delegates to the implementation in search_module_impl.

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
        from engine.search_module_impl import get_branching_factor
        return get_branching_factor(self, current_depth, max_depth)

    def _is_timed_out(self) -> bool:
        """Check if the search has exceeded the timeout.

        Returns
        -------
        bool
            True if elapsed time exceeds the configured timeout.
        """
        from engine.search_module_impl import is_timed_out
        return is_timed_out(self)

    def cancel_search(self) -> None:
        """Cancel an ongoing extended search.

        Sets the _cancel_requested flag. The search loop will check this
        flag and return the best result from the deepest fully completed
        depth level, equivalent to timeout behavior.

        This method is safe to call from another thread.
        """
        self._cancel_requested = True

    def _is_cancelled_or_timed_out(self) -> bool:
        """Check if search should stop due to cancel or timeout.

        Returns
        -------
        bool
            True if cancel requested or timeout exceeded.
        """
        if self._cancel_requested:
            return True
        return self._is_timed_out()

    def _get_extended_branching_factor(self, current_depth: int, max_depth: int) -> int:
        """Determine branching factor for extended search with reduction at deeper levels.

        Implements branching factor reduction: 5 → 4 → 3 as depth increases.

        Parameters
        ----------
        current_depth : int
            Current depth in the search tree.
        max_depth : int
            Total maximum search depth.

        Returns
        -------
        int
            Number of scenarios to generate (3, 4, or 5).
        """
        if current_depth <= 2:
            return 5
        elif current_depth <= 4:
            return 4
        else:
            return 3

    def _run_extended_search(
        self,
        state: MarketState,
        config: ExtendedSearchConfig,
        progress_callback: Optional[Callable] = None,
    ) -> "ExtendedDecisionReport":
        """Run extended search with iterative deepening and cancellation support.

        The extended search process:
        1. Run standard search (depth 3, 5s timeout) as baseline comparison
        2. Iteratively deepen from depth 3 up to configured depth
        3. Report progress via callback at each depth level
        4. On cancel/timeout: return best from deepest fully completed level
        5. Build ExtendedDecisionReport with comparison to standard search

        Parameters
        ----------
        state : MarketState
            Current market state to analyze.
        config : ExtendedSearchConfig
            Extended search configuration (depth, timeout).
        progress_callback : callable, optional
            Function called at each depth level with ExtendedSearchProgress.

        Returns
        -------
        ExtendedDecisionReport
            Extended decision report with comparison to standard search.
        """
        from engine.extended_search import (
            ExtendedDecisionReport,
            ExtendedSearchProgress,
        )

        # Reset extended search state
        self._cancel_requested = False
        self._start_time = time.time()
        self._timeout = config.timeout
        self._nodes_evaluated = 0
        self._timed_out = False

        # ======================================================================
        # Step 1: Run standard search (depth 3, 5s) for baseline comparison
        # ======================================================================
        standard_result = self._run_standard_baseline(state)

        # Check if we should abort early
        if self._is_cancelled_or_timed_out():
            return self._build_extended_report(
                state=state,
                best_action=standard_result.best_action,
                best_score=standard_result.best_score,
                action_scores=standard_result.action_scores,
                effective_depth=standard_result.depth_reached,
                total_nodes=self._nodes_evaluated,
                elapsed=time.time() - self._start_time,
                standard_result=standard_result,
                root_children=standard_result.root_children,
            )

        # ======================================================================
        # Step 2: Iterative deepening from depth 3 to configured max depth
        # ======================================================================
        best_completed_action = standard_result.best_action
        best_completed_score = standard_result.best_score
        best_completed_action_scores = standard_result.action_scores
        best_completed_depth = standard_result.depth_reached
        best_completed_root_children = standard_result.root_children

        for depth_level in range(3, config.depth + 1):
            if self._is_cancelled_or_timed_out():
                break

            # Report progress at start of each depth level
            if progress_callback is not None:
                elapsed = time.time() - self._start_time
                remaining = self._estimate_remaining_time(
                    elapsed, depth_level, config.depth
                )
                progress = ExtendedSearchProgress(
                    current_depth=depth_level,
                    max_depth=config.depth,
                    nodes_evaluated=self._nodes_evaluated,
                    elapsed_seconds=elapsed,
                    estimated_remaining_seconds=remaining,
                    best_action_so_far=best_completed_action,
                    best_score_so_far=best_completed_score,
                )
                progress_callback(progress)

            # Run search at this depth level
            depth_result = self._search_at_depth(state, depth_level)

            if depth_result is not None:
                # This depth level was fully evaluated
                best_completed_action = depth_result.best_action
                best_completed_score = depth_result.best_score
                best_completed_action_scores = depth_result.action_scores
                best_completed_depth = depth_level
                best_completed_root_children = depth_result.root_children
            else:
                # Search was interrupted at this level — use previous results
                break

        # ======================================================================
        # Step 3: Build final ExtendedDecisionReport
        # ======================================================================
        elapsed = time.time() - self._start_time

        # Final progress report
        if progress_callback is not None:
            progress = ExtendedSearchProgress(
                current_depth=best_completed_depth,
                max_depth=config.depth,
                nodes_evaluated=self._nodes_evaluated,
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=0.0,
                best_action_so_far=best_completed_action,
                best_score_so_far=best_completed_score,
            )
            progress_callback(progress)

        return self._build_extended_report(
            state=state,
            best_action=best_completed_action,
            best_score=best_completed_score,
            action_scores=best_completed_action_scores,
            effective_depth=best_completed_depth,
            total_nodes=self._nodes_evaluated,
            elapsed=elapsed,
            standard_result=standard_result,
            root_children=best_completed_root_children,
        )

    def _run_standard_baseline(self, state: MarketState) -> "SearchResult":
        """Run standard search (depth 3, 5s) as a baseline for comparison.

        Uses a temporary timeout of 5 seconds and depth 3 regardless of
        the extended search configuration.

        Parameters
        ----------
        state : MarketState
            Current market state.

        Returns
        -------
        SearchResult
            Standard search result for baseline comparison.
        """
        # Save current timeout, temporarily set to standard 5s
        original_timeout = self._timeout
        original_start = self._start_time

        self._start_time = time.time()
        self._timeout = 5.0
        self._timed_out = False

        result = self.search(state, depth=3)

        # Restore extended timeout settings
        self._timeout = original_timeout
        self._start_time = original_start
        self._timed_out = False

        return result

    def _search_at_depth(
        self, state: MarketState, depth: int
    ) -> Optional["SearchResult"]:
        """Run a single search at a specific depth level for extended search.

        Uses extended branching factor reduction and checks for cancel/timeout.
        Returns None if the search could not be fully completed at this depth.

        Parameters
        ----------
        state : MarketState
            Current market state.
        depth : int
            Depth to search at.

        Returns
        -------
        SearchResult or None
            Complete result if this depth was fully evaluated, None if interrupted.
        """
        # Reset per-depth state (but keep cumulative nodes_evaluated)
        self._timed_out = False

        alpha = -float("inf")
        beta = float("inf")
        action_scores: dict = {}
        best_action = Action.HOLD
        best_score = -float("inf")
        root_children: List[SearchNode] = []

        actions = [Action.BUY, Action.HOLD, Action.SELL]
        all_actions_completed = True

        for action in actions:
            if self._is_cancelled_or_timed_out():
                all_actions_completed = False
                self._timed_out = True
                break

            # Generate scenarios with extended branching factor
            num_scenarios = self._get_extended_branching_factor(
                current_depth=0, max_depth=depth
            )
            try:
                scenarios = self._scenario_generator.generate(
                    state, action, num_scenarios=num_scenarios
                )
            except EngineError as e:
                logger.warning(
                    f"Scenario generation failed for {action.value} at depth {depth}: {e}"
                )
                action_scores[action] = 0.0
                action_node = SearchNode(
                    state=state, action=action, score=0.0, depth=0
                )
                root_children.append(action_node)
                continue

            action_node = SearchNode(state=state, action=action, depth=0)

            # Minimax over scenarios (market is minimizing)
            min_score = float("inf")
            action_fully_evaluated = True

            for scenario_state in scenarios:
                if self._is_cancelled_or_timed_out():
                    action_fully_evaluated = False
                    self._timed_out = True
                    break

                score = self._minimax(
                    state=scenario_state,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    is_maximizing=True,
                    current_depth=1,
                    max_depth=depth,
                )
                min_score = min(min_score, score)

                # Alpha-beta pruning
                beta = min(beta, min_score)
                if beta <= alpha:
                    break

                child_node = SearchNode(
                    state=scenario_state, score=score, depth=1
                )
                action_node.children.append(child_node)

            if not action_fully_evaluated:
                all_actions_completed = False
                break

            # The action's score is the worst-case across scenarios
            action_score = min_score if min_score != float("inf") else 0.0
            action_scores[action] = action_score
            action_node.score = action_score
            root_children.append(action_node)

            if action_score > best_score:
                best_score = action_score
                best_action = action

            alpha = max(alpha, best_score)

        if not all_actions_completed:
            # Depth not fully completed - return None to signal interruption
            return None

        # If no action was fully evaluated, shouldn't happen since all_actions_completed
        if not action_scores:
            return None

        return SearchResult(
            best_action=best_action,
            best_score=best_score,
            action_scores=action_scores,
            nodes_evaluated=self._nodes_evaluated,
            depth_reached=depth,
            timed_out=False,
            root_children=root_children,
        )

    def _estimate_remaining_time(
        self, elapsed: float, current_depth: int, max_depth: int
    ) -> float:
        """Estimate remaining search time based on elapsed time and depth progress.

        Uses exponential growth model: each additional depth level takes
        approximately branching_factor times longer than the previous.

        Parameters
        ----------
        elapsed : float
            Time elapsed so far in seconds.
        current_depth : int
            Depth currently being explored.
        max_depth : int
            Maximum configured depth.

        Returns
        -------
        float
            Estimated remaining seconds. Returns -1.0 if estimation
            is not yet available.
        """
        if current_depth <= 3 or elapsed < 1.0:
            return -1.0  # Not enough data to estimate

        # Estimate based on exponential branching
        # Average branching factor for remaining levels
        remaining_levels = max_depth - current_depth
        if remaining_levels <= 0:
            return 0.0

        # Rough exponential: each depth multiplies time by ~branching factor
        # Average branching factor across extended search is ~4
        avg_branching = 4.0
        time_per_level = elapsed / max(1, current_depth - 2)  # Time per completed level
        estimated_remaining = 0.0
        for i in range(remaining_levels):
            time_per_level *= avg_branching
            estimated_remaining += time_per_level

        # Cap at remaining timeout
        remaining_timeout = self._timeout - elapsed
        return min(estimated_remaining, max(0.0, remaining_timeout))

    def _build_extended_report(
        self,
        state: MarketState,
        best_action: Action,
        best_score: float,
        action_scores: dict,
        effective_depth: int,
        total_nodes: int,
        elapsed: float,
        standard_result: "SearchResult",
        root_children: List[SearchNode],
    ) -> "ExtendedDecisionReport":
        """Build an ExtendedDecisionReport from search results.

        Parameters
        ----------
        state : MarketState
            The market state analyzed.
        best_action : Action
            Best action from the extended search.
        best_score : float
            Score of the best action.
        action_scores : dict
            Mapping of actions to scores.
        effective_depth : int
            Deepest fully evaluated depth.
        total_nodes : int
            Total leaf nodes evaluated.
        elapsed : float
            Total time elapsed.
        standard_result : SearchResult
            Standard search result for comparison.
        root_children : list of SearchNode
            Root children from the deepest complete search.

        Returns
        -------
        ExtendedDecisionReport
            Complete extended decision report.
        """
        from engine.extended_search import ExtendedDecisionReport

        # Compute confidence from the best search result
        from engine.search_module_impl import compute_confidence

        best_search_result = SearchResult(
            best_action=best_action,
            best_score=best_score,
            action_scores=action_scores,
            nodes_evaluated=total_nodes,
            depth_reached=effective_depth,
            timed_out=False,
            root_children=root_children,
        )
        confidence = compute_confidence(self._config, best_search_result)

        # Compute standard confidence
        standard_confidence = compute_confidence(self._config, standard_result)

        # Determine if recommendations agree
        recommendations_agree = best_action == standard_result.best_action

        # Extract top scenarios and indicators using report generator
        from engine.search_module_impl import (
            extract_top_scenarios,
            compute_indicator_contributions,
        )

        top_scenarios = extract_top_scenarios(self._config, best_search_result)
        top_indicators = compute_indicator_contributions(self._config, state)

        return ExtendedDecisionReport(
            symbol=state.symbol,
            recommended_action=best_action,
            confidence=confidence,
            position_score=best_score,
            top_scenarios=top_scenarios,
            top_indicators=top_indicators,
            total_nodes_evaluated=total_nodes,
            effective_depth_reached=effective_depth,
            time_elapsed_seconds=elapsed,
            standard_recommendation=standard_result.best_action,
            standard_confidence=standard_confidence,
            standard_position_score=standard_result.best_score,
            recommendations_agree=recommendations_agree,
        )


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

        Delegates to the implementation in search_module_impl.

        Parameters
        ----------
        search_result : SearchResult
            The search result with action_scores and root_children.

        Returns
        -------
        float
            Confidence in [0.0, 1.0].
        """
        from engine.search_module_impl import compute_confidence
        return compute_confidence(self._config, search_result)

    def _extract_top_scenarios(self, search_result: SearchResult) -> List[ScenarioResult]:
        """Extract top N scenarios from root children, sorted by score.

        Delegates to the implementation in search_module_impl.

        Parameters
        ----------
        search_result : SearchResult
            The search result containing root_children.

        Returns
        -------
        List[ScenarioResult]
            Top scenarios (up to top_scenarios_report, default 3).
        """
        from engine.search_module_impl import extract_top_scenarios
        return extract_top_scenarios(self._config, search_result)

    def _describe_scenario(
        self, action_sequence: List[Action], score: float
    ) -> str:
        """Generate a human-readable description for a scenario.

        Delegates to the implementation in search_module_impl.

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
        from engine.search_module_impl import describe_scenario
        return describe_scenario(action_sequence, score)

    def _compute_indicator_contributions(
        self, state: MarketState
    ) -> List[IndicatorContribution]:
        """Compute top N indicator contributions to the position score.

        Delegates to the implementation in search_module_impl.

        Parameters
        ----------
        state : MarketState
            Current market state with indicator values.

        Returns
        -------
        List[IndicatorContribution]
            Top 5 indicators sorted by absolute contribution (descending).
        """
        from engine.search_module_impl import compute_indicator_contributions
        return compute_indicator_contributions(self._config, state)


# ==============================================================================
# Helper functions (public API preserved)
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
    from engine.search_module_impl import forward_fill_2d
    return forward_fill_2d(arr)
