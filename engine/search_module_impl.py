"""
Search Module Implementation: Heavy computation helpers.

Contains the minimax algorithm internals and utility functions extracted
from search_module.py to keep the main module under 700 lines.

These functions are NOT part of the public API — they are called by
SearchModule and DecisionReportGenerator in the main module.
"""

import logging
import time
from typing import List, Optional

import numpy as np

from engine.config import (
    Action,
    EngineError,
    IndicatorContribution,
    ScenarioResult,
    SearchConfig,
)
from engine.market_state import INDICATOR_COLUMNS, MarketState

logger = logging.getLogger(__name__)


# ==============================================================================
# Minimax implementation
# ==============================================================================


def minimax(
    search_module,
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
    search_module : SearchModule
        The search module instance (provides access to state and helpers).
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
    if is_timed_out(search_module):
        search_module._timed_out = True
        # Return current evaluation as best estimate
        return evaluate_leaf(search_module, state)

    # Leaf node: evaluate with model
    if depth <= 0:
        return evaluate_leaf(search_module, state)

    if is_maximizing:
        return _minimax_maximizing(
            search_module, state, depth, alpha, beta, current_depth, max_depth
        )
    else:
        return _minimax_minimizing(
            search_module, state, depth, alpha, beta, current_depth, max_depth
        )


def _minimax_maximizing(
    search_module,
    state: MarketState,
    depth: int,
    alpha: float,
    beta: float,
    current_depth: int,
    max_depth: int,
) -> float:
    """Maximizing player (trader) branch of minimax.

    Parameters
    ----------
    search_module : SearchModule
        The search module instance.
    state : MarketState
        Current state to expand.
    depth : int
        Remaining depth to search.
    alpha : float
        Best value the maximizer can guarantee.
    beta : float
        Best value the minimizer can guarantee.
    current_depth : int
        Current depth in the tree.
    max_depth : int
        Total maximum search depth.

    Returns
    -------
    float
        The maximum minimax score from this subtree.
    """
    max_eval = -float("inf")
    actions = [Action.BUY, Action.HOLD, Action.SELL]

    for action in actions:
        if is_timed_out(search_module):
            search_module._timed_out = True
            break

        # Generate scenarios for this action
        num_scenarios = get_branching_factor(search_module, current_depth, max_depth)
        try:
            scenarios = search_module._scenario_generator.generate(
                state, action, num_scenarios=num_scenarios
            )
        except EngineError:
            continue

        # Market responds (minimizing over scenarios)
        for scenario_state in scenarios:
            if is_timed_out(search_module):
                search_module._timed_out = True
                break

            eval_score = minimax(
                search_module,
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


def _minimax_minimizing(
    search_module,
    state: MarketState,
    depth: int,
    alpha: float,
    beta: float,
    current_depth: int,
    max_depth: int,
) -> float:
    """Minimizing player (market) branch of minimax.

    Parameters
    ----------
    search_module : SearchModule
        The search module instance.
    state : MarketState
        Current state to expand.
    depth : int
        Remaining depth to search.
    alpha : float
        Best value the maximizer can guarantee.
    beta : float
        Best value the minimizer can guarantee.
    current_depth : int
        Current depth in the tree.
    max_depth : int
        Total maximum search depth.

    Returns
    -------
    float
        The minimum minimax score from this subtree.
    """
    min_eval = float("inf")
    actions = [Action.BUY, Action.HOLD, Action.SELL]

    for action in actions:
        if is_timed_out(search_module):
            search_module._timed_out = True
            break

        # Generate scenarios for this action (market movement)
        num_scenarios = get_branching_factor(search_module, current_depth, max_depth)
        try:
            scenarios = search_module._scenario_generator.generate(
                state, action, num_scenarios=num_scenarios
            )
        except EngineError:
            continue

        for scenario_state in scenarios:
            if is_timed_out(search_module):
                search_module._timed_out = True
                break

            eval_score = minimax(
                search_module,
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


# ==============================================================================
# Leaf evaluation
# ==============================================================================


def evaluate_leaf(search_module, state: MarketState) -> float:
    """Evaluate a leaf node state using the model manager.

    Converts the MarketState to a feature array and runs inference
    through the ModelManager.predict() API.

    Parameters
    ----------
    search_module : SearchModule
        The search module instance.
    state : MarketState
        The leaf state to evaluate.

    Returns
    -------
    float
        Position score in [-1, +1].
    """
    search_module._nodes_evaluated += 1

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
        features = forward_fill_2d(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict: ModelManager expects shape (batch, lookback, num_features)
        # or (lookback, num_features) for single sample
        scores = search_module._model_manager.predict(features)

        # scores shape: (1, 1) → extract scalar
        score = float(scores.flatten()[0])

        # Clamp to [-1, 1] for safety
        return max(-1.0, min(1.0, score))

    except Exception as e:
        logger.warning(f"Leaf evaluation failed: {e}")
        return 0.0


# ==============================================================================
# Branching factor
# ==============================================================================


def get_branching_factor(search_module, current_depth: int, max_depth: int) -> int:
    """Determine the number of scenarios to generate at this depth.

    Implements adaptive branching: when depth > adaptive_depth_threshold
    (default 3), reduces scenarios to maintain the 5-second constraint.

    Parameters
    ----------
    search_module : SearchModule
        The search module instance.
    current_depth : int
        Current depth in the search tree.
    max_depth : int
        Total maximum search depth.

    Returns
    -------
    int
        Number of scenarios to generate (3, 5, or 7).
    """
    config = search_module._config
    # If max_depth exceeds the adaptive threshold, reduce branching
    # at deeper levels to maintain time constraint
    if max_depth > config.adaptive_depth_threshold:
        if current_depth >= config.adaptive_depth_threshold:
            return config.adaptive_scenario_count  # 3
        else:
            # Even at shallow levels, reduce slightly for deep searches
            return config.default_scenarios  # 5
    else:
        # Standard branching at default depth
        if current_depth >= 2:
            # Slight reduction at deeper levels even for depth=3
            return config.adaptive_scenario_count  # 3
        return config.default_scenarios  # 5


# ==============================================================================
# Timeout check
# ==============================================================================


def is_timed_out(search_module) -> bool:
    """Check if the search has exceeded the timeout.

    Parameters
    ----------
    search_module : SearchModule
        The search module instance.

    Returns
    -------
    bool
        True if elapsed time exceeds the configured timeout.
    """
    elapsed = time.time() - search_module._start_time
    return elapsed >= search_module._timeout


# ==============================================================================
# Report generation helpers
# ==============================================================================


def compute_confidence(config, search_result) -> float:
    """Compute confidence from how much scenarios agree.

    Confidence measures the agreement level among root-level action scores.
    High confidence means one action clearly dominates; low confidence
    means actions have similar scores (ambiguous situation).

    The confidence is computed using both action score spread and child
    score variance. Lower spread AND lower variance → higher confidence.

    Parameters
    ----------
    config : SearchConfig
        Search configuration.
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


def extract_top_scenarios(config, search_result) -> List[ScenarioResult]:
    """Extract top N scenarios from root children, sorted by score.

    Each scenario represents an action node from the root level with
    its best evaluated child path.

    Parameters
    ----------
    config : SearchConfig
        Search configuration.
    search_result : SearchResult
        The search result containing root_children.

    Returns
    -------
    List[ScenarioResult]
        Top scenarios (up to top_scenarios_report, default 3).
    """
    top_n = config.top_scenarios_report  # Default 3
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
        description = describe_scenario(action_sequence, root_child.score)

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


def describe_scenario(action_sequence: List[Action], score: float) -> str:
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


def compute_indicator_contributions(
    config, state: MarketState
) -> List[IndicatorContribution]:
    """Compute top N indicator contributions to the position score.

    Contributions are estimated by analyzing which indicators deviate
    most from their neutral values (midpoint of their typical range).
    Indicators with larger absolute deviations from neutral contribute
    more to the score differentiation.

    Parameters
    ----------
    config : SearchConfig
        Search configuration.
    state : MarketState
        Current market state with indicator values.

    Returns
    -------
    List[IndicatorContribution]
        Top 5 indicators sorted by absolute contribution (descending).
    """
    top_n = config.top_indicators_report  # Default 5

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


def forward_fill_2d(arr: np.ndarray) -> np.ndarray:
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
