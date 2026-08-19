"""
Extended Search module for production trading decisions.

Provides configuration, progress tracking, and enhanced decision reports
for extended search mode (depth up to 10, timeout up to 30 minutes).

Extended search is intended for real trading decisions where thoroughness
is prioritized over speed. It includes comparison with standard search
results and real-time progress reporting.

Requirements: 17.1, 17.7, 17.8
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from engine.config import (
    Action,
    ConfigError,
    DecisionReport,
    IndicatorContribution,
    ScenarioResult,
)


# ==============================================================================
# Extended Search Configuration
# ==============================================================================


@dataclass
class ExtendedSearchConfig:
    """Configuration for extended production search.

    Extended search allows deeper exploration (depth 3-10) and longer
    timeout (60-1800 seconds) compared to standard search (depth 3, 5s).

    Attributes
    ----------
    depth : int
        Search depth, must be in range [3, 10]. Default is 10.
    timeout : float
        Maximum search time in seconds, must be in range [60, 1800].
        Default is 1800 (30 minutes).
    enabled : bool
        Whether extended search mode is enabled. Default is False.

    Raises
    ------
    ConfigError
        If depth is not in [3, 10] or timeout is not in [60, 1800].
    """

    depth: int = 10
    timeout: float = 1800.0
    enabled: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if not (3 <= self.depth <= 10):
            raise ConfigError(
                f"Extended search depth must be between 3 and 10, got {self.depth}",
                error_code="INVALID_EXTENDED_DEPTH",
                details={"depth": self.depth, "valid_range": [3, 10]},
            )
        if not (60.0 <= self.timeout <= 1800.0):
            raise ConfigError(
                f"Extended search timeout must be between 60 and 1800 seconds, got {self.timeout}",
                error_code="INVALID_EXTENDED_TIMEOUT",
                details={"timeout": self.timeout, "valid_range": [60.0, 1800.0]},
            )


# ==============================================================================
# Extended Search Progress
# ==============================================================================


@dataclass
class ExtendedSearchProgress:
    """Real-time progress information for an ongoing extended search.

    Reported via callback at each depth level during iterative deepening.

    Attributes
    ----------
    current_depth : int
        The depth level currently being explored.
    max_depth : int
        The maximum configured depth for this search.
    nodes_evaluated : int
        Total number of leaf nodes evaluated so far.
    elapsed_seconds : float
        Time elapsed since search started, in seconds.
    estimated_remaining_seconds : float
        Estimated time remaining to complete the search, in seconds.
        May be -1.0 if estimation is not yet available.
    best_action_so_far : Action or None
        The best action found from the deepest fully completed level.
        None if no level has been fully evaluated yet.
    best_score_so_far : float or None
        The score of the best action found so far.
        None if no level has been fully evaluated yet.
    """

    current_depth: int = 0
    max_depth: int = 10
    nodes_evaluated: int = 0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = -1.0
    best_action_so_far: Optional[Action] = None
    best_score_so_far: Optional[float] = None


# ==============================================================================
# Extended Decision Report
# ==============================================================================


@dataclass
class ExtendedDecisionReport(DecisionReport):
    """Extended decision report with comparison to standard search.

    Extends DecisionReport with additional fields for the extended search
    results, including node count, effective depth, elapsed time, and a
    comparison between the extended and standard search recommendations.

    Attributes
    ----------
    total_nodes_evaluated : int
        Total number of leaf nodes evaluated during the extended search.
    effective_depth_reached : int
        The deepest level that was fully evaluated before completion/timeout.
    time_elapsed_seconds : float
        Total wall-clock time spent on the extended search.
    standard_recommendation : Action or None
        The action that standard search (depth 3, 5s) would have recommended.
    standard_confidence : float or None
        The confidence score from the standard search.
    standard_position_score : float or None
        The position score from the standard search.
    recommendations_agree : bool
        Whether extended and standard search recommend the same action.
    """

    total_nodes_evaluated: int = 0
    effective_depth_reached: int = 0
    time_elapsed_seconds: float = 0.0
    standard_recommendation: Optional[Action] = None
    standard_confidence: Optional[float] = None
    standard_position_score: Optional[float] = None
    recommendations_agree: bool = True
