"""
Property-based tests for Scenario Generator Directional Accuracy Measurement.

Tests the following correctness property from the design document:
- Property 24: Scenario generator directional accuracy measurement
  - accuracy = count(sign match) / N, always in [0.0, 1.0]
  - perfect match = 1.0
  - <100 samples never triggers B→A transition

**Validates: Requirements 16.5**
"""

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.progressive_trainer import (
    PhaseTransitionCriteria,
    ProgressiveTrainer,
    ProgressiveTrainingConfig,
    TrainingPhase,
)


# ---------------------------------------------------------------------------
# Custom strategies for directional accuracy tests
# ---------------------------------------------------------------------------


@st.composite
def prediction_actual_pairs(draw, min_size=1, max_size=300):
    """
    Generate lists of prediction/actual pairs with various distributions.

    Values are non-zero floats to avoid ambiguity in sign matching,
    unless specifically testing zero-handling.
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    predictions = draw(
        st.lists(
            st.floats(min_value=-100.0, max_value=100.0,
                      allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    actuals = draw(
        st.lists(
            st.floats(min_value=-100.0, max_value=100.0,
                      allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    return predictions, actuals


@st.composite
def perfect_match_pairs(draw, min_size=1, max_size=300):
    """
    Generate prediction/actual pairs where signs always match.

    For each pair, both values have the same sign (both positive,
    both negative, or both zero).
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    predictions = []
    actuals = []

    for _ in range(n):
        sign_choice = draw(st.sampled_from(["positive", "negative", "zero"]))
        if sign_choice == "positive":
            pred = draw(st.floats(min_value=0.001, max_value=100.0,
                                  allow_nan=False, allow_infinity=False))
            actual = draw(st.floats(min_value=0.001, max_value=100.0,
                                    allow_nan=False, allow_infinity=False))
        elif sign_choice == "negative":
            pred = draw(st.floats(min_value=-100.0, max_value=-0.001,
                                  allow_nan=False, allow_infinity=False))
            actual = draw(st.floats(min_value=-100.0, max_value=-0.001,
                                    allow_nan=False, allow_infinity=False))
        else:
            pred = 0.0
            actual = 0.0
        predictions.append(pred)
        actuals.append(actual)

    return predictions, actuals


@st.composite
def small_sample_pairs(draw):
    """
    Generate prediction/actual pairs with fewer than 100 samples.
    Even with perfect accuracy, should never trigger B→A transition.
    """
    n = draw(st.integers(min_value=1, max_value=99))
    predictions = draw(
        st.lists(
            st.floats(min_value=0.001, max_value=100.0,
                      allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    # Make actuals have same sign as predictions (perfect match)
    actuals = draw(
        st.lists(
            st.floats(min_value=0.001, max_value=100.0,
                      allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    return predictions, actuals


# ---------------------------------------------------------------------------
# Helper: compute expected accuracy manually
# ---------------------------------------------------------------------------


def compute_expected_accuracy(predictions, actuals):
    """Manually compute directional accuracy for verification."""
    n = len(predictions)
    if n == 0:
        return 0.0
    correct = sum(
        1
        for pred, actual in zip(predictions, actuals)
        if (pred > 0 and actual > 0)
        or (pred < 0 and actual < 0)
        or (pred == 0 and actual == 0)
    )
    return correct / n


# ---------------------------------------------------------------------------
# Property 24: Scenario generator directional accuracy measurement
# ---------------------------------------------------------------------------


class TestProperty24DirectionalAccuracyMeasurement:
    """
    Property 24: Scenario generator directional accuracy measurement.

    For any list of predictions and actuals:
    1. Accuracy = count(sign(pred) == sign(actual)) / N
    2. Accuracy is always in [0.0, 1.0]
    3. Perfect sign match → accuracy = 1.0
    4. If N < 100 samples, B→A transition is NEVER triggered

    **Validates: Requirements 16.5**
    """

    @given(data=prediction_actual_pairs(min_size=1, max_size=300))
    @settings(max_examples=100, deadline=None)
    def test_accuracy_always_in_unit_interval(self, data):
        """
        For any predictions/actuals pair, the computed accuracy is
        always in [0.0, 1.0].

        **Validates: Requirements 16.5**
        """
        predictions, actuals = data

        trainer = ProgressiveTrainer()
        # Set phase to B so transition logic is meaningful
        trainer.current_phase = TrainingPhase.PHASE_B

        trainer.check_b_to_a_transition(predictions, actuals)

        accuracy = trainer.transition_criteria.scenario_accuracy
        sample_count = trainer.transition_criteria.accuracy_sample_count

        # If N < required_sample_count, accuracy may remain at default 0.0
        # but sample_count should still be tracked
        if len(predictions) >= trainer.transition_criteria.required_sample_count:
            assert 0.0 <= accuracy <= 1.0, (
                f"Accuracy {accuracy} is outside [0.0, 1.0] "
                f"for {len(predictions)} samples"
            )
        else:
            # Below threshold, accuracy not computed (stays 0.0)
            assert accuracy == 0.0 or (0.0 <= accuracy <= 1.0)

        assert sample_count == len(predictions)

    @given(data=prediction_actual_pairs(min_size=100, max_size=300))
    @settings(max_examples=100, deadline=None)
    def test_accuracy_equals_sign_match_count_over_n(self, data):
        """
        Accuracy is computed exactly as count(sign match) / N.

        **Validates: Requirements 16.5**
        """
        predictions, actuals = data

        trainer = ProgressiveTrainer()
        trainer.current_phase = TrainingPhase.PHASE_B

        trainer.check_b_to_a_transition(predictions, actuals)

        computed_accuracy = trainer.transition_criteria.scenario_accuracy
        expected_accuracy = compute_expected_accuracy(predictions, actuals)

        assert abs(computed_accuracy - expected_accuracy) < 1e-10, (
            f"Computed accuracy {computed_accuracy} != expected {expected_accuracy} "
            f"for {len(predictions)} samples"
        )

    @given(data=perfect_match_pairs(min_size=100, max_size=300))
    @settings(max_examples=100, deadline=None)
    def test_perfect_match_gives_accuracy_one(self, data):
        """
        When all predictions have the same sign as their corresponding
        actuals, accuracy must equal 1.0.

        **Validates: Requirements 16.5**
        """
        predictions, actuals = data

        trainer = ProgressiveTrainer()
        trainer.current_phase = TrainingPhase.PHASE_B

        trainer.check_b_to_a_transition(predictions, actuals)

        accuracy = trainer.transition_criteria.scenario_accuracy

        assert accuracy == 1.0, (
            f"Perfect sign match should give accuracy 1.0 but got {accuracy}. "
            f"N={len(predictions)}"
        )

    @given(data=small_sample_pairs())
    @settings(max_examples=100, deadline=None)
    def test_fewer_than_100_samples_never_triggers_transition(self, data):
        """
        When N < 100 (the required_sample_count), the B→A transition
        is NEVER triggered, regardless of how high the accuracy would be.

        **Validates: Requirements 16.5**
        """
        predictions, actuals = data
        assume(len(predictions) < 100)

        trainer = ProgressiveTrainer()
        trainer.current_phase = TrainingPhase.PHASE_B

        result = trainer.check_b_to_a_transition(predictions, actuals)

        assert result is False, (
            f"B→A transition should not trigger with N={len(predictions)} < 100 "
            f"samples, but check_b_to_a_transition returned True"
        )
        assert trainer.transition_criteria.transition_available is False, (
            f"transition_available should be False with N={len(predictions)} < 100"
        )

    @given(
        data=prediction_actual_pairs(min_size=100, max_size=300),
    )
    @settings(max_examples=100, deadline=None)
    def test_transition_triggered_iff_accuracy_above_threshold(self, data):
        """
        When N >= 100, the B→A transition is triggered if and only if
        accuracy > 0.6 (the required_accuracy threshold).

        **Validates: Requirements 16.5**
        """
        predictions, actuals = data
        assume(len(predictions) >= 100)

        trainer = ProgressiveTrainer()
        trainer.current_phase = TrainingPhase.PHASE_B

        result = trainer.check_b_to_a_transition(predictions, actuals)

        expected_accuracy = compute_expected_accuracy(predictions, actuals)

        if expected_accuracy > 0.6:
            assert result is True, (
                f"B→A transition should trigger with accuracy={expected_accuracy:.4f} > 0.6 "
                f"and N={len(predictions)} >= 100"
            )
            assert trainer.transition_criteria.transition_available is True
        else:
            assert result is False, (
                f"B→A transition should NOT trigger with accuracy={expected_accuracy:.4f} <= 0.6 "
                f"and N={len(predictions)} >= 100"
            )
