"""
Property-based tests for Progressive Training Phase Transitions.

Tests the following correctness property from the design document:
- Property 22: Training phase transition criteria correctness
  - C→B transition: flagged iff model Sharpe > 2/4 strategies for 3+
    consecutive rounds AND user not confirmed
  - B→A transition: flagged iff >60% accuracy on 100+ samples

**Validates: Requirements 16.3, 16.5, 16.9**
"""

from datetime import datetime

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.config import EngineError
from engine.progressive_trainer import (
    PhaseTransitionCriteria,
    ProgressiveTrainer,
    ProgressiveTrainingConfig,
    TrainingPhase,
    ValidationRound,
)


# ---------------------------------------------------------------------------
# Custom Hypothesis strategies for phase transition tests
# ---------------------------------------------------------------------------


@st.composite
def validation_round_strategy(draw, round_number=None):
    """Generate a ValidationRound with random Sharpe ratios.

    Generates a model Sharpe and 4 strategy Sharpes to test C→B logic.
    """
    if round_number is None:
        round_number = draw(st.integers(min_value=1, max_value=100))

    model_sharpe = draw(
        st.floats(min_value=-3.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    strategy_sharpes = draw(
        st.lists(
            st.floats(min_value=-3.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=4,
            max_size=4,
        )
    )
    model_version = f"v{round_number}"

    return ValidationRound(
        round_number=round_number,
        model_version=model_version,
        model_sharpe=model_sharpe,
        strategy_sharpes=strategy_sharpes,
        phase=TrainingPhase.PHASE_C,
        timestamp=datetime.now(),
    )


@st.composite
def c_to_b_scenario_strategy(draw):
    """Generate a full scenario of multiple validation rounds for C→B testing.

    Returns a list of ValidationRounds and whether the transition should trigger.
    """
    num_rounds = draw(st.integers(min_value=1, max_value=10))
    rounds = []
    for i in range(num_rounds):
        vr = draw(validation_round_strategy(round_number=i + 1))
        rounds.append(vr)
    return rounds


@st.composite
def directional_accuracy_strategy(draw):
    """Generate predictions and actuals for B→A transition testing.

    Returns predictions list, actuals list.
    """
    num_samples = draw(st.integers(min_value=0, max_value=300))
    predictions = draw(
        st.lists(
            st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=num_samples,
            max_size=num_samples,
        )
    )
    actuals = draw(
        st.lists(
            st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=num_samples,
            max_size=num_samples,
        )
    )
    return {"predictions": predictions, "actuals": actuals}


# ---------------------------------------------------------------------------
# Property 22: Training phase transition criteria correctness
# ---------------------------------------------------------------------------


class TestProperty22PhaseTransitionCriteria:
    """
    Property 22: Training phase transition criteria correctness.

    C→B transition: flagged iff model Sharpe > 2/4 strategies for 3+
    consecutive rounds AND user not confirmed.

    B→A transition: flagged iff >60% accuracy on 100+ samples.

    **Validates: Requirements 16.3, 16.5, 16.9**
    """

    # ==================================================================
    # C → B Transition Tests
    # ==================================================================

    @given(rounds=c_to_b_scenario_strategy())
    @settings(max_examples=50, deadline=None)
    def test_c_to_b_transition_flagged_only_after_3_consecutive_wins(self, rounds):
        """
        The C→B transition is flagged as available if and only if the model
        achieves Sharpe > at least 2/4 strategies for 3+ consecutive rounds.

        We simulate the sequence of rounds and verify that:
        - transition_available becomes True exactly when the consecutive
          win streak reaches 3
        - transition_available stays False when streak hasn't reached 3

        **Validates: Requirements 16.3**
        """
        trainer = ProgressiveTrainer(config=ProgressiveTrainingConfig())

        # Track consecutive wins manually to verify
        expected_streak = 0

        for vr in rounds:
            # Compute how many strategies the model beats
            strategies_beaten = sum(
                1 for s in vr.strategy_sharpes if vr.model_sharpe > s
            )

            # Update expected streak
            if strategies_beaten >= 2:
                expected_streak += 1
            else:
                expected_streak = 0

            result = trainer.check_c_to_b_transition(vr)

            # Verify the result matches our expected logic
            if expected_streak >= 3:
                assert result is True, (
                    f"Expected transition to be flagged: streak={expected_streak}, "
                    f"strategies_beaten={strategies_beaten}"
                )
                assert trainer.transition_criteria.transition_available is True
            else:
                assert result is False, (
                    f"Transition should NOT be flagged: streak={expected_streak}, "
                    f"strategies_beaten={strategies_beaten}"
                )

    @given(
        model_sharpe=st.floats(min_value=-3.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        strategy_sharpes=st.lists(
            st.floats(min_value=-3.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=4,
            max_size=4,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_c_to_b_single_round_never_triggers(self, model_sharpe, strategy_sharpes):
        """
        A single validation round should never trigger the C→B transition,
        since 3 consecutive rounds are required.

        **Validates: Requirements 16.3**
        """
        trainer = ProgressiveTrainer(config=ProgressiveTrainingConfig())

        vr = ValidationRound(
            round_number=1,
            model_version="v1",
            model_sharpe=model_sharpe,
            strategy_sharpes=strategy_sharpes,
            phase=TrainingPhase.PHASE_C,
            timestamp=datetime.now(),
        )
        result = trainer.check_c_to_b_transition(vr)

        # A single round can never reach 3 consecutive wins
        assert result is False
        assert trainer.transition_criteria.transition_available is False

    @given(
        winning_sharpe=st.floats(min_value=2.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_c_to_b_three_consecutive_winning_rounds_triggers(self, winning_sharpe):
        """
        When the model beats >=2 strategies for exactly 3 consecutive rounds,
        the transition must be flagged as available.

        **Validates: Requirements 16.3**
        """
        trainer = ProgressiveTrainer(config=ProgressiveTrainingConfig())

        # Create 3 rounds where model always beats at least 2 strategies
        # Strategy sharpes are all lower than the model
        low_sharpes = [winning_sharpe - 1.0, winning_sharpe - 0.5,
                       winning_sharpe - 2.0, winning_sharpe + 1.0]
        # Model beats 3 out of 4 strategies (all except the last one)

        for i in range(3):
            vr = ValidationRound(
                round_number=i + 1,
                model_version=f"v{i + 1}",
                model_sharpe=winning_sharpe,
                strategy_sharpes=low_sharpes,
                phase=TrainingPhase.PHASE_C,
                timestamp=datetime.now(),
            )
            result = trainer.check_c_to_b_transition(vr)

        # After 3 consecutive wins (beating 3/4 strategies each time), transition flags
        assert result is True
        assert trainer.transition_criteria.transition_available is True

    @given(
        model_sharpe=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        loss_round_position=st.integers(min_value=0, max_value=2),
    )
    @settings(max_examples=30, deadline=None)
    def test_c_to_b_streak_resets_on_losing_round(self, model_sharpe, loss_round_position):
        """
        If the model fails to beat >=2 strategies in any round, the
        consecutive win streak resets to 0.

        **Validates: Requirements 16.3**
        """
        trainer = ProgressiveTrainer(config=ProgressiveTrainingConfig())

        # Setup: strategy sharpes that the model always loses to (all higher)
        losing_sharpes = [model_sharpe + 1.0, model_sharpe + 2.0,
                          model_sharpe + 3.0, model_sharpe + 4.0]
        # Model beats 0 strategies

        # Winning sharpes (model beats 3/4)
        winning_sharpes = [model_sharpe - 1.0, model_sharpe - 0.5,
                           model_sharpe - 2.0, model_sharpe + 10.0]

        # Submit 3 rounds, but one of them is a loss
        for i in range(3):
            if i == loss_round_position:
                sharpes = losing_sharpes
            else:
                sharpes = winning_sharpes

            vr = ValidationRound(
                round_number=i + 1,
                model_version=f"v{i + 1}",
                model_sharpe=model_sharpe,
                strategy_sharpes=sharpes,
                phase=TrainingPhase.PHASE_C,
                timestamp=datetime.now(),
            )
            trainer.check_c_to_b_transition(vr)

        # The loss breaks the streak, so transition should NOT be available
        assert trainer.transition_criteria.transition_available is False

    # ==================================================================
    # B → A Transition Tests
    # ==================================================================

    @given(data=directional_accuracy_strategy())
    @settings(max_examples=50, deadline=None)
    def test_b_to_a_transition_criteria_correctness(self, data):
        """
        B→A transition is flagged iff directional accuracy > 60% on 100+
        samples. Verify the logic matches expected behavior for any
        combination of predictions and actuals.

        **Validates: Requirements 16.5**
        """
        predictions = data["predictions"]
        actuals = data["actuals"]

        trainer = ProgressiveTrainer(config=ProgressiveTrainingConfig())
        result = trainer.check_b_to_a_transition(predictions, actuals)

        n = len(predictions)

        if n < 100:
            # Insufficient samples → never triggers
            assert result is False
            assert trainer.transition_criteria.transition_available is False
        else:
            # Compute expected accuracy
            correct = sum(
                1
                for pred, actual in zip(predictions, actuals)
                if (pred > 0 and actual > 0)
                or (pred < 0 and actual < 0)
                or (pred == 0 and actual == 0)
            )
            expected_accuracy = correct / n

            if expected_accuracy > 0.60:
                assert result is True
                assert trainer.transition_criteria.transition_available is True
                assert trainer.transition_criteria.scenario_accuracy == expected_accuracy
            else:
                assert result is False
                # transition_available may have been set by a previous call, but fresh trainer → False
                assert trainer.transition_criteria.scenario_accuracy == expected_accuracy

    @given(
        num_samples=st.integers(min_value=0, max_value=99),
    )
    @settings(max_examples=30, deadline=None)
    def test_b_to_a_insufficient_samples_never_triggers(self, num_samples):
        """
        With fewer than 100 samples, the B→A transition is never triggered
        regardless of accuracy.

        **Validates: Requirements 16.5**
        """
        trainer = ProgressiveTrainer(config=ProgressiveTrainingConfig())

        # Even with perfect predictions (all match), <100 samples → no trigger
        predictions = [1.0] * num_samples
        actuals = [1.0] * num_samples

        result = trainer.check_b_to_a_transition(predictions, actuals)

        assert result is False
        assert trainer.transition_criteria.transition_available is False

    @given(
        num_samples=st.integers(min_value=100, max_value=300),
    )
    @settings(max_examples=30, deadline=None)
    def test_b_to_a_perfect_accuracy_always_triggers(self, num_samples):
        """
        With 100+ samples and perfect directional accuracy (100%),
        the B→A transition must be flagged.

        **Validates: Requirements 16.5**
        """
        trainer = ProgressiveTrainer(config=ProgressiveTrainingConfig())

        # Perfect predictions: all positive → all match
        predictions = [1.0] * num_samples
        actuals = [2.0] * num_samples

        result = trainer.check_b_to_a_transition(predictions, actuals)

        assert result is True
        assert trainer.transition_criteria.transition_available is True
        assert trainer.transition_criteria.scenario_accuracy == 1.0

    @given(
        num_samples=st.integers(min_value=100, max_value=300),
    )
    @settings(max_examples=30, deadline=None)
    def test_b_to_a_zero_accuracy_never_triggers(self, num_samples):
        """
        With 100+ samples but 0% accuracy (all directions wrong),
        the B→A transition must NOT be flagged.

        **Validates: Requirements 16.5**
        """
        trainer = ProgressiveTrainer(config=ProgressiveTrainingConfig())

        # All predictions are positive but actuals are negative → 0% accuracy
        predictions = [1.0] * num_samples
        actuals = [-1.0] * num_samples

        result = trainer.check_b_to_a_transition(predictions, actuals)

        assert result is False
        assert trainer.transition_criteria.transition_available is False
        assert trainer.transition_criteria.scenario_accuracy == 0.0

    # ==================================================================
    # User Confirmation Gate (Req 16.9)
    # ==================================================================

    @given(
        winning_sharpe=st.floats(min_value=2.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_transition_requires_user_confirmation(self, winning_sharpe):
        """
        Even after transition criteria are met, the phase does NOT advance
        until confirm_transition() is called (user confirmation gate).

        **Validates: Requirements 16.9**
        """
        trainer = ProgressiveTrainer(config=ProgressiveTrainingConfig())

        # Create 3 winning rounds to trigger transition availability
        low_sharpes = [winning_sharpe - 1.0, winning_sharpe - 0.5,
                       winning_sharpe - 2.0, winning_sharpe - 3.0]

        for i in range(3):
            vr = ValidationRound(
                round_number=i + 1,
                model_version=f"v{i + 1}",
                model_sharpe=winning_sharpe,
                strategy_sharpes=low_sharpes,
                phase=TrainingPhase.PHASE_C,
                timestamp=datetime.now(),
            )
            trainer.check_c_to_b_transition(vr)

        # Transition is available but phase hasn't changed yet
        assert trainer.transition_criteria.transition_available is True
        assert trainer.current_phase == TrainingPhase.PHASE_C

        # Only after explicit confirmation does the phase advance
        new_phase = trainer.confirm_transition()
        assert new_phase == TrainingPhase.PHASE_B
        assert trainer.current_phase == TrainingPhase.PHASE_B

    @given(
        model_sharpe=st.floats(min_value=-3.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        strategy_sharpes=st.lists(
            st.floats(min_value=-3.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=4,
            max_size=4,
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_confirm_transition_raises_when_not_available(self, model_sharpe, strategy_sharpes):
        """
        Calling confirm_transition() when no transition is available
        raises EngineError.

        **Validates: Requirements 16.9**
        """
        trainer = ProgressiveTrainer(config=ProgressiveTrainingConfig())

        # Submit a single round (never enough for transition)
        vr = ValidationRound(
            round_number=1,
            model_version="v1",
            model_sharpe=model_sharpe,
            strategy_sharpes=strategy_sharpes,
            phase=TrainingPhase.PHASE_C,
            timestamp=datetime.now(),
        )
        trainer.check_c_to_b_transition(vr)

        # Attempting to confirm should raise (single round can't trigger)
        try:
            trainer.confirm_transition()
            assert False, "Should have raised EngineError"
        except EngineError as e:
            assert "NO_TRANSITION" in str(e.error_code)
