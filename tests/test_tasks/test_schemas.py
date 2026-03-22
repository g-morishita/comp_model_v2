"""Tests for declarative trial schemas."""

import pytest

from comp_model.data.schema import Event, EventPhase, Trial
from comp_model.tasks.schemas import (
    ASOCIAL_BANDIT_SCHEMA,
    SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA,
    SOCIAL_POST_OUTCOME_SCHEMA,
    SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA,
    SOCIAL_PRE_CHOICE_SCHEMA,
    TrialSchema,
    TrialSchemaStep,
)


def test_asocial_schema_exposes_decision_and_action_indices() -> None:
    """Ensure the asocial schema exposes its decision metadata."""

    assert ASOCIAL_BANDIT_SCHEMA.decision_step_indices == (1,)
    assert ASOCIAL_BANDIT_SCHEMA.action_required_indices == (1,)


def test_social_pre_choice_schema_accepts_matching_trial() -> None:
    """Ensure the pre-choice social schema validates a matching trial."""

    trial = Trial(
        trial_index=0,
        events=(
            Event(
                phase=EventPhase.INPUT,
                event_index=0,
                node_id="main",
                payload={"available_actions": (0, 1)},
            ),
            Event(
                phase=EventPhase.INPUT,
                event_index=1,
                node_id="main",
                actor_id="demonstrator",
                payload={
                    "available_actions": (0, 1),
                    "observation": {"social_action": 1, "social_reward": 0.0},
                },
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=2,
                node_id="main",
                actor_id="demonstrator",
                payload={"action": 1},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=3,
                node_id="main",
                actor_id="demonstrator",
                payload={"reward": 0.0},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=4,
                node_id="main",
                actor_id="demonstrator",
                payload={},
            ),
            Event(phase=EventPhase.UPDATE, event_index=5, node_id="main", payload={}),
            Event(phase=EventPhase.DECISION, event_index=6, node_id="main", payload={"action": 1}),
            Event(phase=EventPhase.OUTCOME, event_index=7, node_id="main", payload={"reward": 1.0}),
            Event(phase=EventPhase.UPDATE, event_index=8, node_id="main", payload={}),
        ),
    )

    SOCIAL_PRE_CHOICE_SCHEMA.validate_trial(trial)


def test_social_post_outcome_schema_rejects_actor_mismatch() -> None:
    """Ensure actor mismatches are rejected with a clear error."""

    trial = Trial(
        trial_index=0,
        events=(
            Event(
                phase=EventPhase.INPUT,
                event_index=0,
                node_id="main",
                payload={"available_actions": (0, 1)},
            ),
            Event(phase=EventPhase.DECISION, event_index=1, node_id="main", payload={"action": 1}),
            Event(phase=EventPhase.OUTCOME, event_index=2, node_id="main", payload={"reward": 1.0}),
            # Wrong: should be subject UPDATE here, not demonstrator
            Event(
                phase=EventPhase.UPDATE,
                event_index=3,
                node_id="main",
                actor_id="demonstrator",
                payload={},
            ),
            Event(
                phase=EventPhase.INPUT,
                event_index=4,
                node_id="main",
                actor_id="demonstrator",
                payload={
                    "available_actions": (0, 1),
                    "observation": {"social_action": 0, "social_reward": 1.0},
                },
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=5,
                node_id="main",
                actor_id="demonstrator",
                payload={"action": 0},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=6,
                node_id="main",
                actor_id="demonstrator",
                payload={"reward": 1.0},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=7,
                node_id="main",
                actor_id="demonstrator",
                payload={},
            ),
            Event(phase=EventPhase.UPDATE, event_index=8, node_id="main", payload={}),
        ),
    )

    with pytest.raises(ValueError, match="actor_id"):
        SOCIAL_POST_OUTCOME_SCHEMA.validate_trial(trial)


def test_schema_rejects_wrong_event_count() -> None:
    """Ensure schema validation checks trial length."""

    trial = Trial(
        trial_index=0,
        events=(
            Event(
                phase=EventPhase.INPUT,
                event_index=0,
                node_id="main",
                payload={"available_actions": (0, 1)},
            ),
        ),
    )

    with pytest.raises(ValueError, match="expected 4 events"):
        ASOCIAL_BANDIT_SCHEMA.validate_trial(trial)


def test_pre_choice_no_self_outcome_schema_outcome_not_observable() -> None:
    """Ensure the subject OUTCOME step has outcome_observable=False."""
    # step 7 is the subject OUTCOME step (index 7 in the 9-step schema)
    subject_outcome_step = SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA.steps[7]
    assert subject_outcome_step.outcome_observable is False


def test_post_outcome_no_self_outcome_schema_outcome_not_observable() -> None:
    """Ensure the subject OUTCOME step has outcome_observable=False."""
    # step 2 is the subject OUTCOME step (index 2 in the 9-step schema)
    subject_outcome_step = SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA.steps[2]
    assert subject_outcome_step.outcome_observable is False


def test_non_subject_input_without_observable_fields_raises() -> None:
    """Ensure TrialSchema rejects non-subject INPUT steps with empty observable_fields."""

    with pytest.raises(ValueError, match="observable_fields"):
        TrialSchema(
            schema_id="bad_schema",
            steps=(
                TrialSchemaStep(EventPhase.INPUT, "main"),
                TrialSchemaStep(EventPhase.INPUT, "main", actor_id="demonstrator"),
                TrialSchemaStep(EventPhase.DECISION, "main", action_required=True),
                TrialSchemaStep(EventPhase.OUTCOME, "main"),
                TrialSchemaStep(EventPhase.UPDATE, "main"),
            ),
        )
