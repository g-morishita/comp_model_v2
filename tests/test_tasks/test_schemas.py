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


def test_asocial_schema_exposes_decision_step_indices() -> None:
    """Ensure the asocial schema exposes its decision metadata."""

    assert ASOCIAL_BANDIT_SCHEMA.decision_step_indices == (1,)


def test_social_pre_choice_schema_accepts_matching_trial() -> None:
    """Ensure the pre-choice social schema validates a matching trial."""

    # New SOCIAL_PRE_CHOICE_SCHEMA: INPUT(demo) DECISION(demo) OUTCOME(demo) UPDATE(demo→demo)
    #   UPDATE(demo→subj) INPUT(subj) DECISION(subj) OUTCOME(subj) UPDATE(subj→subj)
    trial = Trial(
        trial_index=0,
        events=(
            Event(
                phase=EventPhase.INPUT,
                event_index=0,
                node_id="main",
                actor_id="demonstrator",
                payload={"available_actions": (0, 1)},
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=1,
                node_id="main",
                actor_id="demonstrator",
                payload={"action": 1},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=2,
                node_id="main",
                actor_id="demonstrator",
                payload={"reward": 0.0},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=3,
                node_id="main",
                actor_id="demonstrator",
                payload={"choice": 1, "reward": 0.0},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=4,
                node_id="main",
                actor_id="demonstrator",
                payload={"choice": 1, "reward": 0.0},
            ),
            Event(
                phase=EventPhase.INPUT,
                event_index=5,
                node_id="main",
                payload={"available_actions": (0, 1)},
            ),
            Event(phase=EventPhase.DECISION, event_index=6, node_id="main", payload={"action": 1}),
            Event(phase=EventPhase.OUTCOME, event_index=7, node_id="main", payload={"reward": 1.0}),
            Event(
                phase=EventPhase.UPDATE,
                event_index=8,
                node_id="main",
                payload={"choice": 1, "reward": 1.0},
            ),
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
                payload={"available_actions": (0, 1)},
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


def test_pre_choice_no_self_outcome_schema_has_no_subject_outcome_or_self_update() -> None:
    """Schema has 7 steps: no subject OUTCOME and no subject self-UPDATE."""
    schema = SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA
    assert len(schema.steps) == 7
    subject_outcome_steps = [
        s for s in schema.steps if s.phase == EventPhase.OUTCOME and s.actor_id == "subject"
    ]
    assert subject_outcome_steps == []


def test_post_outcome_no_self_outcome_schema_has_no_subject_outcome_or_self_update() -> None:
    """Schema has 7 steps: no subject OUTCOME and no subject self-UPDATE."""
    schema = SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA
    assert len(schema.steps) == 7
    subject_outcome_steps = [
        s for s in schema.steps if s.phase == EventPhase.OUTCOME and s.actor_id == "subject"
    ]
    assert subject_outcome_steps == []


def test_social_update_without_observable_fields_raises() -> None:
    """Ensure TrialSchema rejects social UPDATE steps without observable_fields."""

    with pytest.raises(ValueError, match="observable_fields"):
        TrialSchema(
            schema_id="bad_schema",
            steps=(
                TrialSchemaStep(EventPhase.INPUT, "main", actor_id="demonstrator"),
                TrialSchemaStep(EventPhase.DECISION, "main", actor_id="demonstrator"),
                TrialSchemaStep(EventPhase.OUTCOME, "main", actor_id="demonstrator"),
                TrialSchemaStep(
                    EventPhase.UPDATE, "main", actor_id="demonstrator", learner_id="demonstrator"
                ),
                # Social UPDATE without observable_fields — should raise
                TrialSchemaStep(
                    EventPhase.UPDATE,
                    "main",
                    actor_id="demonstrator",
                    learner_id="subject",
                    # observable_fields intentionally omitted
                ),
                TrialSchemaStep(EventPhase.INPUT, "main"),
                TrialSchemaStep(EventPhase.DECISION, "main"),
                TrialSchemaStep(EventPhase.OUTCOME, "main"),
                TrialSchemaStep(EventPhase.UPDATE, "main"),
            ),
        )
