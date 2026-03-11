"""Tests for declarative trial schemas."""

import pytest

from comp_model.data.schema import Event, EventPhase, Trial
from comp_model.tasks.schemas import (
    ASOCIAL_BANDIT_SCHEMA,
    SOCIAL_POST_OUTCOME_SCHEMA,
    SOCIAL_PRE_CHOICE_SCHEMA,
)


def test_asocial_schema_exposes_decision_and_action_indices() -> None:
    """Ensure the asocial schema exposes its decision metadata.

    Returns
    -------
    None
        This test asserts schema properties only.
    """

    assert ASOCIAL_BANDIT_SCHEMA.decision_step_indices == (1,)
    assert ASOCIAL_BANDIT_SCHEMA.action_required_indices == (1,)


def test_social_pre_choice_schema_accepts_matching_trial() -> None:
    """Ensure the pre-choice social schema validates a matching trial.

    Returns
    -------
    None
        This test only checks successful validation.
    """

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
            Event(phase=EventPhase.DECISION, event_index=2, node_id="main", payload={"action": 1}),
            Event(phase=EventPhase.OUTCOME, event_index=3, node_id="main", payload={"reward": 1.0}),
            Event(phase=EventPhase.UPDATE, event_index=4, node_id="main", payload={}),
        ),
    )

    SOCIAL_PRE_CHOICE_SCHEMA.validate_trial(trial)


def test_social_post_outcome_schema_rejects_actor_mismatch() -> None:
    """Ensure actor mismatches are rejected with a clear error.

    Returns
    -------
    None
        This test raises on schema mismatch.
    """

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
            Event(
                phase=EventPhase.INPUT,
                event_index=3,
                node_id="main",
                payload={
                    "available_actions": (0, 1),
                    "observation": {"social_action": 0, "social_reward": 1.0},
                },
            ),
            Event(phase=EventPhase.UPDATE, event_index=4, node_id="main", payload={}),
        ),
    )

    with pytest.raises(ValueError, match="actor_id"):
        SOCIAL_POST_OUTCOME_SCHEMA.validate_trial(trial)


def test_schema_rejects_wrong_event_count() -> None:
    """Ensure schema validation checks trial length.

    Returns
    -------
    None
        This test raises on mismatched event count.
    """

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
