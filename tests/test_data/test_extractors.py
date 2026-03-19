"""Tests for schema-driven decision view extraction."""

from comp_model.data.extractors import extract_decision_views
from comp_model.data.schema import Event, EventPhase, Trial
from comp_model.tasks.schemas import (
    ASOCIAL_BANDIT_SCHEMA,
    SOCIAL_POST_OUTCOME_SCHEMA,
    SOCIAL_PRE_CHOICE_SCHEMA,
)


def test_extract_asocial_trial_view() -> None:
    """Ensure asocial trials extract choice and reward correctly.

    Returns
    -------
    None
        This test asserts extracted scalar fields.
    """

    trial = Trial(
        trial_index=0,
        events=(
            Event(
                phase=EventPhase.INPUT,
                event_index=0,
                node_id="main",
                payload={"available_actions": (0, 1), "observation": {"cue": "left"}},
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=1,
                node_id="main",
                payload={"action": 1},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=2,
                node_id="main",
                payload={"reward": 0.5},
            ),
            Event(phase=EventPhase.UPDATE, event_index=3, node_id="main", payload={}),
        ),
    )

    (view,) = extract_decision_views(trial, ASOCIAL_BANDIT_SCHEMA)

    assert view.available_actions == (0, 1)
    assert view.choice == 1
    assert view.reward == 0.5
    assert dict(view.observation) == {"cue": "left"}
    assert view.social_action is None


def test_extract_social_pre_choice_view_reads_social_fields() -> None:
    """Ensure pre-choice social input is unpacked into social fields.

    Returns
    -------
    None
        This test asserts extracted demonstrator information.
    """

    trial = Trial(
        trial_index=1,
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
                    "observation": {"social_action": 0, "social_reward": 1.0},
                },
            ),
            Event(phase=EventPhase.UPDATE, event_index=2, node_id="main", payload={}),
            Event(
                phase=EventPhase.DECISION,
                event_index=3,
                node_id="main",
                payload={"action": 1},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=4,
                node_id="main",
                payload={"reward": 0.0},
            ),
            Event(phase=EventPhase.UPDATE, event_index=5, node_id="main", payload={}),
        ),
    )

    (view,) = extract_decision_views(trial, SOCIAL_PRE_CHOICE_SCHEMA)

    assert view.choice == 1
    assert view.social_action == 0
    assert view.social_reward == 1.0


def test_extract_social_post_outcome_view_ignores_position_difference() -> None:
    """Ensure post-outcome social input extracts the same flat fields.

    Returns
    -------
    None
        This test asserts schema-order agnostic extraction.
    """

    trial = Trial(
        trial_index=2,
        events=(
            Event(
                phase=EventPhase.INPUT,
                event_index=0,
                node_id="main",
                payload={"available_actions": (0, 1)},
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=1,
                node_id="main",
                payload={"action": 0},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=2,
                node_id="main",
                payload={"reward": 1.0},
            ),
            Event(
                phase=EventPhase.INPUT,
                event_index=3,
                node_id="main",
                actor_id="demonstrator",
                payload={
                    "available_actions": (0, 1),
                    "observation": {"social_action": 1, "social_reward": 0.0},
                },
            ),
            Event(phase=EventPhase.UPDATE, event_index=4, node_id="main", payload={}),
        ),
    )

    (view,) = extract_decision_views(trial, SOCIAL_POST_OUTCOME_SCHEMA)

    assert view.choice == 0
    assert view.reward == 1.0
    assert view.social_action == 1
    assert view.social_reward == 0.0
