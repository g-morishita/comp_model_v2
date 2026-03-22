"""Tests for schema-driven trial replay and decision view extraction."""

from comp_model.data.extractors import replay_trial_steps
from comp_model.data.schema import Event, EventPhase, Trial
from comp_model.tasks.schemas import (
    ASOCIAL_BANDIT_SCHEMA,
    SOCIAL_POST_OUTCOME_DEMO_LEARNS_SCHEMA,
    SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA,
    SOCIAL_POST_OUTCOME_SCHEMA,
    SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
    SOCIAL_PRE_CHOICE_DEMO_LEARNS_SCHEMA,
    SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA,
    SOCIAL_PRE_CHOICE_SCHEMA,
)


def _make_asocial_trial(
    trial_index: int = 0,
    action: int = 1,
    reward: float = 0.5,
) -> Trial:
    return Trial(
        trial_index=trial_index,
        events=(
            Event(
                phase=EventPhase.INPUT,
                event_index=0,
                node_id="main",
                payload={"available_actions": (0, 1), "observation": {"cue": "left"}},
            ),
            Event(
                phase=EventPhase.DECISION, event_index=1, node_id="main", payload={"action": action}
            ),
            Event(
                phase=EventPhase.OUTCOME, event_index=2, node_id="main", payload={"reward": reward}
            ),
            Event(phase=EventPhase.UPDATE, event_index=3, node_id="main", payload={}),
        ),
    )


def _make_social_pre_choice_trial(
    trial_index: int = 0,
    subject_action: int = 1,
    subject_reward: float = 0.0,
    demo_action: int = 0,
    demo_reward: float = 1.0,
) -> Trial:
    return Trial(
        trial_index=trial_index,
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
                    "observation": {"social_action": demo_action, "social_reward": demo_reward},
                },
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=2,
                node_id="main",
                actor_id="demonstrator",
                payload={"action": demo_action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=3,
                node_id="main",
                actor_id="demonstrator",
                payload={"reward": demo_reward},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=4,
                node_id="main",
                actor_id="demonstrator",
                payload={},
            ),
            Event(phase=EventPhase.UPDATE, event_index=5, node_id="main", payload={}),
            Event(
                phase=EventPhase.DECISION,
                event_index=6,
                node_id="main",
                payload={"action": subject_action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=7,
                node_id="main",
                payload={"reward": subject_reward},
            ),
            Event(phase=EventPhase.UPDATE, event_index=8, node_id="main", payload={}),
        ),
    )


def _make_social_post_outcome_trial(
    trial_index: int = 0,
    subject_action: int = 0,
    subject_reward: float = 1.0,
    demo_action: int = 1,
    demo_reward: float = 0.0,
) -> Trial:
    return Trial(
        trial_index=trial_index,
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
                payload={"action": subject_action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=2,
                node_id="main",
                payload={"reward": subject_reward},
            ),
            Event(phase=EventPhase.UPDATE, event_index=3, node_id="main", payload={}),
            Event(
                phase=EventPhase.INPUT,
                event_index=4,
                node_id="main",
                actor_id="demonstrator",
                payload={
                    "available_actions": (0, 1),
                    "observation": {"social_action": demo_action, "social_reward": demo_reward},
                },
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=5,
                node_id="main",
                actor_id="demonstrator",
                payload={"action": demo_action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=6,
                node_id="main",
                actor_id="demonstrator",
                payload={"reward": demo_reward},
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


# ---------------------------------------------------------------------------
# Asocial schema
# ---------------------------------------------------------------------------


def test_asocial_replay_yields_one_action_and_one_update() -> None:
    trial = _make_asocial_trial(action=1, reward=0.5)
    steps = list(replay_trial_steps(trial, ASOCIAL_BANDIT_SCHEMA))

    event_types = [s[0] for s in steps]
    assert event_types == ["action", "update"]


def test_asocial_action_step_has_choice_and_no_reward() -> None:
    trial = _make_asocial_trial(action=1, reward=0.5)
    _, learner_id, view = next(replay_trial_steps(trial, ASOCIAL_BANDIT_SCHEMA))

    assert view.choice == 1
    assert view.reward is None
    assert view.available_actions == (0, 1)
    assert dict(view.observation) == {"cue": "left"}
    assert learner_id == "subject"


def test_asocial_update_step_has_choice_and_reward() -> None:
    trial = _make_asocial_trial(action=1, reward=0.5)
    steps = list(replay_trial_steps(trial, ASOCIAL_BANDIT_SCHEMA))
    _, learner_id, view = steps[1]

    assert view.choice == 1
    assert view.reward == 0.5
    assert learner_id == "subject"


# ---------------------------------------------------------------------------
# Social pre-choice schema
# ---------------------------------------------------------------------------


def test_social_pre_choice_replay_step_order() -> None:
    trial = _make_social_pre_choice_trial()
    steps = list(replay_trial_steps(trial, SOCIAL_PRE_CHOICE_SCHEMA))

    event_types = [(s[0], s[1]) for s in steps]
    assert event_types == [
        ("update", "demonstrator"),  # demonstrator self-update
        ("update", "subject"),  # subject social-update (before choice)
        ("action", "subject"),  # subject action
        ("update", "subject"),  # subject self-update (after outcome)
    ]


def test_social_pre_choice_subject_social_update_has_no_choice_and_no_reward() -> None:
    """Subject social update fires before choice — choice and own reward are None."""
    trial = _make_social_pre_choice_trial(demo_action=0, demo_reward=1.0)
    steps = list(replay_trial_steps(trial, SOCIAL_PRE_CHOICE_SCHEMA))

    _, learner_id, view = steps[1]  # subject social update
    assert learner_id == "subject"
    assert view.choice is None
    assert view.reward is None
    assert view.social_action == 0
    assert view.social_reward == 1.0


def test_social_pre_choice_action_step_has_updated_social_info() -> None:
    """Action step carries accumulated social info so kernels can use it."""
    trial = _make_social_pre_choice_trial(subject_action=1, demo_action=0, demo_reward=1.0)
    steps = list(replay_trial_steps(trial, SOCIAL_PRE_CHOICE_SCHEMA))

    _, learner_id, view = steps[2]  # action step
    assert learner_id == "subject"
    assert view.choice == 1
    assert view.reward is None
    assert view.social_action == 0
    assert view.social_reward == 1.0


def test_social_pre_choice_self_update_has_choice_and_reward() -> None:
    trial = _make_social_pre_choice_trial(subject_action=1, subject_reward=0.0)
    steps = list(replay_trial_steps(trial, SOCIAL_PRE_CHOICE_SCHEMA))

    _, learner_id, view = steps[3]  # subject self-update
    assert learner_id == "subject"
    assert view.choice == 1
    assert view.reward == 0.0


def test_social_pre_choice_demonstrator_update_has_demo_choice_and_reward() -> None:
    trial = _make_social_pre_choice_trial(demo_action=0, demo_reward=1.0)
    steps = list(replay_trial_steps(trial, SOCIAL_PRE_CHOICE_SCHEMA))

    _, learner_id, view = steps[0]  # demonstrator self-update
    assert learner_id == "demonstrator"
    assert view.choice == 0
    assert view.reward == 1.0


# ---------------------------------------------------------------------------
# Social pre-choice action-only schema — observable_fields filtering
# ---------------------------------------------------------------------------


def test_action_only_schema_excludes_social_reward_from_update() -> None:
    """observable_fields={"action"} should produce social_reward=None."""
    trial = _make_social_pre_choice_trial(demo_action=0, demo_reward=1.0)
    steps = list(replay_trial_steps(trial, SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA))

    _, learner_id, view = steps[1]  # subject social update
    assert learner_id == "subject"
    assert view.social_action == 0
    assert view.social_reward is None  # reward filtered out


# ---------------------------------------------------------------------------
# Social post-outcome schema
# ---------------------------------------------------------------------------


def test_social_post_outcome_replay_step_order() -> None:
    trial = _make_social_post_outcome_trial()
    steps = list(replay_trial_steps(trial, SOCIAL_POST_OUTCOME_SCHEMA))

    event_types = [(s[0], s[1]) for s in steps]
    assert event_types == [
        ("action", "subject"),  # subject acts first
        ("update", "subject"),  # subject self-update
        ("update", "demonstrator"),  # demonstrator self-update
        ("update", "subject"),  # subject social-update
    ]


def test_social_post_outcome_subject_self_update_has_no_social_info() -> None:
    """Subject self-update fires before seeing demonstrator — no social info yet."""
    trial = _make_social_post_outcome_trial(
        subject_action=0, subject_reward=1.0, demo_action=1, demo_reward=0.0
    )
    steps = list(replay_trial_steps(trial, SOCIAL_POST_OUTCOME_SCHEMA))

    _, learner_id, view = steps[1]  # subject self-update
    assert learner_id == "subject"
    assert view.choice == 0
    assert view.reward == 1.0
    assert view.social_action is None
    assert view.social_reward is None


def test_social_post_outcome_social_update_has_social_info() -> None:
    trial = _make_social_post_outcome_trial(demo_action=1, demo_reward=0.0)
    steps = list(replay_trial_steps(trial, SOCIAL_POST_OUTCOME_SCHEMA))

    _, learner_id, view = steps[3]  # subject social update
    assert learner_id == "subject"
    assert view.social_action == 1
    assert view.social_reward == 0.0


# ---------------------------------------------------------------------------
# No-self-outcome schemas — no subject OUTCOME or subject self-UPDATE step
# ---------------------------------------------------------------------------


def _make_social_pre_choice_no_self_outcome_trial(
    trial_index: int = 0,
    subject_action: int = 1,
    demo_action: int = 0,
    demo_reward: float = 1.0,
) -> Trial:
    return Trial(
        trial_index=trial_index,
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
                    "observation": {"social_action": demo_action, "social_reward": demo_reward},
                },
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=2,
                node_id="main",
                actor_id="demonstrator",
                payload={"action": demo_action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=3,
                node_id="main",
                actor_id="demonstrator",
                payload={"reward": demo_reward},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=4,
                node_id="main",
                actor_id="demonstrator",
                payload={},
            ),
            Event(phase=EventPhase.UPDATE, event_index=5, node_id="main", payload={}),
            Event(
                phase=EventPhase.DECISION,
                event_index=6,
                node_id="main",
                payload={"action": subject_action},
            ),
        ),
    )


def _make_social_post_outcome_no_self_outcome_trial(
    trial_index: int = 0,
    subject_action: int = 0,
    demo_action: int = 1,
    demo_reward: float = 0.0,
) -> Trial:
    return Trial(
        trial_index=trial_index,
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
                payload={"action": subject_action},
            ),
            Event(
                phase=EventPhase.INPUT,
                event_index=2,
                node_id="main",
                actor_id="demonstrator",
                payload={
                    "available_actions": (0, 1),
                    "observation": {"social_action": demo_action, "social_reward": demo_reward},
                },
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=3,
                node_id="main",
                actor_id="demonstrator",
                payload={"action": demo_action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=4,
                node_id="main",
                actor_id="demonstrator",
                payload={"reward": demo_reward},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=5,
                node_id="main",
                actor_id="demonstrator",
                payload={},
            ),
            Event(phase=EventPhase.UPDATE, event_index=6, node_id="main", payload={}),
        ),
    )


def test_pre_choice_no_self_outcome_replay_step_order() -> None:
    trial = _make_social_pre_choice_no_self_outcome_trial()
    steps = list(replay_trial_steps(trial, SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA))

    event_types = [(s[0], s[1]) for s in steps]
    assert event_types == [
        ("update", "demonstrator"),  # demonstrator self-update
        ("update", "subject"),  # subject social-update (before choice)
        ("action", "subject"),  # subject action — no self-update follows
    ]


def test_pre_choice_no_self_outcome_still_carries_social_info() -> None:
    """Demo action/reward reaches the subject social-update step."""
    trial = _make_social_pre_choice_no_self_outcome_trial(demo_action=0, demo_reward=1.0)
    steps = list(replay_trial_steps(trial, SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA))

    _, learner_id, view = steps[1]  # subject social update
    assert learner_id == "subject"
    assert view.social_action == 0
    assert view.social_reward == 1.0
    assert view.reward is None


def test_post_outcome_no_self_outcome_replay_step_order() -> None:
    trial = _make_social_post_outcome_no_self_outcome_trial()
    steps = list(replay_trial_steps(trial, SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA))

    event_types = [(s[0], s[1]) for s in steps]
    assert event_types == [
        ("action", "subject"),  # subject acts — no self-update follows
        ("update", "demonstrator"),  # demonstrator self-update
        ("update", "subject"),  # subject social-update
    ]


def test_post_outcome_no_self_outcome_still_carries_social_info() -> None:
    """Demo action/reward reaches the subject social-update step."""
    trial = _make_social_post_outcome_no_self_outcome_trial(demo_action=1, demo_reward=0.0)
    steps = list(replay_trial_steps(trial, SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA))

    _, learner_id, view = steps[2]  # subject social update
    assert learner_id == "subject"
    assert view.social_action == 1
    assert view.social_reward == 0.0


def test_pre_choice_no_self_outcome_demonstrator_update_has_choice_and_reward() -> None:
    """Demonstrator self-update carries the demo choice and reward."""
    trial = _make_social_pre_choice_no_self_outcome_trial(demo_action=0, demo_reward=1.0)
    steps = list(replay_trial_steps(trial, SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA))

    _, learner_id, view = steps[0]  # demonstrator self-update
    assert learner_id == "demonstrator"
    assert view.choice == 0
    assert view.reward == 1.0


def test_post_outcome_no_self_outcome_demonstrator_update_has_choice_and_reward() -> None:
    """Demonstrator self-update carries the demo choice and reward."""
    trial = _make_social_post_outcome_no_self_outcome_trial(demo_action=1, demo_reward=0.0)
    steps = list(replay_trial_steps(trial, SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA))

    _, learner_id, view = steps[1]  # demonstrator self-update
    assert learner_id == "demonstrator"
    assert view.choice == 1
    assert view.reward == 0.0


# ---------------------------------------------------------------------------
# Demo-learns schemas — demonstrator also updates from subject's action+reward
# ---------------------------------------------------------------------------


def _make_social_pre_choice_demo_learns_trial(
    trial_index: int = 0,
    subject_action: int = 1,
    subject_reward: float = 0.0,
    demo_action: int = 0,
    demo_reward: float = 1.0,
) -> Trial:
    """11-event trial for SOCIAL_PRE_CHOICE_DEMO_LEARNS_SCHEMA.

    Each agent self-updates immediately after its own outcome.  The demonstrator
    also social-updates from the subject's action+reward at the end.
    """
    return Trial(
        trial_index=trial_index,
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
                    "observation": {"social_action": demo_action, "social_reward": demo_reward},
                },
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=2,
                node_id="main",
                actor_id="demonstrator",
                payload={"action": demo_action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=3,
                node_id="main",
                actor_id="demonstrator",
                payload={"reward": demo_reward},
            ),
            # Demonstrator self-update immediately after its own outcome.
            Event(
                phase=EventPhase.UPDATE,
                event_index=4,
                node_id="main",
                actor_id="demonstrator",
                payload={},
            ),
            Event(phase=EventPhase.UPDATE, event_index=5, node_id="main", payload={}),
            Event(
                phase=EventPhase.DECISION,
                event_index=6,
                node_id="main",
                payload={"action": subject_action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=7,
                node_id="main",
                payload={"reward": subject_reward},
            ),
            # Subject self-update immediately after its own outcome.
            Event(phase=EventPhase.UPDATE, event_index=8, node_id="main", payload={}),
            # Demonstrator INPUT carrying subject's action+reward.
            Event(
                phase=EventPhase.INPUT,
                event_index=9,
                node_id="main",
                actor_id="demonstrator",
                payload={
                    "available_actions": (0, 1),
                    "observation": {
                        "social_action": subject_action,
                        "social_reward": subject_reward,
                    },
                },
            ),
            # Demonstrator social-update from subject.
            Event(
                phase=EventPhase.UPDATE,
                event_index=10,
                node_id="main",
                actor_id="demonstrator",
                payload={},
            ),
        ),
    )


def _make_social_post_outcome_demo_learns_trial(
    trial_index: int = 0,
    subject_action: int = 0,
    subject_reward: float = 1.0,
    demo_action: int = 1,
    demo_reward: float = 0.0,
) -> Trial:
    """11-event trial for SOCIAL_POST_OUTCOME_DEMO_LEARNS_SCHEMA.

    Order: subject self-update → demo social-update from subject → demo decides →
    demo self-update → subject social-update from demo.

    The demonstrator's social update from the subject happens *before* its decision,
    so the subject's outcome informs the demonstrator's current-trial choice.
    """
    return Trial(
        trial_index=trial_index,
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
                payload={"action": subject_action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=2,
                node_id="main",
                payload={"reward": subject_reward},
            ),
            # Subject self-update.
            Event(phase=EventPhase.UPDATE, event_index=3, node_id="main", payload={}),
            # Demonstrator INPUT carrying subject's action+reward (demo sees subject first).
            Event(
                phase=EventPhase.INPUT,
                event_index=4,
                node_id="main",
                actor_id="demonstrator",
                payload={
                    "available_actions": (0, 1),
                    "observation": {
                        "social_action": subject_action,
                        "social_reward": subject_reward,
                    },
                },
            ),
            # Demonstrator social-update from subject (before demo's own decision).
            Event(
                phase=EventPhase.UPDATE,
                event_index=5,
                node_id="main",
                actor_id="demonstrator",
                payload={},
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=6,
                node_id="main",
                actor_id="demonstrator",
                payload={"action": demo_action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=7,
                node_id="main",
                actor_id="demonstrator",
                payload={"reward": demo_reward},
            ),
            # Demonstrator self-update.
            Event(
                phase=EventPhase.UPDATE,
                event_index=8,
                node_id="main",
                actor_id="demonstrator",
                payload={},
            ),
            # Subject sees demonstrator's action+reward.
            Event(
                phase=EventPhase.INPUT,
                event_index=9,
                node_id="main",
                actor_id="demonstrator",
                payload={
                    "available_actions": (0, 1),
                    "observation": {"social_action": demo_action, "social_reward": demo_reward},
                },
            ),
            # Subject social-update from demonstrator.
            Event(phase=EventPhase.UPDATE, event_index=10, node_id="main", payload={}),
        ),
    )


def test_pre_choice_demo_learns_replay_step_order() -> None:
    trial = _make_social_pre_choice_demo_learns_trial()
    steps = list(replay_trial_steps(trial, SOCIAL_PRE_CHOICE_DEMO_LEARNS_SCHEMA))

    event_types = [(s[0], s[1]) for s in steps]
    assert event_types == [
        ("update", "demonstrator"),  # demo self-update (own outcome)
        ("update", "subject"),  # subject social-update (from demo, before own decision)
        ("action", "subject"),  # subject acts
        ("update", "subject"),  # subject self-update (own outcome)
        ("update", "demonstrator"),  # demo social-update from subject
    ]


def test_pre_choice_demo_learns_demo_self_update() -> None:
    """Demonstrator self-UPDATE (immediately after own outcome) carries own reward."""
    trial = _make_social_pre_choice_demo_learns_trial(
        subject_action=1, subject_reward=0.5, demo_action=0, demo_reward=1.0
    )
    steps = list(replay_trial_steps(trial, SOCIAL_PRE_CHOICE_DEMO_LEARNS_SCHEMA))

    _, learner_id, view = steps[0]  # demo self-update
    assert learner_id == "demonstrator"
    assert view.reward == 1.0  # demo's own reward


def test_pre_choice_demo_learns_demo_social_update_from_subject() -> None:
    """Demonstrator social-UPDATE (at end) carries subject's action+reward, reward=None."""
    trial = _make_social_pre_choice_demo_learns_trial(
        subject_action=1, subject_reward=0.5, demo_action=0, demo_reward=1.0
    )
    steps = list(replay_trial_steps(trial, SOCIAL_PRE_CHOICE_DEMO_LEARNS_SCHEMA))

    _, learner_id, view = steps[4]  # demo social-update from subject
    assert learner_id == "demonstrator"
    assert view.social_action == 1  # subject's choice
    assert view.social_reward == 0.5  # subject's reward
    assert view.reward is None  # own reward was already consumed in self-update


def test_post_outcome_demo_learns_replay_step_order() -> None:
    trial = _make_social_post_outcome_demo_learns_trial()
    steps = list(replay_trial_steps(trial, SOCIAL_POST_OUTCOME_DEMO_LEARNS_SCHEMA))

    event_types = [(s[0], s[1]) for s in steps]
    assert event_types == [
        ("action", "subject"),  # subject acts
        ("update", "subject"),  # subject self-update (own reward)
        ("update", "demonstrator"),  # demo social-update from subject (before demo decides)
        ("update", "demonstrator"),  # demo self-update (own reward)
        ("update", "subject"),  # subject social-update from demo
    ]


def test_post_outcome_demo_learns_subject_self_update() -> None:
    """Subject self-UPDATE has own reward and no social info."""
    trial = _make_social_post_outcome_demo_learns_trial(
        subject_action=0, subject_reward=1.0, demo_action=1, demo_reward=0.0
    )
    steps = list(replay_trial_steps(trial, SOCIAL_POST_OUTCOME_DEMO_LEARNS_SCHEMA))

    _, learner_id, view = steps[1]  # subject self-update
    assert learner_id == "subject"
    assert view.reward == 1.0
    assert view.social_action is None


def test_post_outcome_demo_learns_demo_social_update_from_subject() -> None:
    """Demonstrator social-UPDATE (before decision) carries subject's action+reward."""
    trial = _make_social_post_outcome_demo_learns_trial(
        subject_action=0, subject_reward=1.0, demo_action=1, demo_reward=0.0
    )
    steps = list(replay_trial_steps(trial, SOCIAL_POST_OUTCOME_DEMO_LEARNS_SCHEMA))

    _, learner_id, view = steps[2]  # demo social-update
    assert learner_id == "demonstrator"
    assert view.social_action == 0  # subject's choice
    assert view.social_reward == 1.0  # subject's reward
    assert view.reward is None  # demo hasn't had its own outcome yet


def test_post_outcome_demo_learns_subject_social_update_from_demo() -> None:
    """Subject social-UPDATE (at the end) carries demonstrator's action+reward."""
    trial = _make_social_post_outcome_demo_learns_trial(
        subject_action=0, subject_reward=1.0, demo_action=1, demo_reward=0.0
    )
    steps = list(replay_trial_steps(trial, SOCIAL_POST_OUTCOME_DEMO_LEARNS_SCHEMA))

    _, learner_id, view = steps[4]  # subject social-update
    assert learner_id == "subject"
    assert view.social_action == 1  # demo's choice
    assert view.social_reward == 0.0  # demo's reward
    assert view.reward is None
