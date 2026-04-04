"""Tests for MLE replay objectives."""

import math
from typing import Any, cast

from comp_model.data.schema import Block, Event, EventPhase, SubjectData, Trial
from comp_model.inference.mle.objective import log_likelihood_simple
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.models.kernels.social_rl_self_reward_demo_reward_sticky import (
    SocialRlSelfRewardDemoRewardStickyKernel,
)
from comp_model.models.kernels.transforms import get_transform
from comp_model.tasks.schemas import (
    ASOCIAL_BANDIT_SCHEMA,
    SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA,
)


def _subject_data() -> SubjectData:
    """Create a small hand-built subject dataset for replay tests.

    Returns
    -------
    SubjectData
        Subject data with two deterministic asocial trials.
    """

    return SubjectData(
        subject_id="s1",
        blocks=(
            Block(
                block_index=0,
                condition="baseline",
                schema_id="asocial_bandit",
                trials=(
                    Trial(
                        trial_index=0,
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
                                payload={"action": 1},
                            ),
                            Event(
                                phase=EventPhase.OUTCOME,
                                event_index=2,
                                node_id="main",
                                payload={"reward": 1.0},
                            ),
                            Event(
                                phase=EventPhase.UPDATE,
                                event_index=3,
                                node_id="main",
                                payload={"choice": 1, "reward": 1.0},
                            ),
                        ),
                    ),
                    Trial(
                        trial_index=1,
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
                                payload={"action": 1},
                            ),
                            Event(
                                phase=EventPhase.OUTCOME,
                                event_index=2,
                                node_id="main",
                                payload={"reward": 0.0},
                            ),
                            Event(
                                phase=EventPhase.UPDATE,
                                event_index=3,
                                node_id="main",
                                payload={"choice": 1, "reward": 0.0},
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def _sticky_no_self_outcome_subject_data() -> SubjectData:
    """Create a two-trial no-self-outcome social dataset with repeated choices."""

    trials = []
    for trial_index in range(2):
        trials.append(
            Trial(
                trial_index=trial_index,
                events=(
                    Event(
                        phase=EventPhase.INPUT,
                        event_index=0,
                        node_id="main",
                        actor_id="subject",
                        payload={"available_actions": (0, 1)},
                    ),
                    Event(
                        phase=EventPhase.DECISION,
                        event_index=1,
                        node_id="main",
                        actor_id="subject",
                        payload={"action": 1},
                    ),
                    Event(
                        phase=EventPhase.INPUT,
                        event_index=2,
                        node_id="main",
                        actor_id="demonstrator",
                        payload={"available_actions": (0, 1)},
                    ),
                    Event(
                        phase=EventPhase.DECISION,
                        event_index=3,
                        node_id="main",
                        actor_id="demonstrator",
                        payload={"action": 0},
                    ),
                    Event(
                        phase=EventPhase.OUTCOME,
                        event_index=4,
                        node_id="main",
                        actor_id="demonstrator",
                        payload={"reward": 0.5},
                    ),
                    Event(
                        phase=EventPhase.UPDATE,
                        event_index=5,
                        node_id="main",
                        actor_id="demonstrator",
                        payload={"choice": 0, "reward": 0.5},
                    ),
                    Event(
                        phase=EventPhase.UPDATE,
                        event_index=6,
                        node_id="main",
                        actor_id="demonstrator",
                        payload={"choice": 0, "reward": 0.5},
                    ),
                ),
            )
        )

    return SubjectData(
        subject_id="sticky-no-self-outcome",
        blocks=(
            Block(
                block_index=0,
                condition="baseline",
                schema_id=SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA.schema_id,
                trials=tuple(trials),
            ),
        ),
    )


def test_log_likelihood_simple_returns_finite_value() -> None:
    """Ensure replay likelihood produces a finite scalar.

    Returns
    -------
    None
        This test asserts finiteness of the replay objective.
    """

    kernel = AsocialQLearningKernel()
    log_likelihood = log_likelihood_simple(
        cast("Any", kernel),
        _subject_data(),
        {"alpha": 0.0, "beta": 1.0},
        ASOCIAL_BANDIT_SCHEMA,
    )

    assert math.isfinite(log_likelihood)


def test_log_likelihood_simple_tracks_stickiness_on_no_self_outcome_schema() -> None:
    """Decision-time choice memory should affect replay on no-self-outcome schemas."""

    kernel = SocialRlSelfRewardDemoRewardStickyKernel()
    raw_params = {
        "alpha_self": get_transform("sigmoid").inverse(0.5),
        "alpha_other": get_transform("sigmoid").inverse(0.5),
        "beta": get_transform("softplus").inverse(1.0),
        "stickiness": 3.0,
    }

    log_likelihood = log_likelihood_simple(
        cast("Any", kernel),
        _sticky_no_self_outcome_subject_data(),
        raw_params,
        SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA,
    )

    expected = math.log(0.5) + math.log(1.0 / (1.0 + math.exp(-3.0)))
    assert math.isclose(log_likelihood, expected, rel_tol=1e-12, abs_tol=1e-12)
