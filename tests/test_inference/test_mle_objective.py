"""Tests for MLE replay objectives."""

import math

from comp_model.data.schema import Block, Event, EventPhase, SubjectData, Trial
from comp_model.inference.mle.objective import log_likelihood_simple
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA


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


def test_log_likelihood_simple_returns_finite_value() -> None:
    """Ensure replay likelihood produces a finite scalar.

    Returns
    -------
    None
        This test asserts finiteness of the replay objective.
    """

    kernel = AsocialQLearningKernel()
    log_likelihood = log_likelihood_simple(
        kernel,
        _subject_data(),
        {"alpha": 0.0, "beta": 1.0},
        ASOCIAL_BANDIT_SCHEMA,
    )

    assert math.isfinite(log_likelihood)
