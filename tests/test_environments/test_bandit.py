"""Tests for the stationary bandit environment."""

import numpy as np

from comp_model.data.schema import EventPhase
from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA
from comp_model.tasks.spec import BlockSpec


def test_bandit_environment_emits_schema_matching_events() -> None:
    """Ensure the bandit environment emits the planned event sequence.

    Returns
    -------
    None
        This test asserts emitted phases and payload content.
    """

    environment = StationaryBanditEnvironment(n_actions=2, reward_probs=(1.0, 0.0))
    block_spec = BlockSpec(
        condition="baseline",
        n_trials=1,
        schema=ASOCIAL_BANDIT_SCHEMA,
        metadata={"n_actions": 2},
    )
    environment.reset(block_spec, rng=np.random.default_rng(0))

    input_event = environment.step()[0]
    decision_event = environment.step(action=0)[0]
    outcome_event = environment.step()[0]
    update_event = environment.step()[0]

    assert input_event.phase == EventPhase.INPUT
    assert decision_event.payload["action"] == 0
    assert outcome_event.payload["reward"] == 1.0
    assert update_event.phase == EventPhase.UPDATE
