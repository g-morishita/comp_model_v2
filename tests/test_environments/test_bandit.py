"""Tests for the stationary bandit environment."""

import numpy as np
import pytest

from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA
from comp_model.tasks.spec import BlockSpec


def _reset_env(reward_probs: tuple[float, ...], seed: int = 0) -> StationaryBanditEnvironment:
    env = StationaryBanditEnvironment(n_actions=len(reward_probs), reward_probs=reward_probs)
    block_spec = BlockSpec(
        condition="baseline",
        n_trials=1,
        schema=ASOCIAL_BANDIT_SCHEMA,
        metadata={"n_actions": len(reward_probs)},
    )
    env.reset(block_spec, rng=np.random.default_rng(seed))
    return env


def test_bandit_step_returns_float_reward() -> None:
    env = _reset_env(reward_probs=(1.0, 0.0))
    reward = env.step(0)
    assert reward == 1.0


def test_bandit_step_zero_prob_arm_returns_zero() -> None:
    env = _reset_env(reward_probs=(1.0, 0.0))
    reward = env.step(1)
    assert reward == 0.0


def test_bandit_step_requires_reset_first() -> None:
    env = StationaryBanditEnvironment(n_actions=2, reward_probs=(0.5, 0.5))
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_bandit_step_is_stochastic_for_intermediate_prob() -> None:
    env = _reset_env(reward_probs=(0.5, 0.5), seed=42)
    rewards = [env.step(0) for _ in range(100)]
    assert any(r == 1.0 for r in rewards)
    assert any(r == 0.0 for r in rewards)
