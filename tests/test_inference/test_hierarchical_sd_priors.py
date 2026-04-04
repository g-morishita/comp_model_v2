"""Tests for configurable hierarchical SD prior export through Stan adapters."""

from __future__ import annotations

from typing import Any

import pytest

from comp_model.data.schema import Block, Dataset, Event, EventPhase, SubjectData, Trial
from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.inference.bayes.stan.adapters import (
    SocialRlSelfRewardDemoActionMixtureStickyStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.asocial_q_learning import (
    AsocialQLearningStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.asocial_rl_asymmetric import (
    AsocialRlAsymmetricStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.asocial_rl_sticky import (
    AsocialRlStickyStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_demo_mixture import (
    SocialRlDemoMixtureStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_demo_reward import (
    SocialRlDemoRewardStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_demo_reward_sticky import (
    SocialRlDemoRewardStickyStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_self_reward_demo_action_mixture import (
    SocialRlSelfRewardDemoActionMixtureStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_self_reward_demo_mixture import (
    SocialRlSelfRewardDemoMixtureStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_self_reward_demo_mixture_sticky import (
    SocialRlSelfRewardDemoMixtureStickyStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_self_reward_demo_reward import (
    SocialRlSelfRewardDemoRewardStanAdapter,
)
from comp_model.inference.config import HierarchyStructure
from comp_model.models.condition.shared_delta import SharedDeltaLayout
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.models.kernels.asocial_rl_sticky import AsocialRlStickyKernel
from comp_model.runtime.engine import SimulationConfig, simulate_dataset
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA, SOCIAL_PRE_CHOICE_SCHEMA
from comp_model.tasks.spec import BlockSpec, TaskSpec


def _asocial_dataset(include_conditions: bool) -> Dataset:
    """Create a small asocial dataset for hierarchical adapter tests."""

    blocks = [
        BlockSpec(
            condition="baseline",
            n_trials=2,
            schema=ASOCIAL_BANDIT_SCHEMA,
            metadata={"n_actions": 2},
        )
    ]
    if include_conditions:
        blocks.append(
            BlockSpec(
                condition="social",
                n_trials=2,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            )
        )

    task = TaskSpec(task_id="hierarchical-sd-priors", blocks=tuple(blocks))
    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    return simulate_dataset(
        task=task,
        env_factory=lambda: StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params_per_subject={"s1": params, "s2": params},
        config=SimulationConfig(seed=23),
    )


def _asocial_sticky_dataset(include_conditions: bool) -> Dataset:
    """Create a small asocial sticky dataset for hierarchical adapter tests."""

    blocks = [
        BlockSpec(
            condition="baseline",
            n_trials=2,
            schema=ASOCIAL_BANDIT_SCHEMA,
            metadata={"n_actions": 2},
        )
    ]
    if include_conditions:
        blocks.append(
            BlockSpec(
                condition="social",
                n_trials=2,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            )
        )

    task = TaskSpec(task_id="hierarchical-sd-priors-sticky", blocks=tuple(blocks))
    kernel = AsocialRlStickyKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0, "stickiness": 0.0})
    return simulate_dataset(
        task=task,
        env_factory=lambda: StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params_per_subject={"s1": params, "s2": params},
        config=SimulationConfig(seed=23),
    )


def _social_trial(
    trial_index: int,
    *,
    action: int,
    reward: float,
    social_action: int,
    social_reward: float,
) -> Trial:
    """Create one minimal social trial matching ``SOCIAL_PRE_CHOICE_SCHEMA``."""

    return Trial(
        trial_index=trial_index,
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
                payload={"action": social_action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=2,
                node_id="main",
                actor_id="demonstrator",
                payload={"reward": social_reward},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=3,
                node_id="main",
                actor_id="demonstrator",
                payload={"choice": social_action, "reward": social_reward},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=4,
                node_id="main",
                actor_id="demonstrator",
                payload={"choice": social_action, "reward": social_reward},
            ),
            Event(
                phase=EventPhase.INPUT,
                event_index=5,
                node_id="main",
                payload={"available_actions": (0, 1)},
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=6,
                node_id="main",
                payload={"action": action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=7,
                node_id="main",
                payload={"reward": reward},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=8,
                node_id="main",
                payload={"choice": action, "reward": reward},
            ),
        ),
    )


def _social_subject(subject_id: str, include_conditions: bool) -> SubjectData:
    """Create a small social subject for hierarchical adapter tests."""

    blocks = [
        Block(
            block_index=0,
            condition="baseline",
            schema_id="social_pre_choice",
            trials=(
                _social_trial(
                    0,
                    action=0,
                    reward=1.0,
                    social_action=1,
                    social_reward=0.0,
                ),
            ),
        )
    ]
    if include_conditions:
        blocks.append(
            Block(
                block_index=1,
                condition="social",
                schema_id="social_pre_choice",
                trials=(
                    _social_trial(
                        0,
                        action=1,
                        reward=0.0,
                        social_action=0,
                        social_reward=1.0,
                    ),
                ),
            )
        )

    return SubjectData(subject_id=subject_id, blocks=tuple(blocks))


def _social_dataset(include_conditions: bool) -> Dataset:
    """Create a small social dataset for hierarchical adapter tests."""

    return Dataset(
        subjects=(
            _social_subject("social-s1", include_conditions),
            _social_subject("social-s2", include_conditions),
        )
    )


_ASOCIAL_ADAPTER_CASES = [
    (
        "asocial_q_learning",
        AsocialQLearningStanAdapter(),
        _asocial_dataset,
        ASOCIAL_BANDIT_SCHEMA,
    ),
    (
        "asocial_rl_sticky",
        AsocialRlStickyStanAdapter(),
        _asocial_sticky_dataset,
        ASOCIAL_BANDIT_SCHEMA,
    ),
    (
        "asocial_rl_asymmetric",
        AsocialRlAsymmetricStanAdapter(),
        _asocial_dataset,
        ASOCIAL_BANDIT_SCHEMA,
    ),
]

_SOCIAL_ADAPTER_CASES = [
    (
        "social_rl_demo_reward",
        SocialRlDemoRewardStanAdapter(),
        _social_dataset,
        SOCIAL_PRE_CHOICE_SCHEMA,
    ),
    (
        "social_rl_demo_reward_sticky",
        SocialRlDemoRewardStickyStanAdapter(),
        _social_dataset,
        SOCIAL_PRE_CHOICE_SCHEMA,
    ),
    (
        "social_rl_self_reward_demo_reward",
        SocialRlSelfRewardDemoRewardStanAdapter(),
        _social_dataset,
        SOCIAL_PRE_CHOICE_SCHEMA,
    ),
    (
        "social_rl_demo_mixture",
        SocialRlDemoMixtureStanAdapter(),
        _social_dataset,
        SOCIAL_PRE_CHOICE_SCHEMA,
    ),
    (
        "social_rl_self_reward_demo_action_mixture",
        SocialRlSelfRewardDemoActionMixtureStanAdapter(),
        _social_dataset,
        SOCIAL_PRE_CHOICE_SCHEMA,
    ),
    (
        "social_rl_self_reward_demo_action_mixture_sticky",
        SocialRlSelfRewardDemoActionMixtureStickyStanAdapter(),
        _social_dataset,
        SOCIAL_PRE_CHOICE_SCHEMA,
    ),
    (
        "social_rl_self_reward_demo_mixture",
        SocialRlSelfRewardDemoMixtureStanAdapter(),
        _social_dataset,
        SOCIAL_PRE_CHOICE_SCHEMA,
    ),
    (
        "social_rl_self_reward_demo_mixture_sticky",
        SocialRlSelfRewardDemoMixtureStickyStanAdapter(),
        _social_dataset,
        SOCIAL_PRE_CHOICE_SCHEMA,
    ),
]


@pytest.mark.parametrize(
    ("adapter", "dataset_factory", "schema"),
    [(case[1], case[2], case[3]) for case in _ASOCIAL_ADAPTER_CASES + _SOCIAL_ADAPTER_CASES],
    ids=[case[0] for case in _ASOCIAL_ADAPTER_CASES + _SOCIAL_ADAPTER_CASES],
)
def test_study_subject_adapters_export_sd_prior_keys(
    adapter: Any,
    dataset_factory: Any,
    schema: Any,
) -> None:
    """Ensure study-level hierarchical adapters export SD prior metadata."""

    dataset = dataset_factory(False)

    stan_data = adapter.build_stan_data(dataset, schema, HierarchyStructure.STUDY_SUBJECT)

    expected = {
        f"sd_{parameter.name}_prior_family" for parameter in adapter.kernel_spec().parameter_specs
    }
    unexpected_delta = {
        f"sd_{parameter.name}_delta_prior_family"
        for parameter in adapter.kernel_spec().parameter_specs
    }
    unexpected_mean_delta = {
        f"{parameter.name}_delta_prior_family"
        for parameter in adapter.kernel_spec().parameter_specs
    }

    assert expected.issubset(stan_data)
    assert unexpected_delta.isdisjoint(stan_data)
    assert unexpected_mean_delta.isdisjoint(stan_data)


@pytest.mark.parametrize(
    ("adapter", "dataset_factory", "schema"),
    [(case[1], case[2], case[3]) for case in _ASOCIAL_ADAPTER_CASES + _SOCIAL_ADAPTER_CASES],
    ids=[case[0] for case in _ASOCIAL_ADAPTER_CASES + _SOCIAL_ADAPTER_CASES],
)
def test_study_subject_condition_adapters_export_shared_and_delta_prior_keys(
    adapter: Any,
    dataset_factory: Any,
    schema: Any,
) -> None:
    """Ensure study+condition hierarchical adapters export delta mean and SD priors."""

    dataset = dataset_factory(True)
    layout = SharedDeltaLayout(
        kernel_spec=adapter.kernel_spec(),
        conditions=("baseline", "social"),
        baseline_condition="baseline",
    )

    stan_data = adapter.build_stan_data(
        dataset,
        schema,
        HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
        layout=layout,
    )

    expected_shared = {
        f"sd_{parameter.name}_prior_family" for parameter in adapter.kernel_spec().parameter_specs
    }
    expected_delta = {
        f"sd_{parameter.name}_delta_prior_family"
        for parameter in adapter.kernel_spec().parameter_specs
    }
    expected_mean_delta = {
        f"{parameter.name}_delta_prior_family"
        for parameter in adapter.kernel_spec().parameter_specs
    }

    assert expected_shared.issubset(stan_data)
    assert expected_delta.issubset(stan_data)
    assert expected_mean_delta.issubset(stan_data)
