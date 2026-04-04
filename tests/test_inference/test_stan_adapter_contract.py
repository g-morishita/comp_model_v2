"""Regression tests for Stan adapter posterior-parameter contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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

if TYPE_CHECKING:
    from comp_model.inference.bayes.stan.adapters.base import StanAdapter

_ADAPTER_CASES: list[tuple[str, StanAdapter, set[str]]] = [
    (
        "asocial_q_learning",
        AsocialQLearningStanAdapter(),
        {
            "alpha_shared_z",
            "beta_shared_z",
            "alpha_delta_z",
            "beta_delta_z",
        },
    ),
    (
        "asocial_rl_asymmetric",
        AsocialRlAsymmetricStanAdapter(),
        {
            "alpha_pos_shared_z",
            "alpha_neg_shared_z",
            "beta_shared_z",
            "alpha_pos_delta_z",
            "alpha_neg_delta_z",
            "beta_delta_z",
        },
    ),
    (
        "asocial_rl_sticky",
        AsocialRlStickyStanAdapter(),
        {
            "alpha_shared_z",
            "beta_shared_z",
            "stickiness_shared_z",
            "alpha_delta_z",
            "beta_delta_z",
            "stickiness_delta_z",
        },
    ),
    (
        "social_rl_demo_mixture",
        SocialRlDemoMixtureStanAdapter(),
        {
            "alpha_other_outcome_shared_z",
            "alpha_other_action_shared_z",
            "w_imitation_shared_z",
            "beta_shared_z",
            "alpha_other_outcome_delta_z",
            "alpha_other_action_delta_z",
            "w_imitation_delta_z",
            "beta_delta_z",
        },
    ),
    (
        "social_rl_self_reward_demo_action_mixture",
        SocialRlSelfRewardDemoActionMixtureStanAdapter(),
        {
            "alpha_self_shared_z",
            "alpha_other_action_shared_z",
            "w_imitation_shared_z",
            "beta_shared_z",
            "alpha_self_delta_z",
            "alpha_other_action_delta_z",
            "w_imitation_delta_z",
            "beta_delta_z",
        },
    ),
    (
        "social_rl_self_reward_demo_mixture",
        SocialRlSelfRewardDemoMixtureStanAdapter(),
        {
            "alpha_self_shared_z",
            "alpha_other_outcome_shared_z",
            "alpha_other_action_shared_z",
            "w_imitation_shared_z",
            "beta_shared_z",
            "alpha_self_delta_z",
            "alpha_other_outcome_delta_z",
            "alpha_other_action_delta_z",
            "w_imitation_delta_z",
            "beta_delta_z",
        },
    ),
    (
        "social_rl_self_reward_demo_mixture_sticky",
        SocialRlSelfRewardDemoMixtureStickyStanAdapter(),
        {
            "alpha_self_shared_z",
            "alpha_other_outcome_shared_z",
            "alpha_other_action_shared_z",
            "w_imitation_shared_z",
            "beta_shared_z",
            "stickiness_shared_z",
            "alpha_self_delta_z",
            "alpha_other_outcome_delta_z",
            "alpha_other_action_delta_z",
            "w_imitation_delta_z",
            "beta_delta_z",
            "stickiness_delta_z",
        },
    ),
    (
        "social_rl_demo_reward",
        SocialRlDemoRewardStanAdapter(),
        {
            "alpha_other_shared_z",
            "beta_shared_z",
            "alpha_other_delta_z",
            "beta_delta_z",
        },
    ),
    (
        "social_rl_demo_reward_sticky",
        SocialRlDemoRewardStickyStanAdapter(),
        {
            "alpha_other_shared_z",
            "beta_shared_z",
            "stickiness_shared_z",
            "alpha_other_delta_z",
            "beta_delta_z",
            "stickiness_delta_z",
        },
    ),
    (
        "social_rl_self_reward_demo_reward",
        SocialRlSelfRewardDemoRewardStanAdapter(),
        {
            "alpha_self_shared_z",
            "alpha_other_shared_z",
            "beta_shared_z",
            "alpha_self_delta_z",
            "alpha_other_delta_z",
            "beta_delta_z",
        },
    ),
]


@pytest.mark.parametrize(
    ("adapter", "expected_extra"),
    [(case[1], case[2]) for case in _ADAPTER_CASES],
    ids=[case[0] for case in _ADAPTER_CASES],
)
def test_conditioned_hierarchies_split_population_and_extra_posterior_names(
    adapter: StanAdapter,
    expected_extra: set[str],
) -> None:
    """Conditioned hierarchies should separate population and latent outputs."""

    subject_population = adapter.population_param_names(HierarchyStructure.SUBJECT_BLOCK_CONDITION)
    subject_extra = adapter.extra_posterior_param_names(HierarchyStructure.SUBJECT_BLOCK_CONDITION)
    study_population = adapter.population_param_names(
        HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION
    )
    study_extra = adapter.extra_posterior_param_names(
        HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION
    )

    assert subject_population == ()
    assert set(subject_extra) == expected_extra
    assert set(study_extra) == expected_extra
    assert study_population
    assert expected_extra.isdisjoint(study_population)
    assert all(
        name.startswith(("mu_", "sd_")) or name.endswith("_pop") for name in study_population
    )


@pytest.mark.parametrize(
    ("adapter", "hierarchy"),
    [
        (case[1], hierarchy)
        for case in _ADAPTER_CASES
        for hierarchy in (
            HierarchyStructure.SUBJECT_SHARED,
            HierarchyStructure.STUDY_SUBJECT,
        )
    ],
    ids=[
        f"{case[0]}-{hierarchy.value}"
        for case in _ADAPTER_CASES
        for hierarchy in (
            HierarchyStructure.SUBJECT_SHARED,
            HierarchyStructure.STUDY_SUBJECT,
        )
    ],
)
def test_non_conditioned_hierarchies_have_no_extra_posterior_names(
    adapter: StanAdapter,
    hierarchy: HierarchyStructure,
) -> None:
    """Non-conditioned hierarchies should not expose extra posterior names."""

    assert adapter.extra_posterior_param_names(hierarchy) == ()
