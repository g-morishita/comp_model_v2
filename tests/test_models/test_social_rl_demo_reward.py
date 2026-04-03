"""Tests for the social demonstrator-reward-only RL kernel."""

import math

from comp_model.data.extractors import DecisionTrialView
from comp_model.models.kernels.social_rl_demo_reward import SocialRlDemoRewardKernel


def test_social_demo_reward_kernel_reports_social_requirement() -> None:
    """The kernel should declare that it consumes social information."""

    assert SocialRlDemoRewardKernel.spec().requires_social is True


def test_social_demo_reward_kernel_ignores_self_updates() -> None:
    """Self UPDATE steps should leave Q-values unchanged."""

    kernel = SocialRlDemoRewardKernel()
    params = kernel.parse_params({"alpha_other": 0.0, "beta": 1.0})
    state = kernel.initial_state(2, params)
    self_view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
        action=0,
        reward=1.0,
    )

    updated_state = kernel.update(state, self_view, params)

    assert updated_state.q_values == state.q_values


def test_social_demo_reward_kernel_updates_from_demonstrator_outcomes() -> None:
    """Social UPDATE steps should change the demonstrated action value."""

    kernel = SocialRlDemoRewardKernel()
    params = kernel.parse_params({"alpha_other": 0.0, "beta": 1.0})
    state = kernel.initial_state(2, params)
    social_view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="demonstrator",
        learner_id="subject",
        action=1,
        reward=0.0,
    )

    updated_state = kernel.update(state, social_view, params)

    assert updated_state.q_values[0] == 0.5
    assert updated_state.q_values[1] < 0.5


def test_social_demo_reward_action_probabilities_sum_to_one() -> None:
    """Choice probabilities should remain normalized."""

    kernel = SocialRlDemoRewardKernel()
    params = kernel.parse_params({"alpha_other": 0.0, "beta": 1.0})
    state = kernel.initial_state(2, params)
    state.q_values[1] = 0.8
    view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
    )

    probabilities = kernel.action_probabilities(state, view, params)

    assert math.isclose(sum(probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert probabilities[1] > probabilities[0]
