"""Tests for the sticky social demonstrator-reward-only RL kernel."""

from comp_model.data.extractors import DecisionTrialView
from comp_model.models.kernels.social_rl_demo_reward_sticky import (
    SocialRlDemoRewardStickyKernel,
)


def test_kernel_reports_social_requirement() -> None:
    """Ensure the kernel declares that it consumes full social input."""

    spec = SocialRlDemoRewardStickyKernel.spec()
    assert spec.requires_social is True
    assert spec.required_social_fields == frozenset({"action", "reward"})


def test_kernel_has_three_parameters() -> None:
    """Ensure the kernel declares the extra stickiness parameter."""

    spec = SocialRlDemoRewardStickyKernel.spec()
    param_names = [parameter.name for parameter in spec.parameter_specs]
    assert param_names == ["alpha_other", "beta", "stickiness"]


def test_self_update_stores_previous_own_choice_without_learning() -> None:
    """Self rows should refresh stickiness state without changing Q-values."""

    kernel = SocialRlDemoRewardStickyKernel()
    params = kernel.parse_params({"alpha_other": 0.0, "beta": 1.0, "stickiness": 0.0})
    state = kernel.initial_state(2, params)

    self_view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
        action=1,
        reward=1.0,
    )
    updated = kernel.update(state, self_view, params)

    assert updated.last_self_action == 1
    assert updated.q_values == state.q_values


def test_social_update_preserves_previous_own_choice() -> None:
    """Demonstrator rows should not overwrite the last self action."""

    kernel = SocialRlDemoRewardStickyKernel()
    params = kernel.parse_params({"alpha_other": 0.0, "beta": 1.0, "stickiness": 0.0})
    state = kernel.initial_state(2, params)
    state.last_self_action = 1

    social_view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="demonstrator",
        learner_id="subject",
        action=0,
        reward=1.0,
    )
    updated = kernel.update(state, social_view, params)

    assert updated.last_self_action == 1
    assert updated.q_values[0] > state.q_values[0]
    assert updated.q_values[1] == state.q_values[1]


def test_action_probabilities_favor_previous_own_choice_with_positive_stickiness() -> None:
    """A positive stickiness term should bias repeats when values are equal."""

    kernel = SocialRlDemoRewardStickyKernel()
    params = kernel.parse_params({"alpha_other": 0.0, "beta": 1.0, "stickiness": 3.0})
    state = kernel.initial_state(2, params)
    state.last_self_action = 0

    view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
    )
    probs = kernel.action_probabilities(state, view, params)

    assert probs[0] > probs[1]
