"""Tests for the sticky social self-reward + demo-mixture RL kernel."""

from comp_model.data.extractors import DecisionTrialView
from comp_model.models.kernels.social_rl_self_reward_demo_mixture_sticky import (
    SocialRlSelfRewardDemoMixtureStickyKernel,
)


def test_kernel_reports_social_requirement() -> None:
    """Ensure the kernel declares that it consumes full social input."""

    spec = SocialRlSelfRewardDemoMixtureStickyKernel.spec()
    assert spec.requires_social is True
    assert spec.required_social_fields == frozenset({"action", "reward"})


def test_kernel_has_six_parameters() -> None:
    """Ensure the kernel declares the extra stickiness parameter."""

    spec = SocialRlSelfRewardDemoMixtureStickyKernel.spec()
    param_names = [parameter.name for parameter in spec.parameter_specs]
    assert param_names == [
        "alpha_self",
        "alpha_other_outcome",
        "alpha_other_action",
        "w_imitation",
        "beta",
        "stickiness",
    ]


def test_self_update_stores_previous_own_choice() -> None:
    """Self-updates should refresh the perseveration state."""

    kernel = SocialRlSelfRewardDemoMixtureStickyKernel()
    params = kernel.parse_params(
        {
            "alpha_self": 0.0,
            "alpha_other_outcome": 0.0,
            "alpha_other_action": 0.0,
            "w_imitation": 0.0,
            "beta": 1.0,
            "stickiness": 0.0,
        }
    )
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
    assert updated.v_outcome[1] > state.v_outcome[1]


def test_social_update_preserves_previous_own_choice() -> None:
    """Social updates should not overwrite the last self action."""

    kernel = SocialRlSelfRewardDemoMixtureStickyKernel()
    params = kernel.parse_params(
        {
            "alpha_self": 0.0,
            "alpha_other_outcome": 0.0,
            "alpha_other_action": 0.0,
            "w_imitation": 0.0,
            "beta": 1.0,
            "stickiness": 0.0,
        }
    )
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
    assert updated.v_outcome[0] > state.v_outcome[0]
    assert updated.v_tendency[0] > state.v_tendency[0]


def test_action_probabilities_favor_previous_own_choice_with_positive_stickiness() -> None:
    """A positive stickiness term should bias repeats when values are equal."""

    kernel = SocialRlSelfRewardDemoMixtureStickyKernel()
    params = kernel.parse_params(
        {
            "alpha_self": 0.0,
            "alpha_other_outcome": 0.0,
            "alpha_other_action": 0.0,
            "w_imitation": 0.0,
            "beta": 1.0,
            "stickiness": 3.0,
        }
    )
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
