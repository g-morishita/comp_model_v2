"""Tests for the sticky social demo-mixture RL kernel."""

from comp_model.data.extractors import DecisionTrialView
from comp_model.models.kernels.social_rl_demo_mixture_sticky import (
    SocialRlDemoMixtureStickyKernel,
)


def test_kernel_reports_social_requirement() -> None:
    """Ensure the kernel declares that it consumes full social input."""

    spec = SocialRlDemoMixtureStickyKernel.spec()
    assert spec.requires_social is True
    assert spec.required_social_fields == frozenset({"action", "reward"})


def test_kernel_has_five_parameters() -> None:
    """Ensure the kernel declares the extra stickiness parameter."""

    spec = SocialRlDemoMixtureStickyKernel.spec()
    param_names = [parameter.name for parameter in spec.parameter_specs]
    assert param_names == [
        "alpha_other_outcome",
        "alpha_other_action",
        "w_imitation",
        "beta",
        "stickiness",
    ]


def test_observe_decision_stores_previous_own_choice() -> None:
    """Decision-time observation should refresh the perseveration state."""

    kernel = SocialRlDemoMixtureStickyKernel()
    params = kernel.parse_params(
        {
            "alpha_other_outcome": 0.0,
            "alpha_other_action": 0.0,
            "w_imitation": 0.0,
            "beta": 1.0,
            "stickiness": 0.0,
        }
    )
    state = kernel.initial_state(2, params)

    decision_view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
        action=1,
    )
    observed = kernel.observe_decision(state, decision_view, params)

    assert observed.last_self_action == 1
    assert observed.v_outcome == state.v_outcome
    assert observed.v_tendency == state.v_tendency


def test_self_update_preserves_values_but_keeps_last_self_action() -> None:
    """Self-updates should not learn, but they should retain own-choice history."""

    kernel = SocialRlDemoMixtureStickyKernel()
    params = kernel.parse_params(
        {
            "alpha_other_outcome": 0.0,
            "alpha_other_action": 0.0,
            "w_imitation": 0.0,
            "beta": 1.0,
            "stickiness": 0.0,
        }
    )
    state = kernel.initial_state(2, params)
    state.last_self_action = 0

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
    assert updated.v_outcome == state.v_outcome
    assert updated.v_tendency == state.v_tendency


def test_social_update_preserves_previous_own_choice_and_updates_both_systems() -> None:
    """Social updates should update learning systems without overwriting stickiness memory."""

    kernel = SocialRlDemoMixtureStickyKernel()
    params = kernel.parse_params(
        {
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
    assert updated.v_tendency[1] < state.v_tendency[1]


def test_action_probabilities_favor_previous_own_choice_with_positive_stickiness() -> None:
    """A positive stickiness term should bias repeats when mixture values are equal."""

    kernel = SocialRlDemoMixtureStickyKernel()
    params = kernel.parse_params(
        {
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
