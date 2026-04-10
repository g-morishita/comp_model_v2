"""Tests for the sticky social demo-action RL kernel."""

from comp_model.data.extractors import DecisionTrialView
from comp_model.models.kernels.social_rl_demo_action_sticky import (
    SocialRlDemoActionStickyKernel,
)


def test_kernel_reports_social_requirement() -> None:
    """Ensure the kernel declares that it consumes demonstrator action input."""

    spec = SocialRlDemoActionStickyKernel.spec()
    assert spec.requires_social is True
    assert spec.required_social_fields == frozenset({"action"})


def test_kernel_has_three_parameters() -> None:
    """Ensure the kernel declares action-learning, temperature, and stickiness."""

    spec = SocialRlDemoActionStickyKernel.spec()
    param_names = [parameter.name for parameter in spec.parameter_specs]
    assert param_names == ["alpha_other_action", "beta", "stickiness"]


def test_self_update_stores_previous_own_choice_without_changing_tendencies() -> None:
    """Self-updates should refresh the perseveration state only."""

    kernel = SocialRlDemoActionStickyKernel()
    params = kernel.parse_params({"alpha_other_action": 0.0, "beta": 1.0, "stickiness": 0.0})
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
    assert updated.v_tendency == state.v_tendency


def test_social_update_preserves_previous_own_choice_and_updates_tendencies() -> None:
    """Social updates should not overwrite the last self action."""

    kernel = SocialRlDemoActionStickyKernel()
    params = kernel.parse_params({"alpha_other_action": 0.0, "beta": 1.0, "stickiness": 0.0})
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
    assert updated.v_tendency[0] > state.v_tendency[0]
    assert updated.v_tendency[1] < state.v_tendency[1]


def test_action_probabilities_favor_previous_own_choice_with_positive_stickiness() -> None:
    """A positive stickiness term should bias repeats when tendencies are equal."""

    kernel = SocialRlDemoActionStickyKernel()
    params = kernel.parse_params({"alpha_other_action": 0.0, "beta": 1.0, "stickiness": 3.0})
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


def test_self_timeout_preserves_existing_last_self_action() -> None:
    """Timeout self trials should not overwrite the previous own choice."""

    kernel = SocialRlDemoActionStickyKernel()
    params = kernel.parse_params({"alpha_other_action": 0.0, "beta": 1.0, "stickiness": 0.0})
    state = kernel.initial_state(2, params)
    state.last_self_action = 0

    timeout_view = DecisionTrialView(
        trial_index=1,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
        action=None,
        reward=None,
    )
    updated = kernel.update(state, timeout_view, params)

    assert updated.last_self_action == 0
    assert updated.v_tendency == state.v_tendency
