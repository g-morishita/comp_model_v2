"""Tests for the social demo-action RL kernel."""

import math

from comp_model.data.extractors import DecisionTrialView
from comp_model.models.kernels.social_rl_demo_action import SocialRlDemoActionKernel


def test_social_demo_action_kernel_reports_social_requirement() -> None:
    """The kernel should declare that it consumes demonstrator action input."""

    spec = SocialRlDemoActionKernel.spec()
    assert spec.requires_social is True
    assert spec.required_social_fields == frozenset({"action"})


def test_social_demo_action_kernel_has_two_parameters() -> None:
    """The kernel should declare the action-learning and temperature parameters."""

    spec = SocialRlDemoActionKernel.spec()
    param_names = [parameter.name for parameter in spec.parameter_specs]
    assert param_names == ["alpha_other_action", "beta"]


def test_social_demo_action_kernel_ignores_self_updates() -> None:
    """Self UPDATE steps should leave action tendencies unchanged."""

    kernel = SocialRlDemoActionKernel()
    params = kernel.parse_params({"alpha_other_action": 0.0, "beta": 1.0})
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

    assert updated_state.v_tendency == state.v_tendency


def test_social_demo_action_kernel_updates_from_demonstrator_actions() -> None:
    """Social UPDATE steps should change the action-tendency state."""

    kernel = SocialRlDemoActionKernel()
    params = kernel.parse_params({"alpha_other_action": 0.0, "beta": 1.0})
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

    assert updated_state.v_tendency[0] < state.v_tendency[0]
    assert updated_state.v_tendency[1] > state.v_tendency[1]


def test_social_demo_action_action_probabilities_sum_to_one() -> None:
    """Choice probabilities should remain normalized."""

    kernel = SocialRlDemoActionKernel()
    params = kernel.parse_params({"alpha_other_action": 0.0, "beta": 1.0})
    state = kernel.initial_state(2, params)
    state.v_tendency[1] = 0.8
    state.v_tendency[0] = 0.2
    view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
    )

    probabilities = kernel.action_probabilities(state, view, params)

    assert math.isclose(sum(probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert probabilities[1] > probabilities[0]
