"""Tests for the non-sticky social demo-action bias RL kernel."""

from comp_model.data.extractors import DecisionTrialView
from comp_model.models.kernels.social_rl_demo_action_bias import SocialRlDemoActionBiasKernel


def test_kernel_reports_social_requirement() -> None:
    """Ensure the kernel declares that it consumes demonstrator action input."""

    spec = SocialRlDemoActionBiasKernel.spec()
    assert spec.requires_social is True
    assert spec.required_social_fields == frozenset({"action"})


def test_kernel_has_one_parameter() -> None:
    """Ensure the kernel declares only the demo-bias parameter."""

    spec = SocialRlDemoActionBiasKernel.spec()
    param_names = [parameter.name for parameter in spec.parameter_specs]
    assert param_names == ["demo_bias"]


def test_social_update_stores_previous_demo_action() -> None:
    """Social updates should refresh the most recent demonstrator action."""

    kernel = SocialRlDemoActionBiasKernel()
    params = kernel.parse_params({"demo_bias": 0.0})
    state = kernel.initial_state(2, params)

    social_view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="demonstrator",
        learner_id="subject",
        action=1,
        reward=1.0,
    )
    updated = kernel.update(state, social_view, params)

    assert updated.last_demo_action == 1


def test_self_update_preserves_demo_action_memory() -> None:
    """Self updates should not discard the stored demonstrator action."""

    kernel = SocialRlDemoActionBiasKernel()
    params = kernel.parse_params({"demo_bias": 0.0})
    state = kernel.initial_state(2, params)
    state.last_demo_action = 1

    self_view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
        action=0,
        reward=1.0,
    )
    updated = kernel.update(state, self_view, params)

    assert updated.last_demo_action == 1


def test_action_probabilities_favor_latest_demo_action_with_positive_demo_bias() -> None:
    """A positive demo-bias term should favor the observed demonstrator action."""

    kernel = SocialRlDemoActionBiasKernel()
    params = kernel.parse_params({"demo_bias": 2.5})
    state = kernel.initial_state(2, params)
    state.last_demo_action = 1

    view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
    )
    probs = kernel.action_probabilities(state, view, params)

    assert probs[1] > probs[0]


def test_social_timeout_preserves_existing_demo_action_memory() -> None:
    """Timeout social trials should not overwrite the previous demo choice."""

    kernel = SocialRlDemoActionBiasKernel()
    params = kernel.parse_params({"demo_bias": 0.0})
    state = kernel.initial_state(2, params)
    state.last_demo_action = 1

    timeout_view = DecisionTrialView(
        trial_index=1,
        available_actions=(0, 1),
        actor_id="demonstrator",
        learner_id="subject",
        action=None,
        reward=None,
    )
    updated = kernel.update(state, timeout_view, params)

    assert updated.last_demo_action == 1
