"""Tests for the social demo-mixture RL kernel."""

from comp_model.data.extractors import DecisionTrialView
from comp_model.models.kernels.social_rl_demo_mixture import (
    SocialRlDemoMixtureKernel,
)


def test_kernel_reports_social_requirement() -> None:
    """Ensure the kernel declares that it consumes social input."""
    spec = SocialRlDemoMixtureKernel.spec()
    assert spec.requires_social is True
    assert spec.required_social_fields == frozenset({"action", "reward"})


def test_kernel_has_four_parameters() -> None:
    """Ensure the kernel declares exactly four free parameters."""
    spec = SocialRlDemoMixtureKernel.spec()
    param_names = [p.name for p in spec.parameter_specs]
    assert param_names == ["alpha_other_outcome", "alpha_other_action", "w_imitation", "beta"]


def test_self_update_is_noop() -> None:
    """Self-UPDATE steps must leave state unchanged.

    This kernel does not learn from the subject's own outcomes.
    """
    kernel = SocialRlDemoMixtureKernel()
    params = kernel.parse_params(
        {"alpha_other_outcome": 0.0, "alpha_other_action": 0.0, "w_imitation": 0.0, "beta": 1.0}
    )
    state = kernel.initial_state(2, params)

    self_view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
        action=0,
        reward=1.0,
    )
    updated = kernel.update(state, self_view, params)

    assert updated.v_outcome == state.v_outcome
    assert updated.v_tendency == state.v_tendency
    assert updated is state  # same object returned


def test_social_update_modifies_both_systems() -> None:
    """Social UPDATE steps must update both v_outcome and v_tendency."""
    kernel = SocialRlDemoMixtureKernel()
    params = kernel.parse_params(
        {"alpha_other_outcome": 0.0, "alpha_other_action": 0.0, "w_imitation": 0.0, "beta": 1.0}
    )
    state = kernel.initial_state(2, params)
    original_v_outcome = list(state.v_outcome)
    original_v_tendency = list(state.v_tendency)

    social_view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="demonstrator",
        learner_id="subject",
        action=0,
        reward=1.0,
    )
    updated = kernel.update(state, social_view, params)

    # With alpha=0.5 (sigmoid(0)=0.5), values should change
    assert updated.v_outcome[0] > original_v_outcome[0]
    assert updated.v_tendency[0] > original_v_tendency[0]
    assert updated.v_tendency[1] < original_v_tendency[1]


def test_action_probabilities_combine_systems() -> None:
    """Action probabilities must combine v_outcome and v_tendency via w_imitation."""
    kernel = SocialRlDemoMixtureKernel()
    # w_imitation at ~1 means v_tendency dominates
    params = kernel.parse_params(
        {"alpha_other_outcome": 0.0, "alpha_other_action": 0.0, "w_imitation": 5.0, "beta": 1.0}
    )
    state = kernel.initial_state(2, params)
    # Manually set v_tendency to favour action 0
    state.v_tendency = [0.9, 0.1]

    view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
    )
    probs = kernel.action_probabilities(state, view, params)

    assert probs[0] > probs[1]
