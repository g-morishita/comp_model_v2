"""Tests for the social self-reward + demo-action mixture RL kernel."""

from comp_model.data.extractors import DecisionTrialView
from comp_model.models.kernels.social_rl_self_reward_demo_action_mixture import (
    SocialRlSelfRewardDemoActionMixtureKernel,
)


def test_kernel_reports_social_requirement() -> None:
    """Ensure the kernel declares that it consumes social input."""
    spec = SocialRlSelfRewardDemoActionMixtureKernel.spec()
    assert spec.requires_social is True
    assert spec.required_social_fields == frozenset({"action"})


def test_kernel_has_four_parameters() -> None:
    """Ensure the kernel declares exactly four free parameters."""
    spec = SocialRlSelfRewardDemoActionMixtureKernel.spec()
    param_names = [p.name for p in spec.parameter_specs]
    assert param_names == ["alpha_self", "alpha_other_action", "w_imitation", "beta"]


def test_self_update_modifies_v_outcome() -> None:
    """Self-UPDATE steps must update v_outcome via alpha_self."""
    kernel = SocialRlSelfRewardDemoActionMixtureKernel()
    params = kernel.parse_params(
        {"alpha_self": 0.0, "alpha_other_action": 0.0, "w_imitation": 0.0, "beta": 1.0}
    )
    state = kernel.initial_state(2, params)
    original_v_outcome = list(state.v_outcome)
    original_v_tendency = list(state.v_tendency)

    self_view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
        action=0,
        reward=1.0,
    )
    updated = kernel.update(state, self_view, params)

    # With alpha=0.5 (sigmoid(0)=0.5), v_outcome should change
    assert updated.v_outcome[0] > original_v_outcome[0]
    # v_tendency should be unchanged by self-update
    assert updated.v_tendency == original_v_tendency


def test_social_update_modifies_only_v_tendency() -> None:
    """Social UPDATE steps must update v_tendency only, not v_outcome."""
    kernel = SocialRlSelfRewardDemoActionMixtureKernel()
    params = kernel.parse_params(
        {"alpha_self": 0.0, "alpha_other_action": 0.0, "w_imitation": 0.0, "beta": 1.0}
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

    # v_outcome should NOT change (no demo reward learning)
    assert updated.v_outcome == original_v_outcome
    # With alpha=0.5 (sigmoid(0)=0.5), v_tendency should change
    assert updated.v_tendency[0] > original_v_tendency[0]
    assert updated.v_tendency[1] < original_v_tendency[1]


def test_action_probabilities_combine_systems() -> None:
    """Action probabilities must combine v_outcome and v_tendency via w_imitation."""
    kernel = SocialRlSelfRewardDemoActionMixtureKernel()
    # w_imitation at ~1 means v_tendency dominates
    params = kernel.parse_params(
        {"alpha_self": 0.0, "alpha_other_action": 0.0, "w_imitation": 5.0, "beta": 1.0}
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
