"""Tests for the social observed-outcome Q-learning kernel."""

from comp_model.data.extractors import DecisionTrialView
from comp_model.models.kernels.social_rl_self_reward_demo_reward import (
    SocialRlSelfRewardDemoRewardKernel,
)


def test_social_kernel_reports_social_requirement() -> None:
    """Ensure the social kernel declares that it consumes social input.

    Returns
    -------
    None
        This test asserts static kernel metadata.
    """

    assert SocialRlSelfRewardDemoRewardKernel.spec().requires_social is True


def test_social_kernel_updates_self_and_social_q_values() -> None:
    """Ensure the social kernel updates both self and demonstrator actions.

    Returns
    -------
    None
        This test asserts both update paths.
    """

    kernel = SocialRlSelfRewardDemoRewardKernel()
    params = kernel.parse_params({"alpha_self": 0.0, "alpha_other": 0.0, "beta": 1.0})
    state = kernel.initial_state(2, params)
    view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        choice=0,
        reward=1.0,
        social_action=1,
        social_reward=0.0,
    )

    next_state = kernel.next_state(state, view, params)

    assert next_state.q_values[0] > 0.5
    assert next_state.q_values[1] < 0.5
