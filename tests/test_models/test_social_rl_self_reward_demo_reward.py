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
    """Ensure the social kernel correctly applies self and social update steps.

    Two separate update calls are made — one self-update (actor == learner)
    and one social update (actor != learner) — mirroring how the engine and
    extractor emit one view per UPDATE event.

    Returns
    -------
    None
        This test asserts both update paths.
    """

    kernel = SocialRlSelfRewardDemoRewardKernel()
    params = kernel.parse_params({"alpha_self": 0.0, "alpha_other": 0.0, "beta": 1.0})
    state = kernel.initial_state(2, params)

    # Self-update: subject chose action 0 and received reward 1.0.
    self_view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
        action=0,
        reward=1.0,
    )
    state = kernel.update(state, self_view, params)

    # Social update: demonstrator chose action 1 and received reward 0.0.
    social_view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="demonstrator",
        learner_id="subject",
        action=1,
        reward=0.0,
    )
    updated_state = kernel.update(state, social_view, params)

    assert updated_state.q_values[0] > 0.5
    assert updated_state.q_values[1] < 0.5
