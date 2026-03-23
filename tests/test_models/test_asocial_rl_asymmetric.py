"""Tests for the asocial asymmetric RL kernel."""

import math

from comp_model.data.extractors import DecisionTrialView
from comp_model.models.kernels.asocial_rl_asymmetric import (
    AsocialRlAsymmetricKernel,
    AsocialRlAsymmetricState,
)


def test_asymmetric_kernel_action_probabilities_sum_to_one() -> None:
    """Ensure asymmetric RL action probabilities are normalized.

    Returns
    -------
    None
        This test asserts probability normalization.
    """

    kernel = AsocialRlAsymmetricKernel()
    params = kernel.parse_params({"alpha_pos": 0.0, "alpha_neg": 0.0, "beta": 1.0})
    state = AsocialRlAsymmetricState(q_values=[0.25, 0.75])
    view = DecisionTrialView(trial_index=0, available_actions=(0, 1), choice=1)

    probabilities = kernel.action_probabilities(state, view, params)

    assert math.isclose(sum(probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert probabilities[1] > probabilities[0]


def test_asymmetric_kernel_positive_rpe_uses_alpha_pos() -> None:
    """Ensure positive prediction errors use alpha_pos.

    Returns
    -------
    None
        This test asserts the asymmetric update for positive RPEs.
    """

    kernel = AsocialRlAsymmetricKernel()
    # alpha_pos = 0.5 (sigmoid(0) ≈ 0.5), alpha_neg ≈ 0 (sigmoid(-10) ≈ 0)
    params = kernel.parse_params({"alpha_pos": 0.0, "alpha_neg": -10.0, "beta": 1.0})
    state = kernel.initial_state(2, params)
    view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        choice=1,
        reward=1.0,  # reward > Q[1]=0.5 → positive RPE
    )

    updated_state = kernel.update(state, view, params)

    assert updated_state.q_values[0] == 0.5
    assert updated_state.q_values[1] > 0.5


def test_asymmetric_kernel_negative_rpe_uses_alpha_neg() -> None:
    """Ensure negative prediction errors use alpha_neg.

    Returns
    -------
    None
        This test asserts the asymmetric update for negative RPEs.
    """

    kernel = AsocialRlAsymmetricKernel()
    # alpha_pos ≈ 0 (sigmoid(-10)), alpha_neg = 0.5 (sigmoid(0))
    params = kernel.parse_params({"alpha_pos": -10.0, "alpha_neg": 0.0, "beta": 1.0})
    state = kernel.initial_state(2, params)
    view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        choice=1,
        reward=0.0,  # reward < Q[1]=0.5 → negative RPE
    )

    updated_state = kernel.update(state, view, params)

    assert updated_state.q_values[0] == 0.5
    assert updated_state.q_values[1] < 0.5


def test_asymmetric_kernel_ignores_social_update_step() -> None:
    """Ensure the asymmetric kernel leaves Q-values unchanged on social UPDATE steps.

    When fitted to data collected under a social schema for model comparison,
    the replay engine emits social UPDATE steps with ``choice=None``.  The
    asocial kernel must ignore these so only the participant's own experience
    drives learning.

    Returns
    -------
    None
        This test asserts that social UPDATE steps are skipped.
    """

    kernel = AsocialRlAsymmetricKernel()
    params = kernel.parse_params({"alpha_pos": 0.0, "alpha_neg": 0.0, "beta": 1.0})
    state = kernel.initial_state(2, params)
    social_view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        choice=None,  # social UPDATE: demonstrator's step, no participant choice
        reward=None,
        social_action=0,
        social_reward=1.0,
    )

    updated_state = kernel.update(state, social_view, params)

    assert updated_state.q_values == state.q_values
