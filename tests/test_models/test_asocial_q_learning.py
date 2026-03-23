"""Tests for the asocial Q-learning kernel."""

import math

from comp_model.data.extractors import DecisionTrialView
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel, QState


def test_asocial_kernel_action_probabilities_sum_to_one() -> None:
    """Ensure asocial action probabilities are normalized.

    Returns
    -------
    None
        This test asserts probability normalization.
    """

    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    state = QState(q_values=[0.25, 0.75])
    view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
        action=1,
    )

    probabilities = kernel.action_probabilities(state, view, params)

    assert math.isclose(sum(probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert probabilities[1] > probabilities[0]


def test_asocial_kernel_updates_chosen_q_value() -> None:
    """Ensure only the chosen action's Q-value is updated.

    Returns
    -------
    None
        This test asserts the Q-learning update rule.
    """

    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    state = kernel.initial_state(2, params)
    view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
        action=1,
        reward=1.0,
    )

    updated_state = kernel.update(state, view, params)

    assert updated_state.q_values[0] == 0.5
    assert updated_state.q_values[1] > 0.5


def test_asocial_kernel_ignores_social_update_step() -> None:
    """Ensure the asocial kernel leaves Q-values unchanged on social UPDATE steps.

    When the asocial kernel is fitted to data collected under a social schema
    (e.g. for model comparison), the replay engine emits UPDATE steps that
    carry the demonstrator's outcome rather than the participant's own
    choice.  These steps have ``choice=None`` and the kernel must ignore them
    so that only the participant's own experience drives learning.

    Returns
    -------
    None
        This test asserts that social UPDATE steps are skipped.
    """

    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.5, "beta": 1.0})
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
