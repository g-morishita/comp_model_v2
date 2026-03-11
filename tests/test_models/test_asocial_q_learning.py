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
    view = DecisionTrialView(trial_index=0, available_actions=(0, 1), choice=1)

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
        choice=1,
        reward=1.0,
    )

    next_state = kernel.next_state(state, view, params)

    assert next_state.q_values[0] == 0.5
    assert next_state.q_values[1] > 0.5
