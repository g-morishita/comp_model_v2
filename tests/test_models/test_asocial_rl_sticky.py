"""Tests for the asocial sticky RL kernel."""

import math

from comp_model.data.extractors import DecisionTrialView
from comp_model.models.kernels.asocial_rl_sticky import (
    AsocialRlStickyKernel,
    AsocialRlStickyState,
)


def test_asocial_sticky_action_probabilities_sum_to_one() -> None:
    """Asocial sticky action probabilities should remain normalized."""

    kernel = AsocialRlStickyKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0, "stickiness": 0.0})
    state = AsocialRlStickyState(q_values=[0.25, 0.75], last_self_action=None)
    view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
    )

    probabilities = kernel.action_probabilities(state, view, params)

    assert math.isclose(sum(probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert probabilities[1] > probabilities[0]


def test_asocial_sticky_positive_stickiness_biases_repeats() -> None:
    """Positive stickiness should favor the previous own choice when values are equal."""

    kernel = AsocialRlStickyKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0, "stickiness": 2.5})
    state = AsocialRlStickyState(q_values=[0.5, 0.5], last_self_action=1)
    view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
    )

    probabilities = kernel.action_probabilities(state, view, params)

    assert probabilities[1] > probabilities[0]


def test_asocial_sticky_updates_q_value_and_previous_choice() -> None:
    """Self-updates should apply Q-learning and store the most recent own action."""

    kernel = AsocialRlStickyKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0, "stickiness": 0.0})
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
    assert updated_state.last_self_action == 1


def test_asocial_sticky_stores_previous_choice_without_reward() -> None:
    """Choice history should update even when the trial omits feedback."""

    kernel = AsocialRlStickyKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0, "stickiness": 0.0})
    state = kernel.initial_state(2, params)
    view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="subject",
        learner_id="subject",
        action=0,
        reward=None,
    )

    updated_state = kernel.update(state, view, params)

    assert updated_state.q_values == state.q_values
    assert updated_state.last_self_action == 0


def test_asocial_sticky_ignores_social_update_step() -> None:
    """Social UPDATE steps should leave both values and previous-choice state unchanged."""

    kernel = AsocialRlStickyKernel()
    params = kernel.parse_params({"alpha": 0.5, "beta": 1.0, "stickiness": 0.2})
    state = AsocialRlStickyState(q_values=[0.5, 0.5], last_self_action=1)
    social_view = DecisionTrialView(
        trial_index=0,
        available_actions=(0, 1),
        actor_id="demonstrator",
        learner_id="subject",
        action=0,
        reward=1.0,
    )

    updated_state = kernel.update(state, social_view, params)

    assert updated_state.q_values == state.q_values
    assert updated_state.last_self_action == state.last_self_action
