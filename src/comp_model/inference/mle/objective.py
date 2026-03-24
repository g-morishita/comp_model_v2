"""Replay log-likelihood functions for maximum-likelihood estimation.

These functions replay observed event traces through a model kernel using the
same extracted decision views used during simulation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from comp_model.data.extractors import replay_trial_steps
from comp_model.data.schema import EventPhase, SubjectData

if TYPE_CHECKING:
    from comp_model.models.condition.shared_delta import SharedDeltaLayout
    from comp_model.models.kernels.base import ModelKernel
    from comp_model.tasks.schemas import TrialSchema


def log_likelihood_simple(
    kernel: ModelKernel[object, object],
    subject_data: SubjectData,
    raw_params: dict[str, float],
    schema: TrialSchema,
) -> float:
    """Compute single-subject replay log-likelihood without condition structure.

    Parameters
    ----------
    kernel
        Model kernel used for replay.
    subject_data
        Subject data being evaluated.
    raw_params
        Unconstrained parameter values keyed by kernel parameter name.
    schema
        Trial schema shared across evaluated trials.

    Returns
    -------
    float
        Total log-likelihood across all decision views.

    Notes
    -----
    Replay parses the unconstrained parameters once, initializes latent state,
    loops over blocks and trials in observed order, and accumulates
    ``log(p(choice))`` from the action-probability vector aligned to
    ``view.available_actions``. Probabilities are clipped away from zero before
    taking logs to avoid ``-inf`` from numerical underflow.
    """

    params = kernel.parse_params(raw_params)
    reset_policy = kernel.spec().state_reset_policy
    n_actions = _infer_n_actions_from_data(subject_data)
    total_log_likelihood = 0.0
    state = kernel.initial_state(n_actions, params)

    for block_index, block in enumerate(subject_data.blocks):
        if reset_policy == "per_block" and block_index > 0:
            state = kernel.initial_state(n_actions, params)

        for trial in block.trials:
            for event_type, learner_id, view in replay_trial_steps(trial, schema):
                if event_type == EventPhase.DECISION and learner_id == "subject":
                    probabilities = kernel.action_probabilities(state, view, params)
                    choice_index = view.available_actions.index(view.action)
                    total_log_likelihood += math.log(max(probabilities[choice_index], 1e-15))
                elif event_type == EventPhase.UPDATE and learner_id == "subject":
                    state = kernel.update(state, view, params)

    return total_log_likelihood


def _infer_n_actions_from_data(subject_data: SubjectData) -> int:
    """Infer the action count from subject event data.

    Parameters
    ----------
    subject_data
        Subject data whose first INPUT event is inspected.

    Returns
    -------
    int
        Number of available actions.

    Raises
    ------
    ValueError
        Raised when no INPUT event with available actions is found.

    Notes
    -----
    This helper scans the observed data rather than trusting task metadata so
    replay can work directly from canonical subject traces.
    """

    for block in subject_data.blocks:
        for trial in block.trials:
            for event in trial.events:
                if event.phase == EventPhase.INPUT and "available_actions" in event.payload:
                    return len(event.payload["available_actions"])
    raise ValueError("Cannot infer n_actions: no INPUT events found")


def log_likelihood_conditioned(
    kernel: ModelKernel[object, object],
    layout: SharedDeltaLayout,
    subject_data: SubjectData,
    raw_params: dict[str, float],
    schema: TrialSchema,
) -> float:
    """Compute replay log-likelihood with shared-plus-delta condition parameters.

    Parameters
    ----------
    kernel
        Model kernel used for replay.
    layout
        Shared-plus-delta parameter layout across conditions.
    subject_data
        Subject data being evaluated.
    raw_params
        Unconstrained layout parameter values keyed by layout names.
    schema
        Trial schema shared across evaluated trials.

    Returns
    -------
    float
        Total log-likelihood across all decision views.

    Notes
    -----
    Conditioned replay changes parameters by block condition, but it still follows the
    kernel's state reset policy. With ``state_reset_policy="continuous"``, latent
    state carries across condition boundaries within a subject.

    For each block, the shared-plus-delta layout reconstructs that block's
    unconstrained parameter vector before replay continues through the block's
    trials.
    """

    reset_policy = kernel.spec().state_reset_policy
    n_actions = _infer_n_actions_from_data(subject_data)
    total_log_likelihood = 0.0
    state: object | None = None

    for block in subject_data.blocks:
        condition_params = kernel.parse_params(layout.reconstruct(raw_params, block.condition))
        if state is None or reset_policy == "per_block":
            state = kernel.initial_state(n_actions, condition_params)

        for trial in block.trials:
            for event_type, learner_id, view in replay_trial_steps(trial, schema):
                if event_type == EventPhase.DECISION and learner_id == "subject":
                    probabilities = kernel.action_probabilities(state, view, condition_params)
                    choice_index = view.available_actions.index(view.action)
                    total_log_likelihood += math.log(max(probabilities[choice_index], 1e-15))
                elif event_type == EventPhase.UPDATE and learner_id == "subject":
                    state = kernel.update(state, view, condition_params)

    return total_log_likelihood
