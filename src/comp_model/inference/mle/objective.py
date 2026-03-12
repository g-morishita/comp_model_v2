"""Replay log-likelihood functions for maximum-likelihood estimation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from comp_model.data.extractors import extract_decision_views
from comp_model.data.schema import EventPhase, SubjectData

if TYPE_CHECKING:
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
            for view in extract_decision_views(trial, schema):
                probabilities = kernel.action_probabilities(state, view, params)
                choice_index = view.available_actions.index(view.choice)
                total_log_likelihood += math.log(max(probabilities[choice_index], 1e-15))
                state = kernel.next_state(state, view, params)

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
    """

    for block in subject_data.blocks:
        for trial in block.trials:
            for event in trial.events:
                if event.phase == EventPhase.INPUT and "available_actions" in event.payload:
                    return len(event.payload["available_actions"])
    raise ValueError("Cannot infer n_actions: no INPUT events found")
