"""Unified estimate extraction from MLE and Bayesian fit results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from comp_model.inference.bayes.result import BayesFitResult
    from comp_model.inference.mle.optimize import MleFitResult
    from comp_model.models.condition.shared_delta import SharedDeltaLayout


@dataclass(frozen=True, slots=True)
class SubjectEstimates:
    """Recovered estimates for one subject.

    Attributes
    ----------
    subject_id
        Subject identifier.
    point_estimates
        Point estimates keyed by parameter name.
    posterior_samples
        Full posterior draws per parameter, or ``None`` for MLE.
    converged
        MLE convergence flag, or ``None`` for Bayes.
    """

    subject_id: str
    point_estimates: dict[str, float]
    posterior_samples: dict[str, np.ndarray] | None
    converged: bool | None


@dataclass(frozen=True, slots=True)
class ReplicationEstimates:
    """All estimates from one simulate-fit cycle.

    Attributes
    ----------
    replication_index
        Index of this replication.
    true_params
        Ground-truth constrained parameters per subject.
    subject_estimates
        Recovered estimates per subject.
    """

    replication_index: int
    true_params: dict[str, dict[str, float]]
    subject_estimates: tuple[SubjectEstimates, ...]


def extract_mle_estimates(
    results: Sequence[MleFitResult],
    layout: SharedDeltaLayout | None = None,
) -> tuple[SubjectEstimates, ...]:
    """Extract point estimates from per-subject MLE results.

    Parameters
    ----------
    results
        MLE fit results, one per subject.
    layout
        Optional condition-aware layout for extracting per-condition params.

    Returns
    -------
    tuple[SubjectEstimates, ...]
        Extracted estimates with ``posterior_samples=None``.
    """

    estimates: list[SubjectEstimates] = []
    for result in results:
        if layout is not None and result.params_by_condition is not None:
            point: dict[str, float] = {}
            for condition, params in result.params_by_condition.items():
                for name, value in params.items():
                    point[f"{name}__{condition}"] = value
        else:
            point = dict(result.constrained_params)
        estimates.append(
            SubjectEstimates(
                subject_id=result.subject_id,
                point_estimates=point,
                posterior_samples=None,
                converged=result.converged,
            )
        )
    return tuple(estimates)


def extract_bayes_estimates(
    result: BayesFitResult,
    subject_ids: Sequence[str],
    param_names: Sequence[str],
    layout: SharedDeltaLayout | None = None,
) -> tuple[SubjectEstimates, ...]:
    """Extract posterior means and draws from hierarchical Bayes results.

    Parameters
    ----------
    result
        Bayesian fit result with posterior samples.
    subject_ids
        Subject identifiers in dataset order.
    param_names
        Subject-level parameter names in the Stan model.
    layout
        Optional condition-aware layout.

    Returns
    -------
    tuple[SubjectEstimates, ...]
        Extracted estimates with full posterior draws.
    """

    estimates: list[SubjectEstimates] = []
    for i, sid in enumerate(subject_ids):
        point: dict[str, float] = {}
        draws: dict[str, np.ndarray] = {}

        for name in param_names:
            samples = result.posterior_samples[name]
            if samples.ndim == 2:
                # Shape: (n_draws, n_subjects)
                subject_draws = samples[:, i]
                point[name] = float(np.mean(subject_draws))
                draws[name] = subject_draws
            elif samples.ndim == 3 and layout is not None:
                # Shape: (n_draws, n_subjects, n_conditions)
                for c_idx, condition in enumerate(layout.conditions):
                    key = f"{name}__{condition}"
                    subject_draws = samples[:, i, c_idx]
                    point[key] = float(np.mean(subject_draws))
                    draws[key] = subject_draws
            else:
                subject_draws = samples[:, i]
                point[name] = float(np.mean(subject_draws))
                draws[name] = subject_draws

        estimates.append(
            SubjectEstimates(
                subject_id=sid,
                point_estimates=point,
                posterior_samples=draws,
                converged=None,
            )
        )
    return tuple(estimates)
