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
    population_true_params
        True population-level parameters (mu/sd on unconstrained scale).
        Populated only for hierarchical Bayes fits with unconstrained-scale
        ``ParamDist`` entries. Keys follow the pattern ``mu_{param}_z`` and
        ``sd_{param}_z``.
    population_estimates
        Posterior means of population-level parameters. Same key convention
        as ``population_true_params``.
    population_posterior_samples
        Full posterior draws for population-level parameters, used to compute
        credible-interval coverage.
    """

    replication_index: int
    true_params: dict[str, dict[str, float]]
    subject_estimates: tuple[SubjectEstimates, ...]
    population_true_params: dict[str, float] | None = None
    population_estimates: dict[str, float] | None = None
    population_posterior_samples: dict[str, np.ndarray] | None = None


def extract_population_estimates(
    result: BayesFitResult,
    param_names: Sequence[str],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Extract population-level estimates from a hierarchical Bayes fit.

    Looks for two kinds of population parameters in the posterior:

    1. **Constrained-scale population mean** — ``{param}_pop`` scalars emitted
       by Stan's ``generated quantities`` block (e.g. ``alpha_pos_pop =
       inv_logit(mu_alpha_pos_z)``).  These are on the same scale as the
       per-subject parameters and can be compared to the empirical mean of the
       true constrained parameters across subjects.

    2. **Unconstrained-scale mu / sd** — ``mu_{param}_z`` and ``sd_{param}_z``
       parameters directly sampled by Stan.  These are only meaningful when the
       ``ParamDist`` was specified with ``scale="unconstrained"`` so that the
       true distribution mean and SD are known.

    Parameters
    ----------
    result
        Bayesian fit result with posterior samples.
    param_names
        Subject-level parameter names (e.g. ``["alpha_pos", "beta"]``).

    Returns
    -------
    point_estimates
        Posterior means keyed by ``{param}_pop``, ``mu_{param}_z``, or
        ``sd_{param}_z`` depending on what the Stan program exposes.
    posterior_samples
        Full draw arrays for the same keys (used for coverage computation).
    """

    point: dict[str, float] = {}
    samples: dict[str, np.ndarray] = {}
    for name in param_names:
        # Constrained-scale population mean from generated quantities
        pop_key = f"{name}_pop"
        if pop_key in result.posterior_samples:
            draws = result.posterior_samples[pop_key]
            point[pop_key] = float(np.mean(draws))
            samples[pop_key] = draws

        # Unconstrained-scale population mean and SD (used when scale="unconstrained")
        for prefix in ("mu", "sd"):
            key = f"{prefix}_{name}_z"
            if key in result.posterior_samples:
                draws = result.posterior_samples[key]
                point[key] = float(np.mean(draws))
                samples[key] = draws
    return point, samples


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
            if samples.ndim == 1:
                # Shape: (n_draws,) — shared across subjects (e.g. SUBJECT_SHARED)
                point[name] = float(np.mean(samples))
                draws[name] = samples
            elif samples.ndim == 2:
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
