"""Unified estimate extraction from MLE and Bayesian fit results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from comp_model.recovery.parameter.result import PopulationRecord, SubjectRecord

if TYPE_CHECKING:
    from collections.abc import Sequence

    from comp_model.inference.bayes.result import BayesFitResult
    from comp_model.inference.mle.optimize import MleFitResult
    from comp_model.models.condition.shared_delta import SharedDeltaLayout


def extract_mle_subject_records(
    results: Sequence[MleFitResult],
    true_params: dict[str, dict[str, float]],
    layout: SharedDeltaLayout | None = None,
) -> tuple[SubjectRecord, ...]:
    """Extract subject records from per-subject MLE results.

    Parameters
    ----------
    results
        MLE fit results, one per subject.
    true_params
        Ground-truth constrained parameters keyed by subject id, then by
        parameter name.  Keys may use the ``{name}__{condition}`` convention
        for condition-aware fits.
    layout
        Optional condition-aware layout for extracting per-condition params.

    Returns
    -------
    tuple[SubjectRecord, ...]
        One record per (subject, parameter[, condition]) combination.
    """
    records: list[SubjectRecord] = []
    for result in results:
        sid = result.subject_id
        true_p = true_params[sid]

        if layout is not None and result.params_by_condition is not None:
            for condition, params in result.params_by_condition.items():
                for name, est_val in params.items():
                    true_key = f"{name}__{condition}"
                    true_val = true_p[true_key]
                    records.append(
                        SubjectRecord(
                            subject_id=sid,
                            param_name=name,
                            condition=condition,
                            true_value=true_val,
                            estimated_value=est_val,
                            posterior_draws=None,
                        )
                    )
        else:
            for name, est_val in result.constrained_params.items():
                true_val = true_p[name]
                records.append(
                    SubjectRecord(
                        subject_id=sid,
                        param_name=name,
                        condition=None,
                        true_value=true_val,
                        estimated_value=est_val,
                        posterior_draws=None,
                    )
                )
    return tuple(records)


def extract_bayes_subject_records(
    result: BayesFitResult,
    subject_ids: Sequence[str],
    param_names: Sequence[str],
    true_params: dict[str, dict[str, float]],
    layout: SharedDeltaLayout | None = None,
) -> tuple[SubjectRecord, ...]:
    """Extract subject records from hierarchical Bayes results.

    Parameters
    ----------
    result
        Bayesian fit result with posterior samples.
    subject_ids
        Subject identifiers in dataset order.
    param_names
        Subject-level parameter names in the Stan model.
    true_params
        Ground-truth constrained parameters keyed by subject id, then by
        parameter name.
    layout
        Optional condition-aware layout.

    Returns
    -------
    tuple[SubjectRecord, ...]
        One record per (subject, parameter[, condition]) combination with
        full posterior draws.
    """
    records: list[SubjectRecord] = []
    for i, sid in enumerate(subject_ids):
        true_p = true_params[sid]

        for name in param_names:
            samples = result.posterior_samples[name]
            if samples.ndim == 1:
                # Shape: (n_draws,) — shared across subjects
                est = float(np.mean(samples))
                true_val = true_p[name]
                records.append(
                    SubjectRecord(
                        subject_id=sid,
                        param_name=name,
                        condition=None,
                        true_value=true_val,
                        estimated_value=est,
                        posterior_draws=samples,
                    )
                )
            elif samples.ndim == 2:
                # Shape: (n_draws, n_subjects)
                subject_draws = samples[:, i]
                est = float(np.mean(subject_draws))
                true_val = true_p[name]
                records.append(
                    SubjectRecord(
                        subject_id=sid,
                        param_name=name,
                        condition=None,
                        true_value=true_val,
                        estimated_value=est,
                        posterior_draws=subject_draws,
                    )
                )
            elif samples.ndim == 3 and layout is not None:
                # Shape: (n_draws, n_subjects, n_conditions)
                for c_idx, condition in enumerate(layout.conditions):
                    subject_draws = samples[:, i, c_idx]
                    est = float(np.mean(subject_draws))
                    true_key = f"{name}__{condition}"
                    true_val = true_p[true_key]
                    records.append(
                        SubjectRecord(
                            subject_id=sid,
                            param_name=name,
                            condition=condition,
                            true_value=true_val,
                            estimated_value=est,
                            posterior_draws=subject_draws,
                        )
                    )
            else:
                subject_draws = samples[:, i]
                est = float(np.mean(subject_draws))
                true_val = true_p[name]
                records.append(
                    SubjectRecord(
                        subject_id=sid,
                        param_name=name,
                        condition=None,
                        true_value=true_val,
                        estimated_value=est,
                        posterior_draws=subject_draws,
                    )
                )
    return tuple(records)


def extract_population_records(
    result: BayesFitResult,
    param_names: Sequence[str],
    true_pop: dict[str, float],
) -> tuple[PopulationRecord, ...]:
    """Extract population records from a hierarchical Bayes fit.

    Looks for population parameters in the posterior:

    1. Constrained-scale population mean (``{param}_pop``)
    2. Unconstrained-scale mu / sd (``mu_{param}_z``, ``sd_{param}_z``)

    Parameters
    ----------
    result
        Bayesian fit result with posterior samples.
    param_names
        Subject-level parameter names (e.g. ``["alpha_pos", "beta"]``).
    true_pop
        True population parameter values keyed by the same names that appear
        in the posterior (e.g. ``alpha_pop``, ``mu_alpha_z``).

    Returns
    -------
    tuple[PopulationRecord, ...]
        One record per population parameter found in the posterior.
    """
    records: list[PopulationRecord] = []
    for name in param_names:
        # Constrained-scale population mean from generated quantities
        pop_key = f"{name}_pop"
        if pop_key in result.posterior_samples and pop_key in true_pop:
            draws = result.posterior_samples[pop_key]
            records.append(
                PopulationRecord(
                    param_name=pop_key,
                    true_value=true_pop[pop_key],
                    estimated_value=float(np.mean(draws)),
                    posterior_draws=draws,
                )
            )

        # Unconstrained-scale population mean and SD
        for prefix in ("mu", "sd"):
            key = f"{prefix}_{name}_z"
            if key in result.posterior_samples and key in true_pop:
                draws = result.posterior_samples[key]
                records.append(
                    PopulationRecord(
                        param_name=key,
                        true_value=true_pop[key],
                        estimated_value=float(np.mean(draws)),
                        posterior_draws=draws,
                    )
                )
    return tuple(records)
