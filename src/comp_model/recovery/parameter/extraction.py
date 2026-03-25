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
            elif samples.ndim == 2 and layout is not None:
                # Shape: (n_draws, n_conditions) — SUBJECT_BLOCK_CONDITION
                for c_idx, condition in enumerate(layout.conditions):
                    cond_draws = samples[:, c_idx]
                    est = float(np.mean(cond_draws))
                    true_key = f"{name}__{condition}"
                    true_val = true_p[true_key]
                    records.append(
                        SubjectRecord(
                            subject_id=sid,
                            param_name=name,
                            condition=condition,
                            true_value=true_val,
                            estimated_value=est,
                            posterior_draws=cond_draws,
                        )
                    )
            elif samples.ndim == 2:
                # Shape: (n_draws, n_subjects) — STUDY_SUBJECT
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
    true_pop: dict[str, float],
    layout: SharedDeltaLayout | None = None,
) -> tuple[PopulationRecord, ...]:
    """Extract constrained-scale population records from a hierarchical Bayes fit.

    Iterates over the constrained-scale keys in *true_pop* and extracts a
    record for every key that also exists in the posterior samples. Only
    population mean outputs whose names end with ``"_pop"`` are reported.

    Parameters
    ----------
    result
        Bayesian fit result with posterior samples.
    true_pop
        True population parameter values keyed by the same names that appear
        in the posterior (e.g. ``alpha_pop``, ``alpha_shared_pop``).
    layout
        Optional condition-aware layout used to split vector-valued
        population delta parameters into one record per non-baseline
        condition.

    Returns
    -------
    tuple[PopulationRecord, ...]
        One record per population parameter found in both *true_pop* and
        the posterior.

    Raises
    ------
    ValueError
        If a posterior population parameter is not scalar and no
        condition-aware layout is available to interpret it.
    """
    records: list[PopulationRecord] = []
    nonbaseline_conditions = ()
    if layout is not None:
        nonbaseline_conditions = tuple(
            condition for condition in layout.conditions if condition != layout.baseline_condition
        )

    for key, true_val in true_pop.items():
        if not key.endswith("_pop"):
            continue
        if key in result.posterior_samples:
            draws = np.asarray(result.posterior_samples[key])
            if draws.ndim == 0:
                scalar_draws = draws.reshape(1)
                records.append(
                    PopulationRecord(
                        param_name=key,
                        condition=None,
                        true_value=true_val,
                        estimated_value=float(np.mean(scalar_draws)),
                        posterior_draws=scalar_draws,
                    )
                )
                continue

            if draws.ndim == 1:
                records.append(
                    PopulationRecord(
                        param_name=key,
                        condition=None,
                        true_value=true_val,
                        estimated_value=float(np.mean(draws)),
                        posterior_draws=draws,
                    )
                )
                continue

            if draws.ndim == 2 and layout is not None:
                if draws.shape[1] != len(nonbaseline_conditions):
                    raise ValueError(
                        f"Population posterior {key!r} has shape {draws.shape}, "
                        f"but layout expects {len(nonbaseline_conditions)} "
                        "non-baseline conditions"
                    )
                for c_idx, condition in enumerate(nonbaseline_conditions):
                    condition_draws = draws[:, c_idx]
                    records.append(
                        PopulationRecord(
                            param_name=key,
                            condition=condition,
                            true_value=true_val,
                            estimated_value=float(np.mean(condition_draws)),
                            posterior_draws=condition_draws,
                        )
                    )
                continue

            raise ValueError(
                f"Population posterior {key!r} must be scalar or condition-indexed; "
                f"got shape {draws.shape}"
            )
    return tuple(records)
