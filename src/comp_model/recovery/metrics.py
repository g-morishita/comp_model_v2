"""Recovery metrics computation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from comp_model.recovery.runner import RecoveryResult


@dataclass(frozen=True, slots=True)
class ParameterRecoveryMetrics:
    """Recovery metrics for a single parameter across all replications.

    Attributes
    ----------
    param_name
        Parameter name.
    correlation
        Pearson correlation between true and estimated values.
    rmse
        Root mean squared error.
    bias
        Mean signed error (estimated minus true).
    mean_absolute_error
        Mean absolute error.
    coverage_90
        Fraction of true values inside 90% HDI (Bayes only).
    coverage_95
        Fraction of true values inside 95% HDI (Bayes only).
    n_observations
        Total number of subject-replication pairs.
    """

    param_name: str
    correlation: float
    rmse: float
    bias: float
    mean_absolute_error: float
    coverage_90: float | None
    coverage_95: float | None
    n_observations: int


@dataclass(frozen=True, slots=True)
class RecoveryMetrics:
    """Aggregated metrics across all parameters.

    Attributes
    ----------
    per_parameter
        Metrics keyed by parameter name.
    """

    per_parameter: dict[str, ParameterRecoveryMetrics]


def compute_recovery_metrics(
    result: RecoveryResult,
    transforms: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
) -> RecoveryMetrics:
    """Compute recovery metrics from a completed study.

    Parameters
    ----------
    result
        Completed recovery study result.
    transforms
        Optional mapping from parameter name to a transform applied to both
        true and estimated values before computing metrics.  For example,
        ``{"beta": np.log}`` computes all metrics on the log scale.

    Returns
    -------
    RecoveryMetrics
        Per-parameter recovery metrics pooled across replications.
    """

    pairs: dict[str, list[tuple[float, float]]] = {}
    coverage_data: dict[str, list[tuple[float, np.ndarray]]] = {}

    for replication in result.replications:
        for subject_est in replication.subject_estimates:
            sid = subject_est.subject_id
            true_params = replication.true_params[sid]
            for param_name, true_val in true_params.items():
                if param_name not in subject_est.point_estimates:
                    continue
                est_val = subject_est.point_estimates[param_name]
                pairs.setdefault(param_name, []).append((true_val, est_val))

                if (
                    subject_est.posterior_samples is not None
                    and param_name in subject_est.posterior_samples
                ):
                    draws = subject_est.posterior_samples[param_name]
                    coverage_data.setdefault(param_name, []).append((true_val, draws))

        # Population-level: one observation per replication
        if replication.population_true_params and replication.population_estimates:
            for key, true_val in replication.population_true_params.items():
                if key not in replication.population_estimates:
                    continue
                est_val = replication.population_estimates[key]
                pairs.setdefault(key, []).append((true_val, est_val))
                if (
                    replication.population_posterior_samples is not None
                    and key in replication.population_posterior_samples
                ):
                    draws = replication.population_posterior_samples[key]
                    coverage_data.setdefault(key, []).append((true_val, draws))

    metrics: dict[str, ParameterRecoveryMetrics] = {}
    for param_name, param_pairs in pairs.items():
        true_arr = np.array([p[0] for p in param_pairs])
        est_arr = np.array([p[1] for p in param_pairs])
        n = len(param_pairs)

        fn = transforms.get(param_name) if transforms else None
        if fn is not None:
            true_arr = fn(true_arr)
            est_arr = fn(est_arr)

        errors = est_arr - true_arr
        bias = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))

        if n > 1 and np.std(true_arr) > 0 and np.std(est_arr) > 0:
            correlation = float(np.corrcoef(true_arr, est_arr)[0, 1])
        else:
            correlation = float("nan")

        cov_90: float | None = None
        cov_95: float | None = None
        if param_name in coverage_data:
            cov_90 = _compute_coverage(coverage_data[param_name], 0.90, fn)
            cov_95 = _compute_coverage(coverage_data[param_name], 0.95, fn)

        metrics[param_name] = ParameterRecoveryMetrics(
            param_name=param_name,
            correlation=correlation,
            rmse=rmse,
            bias=bias,
            mean_absolute_error=mae,
            coverage_90=cov_90,
            coverage_95=cov_95,
            n_observations=n,
        )

    return RecoveryMetrics(per_parameter=metrics)


def _compute_coverage(
    data: list[tuple[float, np.ndarray]],
    prob: float,
    fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> float:
    """Compute coverage as fraction of true values inside HDI."""

    n_covered = 0
    for true_val, draws in data:
        if fn is not None:
            true_val = float(fn(np.asarray(true_val)))
            draws = fn(draws)
        lo, hi = _hdi(draws, prob)
        if lo <= true_val <= hi:
            n_covered += 1
    return n_covered / len(data) if data else float("nan")


def _hdi(draws: np.ndarray, prob: float) -> tuple[float, float]:
    """Compute highest density interval via shortest-interval method."""

    sorted_draws = np.sort(draws)
    n = len(sorted_draws)
    interval_size = max(1, math.ceil(prob * n))
    if interval_size >= n:
        return float(sorted_draws[0]), float(sorted_draws[-1])

    widths = sorted_draws[interval_size:] - sorted_draws[: n - interval_size]
    best_idx = int(np.argmin(widths))
    return float(sorted_draws[best_idx]), float(sorted_draws[best_idx + interval_size])
