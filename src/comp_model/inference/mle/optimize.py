"""Scipy-based maximum-likelihood optimization utilities."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from comp_model.inference.mle.objective import log_likelihood_conditioned, log_likelihood_simple
from comp_model.models.kernels.transforms import get_transform

if TYPE_CHECKING:
    from comp_model.data.schema import SubjectData
    from comp_model.models.condition.shared_delta import SharedDeltaLayout
    from comp_model.models.kernels.base import ModelKernel
    from comp_model.tasks.schemas import TrialSchema


@dataclass(frozen=True, slots=True)
class MleOptimizerConfig:
    """Configuration for multi-start MLE optimization.

    Attributes
    ----------
    method
        Optimization algorithm passed to ``scipy.optimize.minimize``.
    n_restarts
        Number of initializations to evaluate.
    seed
        Optional RNG seed used for restart generation.
    tol
        Optional optimizer tolerance.
    z_bounds
        Shared lower and upper bounds on unconstrained parameters.
    max_iter
        Maximum number of optimizer iterations.
    """

    method: str = "L-BFGS-B"
    n_restarts: int = 10
    seed: int | None = 0
    tol: float | None = None
    z_bounds: tuple[float, float] = (-6.0, 6.0)
    max_iter: int = 500


@dataclass(frozen=True, slots=True)
class MleFitResult:
    """Result bundle for a single-subject MLE fit."""

    subject_id: str
    model_id: str
    log_likelihood: float
    n_params: int
    raw_params: dict[str, float]
    constrained_params: dict[str, float]
    aic: float
    bic: float
    n_trials: int
    converged: bool
    n_restarts: int
    all_candidates: tuple[dict[str, float], ...]
    all_log_likelihoods: tuple[float, ...]
    params_by_condition: dict[str, dict[str, float]] | None = None


DEFAULT_MLE_OPTIMIZER_CONFIG = MleOptimizerConfig()


def fit_mle_simple(
    kernel: ModelKernel[object, object],
    subject_data: SubjectData,
    schema: TrialSchema,
    config: MleOptimizerConfig = DEFAULT_MLE_OPTIMIZER_CONFIG,
) -> MleFitResult:
    """Fit a single subject with multi-start MLE on unconstrained parameters.

    Parameters
    ----------
    kernel
        Model kernel to optimize.
    subject_data
        Subject data being fit.
    schema
        Trial schema used for replay extraction.
    config
        Optimizer configuration.

    Returns
    -------
    MleFitResult
        Best fit found across all restart candidates.
    """

    scipy_optimize = cast("Any", importlib.import_module("scipy.optimize"))
    scipy_minimize = scipy_optimize.minimize

    spec = kernel.spec()
    param_names = tuple(parameter.name for parameter in spec.parameter_specs)
    n_params = len(param_names)
    n_trials = sum(len(block.trials) for block in subject_data.blocks)

    default_start = np.array(
        [
            parameter.mle_init.default_unconstrained if parameter.mle_init is not None else 0.0
            for parameter in spec.parameter_specs
        ]
    )

    rng = np.random.default_rng(config.seed)
    starts = [default_start.copy()]
    for _ in range(config.n_restarts - 1):
        starts.append(rng.uniform(config.z_bounds[0], config.z_bounds[1], size=n_params))

    bounds = [(config.z_bounds[0], config.z_bounds[1])] * n_params

    def objective(z_vector: np.ndarray) -> float:
        raw_params = {name: float(value) for name, value in zip(param_names, z_vector, strict=True)}
        log_likelihood = log_likelihood_simple(kernel, subject_data, raw_params, schema)
        if not np.isfinite(log_likelihood):
            return 1e15
        return -log_likelihood

    best_log_likelihood = float("-inf")
    best_raw_params: dict[str, float] = {}
    best_converged = False
    all_candidates: list[dict[str, float]] = []
    all_log_likelihoods: list[float] = []

    for start in starts:
        result: Any = scipy_minimize(
            objective,
            start,
            method=config.method,
            bounds=bounds,
            tol=config.tol,
            options={"maxiter": config.max_iter},
        )
        raw_params = {
            name: float(value)
            for name, value in zip(
                param_names,
                np.asarray(result.x, dtype=float),
                strict=True,
            )
        }
        log_likelihood = -float(result.fun) if np.isfinite(result.fun) else float("-inf")
        all_candidates.append(raw_params)
        all_log_likelihoods.append(log_likelihood)

        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_raw_params = raw_params
            best_converged = bool(result.success)

    constrained_params = {
        parameter.name: get_transform(parameter.transform_id).forward(
            best_raw_params[parameter.name]
        )
        for parameter in spec.parameter_specs
    }
    aic = -2.0 * best_log_likelihood + 2.0 * n_params
    bic = -2.0 * best_log_likelihood + n_params * math.log(max(n_trials, 1))

    return MleFitResult(
        subject_id=subject_data.subject_id,
        model_id=spec.model_id,
        log_likelihood=best_log_likelihood,
        n_params=n_params,
        raw_params=best_raw_params,
        constrained_params=constrained_params,
        aic=aic,
        bic=bic,
        n_trials=n_trials,
        converged=best_converged,
        n_restarts=config.n_restarts,
        all_candidates=tuple(all_candidates),
        all_log_likelihoods=tuple(all_log_likelihoods),
    )


def fit_mle_conditioned(
    kernel: ModelKernel[object, object],
    layout: SharedDeltaLayout,
    subject_data: SubjectData,
    schema: TrialSchema,
    config: MleOptimizerConfig = DEFAULT_MLE_OPTIMIZER_CONFIG,
) -> MleFitResult:
    """Fit a single subject with shared-plus-delta condition parameters.

    Parameters
    ----------
    kernel
        Model kernel to optimize.
    layout
        Shared-plus-delta parameter layout across conditions.
    subject_data
        Subject data being fit.
    schema
        Trial schema used for replay extraction.
    config
        Optimizer configuration.

    Returns
    -------
    MleFitResult
        Best fit found across all restart candidates.
    """

    scipy_optimize = cast("Any", importlib.import_module("scipy.optimize"))
    scipy_minimize = scipy_optimize.minimize

    spec = kernel.spec()
    param_keys = layout.parameter_keys()
    n_params = len(param_keys)
    n_trials = sum(len(block.trials) for block in subject_data.blocks)

    default_start = np.zeros(n_params)
    rng = np.random.default_rng(config.seed)
    starts = [default_start.copy()]
    for _ in range(config.n_restarts - 1):
        starts.append(rng.uniform(config.z_bounds[0], config.z_bounds[1], size=n_params))

    bounds = [(config.z_bounds[0], config.z_bounds[1])] * n_params

    def objective(z_vector: np.ndarray) -> float:
        raw_params = {name: float(value) for name, value in zip(param_keys, z_vector, strict=True)}
        log_likelihood = log_likelihood_conditioned(
            kernel,
            layout,
            subject_data,
            raw_params,
            schema,
        )
        if not np.isfinite(log_likelihood):
            return 1e15
        return -log_likelihood

    best_log_likelihood = float("-inf")
    best_raw_params: dict[str, float] = {}
    best_converged = False
    all_candidates: list[dict[str, float]] = []
    all_log_likelihoods: list[float] = []

    for start in starts:
        result: Any = scipy_minimize(
            objective,
            start,
            method=config.method,
            bounds=bounds,
            tol=config.tol,
            options={"maxiter": config.max_iter},
        )
        raw_params = {
            name: float(value)
            for name, value in zip(
                param_keys,
                np.asarray(result.x, dtype=float),
                strict=True,
            )
        }
        log_likelihood = -float(result.fun) if np.isfinite(result.fun) else float("-inf")
        all_candidates.append(raw_params)
        all_log_likelihoods.append(log_likelihood)

        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_raw_params = raw_params
            best_converged = bool(result.success)

    params_by_condition: dict[str, dict[str, float]] = {}
    for condition in layout.conditions:
        unconstrained = layout.reconstruct(best_raw_params, condition)
        params_by_condition[condition] = {
            parameter.name: get_transform(parameter.transform_id).forward(
                unconstrained[parameter.name]
            )
            for parameter in spec.parameter_specs
        }

    baseline_constrained = params_by_condition[layout.baseline_condition]
    aic = -2.0 * best_log_likelihood + 2.0 * n_params
    bic = -2.0 * best_log_likelihood + n_params * math.log(max(n_trials, 1))

    return MleFitResult(
        subject_id=subject_data.subject_id,
        model_id=spec.model_id,
        log_likelihood=best_log_likelihood,
        n_params=n_params,
        raw_params=best_raw_params,
        constrained_params=baseline_constrained,
        aic=aic,
        bic=bic,
        n_trials=n_trials,
        converged=best_converged,
        n_restarts=config.n_restarts,
        all_candidates=tuple(all_candidates),
        all_log_likelihoods=tuple(all_log_likelihoods),
        params_by_condition=params_by_condition,
    )
