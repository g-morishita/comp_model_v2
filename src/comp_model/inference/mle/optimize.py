"""Scipy-based maximum-likelihood optimization utilities.

MLE fitting is performed on the unconstrained parameter scale using SciPy's
local optimizers and replay-based likelihood evaluation.
"""

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
        Optional lower and upper bounds on unconstrained parameters.
        ``None`` runs unconstrained optimization with normal restart draws.
    restart_scale
        Standard deviation for normal restart draws when ``z_bounds`` is
        ``None``.
    max_iter
        Maximum number of optimizer iterations.

    Notes
    -----
    Restarts use one deterministic default start plus ``n_restarts - 1`` random
    draws. When ``z_bounds`` is set, draws are sampled uniformly inside the
    bounds; when ``None``, draws are sampled from ``Normal(0, restart_scale)``.
    """

    method: str = "L-BFGS-B"
    n_restarts: int = 10
    seed: int | None = 0
    tol: float | None = None
    z_bounds: tuple[float, float] | None = None
    restart_scale: float = 3.0
    max_iter: int = 500


@dataclass(frozen=True, slots=True)
class MleFitResult:
    """Result bundle for a single-subject MLE fit.

    Attributes
    ----------
    subject_id
        Subject identifier whose data were fit.
    model_id
        Kernel identifier reported by :class:`~comp_model.models.kernels.base.ModelKernelSpec`.
    log_likelihood
        Best replay log-likelihood found across restarts.
    n_params
        Number of free unconstrained parameters optimized.
    raw_params
        Best-fitting unconstrained parameters.
    constrained_params
        Best-fitting constrained parameters. For conditioned fits this reports
        the baseline condition's constrained parameters.
    aic
        Akaike information criterion computed from the winning fit.
    bic
        Bayesian information criterion computed from the winning fit.
    n_trials
        Number of observed trials contributing to the objective.
    converged
        Whether the winning SciPy run reported success.
    n_restarts
        Number of restart candidates evaluated.
    all_candidates
        Unconstrained parameter vectors returned by every restart.
    all_log_likelihoods
        Replay log-likelihood values corresponding to ``all_candidates``.
    params_by_condition
        Optional constrained parameters reconstructed per condition for
        conditioned fits.
    """

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

    Notes
    -----
    The optimizer minimizes the negative replay log-likelihood. The default
    start comes from each parameter's ``mle_init.default_unconstrained`` when
    available, followed by random restart points in ``config.z_bounds``. The
    winning unconstrained solution is transformed back to the constrained
    parameter space before reporting AIC and BIC.
    """

    from comp_model.data.validation import validate_subject

    validate_subject(subject_data, schema)

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
        if config.z_bounds is not None:
            starts.append(rng.uniform(config.z_bounds[0], config.z_bounds[1], size=n_params))
        else:
            starts.append(rng.normal(0.0, config.restart_scale, size=n_params))

    bounds: list[tuple[float, float]] | None = None
    if config.z_bounds is not None:
        bounds = [(config.z_bounds[0], config.z_bounds[1])] * n_params

    def objective(z_vector: np.ndarray) -> float:
        """Evaluate the negative replay log-likelihood for one parameter vector.

        Parameters
        ----------
        z_vector
            Candidate unconstrained parameter vector in kernel parameter order.

        Returns
        -------
        float
            Negative log-likelihood objective for ``scipy.optimize.minimize``.
            Non-finite replay scores are converted to a large penalty value.

        Notes
        -----
        Penalizing non-finite values keeps SciPy inside the search rather than
        aborting when a restart wanders into a numerically unstable region.
        """

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

    Notes
    -----
    This routine mirrors :func:`fit_mle_simple`, but the optimized vector lives
    in :class:`~comp_model.models.condition.shared_delta.SharedDeltaLayout`
    order. After optimization, the winning vector is reconstructed separately
    for each condition and transformed into constrained parameter values.
    """

    from comp_model.data.validation import validate_subject

    validate_subject(subject_data, schema)

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
        if config.z_bounds is not None:
            starts.append(rng.uniform(config.z_bounds[0], config.z_bounds[1], size=n_params))
        else:
            starts.append(rng.normal(0.0, config.restart_scale, size=n_params))

    bounds: list[tuple[float, float]] | None = None
    if config.z_bounds is not None:
        bounds = [(config.z_bounds[0], config.z_bounds[1])] * n_params

    def objective(z_vector: np.ndarray) -> float:
        """Evaluate the conditioned negative replay log-likelihood.

        Parameters
        ----------
        z_vector
            Candidate unconstrained parameter vector in layout key order.

        Returns
        -------
        float
            Negative conditioned log-likelihood for ``scipy.optimize.minimize``.
            Non-finite replay scores are converted to a large penalty value.

        Notes
        -----
        The conditioned objective differs from the simple objective only in how
        the raw vector is interpreted before replay.
        """

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
