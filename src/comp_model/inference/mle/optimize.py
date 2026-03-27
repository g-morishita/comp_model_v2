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
    from comp_model.models.kernels.base import ModelKernel, ParameterSpec
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
    restart_lower_bound
        Lower fallback bound used when restart generation needs a missing bound.
        For simple fits this is interpreted on the constrained parameter scale
        when a parameter does not provide a lower bound. For conditioned delta
        terms it is used directly on the unconstrained scale. When left as
        ``None``, it defaults to ``-restart_upper_bound``.
    restart_upper_bound
        Upper fallback bound used when restart generation needs a missing bound.
        For positive-only parameters this lets users cap the randomly sampled
        restart range without storing inference settings on ``ParameterSpec``.
    max_iter
        Maximum number of optimizer iterations.

    Notes
    -----
    Restarts use one deterministic default start plus ``n_restarts - 1`` random
    draws. Simple fits sample each shared parameter uniformly from its
    constrained bounds, filling any open side with the fallback bounds above.
    Conditioned delta terms use the fallback interval directly on the
    unconstrained scale.
    """

    method: str = "L-BFGS-B"
    n_restarts: int = 10
    seed: int | None = 0
    tol: float | None = None
    restart_lower_bound: float | None = None
    restart_upper_bound: float = 3.0
    max_iter: int = 500

    def __post_init__(self) -> None:
        """Validate fallback restart bounds."""

        resolved_lower = (
            -self.restart_upper_bound
            if self.restart_lower_bound is None
            else self.restart_lower_bound
        )
        if not math.isfinite(resolved_lower):
            raise ValueError("restart_lower_bound must be finite when provided")
        if not math.isfinite(self.restart_upper_bound):
            raise ValueError("restart_upper_bound must be finite")
        if resolved_lower >= self.restart_upper_bound:
            raise ValueError("restart_lower_bound must be smaller than restart_upper_bound")


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


def _fallback_restart_lower_bound(config: MleOptimizerConfig) -> float:
    """Resolve the effective fallback lower bound for restart generation."""

    return (
        -config.restart_upper_bound
        if config.restart_lower_bound is None
        else config.restart_lower_bound
    )


def _resolved_restart_interval(
    parameter: ParameterSpec,
    config: MleOptimizerConfig,
) -> tuple[float, float]:
    """Resolve a finite constrained restart interval for one parameter."""

    lower = parameter.bounds[0] if parameter.bounds is not None else None
    upper = parameter.bounds[1] if parameter.bounds is not None else None
    resolved_lower = _fallback_restart_lower_bound(config) if lower is None else lower
    resolved_upper = config.restart_upper_bound if upper is None else upper
    if resolved_lower >= resolved_upper:
        raise ValueError(
            f"Invalid restart interval for parameter {parameter.name!r}: "
            f"{resolved_lower} >= {resolved_upper}. "
            "Widen the optimizer fallback bounds or tighten the parameter bounds."
        )
    return (resolved_lower, resolved_upper)


def _default_unconstrained_start(
    parameter: ParameterSpec,
    config: MleOptimizerConfig,
) -> float:
    """Build the deterministic default start for one kernel parameter."""

    lower, upper = _resolved_restart_interval(parameter, config)
    midpoint = (lower + upper) / 2.0
    return float(get_transform(parameter.transform_id).inverse(midpoint))


def _random_unconstrained_start(
    parameter: ParameterSpec,
    config: MleOptimizerConfig,
    rng: np.random.Generator,
) -> float:
    """Sample one unconstrained restart from parameter bounds."""

    lower, upper = _resolved_restart_interval(parameter, config)
    constrained_draw = float(rng.uniform(lower, upper))
    return float(get_transform(parameter.transform_id).inverse(constrained_draw))


def _conditioned_default_start(
    layout: SharedDeltaLayout,
    config: MleOptimizerConfig,
) -> np.ndarray:
    """Build the deterministic default start for a conditioned layout."""

    parameter_by_name = {
        parameter.name: parameter for parameter in layout.kernel_spec.parameter_specs
    }
    values: list[float] = []
    for key in layout.parameter_keys():
        if key.endswith("__shared_z"):
            parameter_name = key[: -len("__shared_z")]
            values.append(_default_unconstrained_start(parameter_by_name[parameter_name], config))
        else:
            values.append(0.0)
    return np.asarray(values, dtype=float)


def _conditioned_random_start(
    layout: SharedDeltaLayout,
    config: MleOptimizerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one random restart for a conditioned layout."""

    parameter_by_name = {
        parameter.name: parameter for parameter in layout.kernel_spec.parameter_specs
    }
    fallback_lower = _fallback_restart_lower_bound(config)
    values: list[float] = []
    for key in layout.parameter_keys():
        if key.endswith("__shared_z"):
            parameter_name = key[: -len("__shared_z")]
            values.append(
                _random_unconstrained_start(parameter_by_name[parameter_name], config, rng)
            )
        else:
            values.append(float(rng.uniform(fallback_lower, config.restart_upper_bound)))
    return np.asarray(values, dtype=float)


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
    start is the midpoint of each parameter's constrained bounds (with any
    open side filled from the optimizer config), transformed onto the
    unconstrained scale. Random restarts are then sampled uniformly from the
    same constrained intervals. The winning unconstrained solution is
    transformed back to the constrained parameter space before reporting AIC
    and BIC.
    """

    from comp_model.data.compatibility import check_kernel_schema_compatibility
    from comp_model.data.validation import validate_subject

    check_kernel_schema_compatibility(kernel, schema)
    validate_subject(subject_data, schema)

    scipy_optimize = cast("Any", importlib.import_module("scipy.optimize"))
    scipy_minimize = scipy_optimize.minimize

    spec = kernel.spec()
    param_names = tuple(parameter.name for parameter in spec.parameter_specs)
    n_params = len(param_names)
    n_trials = sum(len(block.trials) for block in subject_data.blocks)

    default_start = np.asarray(
        [_default_unconstrained_start(parameter, config) for parameter in spec.parameter_specs],
        dtype=float,
    )

    rng = np.random.default_rng(config.seed)
    starts = [default_start.copy()]
    for _ in range(config.n_restarts - 1):
        starts.append(
            np.asarray(
                [
                    _random_unconstrained_start(parameter, config, rng)
                    for parameter in spec.parameter_specs
                ],
                dtype=float,
            )
        )

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

    from comp_model.data.compatibility import check_kernel_schema_compatibility
    from comp_model.data.validation import validate_subject

    check_kernel_schema_compatibility(kernel, schema)
    validate_subject(subject_data, schema)

    scipy_optimize = cast("Any", importlib.import_module("scipy.optimize"))
    scipy_minimize = scipy_optimize.minimize

    spec = kernel.spec()
    param_keys = layout.parameter_keys()
    n_params = len(param_keys)
    n_trials = sum(len(block.trials) for block in subject_data.blocks)

    default_start = _conditioned_default_start(layout, config)
    rng = np.random.default_rng(config.seed)
    starts = [default_start.copy()]
    for _ in range(config.n_restarts - 1):
        starts.append(_conditioned_random_start(layout, config, rng))

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
