"""Configuration and parameter sampling for recovery analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np  # noqa: TC002 - used at runtime in sample functions
from scipy.stats._distn_infrastructure import (  # noqa: TC002 - used at runtime in dataclass
    rv_continuous_frozen,
    rv_discrete_frozen,
)

from comp_model.models.kernels.transforms import get_transform

if TYPE_CHECKING:
    from collections.abc import Callable

    from comp_model.environments.base import Environment
    from comp_model.inference.config import InferenceConfig
    from comp_model.models.condition.shared_delta import SharedDeltaLayout
    from comp_model.models.kernels.base import ModelKernel
    from comp_model.tasks.schemas import TrialSchema
    from comp_model.tasks.spec import TaskSpec


@dataclass(frozen=True, slots=True)
class FlatParamDist:
    """Flat sampling from a specified distribution.

    Draws one sample per subject directly from ``dist``.  For
    constrained-scale distributions the sample is mapped back to the
    unconstrained scale via the parameter's inverse transform.  For
    unconstrained-scale distributions the sample is returned as-is.

    Use for per-subject recovery (MLE or per-subject Stan) where no
    population-level parameters are estimated.

    Attributes
    ----------
    name
        Parameter name matching ``ParameterSpec.name``.
    dist
        A scipy frozen distribution, e.g. ``stats.uniform(0, 1)`` or
        ``stats.norm(-0.847, 0.5)``.
    scale
        Scale on which ``dist`` is defined.  ``"constrained"`` means the
        sampled value is mapped back to the unconstrained scale via the
        parameter's inverse transform.  ``"unconstrained"`` means the
        sampled value is already on the unconstrained scale.

    Examples
    --------
    >>> from scipy import stats
    >>> FlatParamDist("alpha", stats.uniform(0, 1))
    >>> FlatParamDist("beta", stats.norm(0, 0.5), scale="unconstrained")
    """

    name: str
    dist: rv_continuous_frozen | rv_discrete_frozen
    scale: Literal["constrained", "unconstrained"] = "constrained"

    def sample_unconstrained(self, rng: np.random.Generator, transform_id: str) -> float:
        """Draw one sample and return it on the unconstrained scale.

        Parameters
        ----------
        rng
            Random number generator.
        transform_id
            Transform identifier from the kernel's ``ParameterSpec``.
            Used to convert constrained samples back to the unconstrained
            scale.

        Returns
        -------
        float
            Sampled value on the unconstrained scale.
        """
        value = float(self.dist.rvs(random_state=rng))
        if self.scale == "constrained":
            return get_transform(transform_id).inverse(value)
        return value


@dataclass(frozen=True, slots=True)
class HierarchicalParamDist:
    """Hierarchical sampling on the unconstrained scale.

    Per replication the population parameters are drawn first::

        mu  ~ mu_prior
        sd  ~ sd_prior

    Then each subject's unconstrained value is drawn from::

        z_i ~ Normal(mu, sd)

    Use for hierarchical Stan recovery where population ``mu`` and ``sd``
    are estimated and you want their recovery to be assessed.

    Attributes
    ----------
    name
        Parameter name matching ``ParameterSpec.name``.
    mu_prior
        Prior distribution for the population mean on the unconstrained
        scale, e.g. ``stats.norm(0, 1)``.
    sd_prior
        Prior distribution for the population standard deviation on the
        unconstrained scale, e.g. ``stats.halfnorm(0, 1)``.

    Examples
    --------
    >>> from scipy import stats
    >>> HierarchicalParamDist("alpha",
    ...     mu_prior=stats.norm(0, 1),
    ...     sd_prior=stats.halfnorm(0, 1),
    ... )
    """

    name: str
    mu_prior: rv_continuous_frozen
    sd_prior: rv_continuous_frozen

    def sample_population(self, rng: np.random.Generator) -> tuple[float, float]:
        """Draw population mu and sd from the hyper-priors.

        Parameters
        ----------
        rng
            Random number generator.

        Returns
        -------
        tuple[float, float]
            ``(mu, sd)`` sampled from ``mu_prior`` and ``sd_prior``.
        """
        mu = float(self.mu_prior.rvs(random_state=rng))
        sd = float(self.sd_prior.rvs(random_state=rng))
        return mu, sd


ParamDist = FlatParamDist | HierarchicalParamDist
"""Type alias accepted by recovery configurations.

Either a :class:`FlatParamDist` for flat (non-hierarchical) sampling or a
:class:`HierarchicalParamDist` for hierarchical sampling with
population-level recovery.
"""


@dataclass(frozen=True, slots=True)
class ParameterRecoveryConfig:
    """Full specification for a parameter recovery study.

    Attributes
    ----------
    n_replications
        Number of simulate-fit cycles.
    n_subjects
        Number of subjects per replication.
    param_dists
        Population distributions for each kernel parameter.
    task
        Task specification used for simulation.
    env_factory
        Factory returning a fresh environment per subject.
    kernel
        Model kernel to simulate and fit.
    schema
        Trial schema used for replay extraction.
    inference_config
        Inference configuration describing backend and hierarchy.
    layout
        Optional condition-aware parameter layout.
    adapter
        Optional Stan adapter for Bayesian inference.
    simulation_base_seed
        Base seed for reproducible simulation.
    max_workers
        Maximum parallel workers. ``None`` selects automatically.
    demonstrator_kernel
        Optional demonstrator kernel for social tasks.
    demonstrator_params
        Default demonstrator parameters for all conditions.
    condition_demonstrator_params
        Per-condition demonstrator parameters mapping condition name to params.
        When set, overrides ``demonstrator_params`` for each condition.
        Requires ``demonstrator_kernel`` to be set.
    """

    n_replications: int
    n_subjects: int
    param_dists: tuple[ParamDist, ...]
    task: TaskSpec
    env_factory: Callable[[], Environment]
    kernel: ModelKernel[Any, Any]
    schema: TrialSchema
    inference_config: InferenceConfig
    layout: SharedDeltaLayout | None = None
    adapter: object | None = None
    simulation_base_seed: int = 42
    max_workers: int | None = None
    demonstrator_kernel: ModelKernel[Any, Any] | None = None
    demonstrator_params: Any | None = None
    condition_demonstrator_params: dict[str, Any] | None = None


def sample_true_params(
    param_dists: tuple[ParamDist, ...],
    kernel: ModelKernel[Any, Any],
    n_subjects: int,
    rng: np.random.Generator,
    layout: SharedDeltaLayout | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any], dict[str, float]]:
    """Sample ground-truth parameters from population distributions.

    For :class:`HierarchicalParamDist` entries, population ``mu`` and ``sd``
    are drawn once per call (i.e. per replication) and subjects are then
    sampled from ``Normal(mu, sd)``.  For :class:`FlatParamDist` entries,
    each subject draws independently from the specified distribution.

    Parameters
    ----------
    param_dists
        Population distributions for each parameter.
    kernel
        Model kernel whose transforms and ``parse_params`` are used.
    n_subjects
        Number of subjects to sample.
    rng
        Random number generator.
    layout
        Optional condition-aware layout. When provided, delta distributions
        must also be supplied via ``param_dists`` using names like
        ``alpha__delta``.

    Returns
    -------
    true_params_table
        Ground-truth constrained parameters:
        ``{subject_id: {param_name: value}}``. For condition-aware layouts,
        keys are ``param__condition`` (e.g., ``alpha__easy``).
    params_per_subject
        Parsed kernel parameters for ``simulate_dataset``:
        ``{subject_id: ParamsT}`` for simple layouts, or
        ``{subject_id: {condition: ParamsT}}`` for condition-aware layouts.
    pop_params
        Sampled population parameters on the unconstrained scale.  Keys
        follow Stan naming: ``mu_{name}_z`` / ``sd_{name}_z`` for simple
        layouts, ``mu_{name}_shared_z`` / ``sd_{name}_shared_z`` and
        ``mu_{name}_delta_z__{condition}`` /
        ``sd_{name}_delta_z__{condition}`` for condition-aware layouts.
        Empty when no :class:`HierarchicalParamDist` entries are present.
    """

    spec = kernel.spec()
    dist_by_name: dict[str, ParamDist] = {d.name: d for d in param_dists}

    if layout is not None:
        return _sample_condition_aware(dist_by_name, spec, kernel, n_subjects, rng, layout)
    return _sample_simple(dist_by_name, spec, kernel, n_subjects, rng)


def _sample_z(
    pdist: ParamDist,
    rng: np.random.Generator,
    transform_id: str,
    mu_sd: dict[str, tuple[float, float]],
) -> float:
    """Draw one unconstrained-scale sample from a parameter distribution.

    Parameters
    ----------
    pdist
        Parameter distribution (flat or hierarchical).
    rng
        Random number generator.
    transform_id
        Transform identifier for the parameter.
    mu_sd
        Pre-sampled ``(mu, sd)`` pairs keyed by parameter name, used for
        :class:`HierarchicalParamDist` entries.

    Returns
    -------
    float
        Sampled value on the unconstrained scale.
    """
    if isinstance(pdist, HierarchicalParamDist):
        mu, sd = mu_sd[pdist.name]
        return float(rng.normal(mu, sd))
    return pdist.sample_unconstrained(rng, transform_id)


def _sample_population_params(
    dist_by_name: dict[str, ParamDist],
    param_names: tuple[str, ...],
    rng: np.random.Generator,
    suffix: str = "",
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """Sample population mu/sd for all hierarchical distributions.

    Parameters
    ----------
    dist_by_name
        Parameter distributions keyed by name.
    param_names
        Names of the parameters to iterate over.
    rng
        Random number generator.
    suffix
        Suffix appended to the Stan naming convention keys.  For example,
        ``"_shared"`` produces ``mu_{name}_shared_z``.

    Returns
    -------
    mu_sd
        ``(mu, sd)`` pairs keyed by distribution name.
    pop_params
        Sampled values keyed by Stan convention names.
    """
    mu_sd: dict[str, tuple[float, float]] = {}
    pop_params: dict[str, float] = {}

    for name in param_names:
        pdist = dist_by_name.get(name)
        if pdist is not None and isinstance(pdist, HierarchicalParamDist):
            mu, sd = pdist.sample_population(rng)
            mu_sd[pdist.name] = (mu, sd)
            pop_params[f"mu_{name}{suffix}_z"] = mu
            pop_params[f"sd_{name}{suffix}_z"] = sd

    return mu_sd, pop_params


def _sample_simple(
    dist_by_name: dict[str, ParamDist],
    spec: Any,
    kernel: ModelKernel[Any, Any],
    n_subjects: int,
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, float]], dict[str, Any], dict[str, float]]:
    """Sample subjects without condition structure.

    Parameters
    ----------
    dist_by_name
        Parameter distributions keyed by name.
    spec
        Kernel specification providing ``parameter_specs``.
    kernel
        Model kernel for ``parse_params``.
    n_subjects
        Number of subjects to sample.
    rng
        Random number generator.

    Returns
    -------
    tuple
        ``(true_table, params_per_subject, pop_params)``
    """
    param_names = tuple(ps.name for ps in spec.parameter_specs)
    mu_sd, pop_params = _sample_population_params(dist_by_name, param_names, rng)

    true_table: dict[str, dict[str, float]] = {}
    params_per_subject: dict[str, Any] = {}

    for i in range(n_subjects):
        sid = f"sub_{i:02d}"
        raw: dict[str, float] = {}
        constrained: dict[str, float] = {}
        for param_spec in spec.parameter_specs:
            pdist = dist_by_name[param_spec.name]
            z = _sample_z(pdist, rng, param_spec.transform_id, mu_sd)
            raw[param_spec.name] = z
            constrained[param_spec.name] = get_transform(param_spec.transform_id).forward(z)
        true_table[sid] = constrained
        params_per_subject[sid] = kernel.parse_params(raw)

    return true_table, params_per_subject, pop_params


def _sample_condition_aware(
    dist_by_name: dict[str, ParamDist],
    spec: Any,
    kernel: ModelKernel[Any, Any],
    n_subjects: int,
    rng: np.random.Generator,
    layout: SharedDeltaLayout,
) -> tuple[dict[str, dict[str, float]], dict[str, Any], dict[str, float]]:
    """Sample subjects with shared/delta condition structure.

    Parameters
    ----------
    dist_by_name
        Parameter distributions keyed by name (includes ``{name}__delta``
        entries for delta distributions).
    spec
        Kernel specification providing ``parameter_specs``.
    kernel
        Model kernel for ``parse_params``.
    n_subjects
        Number of subjects to sample.
    rng
        Random number generator.
    layout
        Condition-aware parameter layout.

    Returns
    -------
    tuple
        ``(true_table, params_per_subject, pop_params)``
    """
    param_names = tuple(ps.name for ps in spec.parameter_specs)

    # Sample shared population params
    shared_mu_sd, pop_params = _sample_population_params(
        dist_by_name, param_names, rng, suffix="_shared"
    )

    # Sample delta population params — one (mu, sd) per (param, condition)
    nonbaseline = tuple(c for c in layout.conditions if c != layout.baseline_condition)
    delta_mu_sd: dict[str, tuple[float, float]] = {}  # keyed by "{name}__delta__{condition}"
    for condition in nonbaseline:
        for name in param_names:
            delta_name = f"{name}__delta"
            pdist = dist_by_name.get(delta_name)
            if pdist is not None and isinstance(pdist, HierarchicalParamDist):
                mu, sd = pdist.sample_population(rng)
                key = f"{delta_name}__{condition}"
                delta_mu_sd[key] = (mu, sd)
                pop_params[f"mu_{name}_delta_z__{condition}"] = mu
                pop_params[f"sd_{name}_delta_z__{condition}"] = sd

    # Sample subjects
    true_table: dict[str, dict[str, float]] = {}
    params_per_subject: dict[str, Any] = {}

    for i in range(n_subjects):
        sid = f"sub_{i:02d}"

        # Shared z-values
        shared_z: dict[str, float] = {}
        for param_spec in spec.parameter_specs:
            pdist = dist_by_name[param_spec.name]
            shared_z[param_spec.name] = _sample_z(pdist, rng, param_spec.transform_id, shared_mu_sd)

        # Delta z-values per condition
        delta_z: dict[str, dict[str, float]] = {}
        for condition in nonbaseline:
            delta_z[condition] = {}
            for param_spec in spec.parameter_specs:
                delta_name = f"{param_spec.name}__delta"
                pdist = dist_by_name[delta_name]
                # For hierarchical deltas, use the condition-specific mu_sd
                cond_key = f"{delta_name}__{condition}"
                if isinstance(pdist, HierarchicalParamDist) and cond_key in delta_mu_sd:
                    mu, sd = delta_mu_sd[cond_key]
                    delta_z[condition][param_spec.name] = float(rng.normal(mu, sd))
                else:
                    delta_z[condition][param_spec.name] = _sample_z(
                        pdist, rng, param_spec.transform_id, {}
                    )

        # Combine shared + delta → constrained
        subject_constrained: dict[str, float] = {}
        condition_params: dict[str, Any] = {}
        for condition in layout.conditions:
            raw: dict[str, float] = {}
            for param_spec in spec.parameter_specs:
                z = shared_z[param_spec.name]
                if condition != layout.baseline_condition:
                    z += delta_z[condition][param_spec.name]
                raw[param_spec.name] = z
                c_val = get_transform(param_spec.transform_id).forward(z)
                subject_constrained[f"{param_spec.name}__{condition}"] = c_val
            condition_params[condition] = kernel.parse_params(raw)

        true_table[sid] = subject_constrained
        params_per_subject[sid] = condition_params

    return true_table, params_per_subject, pop_params
