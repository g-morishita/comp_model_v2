"""Configuration and parameter sampling for recovery analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np  # noqa: TC002 - used at runtime in sample functions

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
class ParamDist:
    """Distribution for one parameter on the unconstrained scale.

    Attributes
    ----------
    name
        Parameter name matching ``ParameterSpec.name``.
    mu_unconstrained
        Population mean on unconstrained scale.
    sd_unconstrained
        Population standard deviation on unconstrained scale.
    """

    name: str
    mu_unconstrained: float
    sd_unconstrained: float


@dataclass(frozen=True, slots=True)
class RecoveryStudyConfig:
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


def sample_true_params(
    param_dists: tuple[ParamDist, ...],
    kernel: ModelKernel[Any, Any],
    n_subjects: int,
    rng: np.random.Generator,
    layout: SharedDeltaLayout | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Sample ground-truth parameters from population distributions.

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
    """

    spec = kernel.spec()
    dist_by_name = {d.name: d for d in param_dists}

    if layout is not None:
        return _sample_condition_aware(dist_by_name, spec, kernel, n_subjects, rng, layout)
    return _sample_simple(dist_by_name, spec, kernel, n_subjects, rng)


def _sample_simple(
    dist_by_name: dict[str, ParamDist],
    spec: Any,
    kernel: ModelKernel[Any, Any],
    n_subjects: int,
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    true_table: dict[str, dict[str, float]] = {}
    params_per_subject: dict[str, Any] = {}

    for i in range(n_subjects):
        sid = f"sub_{i:02d}"
        raw: dict[str, float] = {}
        constrained: dict[str, float] = {}
        for param_spec in spec.parameter_specs:
            dist = dist_by_name[param_spec.name]
            z = float(rng.normal(dist.mu_unconstrained, dist.sd_unconstrained))
            raw[param_spec.name] = z
            constrained[param_spec.name] = get_transform(param_spec.transform_id).forward(z)
        true_table[sid] = constrained
        params_per_subject[sid] = kernel.parse_params(raw)

    return true_table, params_per_subject


def _sample_condition_aware(
    dist_by_name: dict[str, ParamDist],
    spec: Any,
    kernel: ModelKernel[Any, Any],
    n_subjects: int,
    rng: np.random.Generator,
    layout: SharedDeltaLayout,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    true_table: dict[str, dict[str, float]] = {}
    params_per_subject: dict[str, Any] = {}

    for i in range(n_subjects):
        sid = f"sub_{i:02d}"
        shared_z: dict[str, float] = {}
        for param_spec in spec.parameter_specs:
            dist = dist_by_name[param_spec.name]
            shared_z[param_spec.name] = float(
                rng.normal(dist.mu_unconstrained, dist.sd_unconstrained)
            )

        delta_z: dict[str, dict[str, float]] = {}
        for condition in layout.conditions:
            if condition == layout.baseline_condition:
                continue
            delta_z[condition] = {}
            for param_spec in spec.parameter_specs:
                delta_name = f"{param_spec.name}__delta"
                dist = dist_by_name[delta_name]
                delta_z[condition][param_spec.name] = float(
                    rng.normal(dist.mu_unconstrained, dist.sd_unconstrained)
                )

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

    return true_table, params_per_subject
