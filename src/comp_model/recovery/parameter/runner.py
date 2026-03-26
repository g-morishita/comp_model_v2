"""Recovery study orchestration with parallel fitting."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from comp_model.data.schema import Block, Dataset, SubjectData
from comp_model.inference.bayes.result import BayesFitResult
from comp_model.inference.dispatch import fit
from comp_model.models.kernels.transforms import get_transform
from comp_model.recovery.parameter.config import sample_true_params
from comp_model.recovery.parameter.extraction import (
    extract_bayes_subject_records,
    extract_mle_subject_records,
    extract_population_records,
)
from comp_model.recovery.parameter.result import (
    ParameterRecoveryResult,
    PopulationLevelResult,
    ReplicationResult,
    SubjectLevelResult,
)
from comp_model.runtime import SimulationConfig, simulate_subject

if TYPE_CHECKING:
    from comp_model.inference.mle.optimize import MleFitResult
    from comp_model.recovery.parameter.config import ParameterRecoveryConfig
    from comp_model.tasks.schemas import TrialSchema
    from comp_model.tasks.spec import TaskSpec


def _check_schema_consistency(task: TaskSpec, schema: TrialSchema) -> None:
    """Verify that every block in *task* uses the expected trial schema.

    Parameters
    ----------
    task
        Task specification whose blocks are inspected.
    schema
        The trial schema that the recovery study expects.

    Raises
    ------
    ValueError
        If any block's schema id does not match *schema*.
    """
    for idx, block_spec in enumerate(task.blocks):
        if block_spec.schema.schema_id != schema.schema_id:
            raise ValueError(
                f"TaskSpec block {idx} (condition={block_spec.condition!r}) uses "
                f"schema {block_spec.schema.schema_id!r}, but the recovery study "
                f"expects {schema.schema_id!r}"
            )


def _build_true_population_values(
    config: ParameterRecoveryConfig,
    pop_params: dict[str, float],
) -> dict[str, float | list[float]]:
    """Build true population values on the same constrained scale used by Stan.

    For simple hierarchies, the true population value is the constrained
    transform of ``mu_{name}_z``. For condition-aware shared-delta hierarchies,
    the baseline uses ``mu_{name}_shared_z`` and each non-baseline condition
    uses ``mu_{name}_shared_z + mu_{name}_delta_z__{condition}``, transformed
    with the kernel's parameter transform.

    Parameters
    ----------
    config
        Recovery configuration containing the kernel parameter transforms and
        optional condition-aware layout.
    pop_params
        Sampled population-level latent parameters on the unconstrained scale,
        keyed by the Stan naming convention used in ``sample_true_params``.

    Returns
    -------
    dict[str, float | list[float]]
        True population values on the constrained scale keyed by the Stan
        population parameter names extracted from fits. Scalar values are used
        for simple hierarchies, and condition-aware hierarchies produce one
        list entry per condition.
    """

    true_pop: dict[str, float | list[float]] = {}
    for param_spec in config.kernel.spec().parameter_specs:
        transform = get_transform(param_spec.transform_id).forward
        if config.layout is None:
            mu_key = f"mu_{param_spec.name}_z"
            if mu_key in pop_params:
                true_pop[f"{param_spec.name}_pop"] = transform(pop_params[mu_key])
            continue

        shared_mu_key = f"mu_{param_spec.name}_shared_z"
        if shared_mu_key not in pop_params:
            continue

        condition_values: list[float] = []
        shared_mu = pop_params[shared_mu_key]
        for condition in config.layout.conditions:
            mu_z = shared_mu
            if condition != config.layout.baseline_condition:
                delta_key = f"mu_{param_spec.name}_delta_z__{condition}"
                if delta_key not in pop_params:
                    break
                mu_z += pop_params[delta_key]
            condition_values.append(transform(mu_z))

        if len(condition_values) == len(config.layout.conditions):
            true_pop[f"{param_spec.name}_pop"] = condition_values

    return true_pop


def run_parameter_recovery(config: ParameterRecoveryConfig) -> ParameterRecoveryResult:
    """Run the full parameter recovery pipeline.

    Parameters
    ----------
    config
        Recovery study configuration.

    Returns
    -------
    ParameterRecoveryResult
        Results from all replications.
    """

    _check_schema_consistency(config.task, config.schema)

    from comp_model.data.compatibility import check_kernel_schema_compatibility

    check_kernel_schema_compatibility(config.kernel, config.schema)

    if config.inference_config.backend == "stan":
        return _run_stan_recovery(config)
    return _run_mle_recovery(config)


def _run_mle_recovery(config: ParameterRecoveryConfig) -> ParameterRecoveryResult:
    """Run recovery with per-subject maximum-likelihood fits.

    Parameters
    ----------
    config
        Recovery study configuration for an MLE backend.

    Returns
    -------
    ParameterRecoveryResult
        Recovery outputs for every replication.
    """

    replications: list[ReplicationResult] = []
    total_fits = config.n_replications * config.n_subjects

    max_workers = config.max_workers
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, config.n_subjects)

    with tqdm(total=total_fits, desc="Recovery (MLE)", unit="subj") as pbar:
        for r in range(config.n_replications):
            rng = np.random.default_rng(config.simulation_base_seed + r)
            true_table, params_per_subject, _pop_params = sample_true_params(
                config.param_dists, config.kernel, config.n_subjects, rng, config.layout
            )
            dataset = _simulate_dataset(
                config, params_per_subject, seed=config.simulation_base_seed + r
            )

            if max_workers > 1:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(
                            _fit_subject_worker,
                            subject_data,
                            config.inference_config,
                            config.kernel,
                            config.schema,
                            config.layout,
                        ): i
                        for i, subject_data in enumerate(dataset.subjects)
                    }
                    results_by_idx: dict[int, MleFitResult] = {}
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        results_by_idx[idx] = future.result()
                        pbar.update(1)
                    results: list[MleFitResult] = [
                        results_by_idx[i] for i in range(len(dataset.subjects))
                    ]
            else:
                results = []
                for s in dataset.subjects:
                    results.append(
                        fit(config.inference_config, config.kernel, s, config.schema, config.layout)  # type: ignore[arg-type]
                    )
                    pbar.update(1)

            subject_records = extract_mle_subject_records(results, true_table, config.layout)
            replications.append(
                ReplicationResult(
                    replication_index=r,
                    subject_level=SubjectLevelResult(records=subject_records),
                    population_level=None,
                )
            )

    return ParameterRecoveryResult(config=config, replications=tuple(replications))


def _run_stan_recovery(config: ParameterRecoveryConfig) -> ParameterRecoveryResult:
    """Run recovery with hierarchical Stan fits.

    Parameters
    ----------
    config
        Recovery study configuration for a Stan backend.

    Returns
    -------
    ParameterRecoveryResult
        Recovery outputs for every replication.
    """

    simulated: list[tuple[int, dict[str, dict[str, float]], Dataset, dict[str, float]]] = []
    for r in range(config.n_replications):
        rng = np.random.default_rng(config.simulation_base_seed + r)
        true_table, params_per_subject, pop_params = sample_true_params(
            config.param_dists, config.kernel, config.n_subjects, rng, config.layout
        )
        dataset = _simulate_dataset(
            config, params_per_subject, seed=config.simulation_base_seed + r
        )
        simulated.append((r, true_table, dataset, pop_params))

    max_workers = config.max_workers
    if max_workers is None:
        n_chains = 4
        stan_config = config.inference_config.stan_config
        if stan_config is not None:
            n_chains = stan_config.n_chains
        cpu = os.cpu_count() or 1
        max_workers = max(1, min(cpu // n_chains, config.n_replications))

    param_names = tuple(p.name for p in config.kernel.spec().parameter_specs)
    subject_ids = [f"sub_{i:02d}" for i in range(config.n_subjects)]

    def _fit_one(
        item: tuple[int, dict[str, dict[str, float]], Dataset, dict[str, float]],
    ) -> ReplicationResult:
        """Fit one simulated replication with the Stan backend.

        Parameters
        ----------
        item
            Tuple containing the replication index, true parameter table,
            simulated dataset, and sampled population parameters.

        Returns
        -------
        ReplicationResult
            Extracted posterior summaries for the replication.

        Raises
        ------
        TypeError
            If dispatch returns a non-Bayesian fit result for a Stan backend.
        """

        r, true_table, dataset, pop_params = item
        result = fit(
            config.inference_config,
            config.kernel,
            dataset,
            config.schema,
            config.layout,
            config.adapter,
        )
        if not isinstance(result, BayesFitResult):
            raise TypeError("Stan recovery requires BayesFitResult from inference dispatch")

        subject_records = extract_bayes_subject_records(
            result,
            subject_ids,
            param_names,
            true_table,
            config.layout,  # type: ignore[arg-type]
        )
        true_pop = _build_true_population_values(config, pop_params)
        pop_records = extract_population_records(result, true_pop, config.layout)

        return ReplicationResult(
            replication_index=r,
            subject_level=SubjectLevelResult(records=subject_records),
            population_level=PopulationLevelResult(records=pop_records) if pop_records else None,
        )

    with tqdm(total=config.n_replications, desc="Recovery (Stan)", unit="rep") as pbar:
        if max_workers > 1 and config.n_replications > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(_fit_one, item): i for i, item in enumerate(simulated)
                }
                results_by_idx: dict[int, ReplicationResult] = {}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results_by_idx[idx] = future.result()
                    pbar.update(1)
                replications = [results_by_idx[i] for i in range(len(simulated))]
        else:
            replications = []
            for item in simulated:
                replications.append(_fit_one(item))
                pbar.update(1)

    return ParameterRecoveryResult(config=config, replications=tuple(replications))


def _simulate_dataset(
    config: ParameterRecoveryConfig,
    params_per_subject: dict[str, Any],
    seed: int,
) -> Dataset:
    """Simulate a dataset for one recovery replication.

    Parameters
    ----------
    config
        Recovery study configuration.
    params_per_subject
        Parsed parameters keyed by subject, optionally nested by condition.
    seed
        Random seed for the simulation.

    Returns
    -------
    Dataset
        Simulated dataset compatible with the configured task layout.
    """

    if config.layout is not None:
        return _simulate_condition_aware(config, params_per_subject, seed=seed)

    return _simulate_simple(config, params_per_subject, seed=seed)


def _simulate_simple(
    config: ParameterRecoveryConfig,
    params_per_subject: dict[str, Any],
    seed: int,
) -> Dataset:
    """Simulate a dataset without condition-specific parameter structure.

    Parameters
    ----------
    config
        Recovery study configuration.
    params_per_subject
        Parsed parameters keyed by subject identifier.
    seed
        Random seed for the simulation.

    Returns
    -------
    Dataset
        Simulated dataset for the requested task.
    """

    from comp_model.runtime.engine import simulate_dataset as sim_dataset

    return sim_dataset(
        task=config.task,
        env_factory=config.env_factory,
        kernel=config.kernel,
        params_per_subject=params_per_subject,
        config=SimulationConfig(seed=seed),
        demonstrator_kernel=config.demonstrator_kernel,
        demonstrator_params=config.demonstrator_params,
    )


def _simulate_condition_aware(
    config: ParameterRecoveryConfig,
    params_per_subject: dict[str, Any],
    seed: int,
) -> Dataset:
    """Simulate a dataset with per-condition parameter dictionaries.

    Parameters
    ----------
    config
        Recovery study configuration.
    params_per_subject
        Parsed parameters keyed by subject, then by condition.
    seed
        Random seed for the simulation.

    Returns
    -------
    Dataset
        Simulated dataset preserving the configured block conditions.
    """

    from comp_model.tasks.spec import TaskSpec

    subjects: list[SubjectData] = []
    for i, (sid, condition_params) in enumerate(params_per_subject.items()):
        blocks: list[Block] = []
        for block_idx, block_spec in enumerate(config.task.blocks):
            condition = block_spec.condition
            env = config.env_factory()
            demo_params = (
                config.condition_demonstrator_params[condition]
                if config.condition_demonstrator_params is not None
                else config.demonstrator_params
            )
            sub = simulate_subject(
                task=TaskSpec(task_id="tmp", blocks=(block_spec,)),
                env=env,
                kernel=config.kernel,
                params=condition_params[condition],
                config=SimulationConfig(seed=seed + i * 1000 + block_idx),
                subject_id=sid,
                demonstrator_kernel=config.demonstrator_kernel,
                demonstrator_params=demo_params,
            )
            blocks.append(
                Block(
                    block_index=block_idx,
                    condition=sub.blocks[0].condition,
                    schema_id=sub.blocks[0].schema_id,
                    trials=sub.blocks[0].trials,
                )
            )
        subjects.append(SubjectData(subject_id=sid, blocks=tuple(blocks)))

    return Dataset(subjects=tuple(subjects))


def _fit_subject_worker(
    subject_data: SubjectData,
    inference_config: Any,
    kernel: Any,
    schema: Any,
    layout: Any,
) -> MleFitResult:
    """Fit one subject in a worker process for MLE recovery.

    Parameters
    ----------
    subject_data
        Subject-level dataset to fit.
    inference_config
        Inference configuration passed through to dispatch.
    kernel
        Model kernel being fit.
    schema
        Trial schema used for replay extraction.
    layout
        Optional condition-aware layout.

    Returns
    -------
    MleFitResult
        Fitted single-subject MLE result.
    """

    return fit(inference_config, kernel, subject_data, schema, layout)  # type: ignore[return-value]
