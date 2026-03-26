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
    true_table: dict[str, dict[str, float]],
    param_names: tuple[str, ...],
    pop_params: dict[str, float],
) -> dict[str, float | list[float]]:
    """Build true population values for recovery summaries.

    Combines constrained-scale population means (computed from subject
    values) with unconstrained-scale mu/sd (from hierarchical sampling).

    Parameters
    ----------
    config
        Recovery study configuration.
    true_table
        Ground-truth constrained subject parameters keyed by subject id.
    param_names
        Subject-level parameter names from the fitted kernel.
    pop_params
        Sampled population parameters from hierarchical sampling, keyed by
        Stan naming convention (e.g. ``mu_alpha_z``, ``sd_alpha_z``).

    Returns
    -------
    dict[str, float | list[float]]
        True population values keyed to Stan output names.  Scalar for
        simple population means and shared z-params.  Lists for
        condition-indexed delta z-params (one entry per non-baseline
        condition).
    """
    true_pop: dict[str, float | list[float]] = {}

    if config.layout is None:
        # Constrained-scale population means
        for name in param_names:
            vals = [true_table[sid][name] for sid in true_table if name in true_table[sid]]
            if vals:
                true_pop[f"{name}_pop"] = float(np.mean(vals))
        # Unconstrained-scale mu/sd from hierarchical sampling
        true_pop.update(pop_params)
        return true_pop

    # Condition-aware hierarchy
    baseline_condition = config.layout.baseline_condition
    nonbaseline = tuple(c for c in config.layout.conditions if c != baseline_condition)

    # Constrained-scale shared population mean (baseline condition mean)
    for name in param_names:
        baseline_key = f"{name}__{baseline_condition}"
        vals = [
            true_table[sid][baseline_key] for sid in true_table if baseline_key in true_table[sid]
        ]
        if vals:
            true_pop[f"{name}_shared_pop"] = float(np.mean(vals))

    # Unconstrained-scale shared mu/sd (scalar)
    for name in param_names:
        for prefix in ("mu_", "sd_"):
            key = f"{prefix}{name}_shared_z"
            if key in pop_params:
                true_pop[key] = pop_params[key]

    # Unconstrained-scale delta mu/sd (list per non-baseline condition)
    for name in param_names:
        for prefix in ("mu_", "sd_"):
            vals_list: list[float] = []
            for condition in nonbaseline:
                pop_key = f"{prefix}{name}_delta_z__{condition}"
                if pop_key in pop_params:
                    vals_list.append(pop_params[pop_key])
            if vals_list:
                true_pop[f"{prefix}{name}_delta_z"] = vals_list

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

        true_pop = _build_true_population_values(config, true_table, param_names, pop_params)
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
