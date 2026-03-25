"""Recovery study orchestration with parallel fitting."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from comp_model.data.schema import Block, Dataset, SubjectData
from comp_model.inference.bayes.result import BayesFitResult
from comp_model.inference.dispatch import fit
from comp_model.recovery.parameter.config import get_true_population_params, sample_true_params
from comp_model.recovery.parameter.extraction import (
    ReplicationEstimates,
    extract_bayes_estimates,
    extract_mle_estimates,
    extract_population_estimates,
)
from comp_model.runtime import SimulationConfig, simulate_subject

if TYPE_CHECKING:
    from comp_model.inference.mle.optimize import MleFitResult
    from comp_model.recovery.parameter.config import RecoveryStudyConfig


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """Complete results from a recovery study.

    Attributes
    ----------
    config
        Study configuration used.
    replications
        Estimates from each replication.
    """

    config: RecoveryStudyConfig
    replications: tuple[ReplicationEstimates, ...]


def run_recovery(config: RecoveryStudyConfig) -> RecoveryResult:
    """Run the full parameter recovery pipeline.

    Parameters
    ----------
    config
        Recovery study configuration.

    Returns
    -------
    RecoveryResult
        Results from all replications.
    """

    if config.inference_config.backend == "stan":
        return _run_stan_recovery(config)
    return _run_mle_recovery(config)


def _run_mle_recovery(config: RecoveryStudyConfig) -> RecoveryResult:
    """Run recovery with per-subject maximum-likelihood fits.

    Parameters
    ----------
    config
        Recovery study configuration for an MLE backend.

    Returns
    -------
    RecoveryResult
        Recovery outputs for every replication.
    """

    replications: list[ReplicationEstimates] = []
    total_fits = config.n_replications * config.n_subjects

    max_workers = config.max_workers
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, config.n_subjects)

    with tqdm(total=total_fits, desc="Recovery (MLE)", unit="subj") as pbar:
        for r in range(config.n_replications):
            rng = np.random.default_rng(config.simulation_base_seed + r)
            true_table, params_per_subject = sample_true_params(
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

            estimates = extract_mle_estimates(results, config.layout)
            replications.append(
                ReplicationEstimates(
                    replication_index=r,
                    true_params=true_table,
                    subject_estimates=estimates,
                )
            )

    return RecoveryResult(config=config, replications=tuple(replications))


def _run_stan_recovery(config: RecoveryStudyConfig) -> RecoveryResult:
    """Run recovery with hierarchical Stan fits.

    Parameters
    ----------
    config
        Recovery study configuration for a Stan backend.

    Returns
    -------
    RecoveryResult
        Recovery outputs for every replication.
    """

    simulated: list[tuple[int, dict[str, dict[str, float]], Dataset]] = []
    for r in range(config.n_replications):
        rng = np.random.default_rng(config.simulation_base_seed + r)
        true_table, params_per_subject = sample_true_params(
            config.param_dists, config.kernel, config.n_subjects, rng, config.layout
        )
        dataset = _simulate_dataset(
            config, params_per_subject, seed=config.simulation_base_seed + r
        )
        simulated.append((r, true_table, dataset))

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
        item: tuple[int, dict[str, dict[str, float]], Dataset],
    ) -> ReplicationEstimates:
        """Fit one simulated replication with the Stan backend.

        Parameters
        ----------
        item
            Tuple containing the replication index, true parameter table, and
            simulated dataset.

        Returns
        -------
        ReplicationEstimates
            Extracted posterior summaries for the replication.

        Raises
        ------
        TypeError
            If dispatch returns a non-Bayesian fit result for a Stan backend.
        """

        r, true_table, dataset = item
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
        estimates = extract_bayes_estimates(
            result,
            subject_ids,
            param_names,
            config.layout,  # type: ignore[arg-type]
        )
        pop_point, pop_samples = extract_population_estimates(result, param_names)

        # Empirical constrained-scale population mean from this replication's
        # true parameters.  Computed from true_table so it is valid for any
        # ParamDist.scale and varies across replications (enabling correlation).
        true_pop: dict[str, float] = {}
        for name in param_names:
            vals = [true_table[sid][name] for sid in true_table if name in true_table[sid]]
            if vals:
                true_pop[f"{name}_pop"] = float(np.mean(vals))

        # Also include unconstrained-scale true mu/sd when ParamDist was
        # specified with scale="unconstrained" (true values are then constants).
        true_pop.update(get_true_population_params(config.param_dists, config.kernel))

        return ReplicationEstimates(
            replication_index=r,
            true_params=true_table,
            subject_estimates=estimates,
            population_true_params=true_pop or None,
            population_estimates=pop_point or None,
            population_posterior_samples=pop_samples or None,
        )

    with tqdm(total=config.n_replications, desc="Recovery (Stan)", unit="rep") as pbar:
        if max_workers > 1 and config.n_replications > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(_fit_one, item): i for i, item in enumerate(simulated)
                }
                results_by_idx: dict[int, ReplicationEstimates] = {}
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

    return RecoveryResult(config=config, replications=tuple(replications))


def _simulate_dataset(
    config: RecoveryStudyConfig,
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

    Returns
    -------
    Dataset
        Simulated dataset compatible with the configured task layout.
    """

    if config.layout is not None:
        return _simulate_condition_aware(config, params_per_subject, seed=seed)

    return _simulate_simple(config, params_per_subject, seed=seed)


def _simulate_simple(
    config: RecoveryStudyConfig,
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
    config: RecoveryStudyConfig,
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
