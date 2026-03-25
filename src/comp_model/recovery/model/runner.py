"""Orchestration engine for model recovery studies."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from comp_model.inference.bayes.result import BayesFitResult
from comp_model.inference.dispatch import fit
from comp_model.recovery.model.criteria import (
    score_candidate_bayes,
    score_candidate_mle,
    select_winner,
)
from comp_model.recovery.parameter.config import sample_true_params
from comp_model.runtime import SimulationConfig
from comp_model.runtime.engine import simulate_dataset

if TYPE_CHECKING:
    from comp_model.data.schema import Dataset
    from comp_model.inference.mle.optimize import MleFitResult
    from comp_model.recovery.model.config import CandidateModelSpec, ModelRecoveryConfig
    from comp_model.tasks.schemas import TrialSchema
    from comp_model.tasks.spec import TaskSpec


@dataclass(frozen=True, slots=True)
class ReplicationResult:
    """Outcome of one simulate-then-fit cycle for a single generating model.

    Attributes
    ----------
    replication_index
        Zero-based index of this replication within the study.
    generating_model
        Name of the generating model that produced the simulated data.
    candidate_scores
        Score for each candidate model (higher = better for all criteria).
    selected_model
        Name of the candidate model that achieved the highest score.
    winner_score
        Score of the selected model.
    second_best_model
        Name of the runner-up candidate, or ``None`` if only one candidate.
    delta_to_second
        ``winner_score - second_score``, or ``None`` if only one candidate.
    """

    replication_index: int
    generating_model: str
    candidate_scores: dict[str, float]
    selected_model: str
    winner_score: float
    second_best_model: str | None
    delta_to_second: float | None


@dataclass(frozen=True, slots=True)
class ModelRecoveryResult:
    """Complete results from a model recovery study.

    Attributes
    ----------
    config
        Study configuration used.
    replications
        Results from all replications across all generating models,
        in ``(generating_model, replication_index)`` order.
    """

    config: ModelRecoveryConfig
    replications: tuple[ReplicationResult, ...]


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


def run_model_recovery(config: ModelRecoveryConfig) -> ModelRecoveryResult:
    """Run the full model recovery pipeline with parallel fitting.

    The pipeline runs in two phases:

    1. **Simulation** (sequential): All ``(generating_model, replication)``
       datasets are simulated upfront.  Simulation is kept sequential to avoid
       pickling the user-supplied ``env_factory``.

    2. **Fitting** (parallel): One job is submitted per
       ``(generating_model, candidate_model, replication)`` triple.  Each job
       fits a single candidate model to a single dataset so the parallelism
       is as fine-grained as possible.  Model selection is performed after all
       scores for a ``(generating_model, replication)`` pair are collected.

    ``config.max_workers`` controls the number of parallel processes in phase 2.
    When ``None``, it defaults to ``min(cpu_count, n_jobs)`` where
    ``n_jobs = n_generating_models x n_candidates x n_replications``.

    Parameters
    ----------
    config
        Model recovery study configuration.

    Returns
    -------
    ModelRecoveryResult
        Results from all replications across all generating models.
    """

    _check_schema_consistency(config.task, config.schema)

    # ------------------------------------------------------------------
    # Phase 1: simulate all datasets (sequential — env_factory may not pickle)
    # ------------------------------------------------------------------
    # Key: (gen_name, rep_idx) → Dataset
    simulated: dict[tuple[str, int], Dataset] = {}
    gen_rep_order: list[tuple[str, int]] = []

    for gen_spec in config.generating_models:
        for rep_idx in range(config.n_replications):
            seed = config.simulation_base_seed + rep_idx
            rng = np.random.default_rng(seed)
            _, params_per_subject = sample_true_params(
                gen_spec.param_dists,
                gen_spec.kernel,
                config.n_subjects,
                rng,
            )
            dataset = simulate_dataset(
                task=config.task,
                env_factory=config.env_factory,
                kernel=gen_spec.kernel,
                params_per_subject=params_per_subject,
                config=SimulationConfig(seed=seed),
            )
            key = (gen_spec.name, rep_idx)
            simulated[key] = dataset
            gen_rep_order.append(key)

    # ------------------------------------------------------------------
    # Phase 2: fit — one job per (generating_model, candidate, replication)
    # ------------------------------------------------------------------
    # Build the flat job list, preserving a stable ordering key.
    jobs: list[tuple[str, str, int]] = [  # (gen_name, cand_name, rep_idx)
        (gen_name, cand.name, rep_idx)
        for gen_name, rep_idx in gen_rep_order
        for cand in config.candidate_models
    ]

    max_workers = config.max_workers
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, len(jobs))

    # scores[gen_name][rep_idx][cand_name] = score
    scores: dict[str, dict[int, dict[str, float]]] = {}
    total_jobs = len(jobs)

    with tqdm(total=total_jobs, desc="Model recovery", unit="fit") as pbar:
        if max_workers > 1 and total_jobs > 1:
            cand_by_name = {c.name: c for c in config.candidate_models}
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_key = {
                    executor.submit(
                        _fit_candidate_job,
                        gen_name,
                        rep_idx,
                        cand_by_name[cand_name],
                        simulated[(gen_name, rep_idx)],
                        config.schema,
                        config.criterion,
                    ): (gen_name, rep_idx, cand_name)
                    for gen_name, cand_name, rep_idx in jobs
                }
                for future in as_completed(future_to_key):
                    gen_name, rep_idx, cand_name = future_to_key[future]
                    score = future.result()
                    scores.setdefault(gen_name, {}).setdefault(rep_idx, {})[cand_name] = score
                    pbar.update(1)
        else:
            cand_by_name = {c.name: c for c in config.candidate_models}
            for gen_name, cand_name, rep_idx in jobs:
                score = _fit_candidate_job(
                    gen_name,
                    rep_idx,
                    cand_by_name[cand_name],
                    simulated[(gen_name, rep_idx)],
                    config.schema,
                    config.criterion,
                )
                scores.setdefault(gen_name, {}).setdefault(rep_idx, {})[cand_name] = score
                pbar.update(1)

    # ------------------------------------------------------------------
    # Assemble ReplicationResult in original (gen, rep) order
    # ------------------------------------------------------------------
    replications: list[ReplicationResult] = []
    for gen_name, rep_idx in gen_rep_order:
        rep_scores = scores[gen_name][rep_idx]
        winner, winner_score, second, delta = select_winner(rep_scores)
        replications.append(
            ReplicationResult(
                replication_index=rep_idx,
                generating_model=gen_name,
                candidate_scores=rep_scores,
                selected_model=winner,
                winner_score=winner_score,
                second_best_model=second,
                delta_to_second=delta,
            )
        )

    return ModelRecoveryResult(
        config=config,
        replications=tuple(replications),
    )


# ---------------------------------------------------------------------------
# Top-level worker (must be importable at module level for pickling)
# ---------------------------------------------------------------------------


def _fit_candidate_job(
    gen_name: str,
    rep_idx: int,
    cand: CandidateModelSpec,
    dataset: Dataset,
    schema: Any,
    criterion: str,
) -> float:
    """Fit one candidate model to one dataset and return its score.

    This function is the unit of work submitted to the process pool.  It runs
    entirely in a single process with no nested parallelism.

    Parameters
    ----------
    gen_name
        Name of the generating model (used only for error messages).
    rep_idx
        Replication index (used only for error messages).
    cand
        Candidate model specification.
    dataset
        Simulated dataset to fit.
    schema
        Trial schema used for replay extraction.
    criterion
        Scoring criterion.

    Returns
    -------
    float
        Score for ``cand`` on ``dataset`` (higher = better).
    """

    if criterion in ("aic", "bic", "log_likelihood"):
        mle_results: list[MleFitResult] = [
            fit(cand.inference_config, cand.kernel, subject_data, schema)  # type: ignore[return-value]
            for subject_data in dataset.subjects
        ]
        return score_candidate_mle(mle_results, criterion)  # type: ignore[arg-type]

    if criterion in ("waic", "loo"):
        result = fit(
            cand.inference_config,
            cand.kernel,
            dataset,
            schema,
            adapter=cand.adapter,
        )
        if not isinstance(result, BayesFitResult):
            raise TypeError(
                f"Candidate '{cand.name}' (gen='{gen_name}', rep={rep_idx}) "
                "returned a non-Bayesian result. "
                "Ensure inference_config.backend == 'stan'."
            )
        return score_candidate_bayes(result, criterion)  # type: ignore[arg-type]

    raise ValueError(f"Unknown criterion: {criterion!r}")
