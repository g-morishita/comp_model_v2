"""Orchestration engine for model recovery studies."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from comp_model.data.schema import Block, Dataset, SubjectData
from comp_model.inference.bayes.result import BayesFitResult
from comp_model.inference.config import HierarchyStructure
from comp_model.inference.dispatch import fit
from comp_model.inference.exceptions import SamplingError
from comp_model.recovery.model.criteria import (
    score_candidate_bayes,
    score_candidate_mle,
    select_winner,
)
from comp_model.recovery.model.result import ModelRecoveryResult, ReplicationResult
from comp_model.recovery.parameter.config import sample_true_params
from comp_model.runtime import SimulationConfig
from comp_model.runtime.engine import simulate_dataset, simulate_subject

if TYPE_CHECKING:
    from comp_model.inference.mle.optimize import MleFitResult
    from comp_model.models.condition.shared_delta import SharedDeltaLayout
    from comp_model.recovery.model.config import (
        CandidateModelSpec,
        GeneratingModelSpec,
        ModelRecoveryConfig,
    )
    from comp_model.tasks.schemas import TrialSchema
    from comp_model.tasks.spec import TaskSpec

_CONDITION_HIERARCHIES = (
    HierarchyStructure.SUBJECT_BLOCK_CONDITION,
    HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
)


def _precompile_stan_models(config: ModelRecoveryConfig) -> None:
    """Pre-compile all unique Stan programs before parallel fitting.

    Parameters
    ----------
    config
        Model recovery configuration containing candidate model specs.

    Notes
    -----
    CmdStanPy compilation is not safe for concurrent access.  When multiple
    ``ProcessPoolExecutor`` workers call ``CmdStanModel`` on the same
    ``.stan`` file simultaneously, one process may delete intermediate build
    artefacts while another still expects them, causing a
    ``FileNotFoundError``.  Compiling once in the main process ensures the
    cached executable is available for all workers.
    """
    stan_candidates = [
        c
        for c in config.candidate_models
        if c.inference_config.backend == "stan" and c.adapter is not None
    ]
    if not stan_candidates:
        return

    try:
        cmdstanpy = importlib.import_module("cmdstanpy")
    except ModuleNotFoundError:
        return

    seen: set[str] = set()
    for cand in stan_candidates:
        adapter: Any = cand.adapter
        stan_file: str = adapter.stan_program_path(cand.inference_config.hierarchy)
        if stan_file in seen:
            continue
        seen.add(stan_file)
        functions_dir = str(Path(stan_file).parent / "functions")
        cmdstanpy.CmdStanModel(
            stan_file=stan_file,
            stanc_options={"include-paths": [functions_dir]},
        )


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


def _validate_layout_for_kernel(
    *,
    owner: str,
    kernel: Any,
    layout: SharedDeltaLayout | None,
) -> None:
    """Ensure an optional layout matches the kernel it is paired with."""

    if layout is None:
        return

    kernel_spec = kernel.spec()
    if layout.kernel_spec != kernel_spec:
        raise ValueError(
            f"{owner} layout kernel_spec {layout.kernel_spec.model_id!r} does not match "
            f"kernel {kernel_spec.model_id!r}"
        )


def _check_layout_consistency(config: ModelRecoveryConfig) -> None:
    """Validate layout usage across generating and candidate models."""

    for gen_spec in config.generating_models:
        _validate_layout_for_kernel(
            owner=f"Generating model {gen_spec.name!r}",
            kernel=gen_spec.kernel,
            layout=gen_spec.layout,
        )

    for cand_spec in config.candidate_models:
        _validate_layout_for_kernel(
            owner=f"Candidate model {cand_spec.name!r}",
            kernel=cand_spec.kernel,
            layout=cand_spec.layout,
        )
        hierarchy = cand_spec.inference_config.hierarchy
        if hierarchy in _CONDITION_HIERARCHIES and cand_spec.layout is None:
            raise ValueError(
                f"Candidate model {cand_spec.name!r} uses condition-aware hierarchy "
                f"{hierarchy.value!r} but layout=None"
            )
        if hierarchy not in _CONDITION_HIERARCHIES and cand_spec.layout is not None:
            raise ValueError(
                f"Candidate model {cand_spec.name!r} provides a layout, but hierarchy "
                f"{hierarchy.value!r} is not condition-aware"
            )


def _with_console_suppressed(cand: CandidateModelSpec) -> CandidateModelSpec:
    """Return a copy of *cand* with Stan console output disabled.

    When fitting candidates in parallel without a ``log_dir``, Stan's
    raw progress text would interleave on the terminal.  This helper
    sets ``show_console=False`` so parallel workers stay quiet while
    the job-level ``tqdm`` bar still reports overall progress.

    If the candidate does not use a Stan backend the spec is returned
    unchanged.

    Parameters
    ----------
    cand
        Original candidate model specification.

    Returns
    -------
    CandidateModelSpec
        A shallow copy with ``show_console=False`` on the Stan config,
        or the original spec if no Stan config is present.
    """
    stan_cfg = getattr(cand.inference_config, "stan_config", None)
    if stan_cfg is None or not stan_cfg.show_console:
        return cand
    quiet_stan = dataclasses.replace(stan_cfg, show_console=False)
    quiet_inf = dataclasses.replace(cand.inference_config, stan_config=quiet_stan)
    return dataclasses.replace(cand, inference_config=quiet_inf)


def _simulate_generated_dataset(
    config: ModelRecoveryConfig,
    gen_spec: GeneratingModelSpec,
    params_per_subject: dict[str, Any],
    seed: int,
) -> Dataset:
    """Simulate one generated dataset for a model recovery replication."""

    if gen_spec.layout is None:
        return simulate_dataset(
            task=config.task,
            env_factory=config.env_factory,
            kernel=gen_spec.kernel,
            params_per_subject=params_per_subject,
            config=SimulationConfig(seed=seed),
            demonstrator_kernel=config.demonstrator_kernel,
            demonstrator_params=config.demonstrator_params,
        )

    from comp_model.tasks.spec import TaskSpec

    subjects: list[SubjectData] = []
    for subject_offset, (subject_id, condition_params) in enumerate(params_per_subject.items()):
        blocks: list[Block] = []
        for block_index, block_spec in enumerate(config.task.blocks):
            condition = block_spec.condition
            if condition not in condition_params:
                raise ValueError(
                    f"Generating model {gen_spec.name!r} is missing parameters for "
                    f"task condition {condition!r}"
                )
            # Resolve per-condition demonstrator params
            demo_params = config.demonstrator_params
            if config.condition_demonstrator_params is not None:
                if condition not in config.condition_demonstrator_params:
                    raise ValueError(
                        f"Generating model {gen_spec.name!r} is missing demonstrator "
                        f"parameters for task condition {condition!r} in "
                        f"'condition_demonstrator_params'"
                    )
                demo_params = config.condition_demonstrator_params[condition]
            env = config.env_factory()
            subject = simulate_subject(
                task=TaskSpec(task_id="tmp", blocks=(block_spec,)),
                env=env,
                kernel=gen_spec.kernel,
                params=condition_params[condition],
                config=SimulationConfig(seed=seed + subject_offset * 1000 + block_index),
                subject_id=subject_id,
                demonstrator_kernel=config.demonstrator_kernel,
                demonstrator_params=demo_params,
            )
            blocks.append(dataclasses.replace(subject.blocks[0], block_index=block_index))
        subjects.append(SubjectData(subject_id=subject_id, blocks=tuple(blocks)))

    return Dataset(subjects=tuple(subjects))


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
    _check_layout_consistency(config)

    from comp_model.data.compatibility import check_kernel_schema_compatibility

    for gen_spec in config.generating_models:
        check_kernel_schema_compatibility(gen_spec.kernel, config.schema)
    for cand_spec in config.candidate_models:
        check_kernel_schema_compatibility(cand_spec.kernel, config.schema)

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
            _, params_per_subject, _ = sample_true_params(
                gen_spec.param_dists,
                gen_spec.kernel,
                config.n_subjects,
                rng,
                gen_spec.layout,
            )
            dataset = _simulate_generated_dataset(config, gen_spec, params_per_subject, seed)
            key = (gen_spec.name, rep_idx)
            simulated[key] = dataset
            gen_rep_order.append(key)

    # ------------------------------------------------------------------
    # Phase 1.5: pre-compile Stan programs (avoids parallel race condition)
    # ------------------------------------------------------------------
    _precompile_stan_models(config)

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

    # scores[gen_name][rep_idx][cand_name] = score | None
    scores: dict[str, dict[int, dict[str, float | None]]] = {}
    # Track (gen, rep) pairs where at least one candidate failed.
    failed_pairs: set[tuple[str, int]] = set()
    total_jobs = len(jobs)

    log_dir = config.log_dir
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    with tqdm(total=total_jobs, desc="Model recovery", unit="fit") as pbar:
        if max_workers > 1 and total_jobs > 1:
            # When log_dir is set, keep show_console=True so chain
            # progress is written — stdout/stderr are redirected to
            # per-job files inside the worker.  Without log_dir,
            # suppress console output entirely.
            if log_dir is not None:
                cand_by_name = {c.name: c for c in config.candidate_models}
            else:
                cand_by_name = {
                    c.name: _with_console_suppressed(c) for c in config.candidate_models
                }
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
                        log_dir,
                    ): (gen_name, rep_idx, cand_name)
                    for gen_name, cand_name, rep_idx in jobs
                }
                for future in as_completed(future_to_key):
                    gen_name, rep_idx, cand_name = future_to_key[future]
                    score = future.result()
                    scores.setdefault(gen_name, {}).setdefault(rep_idx, {})[cand_name] = score
                    if score is None:
                        failed_pairs.add((gen_name, rep_idx))
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
                if score is None:
                    failed_pairs.add((gen_name, rep_idx))
                pbar.update(1)

    # ------------------------------------------------------------------
    # Assemble ReplicationResult in original (gen, rep) order
    # ------------------------------------------------------------------
    if failed_pairs:
        _logger.warning(
            "%d replication(s) excluded due to sampling failures: %s",
            len(failed_pairs),
            sorted(failed_pairs),
        )

    replications: list[ReplicationResult] = []
    for gen_name, rep_idx in gen_rep_order:
        if (gen_name, rep_idx) in failed_pairs:
            continue
        rep_scores = {k: v for k, v in scores[gen_name][rep_idx].items() if v is not None}
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


@contextlib.contextmanager
def _redirect_to_log(log_path: Path | None):
    """Context manager that redirects stdout and stderr to *log_path*.

    When *log_path* is ``None`` the context manager is a no-op and
    output goes to the original streams as usual.

    Parameters
    ----------
    log_path
        File to write captured output to, or ``None`` to skip.

    Yields
    ------
    None
    """
    if log_path is None:
        yield
        return
    with open(log_path, "w") as fh:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = fh  # type: ignore[assignment]
        sys.stderr = fh  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


def _fit_candidate_job(
    gen_name: str,
    rep_idx: int,
    cand: CandidateModelSpec,
    dataset: Dataset,
    schema: Any,
    criterion: str,
    log_dir: Path | None = None,
) -> float | None:
    """Fit one candidate model to one dataset and return its score.

    This function is the unit of work submitted to the process pool.  It runs
    entirely in a single process with no nested parallelism.

    When *log_dir* is not ``None``, stdout and stderr are redirected to
    ``<log_dir>/gen=<gen_name>_cand=<cand_name>_rep=<rep_idx>.log`` so
    that Stan chain progress can be inspected with ``tail -f`` without
    interleaving across workers.

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
    log_dir
        Directory for per-job log files.  ``None`` skips redirection.

    Returns
    -------
    float
        Score for ``cand`` on ``dataset`` (higher = better).
    """

    log_path = (
        log_dir / f"gen={gen_name}_cand={cand.name}_rep={rep_idx}.log"
        if log_dir is not None
        else None
    )

    with _redirect_to_log(log_path):
        return _fit_candidate_inner(gen_name, rep_idx, cand, dataset, schema, criterion)


_logger = logging.getLogger(__name__)


def _fit_candidate_inner(
    gen_name: str,
    rep_idx: int,
    cand: CandidateModelSpec,
    dataset: Dataset,
    schema: Any,
    criterion: str,
) -> float | None:
    """Core fitting logic, separated to allow stdout/stderr redirection.

    Returns
    -------
    float or None
        The candidate score, or ``None`` when sampling fails.  A ``None``
        signals that this ``(gen, rep)`` pair should be excluded from
        results rather than biased with an artificial score.
    """
    try:
        return _fit_candidate_core(gen_name, rep_idx, cand, dataset, schema, criterion)
    except SamplingError as exc:
        _logger.warning(
            "Sampling failed for gen=%r cand=%r rep=%d: %s — "
            "this replication will be excluded from results",
            gen_name,
            cand.name,
            rep_idx,
            exc,
        )
        return None


def _fit_candidate_core(
    gen_name: str,
    rep_idx: int,
    cand: CandidateModelSpec,
    dataset: Dataset,
    schema: Any,
    criterion: str,
) -> float:
    """Run the actual fitting and scoring for a single candidate.

    Parameters
    ----------
    gen_name
        Name of the generating model.
    rep_idx
        Replication index.
    cand
        Candidate model specification.
    dataset
        Simulated dataset to fit.
    schema
        Trial schema.
    criterion
        Scoring criterion.

    Returns
    -------
    float
        Score for ``cand`` on ``dataset`` (higher = better).
    """

    if criterion in ("aic", "bic", "log_likelihood"):
        mle_results: list[MleFitResult] = [
            fit(
                cand.inference_config,
                cand.kernel,
                subject_data,
                schema,
                cand.layout,
            )  # type: ignore[return-value]
            for subject_data in dataset.subjects
        ]
        return score_candidate_mle(mle_results, criterion)  # type: ignore[arg-type]

    if criterion == "waic":
        result = fit(
            cand.inference_config,
            cand.kernel,
            dataset,
            schema,
            cand.layout,
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
