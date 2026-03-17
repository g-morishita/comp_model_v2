"""Parameter recovery analysis for within-subject condition-aware models.

Compares per-subject MLE vs. hierarchical Stan (population-level) recovery
as the number of trials increases. If recovery improves with more data the
model is identifiable and the code is correct; persistent bias at all sample
sizes suggests an identifiability or implementation problem.

Ground-truth:
  Condition "easy":   alpha=0.3, beta=2.0  (baseline)
  Condition "hard":   alpha=0.5, beta=1.0

Usage:
  python parameter_recovery_within_subject.py          # MLE only
  python parameter_recovery_within_subject.py --stan   # MLE + hierarchical Stan

Requires: cmdstanpy and a working CmdStan installation (for --stan).
"""

from __future__ import annotations

import sys

import numpy as np

from comp_model.data import Block, Dataset, SubjectData
from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.models import SharedDeltaLayout
from comp_model.models.kernels import AsocialQLearningKernel, QParams
from comp_model.models.kernels.transforms import get_transform
from comp_model.runtime import SimulationConfig, simulate_subject
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec

# ── Configuration ─────────────────────────────────────────────────────────
N_ACTIONS = 2
N_SUBJECTS = 10
TRIAL_COUNTS = [50, 100, 200, 500]

TRUE_PARAMS: dict[str, QParams] = {
    "easy": QParams(alpha=0.3, beta=2.0),
    "hard": QParams(alpha=0.5, beta=1.0),
}

REWARD_PROBS: dict[str, tuple[float, ...]] = {
    "easy": (0.8, 0.2),
    "hard": (0.6, 0.4),
}

kernel = AsocialQLearningKernel()
layout = SharedDeltaLayout(
    kernel_spec=kernel.spec(),
    conditions=("easy", "hard"),
    baseline_condition="easy",
)

sigmoid = get_transform("sigmoid")
softplus = get_transform("softplus")


def simulate_within_subject_dataset(
    n_trials: int,
    n_subjects: int,
    base_seed: int,
) -> Dataset:
    """Simulate a within-subject dataset with the given trial count per block.

    Parameters
    ----------
    n_trials
        Number of trials per condition block.
    n_subjects
        Number of subjects to simulate.
    base_seed
        Base random seed (offset per subject and block).

    Returns
    -------
    Dataset
        Simulated dataset.
    """
    task = TaskSpec(
        task_id="recovery",
        blocks=tuple(
            BlockSpec(
                condition=cond,
                n_trials=n_trials,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": N_ACTIONS},
            )
            for cond in ("easy", "hard")
        ),
    )

    subjects = []
    for i in range(n_subjects):
        sid = f"sub_{i:02d}"
        blocks = []
        for block_idx, block_spec in enumerate(task.blocks):
            cond = block_spec.condition
            env = StationaryBanditEnvironment(
                n_actions=N_ACTIONS,
                reward_probs=REWARD_PROBS[cond],
            )
            sub = simulate_subject(
                task=TaskSpec(task_id="tmp", blocks=(block_spec,)),
                env=env,
                kernel=kernel,
                params=TRUE_PARAMS[cond],
                config=SimulationConfig(seed=base_seed + i * 1000 + block_idx),
                subject_id=sid,
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


def _print_row(
    n_trials: int,
    method: str,
    cond: str,
    true_alpha: float,
    fit_alphas: np.ndarray,
    true_beta: float,
    fit_betas: np.ndarray,
) -> None:
    """Print one row of the recovery table."""
    bias_a = float(np.mean(fit_alphas) - true_alpha)
    rmse_a = float(np.sqrt(np.mean((fit_alphas - true_alpha) ** 2)))
    bias_b = float(np.mean(fit_betas) - true_beta)
    rmse_b = float(np.sqrt(np.mean((fit_betas - true_beta) ** 2)))
    print(
        f"{n_trials:>6}  {method:<5}  {cond:<6}  "
        f"{true_alpha:>7.3f}  {np.mean(fit_alphas):>7.3f}  "
        f"{np.std(fit_alphas):>7.3f}  {bias_a:>+7.3f}  {rmse_a:>7.3f}  "
        f"{true_beta:>7.3f}  {np.mean(fit_betas):>7.3f}  "
        f"{np.std(fit_betas):>7.3f}  {bias_b:>+7.3f}  {rmse_b:>7.3f}"
    )


def run_mle_recovery(
    dataset: Dataset,
) -> dict[str, dict[str, list[float]]]:
    """Run per-subject MLE recovery.

    Parameters
    ----------
    dataset
        Dataset to fit.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        ``{condition: {param_name: [fitted_values_per_subject]}}``
    """
    mle_config = InferenceConfig(
        hierarchy=HierarchyStructure.SUBJECT_BLOCK_CONDITION,
        backend="mle",
        mle_config=MleOptimizerConfig(n_restarts=20, seed=0),
    )

    recovered: dict[str, dict[str, list[float]]] = {
        cond: {"alpha": [], "beta": []} for cond in ("easy", "hard")
    }

    for subject in dataset.subjects:
        result = fit(mle_config, kernel, subject, ASOCIAL_BANDIT_SCHEMA, layout=layout)
        for cond in ("easy", "hard"):
            fit_p = result.params_by_condition[cond]
            recovered[cond]["alpha"].append(float(sigmoid.forward(fit_p["alpha"])))
            recovered[cond]["beta"].append(float(softplus.forward(fit_p["beta"])))

    return recovered


def run_stan_recovery(
    dataset: Dataset,
) -> dict[str, dict[str, list[float]]]:
    """Run hierarchical Stan recovery with population-level estimates.

    Uses STUDY_SUBJECT_BLOCK_CONDITION hierarchy to jointly estimate
    population-level and per-subject parameters with hierarchical shrinkage.

    Parameters
    ----------
    dataset
        Dataset to fit.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        ``{condition: {param_name: [per-subject posterior means]}}``
    """
    from comp_model.inference.bayes.stan import AsocialQLearningStanAdapter, StanFitConfig

    stan_config = InferenceConfig(
        hierarchy=HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
        backend="stan",
        stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
    )
    adapter = AsocialQLearningStanAdapter()
    result = fit(
        stan_config, kernel, dataset, ASOCIAL_BANDIT_SCHEMA, layout=layout, adapter=adapter
    )

    recovered: dict[str, dict[str, list[float]]] = {
        cond: {"alpha": [], "beta": []} for cond in ("easy", "hard")
    }

    # alpha, beta have shape (n_draws, N_SUBJECTS, C)
    alpha_samples = result.posterior_samples["alpha"]
    beta_samples = result.posterior_samples["beta"]

    for c_idx, cond in enumerate(layout.conditions):
        for s_idx in range(len(dataset.subjects)):
            recovered[cond]["alpha"].append(float(np.mean(alpha_samples[:, s_idx, c_idx])))
            recovered[cond]["beta"].append(float(np.mean(beta_samples[:, s_idx, c_idx])))

    return recovered


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_stan = "--stan" in sys.argv

    header = (
        f"{'Trials':>6}  {'Meth':<5}  {'Cond':<6}  "
        f"{'True a':>7}  {'Mean a':>7}  {'SD a':>7}  {'Bias a':>7}  {'RMSE a':>7}  "
        f"{'True b':>7}  {'Mean b':>7}  {'SD b':>7}  {'Bias b':>7}  {'RMSE b':>7}"
    )
    methods = "MLE" + (" + Stan" if run_stan else "")
    print(f"Parameter Recovery: Within-Subject Condition-Aware ({methods})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for n_trials in TRIAL_COUNTS:
        dataset = simulate_within_subject_dataset(n_trials, N_SUBJECTS, base_seed=42)

        # Per-subject MLE
        mle_recovered = run_mle_recovery(dataset)
        for cond in ("easy", "hard"):
            _print_row(
                n_trials,
                "MLE",
                cond,
                TRUE_PARAMS[cond].alpha,
                np.array(mle_recovered[cond]["alpha"]),
                TRUE_PARAMS[cond].beta,
                np.array(mle_recovered[cond]["beta"]),
            )

        # Hierarchical Stan (population-level)
        if run_stan:
            stan_recovered = run_stan_recovery(dataset)
            for cond in ("easy", "hard"):
                _print_row(
                    n_trials,
                    "Stan",
                    cond,
                    TRUE_PARAMS[cond].alpha,
                    np.array(stan_recovered[cond]["alpha"]),
                    TRUE_PARAMS[cond].beta,
                    np.array(stan_recovered[cond]["beta"]),
                )

        if n_trials != TRIAL_COUNTS[-1]:
            print()

    print("\nInterpretation:")
    print("  - If RMSE decreases with more trials: model is identifiable, code is correct.")
    print("  - If RMSE stays constant: possible identifiability issue.")
    print("  - If RMSE increases: likely a code bug.")
    if run_stan:
        print("  - If Stan RMSE < MLE RMSE: hierarchical shrinkage helps recovery.")
    else:
        print("  - Run with --stan to also test hierarchical Stan recovery.")
