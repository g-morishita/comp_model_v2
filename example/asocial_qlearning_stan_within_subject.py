"""Simulate asocial Q-learning agents on a stationary bandit, then recover
parameters with per-subject Bayesian Stan inference (no hierarchical pooling).

This is a "within-subject" design: each subject is fit independently with
weakly informative priors. Useful when you don't want to assume subjects are
drawn from a shared population distribution.

Ground-truth: alpha=0.3, beta=2.0 (same for all subjects)
Requires: cmdstanpy and a working CmdStan installation.
"""

from pathlib import Path

import numpy as np

from comp_model.data import Dataset
from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.bayes.stan import AsocialQLearningStanAdapter, StanFitConfig
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.io import save_dataset_to_csv
from comp_model.models.kernels import AsocialQLearningKernel, QParams
from comp_model.runtime import SimulationConfig, simulate_dataset
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec

# ── 1. Define task ──────────────────────────────────────────────────────────
N_ACTIONS = 2
N_TRIALS = 100
N_SUBJECTS = 5

task = TaskSpec(
    task_id="asocial_bandit",
    blocks=(
        BlockSpec(
            condition="learning",
            n_trials=N_TRIALS,
            schema=ASOCIAL_BANDIT_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
    ),
)

# ── 2. Environment ─────────────────────────────────────────────────────────
REWARD_PROBS = (0.8, 0.2)

# ── 3. Ground-truth parameters ─────────────────────────────────────────────
TRUE_ALPHA = 0.3
TRUE_BETA = 2.0
kernel = AsocialQLearningKernel()
true_params = QParams(alpha=TRUE_ALPHA, beta=TRUE_BETA)

# ── 4. Simulate dataset ────────────────────────────────────────────────────
params_per_subject = {f"sub_{i:02d}": true_params for i in range(N_SUBJECTS)}

dataset = simulate_dataset(
    task=task,
    env_factory=lambda: StationaryBanditEnvironment(
        n_actions=N_ACTIONS, reward_probs=REWARD_PROBS
    ),
    kernel=kernel,
    params_per_subject=params_per_subject,
    config=SimulationConfig(seed=42),
)

# ── 5. Save to CSV ─────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "asocial_qlearning_data.csv"
save_dataset_to_csv(dataset, schema=ASOCIAL_BANDIT_SCHEMA, path=csv_path)
print(f"Saved {len(dataset.subjects)} subjects to {csv_path}")

# ── 6. Fit each subject independently with Stan ────────────────────────────
stan_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_SHARED,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
)

adapter = AsocialQLearningStanAdapter()

print(f"\n{'Subject':<12} {'True α':>8} {'Post. α':>10} {'True β':>8} {'Post. β':>10}")
print("-" * 52)

for subject in dataset.subjects:
    result = fit(stan_config, kernel, subject, ASOCIAL_BANDIT_SCHEMA, adapter=adapter)
    alpha_samples = result.posterior_samples["alpha"]
    beta_samples = result.posterior_samples["beta"]
    print(
        f"{subject.subject_id:<12} "
        f"{TRUE_ALPHA:>8.3f} {np.mean(alpha_samples):>10.3f} "
        f"{TRUE_BETA:>8.3f} {np.mean(beta_samples):>10.3f}"
    )

print(f"\nTrue values: alpha={TRUE_ALPHA}, beta={TRUE_BETA}")
print("Done.")
