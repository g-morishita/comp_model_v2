"""Simulate asocial asymmetric RL agents on a stationary bandit, then recover
parameters with per-subject Bayesian Stan inference (no hierarchical pooling).

Ground-truth: alpha_pos=0.5, alpha_neg=0.2, beta=3.0
Requires: cmdstanpy and a working CmdStan installation.
"""

from pathlib import Path

import numpy as np

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.bayes.stan import AsocialRlAsymmetricStanAdapter, StanFitConfig
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.io import save_dataset_to_csv
from comp_model.models.kernels import AsocialRlAsymmetricKernel, AsocialRlAsymmetricParams
from comp_model.runtime import SimulationConfig, simulate_dataset
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec

# ── 1. Define task ──────────────────────────────────────────────────────────
N_ACTIONS = 2
N_TRIALS = 200
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

REWARD_PROBS = (0.8, 0.2)

# ── 2. Ground-truth parameters ─────────────────────────────────────────────
TRUE_ALPHA_POS = 0.5
TRUE_ALPHA_NEG = 0.2
TRUE_BETA = 3.0

kernel = AsocialRlAsymmetricKernel()
true_params = AsocialRlAsymmetricParams(
    alpha_pos=TRUE_ALPHA_POS, alpha_neg=TRUE_ALPHA_NEG, beta=TRUE_BETA
)

# ── 3. Simulate dataset ────────────────────────────────────────────────────
params_per_subject = {f"sub_{i:02d}": true_params for i in range(N_SUBJECTS)}

dataset = simulate_dataset(
    task=task,
    env_factory=lambda: StationaryBanditEnvironment(n_actions=N_ACTIONS, reward_probs=REWARD_PROBS),
    kernel=kernel,
    params_per_subject=params_per_subject,
    config=SimulationConfig(seed=42),
)

# ── 4. Save to CSV ─────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "data.csv"
save_dataset_to_csv(dataset, schema=ASOCIAL_BANDIT_SCHEMA, path=csv_path)
print(f"Saved {len(dataset.subjects)} subjects to {csv_path}")

# ── 5. Fit each subject independently with Stan ────────────────────────────
stan_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_SHARED,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
)

adapter = AsocialRlAsymmetricStanAdapter()

print(
    f"\n{'Subject':<12} {'True a+':>8} {'Post a+':>10} "
    f"{'True a-':>8} {'Post a-':>10} {'True b':>8} {'Post b':>10}"
)
print("-" * 72)

for subject in dataset.subjects:
    result = fit(stan_config, kernel, subject, ASOCIAL_BANDIT_SCHEMA, adapter=adapter)
    ap = np.mean(result.posterior_samples["alpha_pos"])
    an = np.mean(result.posterior_samples["alpha_neg"])
    b = np.mean(result.posterior_samples["beta"])
    print(
        f"{subject.subject_id:<12} "
        f"{TRUE_ALPHA_POS:>8.3f} {ap:>10.3f} "
        f"{TRUE_ALPHA_NEG:>8.3f} {an:>10.3f} "
        f"{TRUE_BETA:>8.3f} {b:>10.3f}"
    )

print(f"\nTrue values: alpha_pos={TRUE_ALPHA_POS}, alpha_neg={TRUE_ALPHA_NEG}, beta={TRUE_BETA}")
print("Done.")
