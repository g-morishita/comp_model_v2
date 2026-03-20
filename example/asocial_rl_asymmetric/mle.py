"""Simulate asocial asymmetric RL agents on a stationary bandit, then recover
parameters with per-subject MLE.

Ground-truth: alpha_pos=0.5, alpha_neg=0.2, beta=3.0
"""

from pathlib import Path

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.io import load_dataset_from_csv, save_dataset_to_csv
from comp_model.models.kernels import AsocialRlAsymmetricKernel, AsocialRlAsymmetricParams
from comp_model.runtime import SimulationConfig, simulate_dataset
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec

# ── 1. Define task ──────────────────────────────────────────────────────────
N_ACTIONS = 2
N_TRIALS = 200
N_SUBJECTS = 5
REWARD_PROBS = (0.8, 0.2)

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

# ── 4. Save and reload CSV ─────────────────────────────────────────────────
csv_path = Path(__file__).parent / "data.csv"
save_dataset_to_csv(dataset, schema=ASOCIAL_BANDIT_SCHEMA, path=csv_path)
print(f"Saved {len(dataset.subjects)} subjects to {csv_path}")

loaded = load_dataset_from_csv(csv_path, schema=ASOCIAL_BANDIT_SCHEMA)

# ── 5. Fit each subject with MLE ───────────────────────────────────────────
mle_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_SHARED,
    backend="mle",
    mle_config=MleOptimizerConfig(n_restarts=10, seed=0),
)

print(
    f"\n{'Subject':<12} {'True a+':>8} {'Fit a+':>8} "
    f"{'True a-':>8} {'Fit a-':>8} {'True b':>8} {'Fit b':>8} {'LL':>10}"
)
print("-" * 80)

for subject in loaded.subjects:
    result = fit(mle_config, kernel, subject, ASOCIAL_BANDIT_SCHEMA)
    cp = result.constrained_params
    print(
        f"{subject.subject_id:<12} "
        f"{TRUE_ALPHA_POS:>8.3f} {cp['alpha_pos']:>8.3f} "
        f"{TRUE_ALPHA_NEG:>8.3f} {cp['alpha_neg']:>8.3f} "
        f"{TRUE_BETA:>8.3f} {cp['beta']:>8.3f} "
        f"{result.log_likelihood:>10.2f}"
    )

print("\nDone.")
