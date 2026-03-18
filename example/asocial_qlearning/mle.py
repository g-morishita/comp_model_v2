"""Simulate asocial Q-learning agents on a stationary bandit, then recover
parameters with per-subject MLE.

Ground-truth: alpha=0.3, beta=2.0
"""

from pathlib import Path

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.io import load_dataset_from_csv, save_dataset_to_csv
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

# ── 2. Create environment ──────────────────────────────────────────────────
REWARD_PROBS = (0.8, 0.2)

# ── 3. Set ground-truth parameters ─────────────────────────────────────────
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
csv_path = Path(__file__).parent / "data.csv"
save_dataset_to_csv(dataset, schema=ASOCIAL_BANDIT_SCHEMA, path=csv_path)
print(f"Saved {len(dataset.subjects)} subjects to {csv_path}")

# ── 6. Load back from CSV (round-trip check) ───────────────────────────────
loaded = load_dataset_from_csv(csv_path, schema=ASOCIAL_BANDIT_SCHEMA)
assert len(loaded.subjects) == N_SUBJECTS

# ── 7. Fit each subject with MLE ───────────────────────────────────────────
mle_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_SHARED,
    backend="mle",
    mle_config=MleOptimizerConfig(n_restarts=10, seed=0),
)

print(f"\n{'Subject':<12} {'True a':>8} {'Fit a':>8} {'True b':>8} {'Fit b':>8} {'LL':>10}")
print("-" * 60)

for subject in loaded.subjects:
    result = fit(mle_config, kernel, subject, ASOCIAL_BANDIT_SCHEMA)
    cp = result.constrained_params
    print(
        f"{subject.subject_id:<12} "
        f"{TRUE_ALPHA:>8.3f} {cp['alpha']:>8.3f} "
        f"{TRUE_BETA:>8.3f} {cp['beta']:>8.3f} "
        f"{result.log_likelihood:>10.2f}"
    )

print("\nDone.")
