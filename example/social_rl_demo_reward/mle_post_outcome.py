"""Simulate social demonstrator-reward learning agents with post-outcome
demonstrator observation, then recover parameters with per-subject MLE.

Ground-truth: alpha_other=0.2, beta=2.0
"""

from pathlib import Path

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.io import load_dataset_from_csv, save_dataset_to_csv
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    QParams,
    SocialRlDemoRewardKernel,
    SocialRlDemoRewardParams,
)
from comp_model.runtime import SimulationConfig, simulate_dataset
from comp_model.tasks import SOCIAL_POST_OUTCOME_SCHEMA, BlockSpec, TaskSpec

# ── 1. Define task ──────────────────────────────────────────────────────────
N_ACTIONS = 2
N_TRIALS = 100
N_SUBJECTS = 5
REWARD_PROBS = (0.8, 0.2)

task = TaskSpec(
    task_id="social_post_outcome",
    blocks=(
        BlockSpec(
            condition="social",
            n_trials=N_TRIALS,
            schema=SOCIAL_POST_OUTCOME_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
    ),
)

# ── 2. Ground-truth parameters ─────────────────────────────────────────────
TRUE_ALPHA_OTHER = 0.2
TRUE_BETA = 2.0

kernel = SocialRlDemoRewardKernel()
true_params = SocialRlDemoRewardParams(alpha_other=TRUE_ALPHA_OTHER, beta=TRUE_BETA)

# ── 3. Simulate dataset ────────────────────────────────────────────────────
params_per_subject = {f"sub_{i:02d}": true_params for i in range(N_SUBJECTS)}

dataset = simulate_dataset(
    task=task,
    env_factory=lambda: StationaryBanditEnvironment(n_actions=N_ACTIONS, reward_probs=REWARD_PROBS),
    kernel=kernel,
    params_per_subject=params_per_subject,
    config=SimulationConfig(seed=42),
    demonstrator_kernel=AsocialQLearningKernel(),
    demonstrator_params=QParams(alpha=0.0, beta=0.0),
)

# ── 4. Save and reload CSV ─────────────────────────────────────────────────
csv_path = Path(__file__).parent / "post_outcome_data.csv"
save_dataset_to_csv(dataset, schema=SOCIAL_POST_OUTCOME_SCHEMA, path=csv_path)
print(f"Saved {len(dataset.subjects)} subjects to {csv_path}")

loaded = load_dataset_from_csv(csv_path, schema=SOCIAL_POST_OUTCOME_SCHEMA)

# ── 5. Fit each subject with MLE ───────────────────────────────────────────
mle_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_SHARED,
    backend="mle",
    mle_config=MleOptimizerConfig(n_restarts=10, seed=0),
)

print(f"\n{'Subject':<12} {'True ao':>8} {'Fit ao':>8} {'True b':>8} {'Fit b':>8} {'LL':>10}")
print("-" * 60)

for subject in loaded.subjects:
    result = fit(mle_config, kernel, subject, SOCIAL_POST_OUTCOME_SCHEMA)
    cp = result.constrained_params
    print(
        f"{subject.subject_id:<12} "
        f"{TRUE_ALPHA_OTHER:>8.3f} {cp['alpha_other']:>8.3f} "
        f"{TRUE_BETA:>8.3f} {cp['beta']:>8.3f} "
        f"{result.log_likelihood:>10.2f}"
    )

print("\nDone.")
