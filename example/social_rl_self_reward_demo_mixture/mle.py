"""Simulate mixture social RL agents and recover parameters with per-subject MLE.

The mixture kernel maintains two independent value systems:
- v_outcome: updated by self reward (alpha_self) and demonstrator reward (alpha_other_outcome)
- v_tendency: updated by demonstrator action frequency (alpha_other_action)

At decision time both are combined via w_imitation before the softmax.

Ground-truth:
    alpha_self=0.3, alpha_other_outcome=0.2, alpha_other_action=0.4,
    w_imitation=0.3, beta=2.0
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
    SocialRlSelfRewardDemoMixtureKernel,
    SocialRlSelfRewardDemoMixtureParams,
)
from comp_model.runtime import SimulationConfig, simulate_dataset
from comp_model.tasks import SOCIAL_PRE_CHOICE_SCHEMA, BlockSpec, TaskSpec

# ── 1. Define task ──────────────────────────────────────────────────────────
N_ACTIONS = 2
N_TRIALS = 1000
N_SUBJECTS = 1
REWARD_PROBS = (0.8, 0.2)

task = TaskSpec(
    task_id="social_pre_choice_mixture",
    blocks=(
        BlockSpec(
            condition="social",
            n_trials=N_TRIALS,
            schema=SOCIAL_PRE_CHOICE_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
    ),
)

# ── 2. Ground-truth parameters ─────────────────────────────────────────────
TRUE_ALPHA_SELF = 0.3
TRUE_ALPHA_OTHER_OUTCOME = 0.2
TRUE_ALPHA_OTHER_ACTION = 0.4
TRUE_W_IMITATION = 0.3
TRUE_BETA = 2.0

kernel = SocialRlSelfRewardDemoMixtureKernel()
true_params = SocialRlSelfRewardDemoMixtureParams(
    alpha_self=TRUE_ALPHA_SELF,
    alpha_other_outcome=TRUE_ALPHA_OTHER_OUTCOME,
    alpha_other_action=TRUE_ALPHA_OTHER_ACTION,
    w_imitation=TRUE_W_IMITATION,
    beta=TRUE_BETA,
)

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
csv_path = Path(__file__).parent / "data.csv"
save_dataset_to_csv(dataset, schema=SOCIAL_PRE_CHOICE_SCHEMA, path=csv_path)
print(f"Saved {len(dataset.subjects)} subjects to {csv_path}")

loaded = load_dataset_from_csv(csv_path, schema=SOCIAL_PRE_CHOICE_SCHEMA)

# ── 5. Fit each subject with MLE ───────────────────────────────────────────
mle_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_SHARED,
    backend="mle",
    mle_config=MleOptimizerConfig(n_restarts=10, seed=0),
)

print(
    f"\n{'Subject':<12} "
    f"{'True as':>8} {'Fit as':>8} "
    f"{'True aoo':>9} {'Fit aoo':>9} "
    f"{'True aoa':>9} {'Fit aoa':>9} "
    f"{'True wi':>8} {'Fit wi':>8} "
    f"{'True b':>7} {'Fit b':>7} "
    f"{'LL':>10}"
)
print("-" * 110)

for subject in loaded.subjects:
    result = fit(mle_config, kernel, subject, SOCIAL_PRE_CHOICE_SCHEMA)
    cp = result.constrained_params
    print(
        f"{subject.subject_id:<12} "
        f"{TRUE_ALPHA_SELF:>8.3f} {cp['alpha_self']:>8.3f} "
        f"{TRUE_ALPHA_OTHER_OUTCOME:>9.3f} {cp['alpha_other_outcome']:>9.3f} "
        f"{TRUE_ALPHA_OTHER_ACTION:>9.3f} {cp['alpha_other_action']:>9.3f} "
        f"{TRUE_W_IMITATION:>8.3f} {cp['w_imitation']:>8.3f} "
        f"{TRUE_BETA:>7.3f} {cp['beta']:>7.3f} "
        f"{result.log_likelihood:>10.2f}"
    )

print("\nDone.")
