"""Simulate social Q-learning agents with post-outcome demonstrator observation,
then recover parameters with hierarchical Stan (NUTS) inference.

Ground-truth: alpha_self=0.3, alpha_other=0.2, beta=2.0
Requires: cmdstanpy and a working CmdStan installation.
"""

from __future__ import annotations

from pathlib import Path

from comp_model.environments import SocialBanditEnvironment, StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.bayes.stan import AsocialQLearningStanAdapter, StanFitConfig
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.io import save_dataset_to_csv
from comp_model.models.kernels import SocialObservedOutcomeQKernel, SocialQParams
from comp_model.models.kernels.transforms import get_transform
from comp_model.runtime import SimulationConfig, simulate_dataset
from comp_model.tasks import SOCIAL_POST_OUTCOME_SCHEMA, BlockSpec, TaskSpec


# ── 1. Define task ──────────────────────────────────────────────────────────
N_ACTIONS = 2
N_TRIALS = 100
N_SUBJECTS = 5

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
TRUE_ALPHA_SELF = 0.3
TRUE_ALPHA_OTHER = 0.2
TRUE_BETA = 2.0
REWARD_PROBS = (0.8, 0.2)

kernel = SocialObservedOutcomeQKernel()
true_params = SocialQParams(
    alpha_self=TRUE_ALPHA_SELF, alpha_other=TRUE_ALPHA_OTHER, beta=TRUE_BETA
)

# ── 3. Simulate dataset ────────────────────────────────────────────────────
params_per_subject = {f"sub_{i:02d}": true_params for i in range(N_SUBJECTS)}

dataset = simulate_dataset(
    task=task,
    env_factory=lambda: SocialBanditEnvironment(
        inner=StationaryBanditEnvironment(n_actions=N_ACTIONS, reward_probs=REWARD_PROBS),
        demonstrator_policy=(0.5, 0.5),
    ),
    kernel=kernel,
    params_per_subject=params_per_subject,
    config=SimulationConfig(seed=42),
)

# ── 4. Save to CSV ─────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "social_post_outcome_data.csv"
save_dataset_to_csv(dataset, schema=SOCIAL_POST_OUTCOME_SCHEMA, path=csv_path)
print(f"Saved {len(dataset.subjects)} subjects to {csv_path}")

# ── 5. Fit with Stan ───────────────────────────────────────────────────────
# NOTE: There is currently only an AsocialQLearningStanAdapter.
# A social Stan adapter and program would need to be implemented for this to
# work end-to-end. This script demonstrates the intended API shape.
print("\nWARNING: No social Stan adapter is currently implemented.")
print("This script shows the intended API but cannot run Stan fitting yet.")
print("Use the MLE variant (social_post_outcome_social_q_mle.py) instead.")
print("\nDone.")
