"""Simulate mixture social RL agents and recover parameters with per-subject
Bayesian Stan inference (no hierarchical pooling).

Each subject is fit independently with weakly informative priors
(SUBJECT_SHARED hierarchy). Useful when you do not want to assume subjects
are drawn from a shared population distribution.

Ground-truth: alpha_self=0.3, alpha_other_outcome=0.2, alpha_other_action=0.4,
              w_imitation=0.3, beta=2.0 (same for all subjects)

Requires: cmdstanpy and a working CmdStan installation.

Usage:
    uv run python example/social_rl_self_reward_demo_mixture/stan_no_pooling.py
"""

from pathlib import Path

import numpy as np

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.bayes.stan import (
    SocialRlSelfRewardDemoMixtureStanAdapter,
    StanFitConfig,
)
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.io import save_dataset_to_csv
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
N_TRIALS = 100
N_SUBJECTS = 5
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

# ── 4. Save to CSV ─────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "data.csv"
save_dataset_to_csv(dataset, schema=SOCIAL_PRE_CHOICE_SCHEMA, path=csv_path)
print(f"Saved {len(dataset.subjects)} subjects to {csv_path}")

# ── 5. Fit each subject independently with Stan ────────────────────────────
stan_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_SHARED,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
)

adapter = SocialRlSelfRewardDemoMixtureStanAdapter()

print(
    f"\n{'Subject':<12} "
    f"{'True as':>8} {'Post as':>10} "
    f"{'True aoo':>9} {'Post aoo':>11} "
    f"{'True aoa':>9} {'Post aoa':>11} "
    f"{'True wi':>8} {'Post wi':>10} "
    f"{'True b':>7} {'Post b':>9}"
)
print("-" * 110)

for subject in dataset.subjects:
    result = fit(stan_config, kernel, subject, SOCIAL_PRE_CHOICE_SCHEMA, adapter=adapter)
    ps = result.posterior_samples
    print(
        f"{subject.subject_id:<12} "
        f"{TRUE_ALPHA_SELF:>8.3f} {np.mean(ps['alpha_self']):>10.3f} "
        f"{TRUE_ALPHA_OTHER_OUTCOME:>9.3f} {np.mean(ps['alpha_other_outcome']):>11.3f} "
        f"{TRUE_ALPHA_OTHER_ACTION:>9.3f} {np.mean(ps['alpha_other_action']):>11.3f} "
        f"{TRUE_W_IMITATION:>8.3f} {np.mean(ps['w_imitation']):>10.3f} "
        f"{TRUE_BETA:>7.3f} {np.mean(ps['beta']):>9.3f}"
    )

print(
    f"\nTrue values: alpha_self={TRUE_ALPHA_SELF}, "
    f"alpha_other_outcome={TRUE_ALPHA_OTHER_OUTCOME}, "
    f"alpha_other_action={TRUE_ALPHA_OTHER_ACTION}, "
    f"w_imitation={TRUE_W_IMITATION}, beta={TRUE_BETA}"
)
print("Done.")
