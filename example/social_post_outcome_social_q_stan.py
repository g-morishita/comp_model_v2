"""Simulate social Q-learning agents with post-outcome demonstrator observation,
then recover parameters with hierarchical Stan (NUTS) inference.

Ground-truth: alpha_self=0.3, alpha_other=0.2, beta=2.0
Requires: cmdstanpy and a working CmdStan installation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from comp_model.environments import SocialBanditEnvironment, StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.bayes.stan import SocialObservedOutcomeQStanAdapter, StanFitConfig
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.io import save_dataset_to_csv
from comp_model.models.kernels import SocialObservedOutcomeQKernel, SocialQParams
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
stan_config = InferenceConfig(
    hierarchy=HierarchyStructure.STUDY_SUBJECT,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
)

adapter = SocialObservedOutcomeQStanAdapter()
result = fit(stan_config, kernel, dataset, SOCIAL_POST_OUTCOME_SCHEMA, adapter=adapter)

# ── 6. Report results ──────────────────────────────────────────────────────
print(f"\nModel: {result.model_id}")
print(f"Hierarchy: {result.hierarchy}")

if "alpha_self_pop" in result.posterior_samples:
    alpha_self_pop = result.posterior_samples["alpha_self_pop"]
    alpha_other_pop = result.posterior_samples["alpha_other_pop"]
    beta_pop = result.posterior_samples["beta_pop"]
    print("\nPopulation posterior means (constrained):")
    print(f"  alpha_self:  {np.mean(alpha_self_pop):.3f}  (true: {TRUE_ALPHA_SELF:.3f})")
    print(f"  alpha_other: {np.mean(alpha_other_pop):.3f}  (true: {TRUE_ALPHA_OTHER:.3f})")
    print(f"  beta:        {np.mean(beta_pop):.3f}  (true: {TRUE_BETA:.3f})")

if "alpha_self" in result.posterior_samples:
    alpha_self_all = result.posterior_samples["alpha_self"]
    alpha_other_all = result.posterior_samples["alpha_other"]
    beta_all = result.posterior_samples["beta"]
    print(f"\n{'Subject':<12} {'True α_s':>8} {'Post α_s':>10} "
          f"{'True α_o':>8} {'Post α_o':>10} "
          f"{'True β':>8} {'Post β':>10}")
    print("-" * 72)
    for i in range(N_SUBJECTS):
        sid = f"sub_{i:02d}"
        print(
            f"{sid:<12} {TRUE_ALPHA_SELF:>8.3f} {np.mean(alpha_self_all[:, i]):>10.3f} "
            f"{TRUE_ALPHA_OTHER:>8.3f} {np.mean(alpha_other_all[:, i]):>10.3f} "
            f"{TRUE_BETA:>8.3f} {np.mean(beta_all[:, i]):>10.3f}"
        )

print("\nDone.")
