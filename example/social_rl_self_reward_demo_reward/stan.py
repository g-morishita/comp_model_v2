"""Simulate social Q-learning agents with pre-choice demonstrator observation,
then recover parameters with hierarchical Stan (NUTS) inference.

Per-subject parameters are sampled from a population distribution:
  alpha_self  ~ sigmoid(Normal(mu_alpha_self_z, sd_alpha_self_z))
  alpha_other ~ sigmoid(Normal(mu_alpha_other_z, sd_alpha_other_z))
  beta        ~ softplus(Normal(mu_beta_z, sd_beta_z))

Requires: cmdstanpy and a working CmdStan installation.
"""

from pathlib import Path

import numpy as np
from scipy.special import expit as sigmoid_vec
from scipy.special import logit as inv_sigmoid_vec

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.bayes.stan import SocialRlSelfRewardDemoRewardStanAdapter, StanFitConfig
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.io import save_dataset_to_csv
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    QParams,
    SocialQParams,
    SocialRlSelfRewardDemoRewardKernel,
)
from comp_model.runtime import SimulationConfig, simulate_dataset
from comp_model.tasks import SOCIAL_PRE_CHOICE_SCHEMA, BlockSpec, TaskSpec


def softplus_vec(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(x))


def inv_softplus_vec(x: np.ndarray) -> np.ndarray:
    return np.log(np.expm1(x))


# ── 1. Define task ──────────────────────────────────────────────────────────
N_ACTIONS = 2
N_TRIALS = 200
N_SUBJECTS = 15
REWARD_PROBS = (0.8, 0.2)

task = TaskSpec(
    task_id="social_pre_choice",
    blocks=(
        BlockSpec(
            condition="social",
            n_trials=N_TRIALS,
            schema=SOCIAL_PRE_CHOICE_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
    ),
)

# ── 2. Sample ground-truth parameters from a population distribution ───────
kernel = SocialRlSelfRewardDemoRewardKernel()

TRUE_MU_ALPHA_SELF_Z = float(inv_sigmoid_vec(0.3))
TRUE_SD_ALPHA_SELF_Z = 0.5
TRUE_MU_ALPHA_OTHER_Z = float(inv_sigmoid_vec(0.2))
TRUE_SD_ALPHA_OTHER_Z = 0.5
TRUE_MU_BETA_Z = float(inv_softplus_vec(np.array(2.0)))
TRUE_SD_BETA_Z = 0.5

rng = np.random.default_rng(123)
true_alpha_self_z = rng.normal(TRUE_MU_ALPHA_SELF_Z, TRUE_SD_ALPHA_SELF_Z, size=N_SUBJECTS)
true_alpha_other_z = rng.normal(TRUE_MU_ALPHA_OTHER_Z, TRUE_SD_ALPHA_OTHER_Z, size=N_SUBJECTS)
true_beta_z = rng.normal(TRUE_MU_BETA_Z, TRUE_SD_BETA_Z, size=N_SUBJECTS)
true_alpha_selfs = sigmoid_vec(true_alpha_self_z)
true_alpha_others = sigmoid_vec(true_alpha_other_z)
true_betas = softplus_vec(true_beta_z)

# ── 3. Simulate dataset ────────────────────────────────────────────────────
params_per_subject = {
    f"sub_{i:02d}": SocialQParams(
        alpha_self=float(true_alpha_selfs[i]),
        alpha_other=float(true_alpha_others[i]),
        beta=float(true_betas[i]),
    )
    for i in range(N_SUBJECTS)
}

print("Ground-truth per-subject parameters:")
for sid, p in params_per_subject.items():
    print(
        f"  {sid}: alpha_self={p.alpha_self:.3f}, "
        f"alpha_other={p.alpha_other:.3f}, beta={p.beta:.3f}"
    )
true_mu_as = float(sigmoid_vec(TRUE_MU_ALPHA_SELF_Z))
true_mu_ao = float(sigmoid_vec(TRUE_MU_ALPHA_OTHER_Z))
true_mu_b = float(softplus_vec(np.array(TRUE_MU_BETA_Z)))
print(
    f"Population: mu_alpha_self={true_mu_as:.3f}, "
    f"mu_alpha_other={true_mu_ao:.3f}, mu_beta={true_mu_b:.3f}"
)

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

# ── 5. Fit with Stan ───────────────────────────────────────────────────────
stan_config = InferenceConfig(
    hierarchy=HierarchyStructure.STUDY_SUBJECT,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
)

adapter = SocialRlSelfRewardDemoRewardStanAdapter()
result = fit(stan_config, kernel, dataset, SOCIAL_PRE_CHOICE_SCHEMA, adapter=adapter)

# ── 6. Report results ──────────────────────────────────────────────────────
print(f"\nModel: {result.model_id}")
print(f"Hierarchy: {result.hierarchy}")

if "alpha_self_pop" in result.posterior_samples:
    alpha_self_pop = result.posterior_samples["alpha_self_pop"]
    alpha_other_pop = result.posterior_samples["alpha_other_pop"]
    beta_pop = result.posterior_samples["beta_pop"]
    print("\nPopulation posterior means (constrained):")
    print(f"  alpha_self:  {np.mean(alpha_self_pop):.3f}  (true: {true_mu_as:.3f})")
    print(f"  alpha_other: {np.mean(alpha_other_pop):.3f}  (true: {true_mu_ao:.3f})")
    print(f"  beta:        {np.mean(beta_pop):.3f}  (true: {true_mu_b:.3f})")

if "alpha_self" in result.posterior_samples:
    alpha_self_all = result.posterior_samples["alpha_self"]
    alpha_other_all = result.posterior_samples["alpha_other"]
    beta_all = result.posterior_samples["beta"]
    print(
        f"\n{'Subject':<12} {'True a_s':>8} {'Post a_s':>10} "
        f"{'True a_o':>8} {'Post a_o':>10} "
        f"{'True b':>8} {'Post b':>10}"
    )
    print("-" * 72)
    for i in range(N_SUBJECTS):
        sid = f"sub_{i:02d}"
        print(
            f"{sid:<12} {true_alpha_selfs[i]:>8.3f} {np.mean(alpha_self_all[:, i]):>10.3f} "
            f"{true_alpha_others[i]:>8.3f} {np.mean(alpha_other_all[:, i]):>10.3f} "
            f"{true_betas[i]:>8.3f} {np.mean(beta_all[:, i]):>10.3f}"
        )

print("\nDone.")
