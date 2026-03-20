"""Simulate asocial asymmetric RL agents on a stationary bandit, then recover
parameters with hierarchical Stan (NUTS) inference.

Per-subject parameters are sampled from a population distribution:
  alpha_pos ~ sigmoid(Normal(mu_alpha_pos_z, sd_alpha_pos_z))
  alpha_neg ~ sigmoid(Normal(mu_alpha_neg_z, sd_alpha_neg_z))
  beta      ~ softplus(Normal(mu_beta_z, sd_beta_z))

Requires: cmdstanpy and a working CmdStan installation.
"""

from pathlib import Path

import numpy as np
from scipy.special import expit as sigmoid_vec
from scipy.special import logit as inv_sigmoid_vec

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.bayes.stan import AsocialRlAsymmetricStanAdapter, StanFitConfig
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.io import save_dataset_to_csv
from comp_model.models.kernels import AsocialRlAsymmetricKernel, AsocialRlAsymmetricParams
from comp_model.runtime import SimulationConfig, simulate_dataset
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec


def softplus_vec(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(x))


def inv_softplus_vec(x: np.ndarray) -> np.ndarray:
    return np.log(np.expm1(x))


# ── 1. Define task ──────────────────────────────────────────────────────────
N_ACTIONS = 2
N_TRIALS = 200
N_SUBJECTS = 15

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

# ── 2. Sample ground-truth parameters from a population distribution ───────
kernel = AsocialRlAsymmetricKernel()

TRUE_MU_ALPHA_POS_Z = float(inv_sigmoid_vec(0.5))  # logit(0.5) = 0.0
TRUE_SD_ALPHA_POS_Z = 0.5
TRUE_MU_ALPHA_NEG_Z = float(inv_sigmoid_vec(0.2))  # logit(0.2) ≈ -1.386
TRUE_SD_ALPHA_NEG_Z = 0.5
TRUE_MU_BETA_Z = float(inv_softplus_vec(np.array(3.0)))
TRUE_SD_BETA_Z = 0.3

rng = np.random.default_rng(123)
true_alpha_pos_z = rng.normal(TRUE_MU_ALPHA_POS_Z, TRUE_SD_ALPHA_POS_Z, size=N_SUBJECTS)
true_alpha_neg_z = rng.normal(TRUE_MU_ALPHA_NEG_Z, TRUE_SD_ALPHA_NEG_Z, size=N_SUBJECTS)
true_beta_z = rng.normal(TRUE_MU_BETA_Z, TRUE_SD_BETA_Z, size=N_SUBJECTS)

true_alpha_pos = sigmoid_vec(true_alpha_pos_z)
true_alpha_neg = sigmoid_vec(true_alpha_neg_z)
true_betas = softplus_vec(true_beta_z)

# ── 3. Simulate dataset ────────────────────────────────────────────────────
params_per_subject = {
    f"sub_{i:02d}": AsocialRlAsymmetricParams(
        alpha_pos=float(true_alpha_pos[i]),
        alpha_neg=float(true_alpha_neg[i]),
        beta=float(true_betas[i]),
    )
    for i in range(N_SUBJECTS)
}

print("Ground-truth per-subject parameters:")
for sid, p in params_per_subject.items():
    print(f"  {sid}: alpha_pos={p.alpha_pos:.3f}, alpha_neg={p.alpha_neg:.3f}, beta={p.beta:.3f}")

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
print(f"\nSaved {len(dataset.subjects)} subjects to {csv_path}")

# ── 5. Fit with Stan ───────────────────────────────────────────────────────
stan_config = InferenceConfig(
    hierarchy=HierarchyStructure.STUDY_SUBJECT,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
)

adapter = AsocialRlAsymmetricStanAdapter()
result = fit(stan_config, kernel, dataset, ASOCIAL_BANDIT_SCHEMA, adapter=adapter)

# ── 6. Report results ──────────────────────────────────────────────────────
print(f"\nModel: {result.model_id}")
print(f"Hierarchy: {result.hierarchy}")

true_mu_alpha_pos = float(sigmoid_vec(TRUE_MU_ALPHA_POS_Z))
true_mu_alpha_neg = float(sigmoid_vec(TRUE_MU_ALPHA_NEG_Z))
true_mu_beta = float(softplus_vec(np.array(TRUE_MU_BETA_Z)))

if "alpha_pos_pop" in result.posterior_samples:
    print("\nPopulation posterior means (constrained):")
    print(
        f"  alpha_pos: {np.mean(result.posterior_samples['alpha_pos_pop']):.3f}"
        f"  (true: {true_mu_alpha_pos:.3f})"
    )
    print(
        f"  alpha_neg: {np.mean(result.posterior_samples['alpha_neg_pop']):.3f}"
        f"  (true: {true_mu_alpha_neg:.3f})"
    )
    print(
        f"  beta:      {np.mean(result.posterior_samples['beta_pop']):.3f}"
        f"  (true: {true_mu_beta:.3f})"
    )

if "alpha_pos" in result.posterior_samples:
    ap_all = result.posterior_samples["alpha_pos"]
    an_all = result.posterior_samples["alpha_neg"]
    beta_all = result.posterior_samples["beta"]
    print(
        f"\n{'Subject':<12} {'True a+':>8} {'Post a+':>10} "
        f"{'True a-':>8} {'Post a-':>10} {'True b':>8} {'Post b':>10}"
    )
    print("-" * 72)
    for i in range(N_SUBJECTS):
        sid = f"sub_{i:02d}"
        print(
            f"{sid:<12} {true_alpha_pos[i]:>8.3f} {np.mean(ap_all[:, i]):>10.3f} "
            f"{true_alpha_neg[i]:>8.3f} {np.mean(an_all[:, i]):>10.3f} "
            f"{true_betas[i]:>8.3f} {np.mean(beta_all[:, i]):>10.3f}"
        )

print("\nDone.")
