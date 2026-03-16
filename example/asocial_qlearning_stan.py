"""Simulate asocial Q-learning agents on a stationary bandit, then recover
parameters with hierarchical Stan (NUTS) inference.

Per-subject parameters are sampled from a population distribution:
  alpha ~ sigmoid(Normal(mu_alpha_z, sd_alpha_z))
  beta  ~ softplus(Normal(mu_beta_z, sd_beta_z))

Requires: cmdstanpy and a working CmdStan installation.
"""

from pathlib import Path

import numpy as np
from scipy.special import expit as sigmoid_vec
from scipy.special import logit as inv_sigmoid_vec

from comp_model.data import Dataset
from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.bayes.stan import AsocialQLearningStanAdapter, StanFitConfig
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.io import save_dataset_to_csv
from comp_model.models.kernels import AsocialQLearningKernel, QParams
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

# ── 2. Environment ─────────────────────────────────────────────────────────
REWARD_PROBS = (0.8, 0.2)

# ── 3. Sample ground-truth parameters from a population distribution ───────
kernel = AsocialQLearningKernel()

# Population-level ground truth (unconstrained scale)
TRUE_MU_ALPHA_Z = float(inv_sigmoid_vec(0.3))   # ≈ -0.847
TRUE_SD_ALPHA_Z = 0.5
TRUE_MU_BETA_Z = float(inv_softplus_vec(np.array(2.0)))   # ≈ 1.687
TRUE_SD_BETA_Z = 0.5

rng = np.random.default_rng(123)
true_alpha_z = rng.normal(TRUE_MU_ALPHA_Z, TRUE_SD_ALPHA_Z, size=N_SUBJECTS)
true_beta_z = rng.normal(TRUE_MU_BETA_Z, TRUE_SD_BETA_Z, size=N_SUBJECTS)
true_alphas = sigmoid_vec(true_alpha_z)
true_betas = softplus_vec(true_beta_z)

# ── 4. Simulate dataset ────────────────────────────────────────────────────
params_per_subject = {
    f"sub_{i:02d}": QParams(alpha=float(true_alphas[i]), beta=float(true_betas[i]))
    for i in range(N_SUBJECTS)
}

print("Ground-truth per-subject parameters:")
for sid, p in params_per_subject.items():
    print(f"  {sid}: alpha={p.alpha:.3f}, beta={p.beta:.3f}")
print(f"Population: mu_alpha={float(sigmoid_vec(TRUE_MU_ALPHA_Z)):.3f}, "
      f"mu_beta={float(softplus_vec(np.array(TRUE_MU_BETA_Z))):.3f}")

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
csv_path = Path(__file__).parent / "asocial_qlearning_data.csv"
save_dataset_to_csv(dataset, schema=ASOCIAL_BANDIT_SCHEMA, path=csv_path)
print(f"Saved {len(dataset.subjects)} subjects to {csv_path}")

# ── 6. Fit with Stan ───────────────────────────────────────────────────────
stan_config = InferenceConfig(
    hierarchy=HierarchyStructure.STUDY_SUBJECT,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
)

adapter = AsocialQLearningStanAdapter()
result = fit(stan_config, kernel, dataset, ASOCIAL_BANDIT_SCHEMA, adapter=adapter)

# ── 7. Report results ──────────────────────────────────────────────────────
print(f"\nModel: {result.model_id}")
print(f"Hierarchy: {result.hierarchy}")

# Population-level
true_mu_alpha = float(sigmoid_vec(TRUE_MU_ALPHA_Z))
true_mu_beta = float(softplus_vec(np.array(TRUE_MU_BETA_Z)))

if "alpha_pop" in result.posterior_samples:
    alpha_pop = result.posterior_samples["alpha_pop"]
    beta_pop = result.posterior_samples["beta_pop"]
    print(f"\nPopulation posterior means (constrained):")
    print(f"  alpha: {np.mean(alpha_pop):.3f}  (true: {true_mu_alpha:.3f})")
    print(f"  beta:  {np.mean(beta_pop):.3f}  (true: {true_mu_beta:.3f})")

# Per-subject (alpha/beta are already constrained in transformed parameters)
if "alpha" in result.posterior_samples:
    alpha_all = result.posterior_samples["alpha"]  # shape: (n_draws, N_SUBJECTS)
    beta_all = result.posterior_samples["beta"]
    print(f"\n{'Subject':<12} {'True α':>8} {'Post. α':>10} {'True β':>8} {'Post. β':>10}")
    print("-" * 52)
    for i in range(N_SUBJECTS):
        sid = f"sub_{i:02d}"
        print(
            f"{sid:<12} {true_alphas[i]:>8.3f} {np.mean(alpha_all[:, i]):>10.3f} "
            f"{true_betas[i]:>8.3f} {np.mean(beta_all[:, i]):>10.3f}"
        )

print("\nDone.")
