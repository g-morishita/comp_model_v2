"""Simulate mixture social RL agents and recover parameters with hierarchical Stan.

Per-subject parameters are sampled from a population distribution:
  alpha_self          ~ sigmoid(Normal(mu_alpha_self_z,          sd_alpha_self_z))
  alpha_other_outcome ~ sigmoid(Normal(mu_alpha_other_outcome_z, sd_alpha_other_outcome_z))
  alpha_other_action  ~ sigmoid(Normal(mu_alpha_other_action_z,  sd_alpha_other_action_z))
  w_imitation         ~ sigmoid(Normal(mu_w_imitation_z,         sd_w_imitation_z))
  beta                ~ softplus(Normal(mu_beta_z,                sd_beta_z))

Requires: cmdstanpy and a working CmdStan installation.

Usage:
    uv run python example/social_rl_self_reward_demo_mixture/stan.py
"""

from pathlib import Path

import numpy as np
from scipy.special import expit as sigmoid_vec
from scipy.special import logit as inv_sigmoid_vec

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


def softplus_vec(x: np.ndarray) -> np.ndarray:
    """Apply softplus element-wise: log(1 + exp(x))."""
    return np.log1p(np.exp(x))


def inv_softplus_vec(x: np.ndarray) -> np.ndarray:
    """Invert softplus element-wise: log(exp(x) - 1)."""
    return np.log(np.expm1(x))


# ── 1. Define task ──────────────────────────────────────────────────────────
N_ACTIONS = 2
N_TRIALS = 200
N_SUBJECTS = 15
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

# ── 2. Sample ground-truth parameters from a population distribution ───────
kernel = SocialRlSelfRewardDemoMixtureKernel()

TRUE_MU_ALPHA_SELF_Z = float(inv_sigmoid_vec(0.3))
TRUE_SD_ALPHA_SELF_Z = 0.5
TRUE_MU_ALPHA_OTHER_OUTCOME_Z = float(inv_sigmoid_vec(0.2))
TRUE_SD_ALPHA_OTHER_OUTCOME_Z = 0.5
TRUE_MU_ALPHA_OTHER_ACTION_Z = float(inv_sigmoid_vec(0.4))
TRUE_SD_ALPHA_OTHER_ACTION_Z = 0.5
TRUE_MU_W_IMITATION_Z = float(inv_sigmoid_vec(0.3))
TRUE_SD_W_IMITATION_Z = 0.5
TRUE_MU_BETA_Z = float(inv_softplus_vec(np.array(2.0)))
TRUE_SD_BETA_Z = 0.5

rng = np.random.default_rng(123)
true_alpha_self_z = rng.normal(TRUE_MU_ALPHA_SELF_Z, TRUE_SD_ALPHA_SELF_Z, size=N_SUBJECTS)
true_alpha_other_outcome_z = rng.normal(
    TRUE_MU_ALPHA_OTHER_OUTCOME_Z, TRUE_SD_ALPHA_OTHER_OUTCOME_Z, size=N_SUBJECTS
)
true_alpha_other_action_z = rng.normal(
    TRUE_MU_ALPHA_OTHER_ACTION_Z, TRUE_SD_ALPHA_OTHER_ACTION_Z, size=N_SUBJECTS
)
true_w_imitation_z = rng.normal(TRUE_MU_W_IMITATION_Z, TRUE_SD_W_IMITATION_Z, size=N_SUBJECTS)
true_beta_z = rng.normal(TRUE_MU_BETA_Z, TRUE_SD_BETA_Z, size=N_SUBJECTS)

true_alpha_selfs = sigmoid_vec(true_alpha_self_z)
true_alpha_other_outcomes = sigmoid_vec(true_alpha_other_outcome_z)
true_alpha_other_actions = sigmoid_vec(true_alpha_other_action_z)
true_w_imitations = sigmoid_vec(true_w_imitation_z)
true_betas = softplus_vec(true_beta_z)

# ── 3. Simulate dataset ────────────────────────────────────────────────────
params_per_subject = {
    f"sub_{i:02d}": SocialRlSelfRewardDemoMixtureParams(
        alpha_self=float(true_alpha_selfs[i]),
        alpha_other_outcome=float(true_alpha_other_outcomes[i]),
        alpha_other_action=float(true_alpha_other_actions[i]),
        w_imitation=float(true_w_imitations[i]),
        beta=float(true_betas[i]),
    )
    for i in range(N_SUBJECTS)
}

print("Ground-truth per-subject parameters:")
for sid, p in params_per_subject.items():
    print(
        f"  {sid}: alpha_self={p.alpha_self:.3f}, "
        f"alpha_other_outcome={p.alpha_other_outcome:.3f}, "
        f"alpha_other_action={p.alpha_other_action:.3f}, "
        f"w_imitation={p.w_imitation:.3f}, beta={p.beta:.3f}"
    )

true_mu_as = float(sigmoid_vec(np.array(TRUE_MU_ALPHA_SELF_Z)))
true_mu_aoo = float(sigmoid_vec(np.array(TRUE_MU_ALPHA_OTHER_OUTCOME_Z)))
true_mu_aoa = float(sigmoid_vec(np.array(TRUE_MU_ALPHA_OTHER_ACTION_Z)))
true_mu_wi = float(sigmoid_vec(np.array(TRUE_MU_W_IMITATION_Z)))
true_mu_b = float(softplus_vec(np.array(TRUE_MU_BETA_Z)))
print(
    f"Population: mu_alpha_self={true_mu_as:.3f}, "
    f"mu_alpha_other_outcome={true_mu_aoo:.3f}, "
    f"mu_alpha_other_action={true_mu_aoa:.3f}, "
    f"mu_w_imitation={true_mu_wi:.3f}, mu_beta={true_mu_b:.3f}"
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

adapter = SocialRlSelfRewardDemoMixtureStanAdapter()
result = fit(stan_config, kernel, dataset, SOCIAL_PRE_CHOICE_SCHEMA, adapter=adapter)

# ── 6. Report results ──────────────────────────────────────────────────────
print(f"\nModel: {result.model_id}")
print(f"Hierarchy: {result.hierarchy}")

pop_keys = [
    ("alpha_self_pop", "alpha_self", true_mu_as),
    ("alpha_other_outcome_pop", "alpha_other_outcome", true_mu_aoo),
    ("alpha_other_action_pop", "alpha_other_action", true_mu_aoa),
    ("w_imitation_pop", "w_imitation", true_mu_wi),
    ("beta_pop", "beta", true_mu_b),
]

print("\nPopulation posterior means (constrained):")
for pop_key, label, true_val in pop_keys:
    if pop_key in result.posterior_samples:
        est = float(np.mean(result.posterior_samples[pop_key]))
        print(f"  {label:<22}: {est:.3f}  (true: {true_val:.3f})")

subject_keys = [
    ("alpha_self", true_alpha_selfs),
    ("alpha_other_outcome", true_alpha_other_outcomes),
    ("alpha_other_action", true_alpha_other_actions),
    ("w_imitation", true_w_imitations),
    ("beta", true_betas),
]

if all(k in result.posterior_samples for k, _ in subject_keys):
    header = f"\n{'Subject':<12} " + " ".join(
        f"{'True ' + k[:6]:>8} {'Post ' + k[:6]:>10}" for k, _ in subject_keys
    )
    print(header)
    print("-" * (12 + 19 * len(subject_keys)))
    for i in range(N_SUBJECTS):
        sid = f"sub_{i:02d}"
        row = f"{sid:<12}"
        for key, true_vals in subject_keys:
            samples = result.posterior_samples[key]
            row += f" {true_vals[i]:>8.3f} {np.mean(samples[:, i]):>10.3f}"
        print(row)

print("\nDone.")
