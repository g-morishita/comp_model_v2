"""Simulate asocial Q-learning agents on a stationary bandit with two
within-subject conditions, then recover parameters with hierarchical Stan
(NUTS) inference using SharedDeltaLayout.

This combines population-level estimation with within-subject conditions:
  - Each subject experiences both "easy" and "hard" conditions
  - Per-subject parameters are drawn from a population distribution
  - The model uses shared baseline + delta parameterization across conditions
  - Population-level mu/sd are estimated for both shared and delta parameters

Hierarchy: STUDY_SUBJECT_BLOCK_CONDITION
  Population:  mu_alpha_shared_z, sd_alpha_shared_z, mu_beta_shared_z, sd_beta_shared_z,
               mu_alpha_delta_z, sd_alpha_delta_z, mu_beta_delta_z, sd_beta_delta_z
  Subject:     alpha[N][C], beta[N][C]  (constrained, per-condition)

Ground-truth:
  Condition "easy":   alpha=0.3, beta=2.0  (baseline)
  Condition "hard":   alpha=0.5, beta=1.0

Requires: cmdstanpy and a working CmdStan installation.
"""

from pathlib import Path

import numpy as np
from scipy.special import expit as sigmoid_vec
from scipy.special import logit as inv_sigmoid_vec

from comp_model.data import Block, Dataset, SubjectData
from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.bayes.stan import AsocialQLearningStanAdapter, StanFitConfig
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.io import save_dataset_to_csv
from comp_model.models import SharedDeltaLayout
from comp_model.models.kernels import AsocialQLearningKernel, QParams
from comp_model.models.kernels.transforms import get_transform
from comp_model.runtime import SimulationConfig, simulate_subject
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec


def softplus_vec(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(x))


def inv_softplus_vec(x: np.ndarray) -> np.ndarray:
    return np.log(np.expm1(x))


# -- 1. Define task with two conditions ----------------------------------------
N_ACTIONS = 2
N_TRIALS = 200
N_SUBJECTS = 15

task = TaskSpec(
    task_id="asocial_bandit_within",
    blocks=(
        BlockSpec(
            condition="easy",
            n_trials=N_TRIALS,
            schema=ASOCIAL_BANDIT_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
        BlockSpec(
            condition="hard",
            n_trials=N_TRIALS,
            schema=ASOCIAL_BANDIT_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
    ),
)

# -- 2. Ground-truth population parameters ------------------------------------
kernel = AsocialQLearningKernel()

# Population-level ground truth for baseline ("easy") on unconstrained scale
TRUE_MU_ALPHA_SHARED_Z = float(inv_sigmoid_vec(0.3))  # baseline alpha
TRUE_SD_ALPHA_SHARED_Z = 0.3
TRUE_MU_BETA_SHARED_Z = float(inv_softplus_vec(np.array(2.0)))  # baseline beta
TRUE_SD_BETA_SHARED_Z = 0.3

# Population-level ground truth for delta (hard - easy) on unconstrained scale
# easy: alpha=0.3 -> alpha_z = inv_sigmoid(0.3) ~ -0.847
# hard: alpha=0.5 -> alpha_z = inv_sigmoid(0.5) = 0.0
# delta_alpha_z = 0.0 - (-0.847) ~ 0.847
TRUE_MU_ALPHA_DELTA_Z = float(inv_sigmoid_vec(0.5) - inv_sigmoid_vec(0.3))
TRUE_SD_ALPHA_DELTA_Z = 0.2

# easy: beta=2.0 -> beta_z = inv_softplus(2.0) ~ 1.687
# hard: beta=1.0 -> beta_z = inv_softplus(1.0) ~ 0.541
# delta_beta_z = 0.541 - 1.687 ~ -1.145
TRUE_MU_BETA_DELTA_Z = float(inv_softplus_vec(np.array(1.0)) - inv_softplus_vec(np.array(2.0)))
TRUE_SD_BETA_DELTA_Z = 0.2

# -- 3. Sample per-subject parameters from population -------------------------
rng = np.random.default_rng(123)

# Shared (baseline) parameters
true_alpha_shared_z = rng.normal(TRUE_MU_ALPHA_SHARED_Z, TRUE_SD_ALPHA_SHARED_Z, size=N_SUBJECTS)
true_beta_shared_z = rng.normal(TRUE_MU_BETA_SHARED_Z, TRUE_SD_BETA_SHARED_Z, size=N_SUBJECTS)

# Delta parameters
true_alpha_delta_z = rng.normal(TRUE_MU_ALPHA_DELTA_Z, TRUE_SD_ALPHA_DELTA_Z, size=N_SUBJECTS)
true_beta_delta_z = rng.normal(TRUE_MU_BETA_DELTA_Z, TRUE_SD_BETA_DELTA_Z, size=N_SUBJECTS)

# Constrained per-subject, per-condition parameters
sigmoid = get_transform("sigmoid")
softplus = get_transform("softplus")

true_alphas_easy = sigmoid_vec(true_alpha_shared_z)
true_betas_easy = softplus_vec(true_beta_shared_z)
true_alphas_hard = sigmoid_vec(true_alpha_shared_z + true_alpha_delta_z)
true_betas_hard = softplus_vec(true_beta_shared_z + true_beta_delta_z)

TRUE_PARAMS_PER_SUBJECT: dict[str, dict[str, QParams]] = {}
for i in range(N_SUBJECTS):
    sid = f"sub_{i:02d}"
    TRUE_PARAMS_PER_SUBJECT[sid] = {
        "easy": QParams(alpha=float(true_alphas_easy[i]), beta=float(true_betas_easy[i])),
        "hard": QParams(alpha=float(true_alphas_hard[i]), beta=float(true_betas_hard[i])),
    }

# -- 4. Set up condition-aware layout -----------------------------------------
layout = SharedDeltaLayout(
    kernel_spec=kernel.spec(),
    conditions=("easy", "hard"),
    baseline_condition="easy",
)

print(f"Layout parameter keys: {layout.parameter_keys()}")

# -- 5. Print ground-truth parameters -----------------------------------------
print("\nGround-truth population parameters (constrained):")
print(f"  Baseline alpha (easy): {float(sigmoid_vec(TRUE_MU_ALPHA_SHARED_Z)):.3f}")
print(f"  Baseline beta  (easy): {float(softplus_vec(np.array(TRUE_MU_BETA_SHARED_Z))):.3f}")
print(
    f"  Hard alpha (via delta): "
    f"{float(sigmoid_vec(TRUE_MU_ALPHA_SHARED_Z + TRUE_MU_ALPHA_DELTA_Z)):.3f}"
)
print(
    f"  Hard beta  (via delta): "
    f"{float(softplus_vec(np.array(TRUE_MU_BETA_SHARED_Z + TRUE_MU_BETA_DELTA_Z))):.3f}"
)

print(f"\n{'Subject':<10} {'Cond':<8} {'True a':>8} {'True b':>8}")
print("-" * 38)
for sid, cond_params in TRUE_PARAMS_PER_SUBJECT.items():
    for cond in ("easy", "hard"):
        p = cond_params[cond]
        print(f"{sid:<10} {cond:<8} {p.alpha:>8.3f} {p.beta:>8.3f}")

# -- 6. Simulate dataset ------------------------------------------------------
REWARD_PROBS = {
    "easy": (0.8, 0.2),
    "hard": (0.6, 0.4),
}

subjects = []
for i in range(N_SUBJECTS):
    sid = f"sub_{i:02d}"
    blocks = []
    for block_idx, block_spec in enumerate(task.blocks):
        condition = block_spec.condition
        env = StationaryBanditEnvironment(n_actions=N_ACTIONS, reward_probs=REWARD_PROBS[condition])
        sub = simulate_subject(
            task=TaskSpec(task_id="tmp", blocks=(block_spec,)),
            env=env,
            kernel=kernel,
            params=TRUE_PARAMS_PER_SUBJECT[sid][condition],
            config=SimulationConfig(seed=42 + i * 1000 + block_idx),
            subject_id=sid,
        )
        blocks.append(
            Block(
                block_index=block_idx,
                condition=sub.blocks[0].condition,
                schema_id=sub.blocks[0].schema_id,
                trials=sub.blocks[0].trials,
            )
        )
    subjects.append(SubjectData(subject_id=sid, blocks=tuple(blocks)))

dataset = Dataset(subjects=tuple(subjects))

# -- 7. Save to CSV -----------------------------------------------------------
csv_path = Path(__file__).parent / "hierarchical_within_subject_data.csv"
save_dataset_to_csv(dataset, schema=ASOCIAL_BANDIT_SCHEMA, path=csv_path)
print(f"\nSaved {len(dataset.subjects)} subjects x {len(task.blocks)} blocks to {csv_path}")

# -- 8. Fit with hierarchical condition-aware Stan -----------------------------
stan_config = InferenceConfig(
    hierarchy=HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
)

adapter = AsocialQLearningStanAdapter()
result = fit(stan_config, kernel, dataset, ASOCIAL_BANDIT_SCHEMA, layout=layout, adapter=adapter)

# -- 9. Report results ---------------------------------------------------------
print(f"\nModel: {result.model_id}")
print(f"Hierarchy: {result.hierarchy}")

# Population-level: shared baseline (constrained)
if "alpha_shared_pop" in result.posterior_samples:
    alpha_shared_pop = result.posterior_samples["alpha_shared_pop"]
    beta_shared_pop = result.posterior_samples["beta_shared_pop"]
    true_pop_alpha = float(sigmoid_vec(TRUE_MU_ALPHA_SHARED_Z))
    true_pop_beta = float(softplus_vec(np.array(TRUE_MU_BETA_SHARED_Z)))
    print("\nPopulation posterior means (baseline condition, constrained):")
    print(f"  alpha_shared: {np.mean(alpha_shared_pop):.3f}  (true: {true_pop_alpha:.3f})")
    print(f"  beta_shared:  {np.mean(beta_shared_pop):.3f}  (true: {true_pop_beta:.3f})")

# Population-level: unconstrained mu/sd
if "mu_alpha_shared_z" in result.posterior_samples:
    print("\nPopulation posterior means (unconstrained scale):")
    for name in [
        "mu_alpha_shared_z",
        "sd_alpha_shared_z",
        "mu_beta_shared_z",
        "sd_beta_shared_z",
    ]:
        post = result.posterior_samples[name]
        true_val = {
            "mu_alpha_shared_z": TRUE_MU_ALPHA_SHARED_Z,
            "sd_alpha_shared_z": TRUE_SD_ALPHA_SHARED_Z,
            "mu_beta_shared_z": TRUE_MU_BETA_SHARED_Z,
            "sd_beta_shared_z": TRUE_SD_BETA_SHARED_Z,
        }[name]
        print(f"  {name}: {np.mean(post):.3f}  (true: {true_val:.3f})")

if "mu_alpha_delta_z" in result.posterior_samples:
    print("\nPopulation delta posterior means (unconstrained scale):")
    for name in [
        "mu_alpha_delta_z",
        "sd_alpha_delta_z",
        "mu_beta_delta_z",
        "sd_beta_delta_z",
    ]:
        post = result.posterior_samples[name]
        true_val = {
            "mu_alpha_delta_z": TRUE_MU_ALPHA_DELTA_Z,
            "sd_alpha_delta_z": TRUE_SD_ALPHA_DELTA_Z,
            "mu_beta_delta_z": TRUE_MU_BETA_DELTA_Z,
            "sd_beta_delta_z": TRUE_SD_BETA_DELTA_Z,
        }[name]
        # delta params may be vectors of size C-1; take first element
        post_mean = float(np.mean(post))
        print(f"  {name}: {post_mean:.3f}  (true: {true_val:.3f})")

# Per-subject, per-condition
if "alpha" in result.posterior_samples:
    # alpha shape: (n_draws, N_SUBJECTS, C)
    alpha_all = result.posterior_samples["alpha"]
    beta_all = result.posterior_samples["beta"]
    print(
        f"\n{'Subject':<10} {'Cond':<8} {'True a':>8} {'Post. a':>10} {'True b':>8} {'Post. b':>10}"
    )
    print("-" * 60)
    for i in range(N_SUBJECTS):
        sid = f"sub_{i:02d}"
        for c_idx, condition in enumerate(layout.conditions):
            true_p = TRUE_PARAMS_PER_SUBJECT[sid][condition]
            post_alpha = float(np.mean(alpha_all[:, i, c_idx]))
            post_beta = float(np.mean(beta_all[:, i, c_idx]))
            print(
                f"{sid:<10} {condition:<8} "
                f"{true_p.alpha:>8.3f} {post_alpha:>10.3f} "
                f"{true_p.beta:>8.3f} {post_beta:>10.3f}"
            )

print("\nDone.")
