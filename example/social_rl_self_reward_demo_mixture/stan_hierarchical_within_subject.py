"""Simulate mixture social RL agents with two within-subject conditions, then
recover parameters with hierarchical Stan (NUTS) inference using SharedDeltaLayout.

Combines population-level estimation with within-subject conditions:
  - Each subject experiences both "social_low" and "social_high" conditions
  - Per-subject parameters are drawn from a population distribution
  - The model uses shared baseline + delta parameterisation across conditions
  - Population-level mu/sd are estimated for both shared and delta parameters

Hierarchy: STUDY_SUBJECT_BLOCK_CONDITION
  Population:  mu/sd for each of alpha_self, alpha_other_outcome,
               alpha_other_action, w_imitation, beta (shared and delta)
  Subject:     all 5 parameters per condition (constrained)

Ground-truth:
  Condition "social_low"  (baseline): alpha_self=0.3, alpha_other_outcome=0.1,
                                       alpha_other_action=0.2, w_imitation=0.2, beta=2.0
  Condition "social_high":             alpha_self=0.3, alpha_other_outcome=0.4,
                                       alpha_other_action=0.6, w_imitation=0.5, beta=2.0

Requires: cmdstanpy and a working CmdStan installation.

Usage:
    uv run python example/social_rl_self_reward_demo_mixture/stan_hierarchical_within_subject.py
"""

from pathlib import Path

import numpy as np
from scipy.special import expit as sigmoid_vec
from scipy.special import logit as inv_sigmoid_vec

from comp_model.data import Block, Dataset, SubjectData
from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.bayes.stan import (
    SocialRlSelfRewardDemoMixtureStanAdapter,
    StanFitConfig,
)
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.io import save_dataset_to_csv
from comp_model.models import SharedDeltaLayout
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    QParams,
    SocialRlSelfRewardDemoMixtureKernel,
    SocialRlSelfRewardDemoMixtureParams,
)
from comp_model.runtime import SimulationConfig, simulate_subject
from comp_model.tasks import SOCIAL_PRE_CHOICE_SCHEMA, BlockSpec, TaskSpec


def softplus_vec(x: np.ndarray) -> np.ndarray:
    """Apply softplus element-wise: log(1 + exp(x))."""
    return np.log1p(np.exp(x))


def inv_softplus_vec(x: np.ndarray) -> np.ndarray:
    """Invert softplus element-wise: log(exp(x) - 1)."""
    return np.log(np.expm1(x))


# ── 1. Define task with two conditions ──────────────────────────────────────
N_ACTIONS = 2
N_TRIALS = 200
N_SUBJECTS = 15
REWARD_PROBS = (0.8, 0.2)

task = TaskSpec(
    task_id="social_mixture_within",
    blocks=(
        BlockSpec(
            condition="social_low",
            n_trials=N_TRIALS,
            schema=SOCIAL_PRE_CHOICE_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
        BlockSpec(
            condition="social_high",
            n_trials=N_TRIALS,
            schema=SOCIAL_PRE_CHOICE_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
    ),
)

# ── 2. Ground-truth population parameters ───────────────────────────────────
kernel = SocialRlSelfRewardDemoMixtureKernel()

# Baseline ("social_low") population means on unconstrained scale
TRUE_MU_ALPHA_SELF_SHARED_Z = float(inv_sigmoid_vec(0.3))
TRUE_SD_ALPHA_SELF_SHARED_Z = 0.3
TRUE_MU_ALPHA_OTHER_OUTCOME_SHARED_Z = float(inv_sigmoid_vec(0.1))
TRUE_SD_ALPHA_OTHER_OUTCOME_SHARED_Z = 0.3
TRUE_MU_ALPHA_OTHER_ACTION_SHARED_Z = float(inv_sigmoid_vec(0.2))
TRUE_SD_ALPHA_OTHER_ACTION_SHARED_Z = 0.3
TRUE_MU_W_IMITATION_SHARED_Z = float(inv_sigmoid_vec(0.2))
TRUE_SD_W_IMITATION_SHARED_Z = 0.3
TRUE_MU_BETA_SHARED_Z = float(inv_softplus_vec(np.array(2.0)))
TRUE_SD_BETA_SHARED_Z = 0.3

# Delta ("social_high" - "social_low") population means on unconstrained scale
TRUE_MU_ALPHA_OTHER_OUTCOME_DELTA_Z = float(inv_sigmoid_vec(0.4) - inv_sigmoid_vec(0.1))
TRUE_SD_ALPHA_OTHER_OUTCOME_DELTA_Z = 0.2
TRUE_MU_ALPHA_OTHER_ACTION_DELTA_Z = float(inv_sigmoid_vec(0.6) - inv_sigmoid_vec(0.2))
TRUE_SD_ALPHA_OTHER_ACTION_DELTA_Z = 0.2
TRUE_MU_W_IMITATION_DELTA_Z = float(inv_sigmoid_vec(0.5) - inv_sigmoid_vec(0.2))
TRUE_SD_W_IMITATION_DELTA_Z = 0.2
# alpha_self and beta do not differ across conditions: delta mean = 0
TRUE_MU_ALPHA_SELF_DELTA_Z = 0.0
TRUE_SD_ALPHA_SELF_DELTA_Z = 0.2
TRUE_MU_BETA_DELTA_Z = 0.0
TRUE_SD_BETA_DELTA_Z = 0.2

# ── 3. Sample per-subject parameters from population ────────────────────────
rng = np.random.default_rng(123)

alpha_self_shared_z = rng.normal(
    TRUE_MU_ALPHA_SELF_SHARED_Z, TRUE_SD_ALPHA_SELF_SHARED_Z, N_SUBJECTS
)
aoo_shared_z = rng.normal(
    TRUE_MU_ALPHA_OTHER_OUTCOME_SHARED_Z, TRUE_SD_ALPHA_OTHER_OUTCOME_SHARED_Z, N_SUBJECTS
)
aoa_shared_z = rng.normal(
    TRUE_MU_ALPHA_OTHER_ACTION_SHARED_Z, TRUE_SD_ALPHA_OTHER_ACTION_SHARED_Z, N_SUBJECTS
)
wi_shared_z = rng.normal(TRUE_MU_W_IMITATION_SHARED_Z, TRUE_SD_W_IMITATION_SHARED_Z, N_SUBJECTS)
beta_shared_z = rng.normal(TRUE_MU_BETA_SHARED_Z, TRUE_SD_BETA_SHARED_Z, N_SUBJECTS)

alpha_self_delta_z = rng.normal(TRUE_MU_ALPHA_SELF_DELTA_Z, TRUE_SD_ALPHA_SELF_DELTA_Z, N_SUBJECTS)
aoo_delta_z = rng.normal(
    TRUE_MU_ALPHA_OTHER_OUTCOME_DELTA_Z, TRUE_SD_ALPHA_OTHER_OUTCOME_DELTA_Z, N_SUBJECTS
)
aoa_delta_z = rng.normal(
    TRUE_MU_ALPHA_OTHER_ACTION_DELTA_Z, TRUE_SD_ALPHA_OTHER_ACTION_DELTA_Z, N_SUBJECTS
)
wi_delta_z = rng.normal(TRUE_MU_W_IMITATION_DELTA_Z, TRUE_SD_W_IMITATION_DELTA_Z, N_SUBJECTS)
beta_delta_z = rng.normal(TRUE_MU_BETA_DELTA_Z, TRUE_SD_BETA_DELTA_Z, N_SUBJECTS)

TRUE_PARAMS_PER_SUBJECT: dict[str, dict[str, SocialRlSelfRewardDemoMixtureParams]] = {}
for i in range(N_SUBJECTS):
    sid = f"sub_{i:02d}"
    TRUE_PARAMS_PER_SUBJECT[sid] = {
        "social_low": SocialRlSelfRewardDemoMixtureParams(
            alpha_self=float(sigmoid_vec(alpha_self_shared_z[i])),
            alpha_other_outcome=float(sigmoid_vec(aoo_shared_z[i])),
            alpha_other_action=float(sigmoid_vec(aoa_shared_z[i])),
            w_imitation=float(sigmoid_vec(wi_shared_z[i])),
            beta=float(softplus_vec(beta_shared_z[i])),
        ),
        "social_high": SocialRlSelfRewardDemoMixtureParams(
            alpha_self=float(sigmoid_vec(alpha_self_shared_z[i] + alpha_self_delta_z[i])),
            alpha_other_outcome=float(sigmoid_vec(aoo_shared_z[i] + aoo_delta_z[i])),
            alpha_other_action=float(sigmoid_vec(aoa_shared_z[i] + aoa_delta_z[i])),
            w_imitation=float(sigmoid_vec(wi_shared_z[i] + wi_delta_z[i])),
            beta=float(softplus_vec(beta_shared_z[i] + beta_delta_z[i])),
        ),
    }

# ── 4. Set up condition-aware layout ────────────────────────────────────────
layout = SharedDeltaLayout(
    kernel_spec=kernel.spec(),
    conditions=("social_low", "social_high"),
    baseline_condition="social_low",
)

print(f"Layout parameter keys: {layout.parameter_keys()}")

# ── 5. Print ground-truth population parameters ─────────────────────────────
print("\nGround-truth population parameters (baseline condition, constrained):")
print(f"  alpha_self:          {float(sigmoid_vec(TRUE_MU_ALPHA_SELF_SHARED_Z)):.3f}")
print(f"  alpha_other_outcome: {float(sigmoid_vec(TRUE_MU_ALPHA_OTHER_OUTCOME_SHARED_Z)):.3f}")
print(f"  alpha_other_action:  {float(sigmoid_vec(TRUE_MU_ALPHA_OTHER_ACTION_SHARED_Z)):.3f}")
print(f"  w_imitation:         {float(sigmoid_vec(TRUE_MU_W_IMITATION_SHARED_Z)):.3f}")
print(f"  beta:                {float(softplus_vec(np.array(TRUE_MU_BETA_SHARED_Z))):.3f}")

# ── 6. Simulate dataset ────────────────────────────────────────────────────
subjects = []
for i in range(N_SUBJECTS):
    sid = f"sub_{i:02d}"
    blocks = []
    for block_idx, block_spec in enumerate(task.blocks):
        condition = block_spec.condition
        sub = simulate_subject(
            task=TaskSpec(task_id="tmp", blocks=(block_spec,)),
            env=StationaryBanditEnvironment(n_actions=N_ACTIONS, reward_probs=REWARD_PROBS),
            kernel=kernel,
            params=TRUE_PARAMS_PER_SUBJECT[sid][condition],
            config=SimulationConfig(seed=42 + i * 1000 + block_idx),
            subject_id=sid,
            demonstrator_kernel=AsocialQLearningKernel(),
            demonstrator_params=QParams(alpha=0.0, beta=0.0),
        )
        blocks.append(
            Block(
                block_index=block_idx,
                condition=sub.blocks[0].condition,
                trials=sub.blocks[0].trials,
            )
        )
    subjects.append(SubjectData(subject_id=sid, blocks=tuple(blocks)))

dataset = Dataset(subjects=tuple(subjects))

# ── 7. Save to CSV ─────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "hierarchical_within_subject_data.csv"
save_dataset_to_csv(dataset, schema=SOCIAL_PRE_CHOICE_SCHEMA, path=csv_path)
print(f"\nSaved {len(dataset.subjects)} subjects x {len(task.blocks)} blocks to {csv_path}")

# ── 8. Fit with hierarchical condition-aware Stan ───────────────────────────
stan_config = InferenceConfig(
    hierarchy=HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
)

adapter = SocialRlSelfRewardDemoMixtureStanAdapter()
result = fit(stan_config, kernel, dataset, SOCIAL_PRE_CHOICE_SCHEMA, layout=layout, adapter=adapter)

# ── 9. Report results ────────────────────────────────────────────────────────
print(f"\nModel: {result.model_id}")
print(f"Hierarchy: {result.hierarchy}")

shared_pop_keys = [
    ("alpha_self_shared_pop", "alpha_self", float(sigmoid_vec(TRUE_MU_ALPHA_SELF_SHARED_Z))),
    (
        "alpha_other_outcome_shared_pop",
        "alpha_other_outcome",
        float(sigmoid_vec(TRUE_MU_ALPHA_OTHER_OUTCOME_SHARED_Z)),
    ),
    (
        "alpha_other_action_shared_pop",
        "alpha_other_action",
        float(sigmoid_vec(TRUE_MU_ALPHA_OTHER_ACTION_SHARED_Z)),
    ),
    ("w_imitation_shared_pop", "w_imitation", float(sigmoid_vec(TRUE_MU_W_IMITATION_SHARED_Z))),
    ("beta_shared_pop", "beta", float(softplus_vec(np.array(TRUE_MU_BETA_SHARED_Z)))),
]

print("\nPopulation posterior means (baseline condition, constrained):")
for pop_key, label, true_val in shared_pop_keys:
    if pop_key in result.posterior_samples:
        est = float(np.mean(result.posterior_samples[pop_key]))
        print(f"  {label:<22}: {est:.3f}  (true: {true_val:.3f})")

param_labels = [
    "alpha_self",
    "alpha_other_outcome",
    "alpha_other_action",
    "w_imitation",
    "beta",
]

if all(p in result.posterior_samples for p in param_labels):
    print(
        f"\n{'Subject':<10} {'Cond':<14} "
        + " ".join(f"{'True ' + p[:5]:>8} {'Post ' + p[:5]:>10}" for p in param_labels)
    )
    print("-" * (10 + 14 + 19 * len(param_labels)))
    for i in range(N_SUBJECTS):
        sid = f"sub_{i:02d}"
        for c_idx, condition in enumerate(layout.conditions):
            true_p = TRUE_PARAMS_PER_SUBJECT[sid][condition]
            true_vals = [
                true_p.alpha_self,
                true_p.alpha_other_outcome,
                true_p.alpha_other_action,
                true_p.w_imitation,
                true_p.beta,
            ]
            # shape: (n_draws, N_SUBJECTS, C)
            post_vals = [
                float(np.mean(result.posterior_samples[p][:, i, c_idx])) for p in param_labels
            ]
            row = f"{sid:<10} {condition:<14} "
            row += " ".join(
                f"{t:>8.3f} {po:>10.3f}" for t, po in zip(true_vals, post_vals, strict=True)
            )
            print(row)

print("\nDone.")
