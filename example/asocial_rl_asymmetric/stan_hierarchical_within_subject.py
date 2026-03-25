"""Simulate asocial asymmetric RL agents on a stationary bandit with two
within-subject conditions, then recover parameters with hierarchical Stan
(NUTS) inference using SharedDeltaLayout.

Hierarchy: STUDY_SUBJECT_BLOCK_CONDITION
  Population:  mu/sd for alpha_pos_shared_z, alpha_neg_shared_z, beta_shared_z
               mu/sd for alpha_pos_delta_z, alpha_neg_delta_z, beta_delta_z
  Subject:     alpha_pos[N][C], alpha_neg[N][C], beta[N][C]

Ground-truth:
  Condition "easy":   alpha_pos=0.5, alpha_neg=0.2, beta=3.0  (baseline)
  Condition "hard":   alpha_pos=0.3, alpha_neg=0.3, beta=2.0

Requires: cmdstanpy and a working CmdStan installation.
"""

from pathlib import Path

import numpy as np
from scipy.special import expit as sigmoid_vec
from scipy.special import logit as inv_sigmoid_vec

from comp_model.data import Block, Dataset, SubjectData
from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.bayes.stan import AsocialRlAsymmetricStanAdapter, StanFitConfig
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.io import save_dataset_to_csv
from comp_model.models import SharedDeltaLayout
from comp_model.models.kernels import AsocialRlAsymmetricKernel, AsocialRlAsymmetricParams
from comp_model.models.kernels.transforms import get_transform
from comp_model.runtime import SimulationConfig, simulate_subject
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec


def softplus_vec(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(x))


def inv_softplus_vec(x: np.ndarray) -> np.ndarray:
    return np.log(np.expm1(x))


# ── 1. Define task with two conditions ──────────────────────────────────────
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

# ── 2. Ground-truth population parameters ───────────────────────────────────
kernel = AsocialRlAsymmetricKernel()

# Baseline ("easy") population means on unconstrained scale
TRUE_MU_APOS_SHARED_Z = float(inv_sigmoid_vec(0.5))  # logit(0.5) = 0.0
TRUE_SD_APOS_SHARED_Z = 0.3
TRUE_MU_ANEG_SHARED_Z = float(inv_sigmoid_vec(0.2))  # logit(0.2) ≈ -1.386
TRUE_SD_ANEG_SHARED_Z = 0.3
TRUE_MU_BETA_SHARED_Z = float(inv_softplus_vec(np.array(3.0)))
TRUE_SD_BETA_SHARED_Z = 0.3

# Delta (hard - easy) population means on unconstrained scale
TRUE_MU_APOS_DELTA_Z = float(inv_sigmoid_vec(0.3) - inv_sigmoid_vec(0.5))
TRUE_SD_APOS_DELTA_Z = 0.2
TRUE_MU_ANEG_DELTA_Z = float(inv_sigmoid_vec(0.3) - inv_sigmoid_vec(0.2))
TRUE_SD_ANEG_DELTA_Z = 0.2
TRUE_MU_BETA_DELTA_Z = float(inv_softplus_vec(np.array(2.0)) - inv_softplus_vec(np.array(3.0)))
TRUE_SD_BETA_DELTA_Z = 0.2

# ── 3. Sample per-subject parameters from population ────────────────────────
rng = np.random.default_rng(123)

apos_shared_z = rng.normal(TRUE_MU_APOS_SHARED_Z, TRUE_SD_APOS_SHARED_Z, size=N_SUBJECTS)
aneg_shared_z = rng.normal(TRUE_MU_ANEG_SHARED_Z, TRUE_SD_ANEG_SHARED_Z, size=N_SUBJECTS)
beta_shared_z = rng.normal(TRUE_MU_BETA_SHARED_Z, TRUE_SD_BETA_SHARED_Z, size=N_SUBJECTS)
apos_delta_z = rng.normal(TRUE_MU_APOS_DELTA_Z, TRUE_SD_APOS_DELTA_Z, size=N_SUBJECTS)
aneg_delta_z = rng.normal(TRUE_MU_ANEG_DELTA_Z, TRUE_SD_ANEG_DELTA_Z, size=N_SUBJECTS)
beta_delta_z = rng.normal(TRUE_MU_BETA_DELTA_Z, TRUE_SD_BETA_DELTA_Z, size=N_SUBJECTS)

sigmoid = get_transform("sigmoid")
softplus = get_transform("softplus")

TRUE_PARAMS_PER_SUBJECT: dict[str, dict[str, AsocialRlAsymmetricParams]] = {}
for i in range(N_SUBJECTS):
    sid = f"sub_{i:02d}"
    TRUE_PARAMS_PER_SUBJECT[sid] = {
        "easy": AsocialRlAsymmetricParams(
            alpha_pos=float(sigmoid_vec(apos_shared_z[i])),
            alpha_neg=float(sigmoid_vec(aneg_shared_z[i])),
            beta=float(softplus_vec(beta_shared_z[i : i + 1])[0]),
        ),
        "hard": AsocialRlAsymmetricParams(
            alpha_pos=float(sigmoid_vec(apos_shared_z[i] + apos_delta_z[i])),
            alpha_neg=float(sigmoid_vec(aneg_shared_z[i] + aneg_delta_z[i])),
            beta=float(softplus_vec((beta_shared_z + beta_delta_z)[i : i + 1])[0]),
        ),
    }

# ── 4. Set up condition-aware layout ────────────────────────────────────────
layout = SharedDeltaLayout(
    kernel_spec=kernel.spec(),
    conditions=("easy", "hard"),
    baseline_condition="easy",
)

print(f"Layout parameter keys: {layout.parameter_keys()}")

# ── 5. Simulate dataset ─────────────────────────────────────────────────────
REWARD_PROBS = {"easy": (0.8, 0.2), "hard": (0.6, 0.4)}

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

# ── 6. Save to CSV ──────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "hierarchical_within_subject_data.csv"
save_dataset_to_csv(dataset, schema=ASOCIAL_BANDIT_SCHEMA, path=csv_path)
print(f"\nSaved {len(dataset.subjects)} subjects x {len(task.blocks)} blocks to {csv_path}")

# ── 7. Fit with hierarchical condition-aware Stan ────────────────────────────
stan_config = InferenceConfig(
    hierarchy=HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
)

adapter = AsocialRlAsymmetricStanAdapter()
result = fit(stan_config, kernel, dataset, ASOCIAL_BANDIT_SCHEMA, layout=layout, adapter=adapter)

# ── 8. Report results ────────────────────────────────────────────────────────
print(f"\nModel: {result.model_id}")
print(f"Hierarchy: {result.hierarchy}")

if "alpha_pos_shared_pop" in result.posterior_samples:
    print("\nPopulation posterior means (baseline condition, constrained):")
    for param, true_z, label in [
        ("alpha_pos_shared_pop", TRUE_MU_APOS_SHARED_Z, "alpha_pos"),
        ("alpha_neg_shared_pop", TRUE_MU_ANEG_SHARED_Z, "alpha_neg"),
        ("beta_shared_pop", TRUE_MU_BETA_SHARED_Z, "beta"),
    ]:
        post_mean = float(np.mean(result.posterior_samples[param]))
        true_constrained = (
            float(sigmoid_vec(true_z))
            if "alpha" in param
            else float(softplus_vec(np.array([true_z]))[0])
        )
        print(f"  {label}: {post_mean:.3f}  (true: {true_constrained:.3f})")

if "mu_alpha_pos_shared_z" in result.posterior_samples:
    print("\nPopulation posterior means (unconstrained scale):")
    truth = {
        "mu_alpha_pos_shared_z": TRUE_MU_APOS_SHARED_Z,
        "sd_alpha_pos_shared_z": TRUE_SD_APOS_SHARED_Z,
        "mu_alpha_neg_shared_z": TRUE_MU_ANEG_SHARED_Z,
        "sd_alpha_neg_shared_z": TRUE_SD_ANEG_SHARED_Z,
        "mu_beta_shared_z": TRUE_MU_BETA_SHARED_Z,
        "sd_beta_shared_z": TRUE_SD_BETA_SHARED_Z,
        "mu_alpha_pos_delta_z": TRUE_MU_APOS_DELTA_Z,
        "sd_alpha_pos_delta_z": TRUE_SD_APOS_DELTA_Z,
        "mu_alpha_neg_delta_z": TRUE_MU_ANEG_DELTA_Z,
        "sd_alpha_neg_delta_z": TRUE_SD_ANEG_DELTA_Z,
        "mu_beta_delta_z": TRUE_MU_BETA_DELTA_Z,
        "sd_beta_delta_z": TRUE_SD_BETA_DELTA_Z,
    }
    for name, true_val in truth.items():
        if name in result.posterior_samples:
            post_mean = float(np.mean(result.posterior_samples[name]))
            print(f"  {name}: {post_mean:.3f}  (true: {true_val:.3f})")

if "alpha_pos" in result.posterior_samples:
    ap_all = result.posterior_samples["alpha_pos"]  # (draws, N, C)
    an_all = result.posterior_samples["alpha_neg"]
    b_all = result.posterior_samples["beta"]
    print(
        f"\n{'Subject':<10} {'Cond':<8} {'True a+':>8} {'Post a+':>10} "
        f"{'True a-':>8} {'Post a-':>10} {'True b':>8} {'Post b':>10}"
    )
    print("-" * 80)
    for i in range(N_SUBJECTS):
        sid = f"sub_{i:02d}"
        for c_idx, condition in enumerate(layout.conditions):
            true_p = TRUE_PARAMS_PER_SUBJECT[sid][condition]
            print(
                f"{sid:<10} {condition:<8} "
                f"{true_p.alpha_pos:>8.3f} {float(np.mean(ap_all[:, i, c_idx])):>10.3f} "
                f"{true_p.alpha_neg:>8.3f} {float(np.mean(an_all[:, i, c_idx])):>10.3f} "
                f"{true_p.beta:>8.3f} {float(np.mean(b_all[:, i, c_idx])):>10.3f}"
            )

print("\nDone.")
