"""Simulate mixture social RL agents with two social conditions, then recover
condition-specific parameters with Stan (NUTS) inference using SharedDeltaLayout.

Within-subject design: each subject experiences both conditions. The model
estimates a shared baseline parameter set plus additive deltas for the
non-baseline condition on the unconstrained scale.

Ground-truth:
  Condition "social_low"  (baseline): alpha_self=0.3, alpha_other_outcome=0.1,
                                       alpha_other_action=0.2, w_imitation=0.2, beta=2.0
  Condition "social_high":             alpha_self=0.3, alpha_other_outcome=0.4,
                                       alpha_other_action=0.6, w_imitation=0.5, beta=2.0

Requires: cmdstanpy and a working CmdStan installation.

Usage:
    uv run python example/social_rl_self_reward_demo_mixture/stan_within_subject.py
"""

from pathlib import Path

import numpy as np

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

# ── 1. Define task with two conditions ──────────────────────────────────────
N_ACTIONS = 3
N_TRIALS = 200
N_SUBJECTS = 5
REWARD_PROBS = (0.25, 0.5, 0.75)

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

# ── 2. Ground-truth parameters per condition ────────────────────────────────
TRUE_PARAMS = {
    "social_low": SocialRlSelfRewardDemoMixtureParams(
        alpha_self=0.3,
        alpha_other_outcome=0.1,
        alpha_other_action=0.2,
        w_imitation=0.2,
        beta=2.0,
    ),
    "social_high": SocialRlSelfRewardDemoMixtureParams(
        alpha_self=0.3,
        alpha_other_outcome=0.4,
        alpha_other_action=0.6,
        w_imitation=0.5,
        beta=2.0,
    ),
}

kernel = SocialRlSelfRewardDemoMixtureKernel()

# ── 3. Set up condition-aware layout ────────────────────────────────────────
layout = SharedDeltaLayout(
    kernel_spec=kernel.spec(),
    conditions=("social_low", "social_high"),
    baseline_condition="social_low",
)

print(f"Layout parameter keys: {layout.parameter_keys()}")

# ── 4. Simulate dataset ────────────────────────────────────────────────────
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
            params=TRUE_PARAMS[condition],
            config=SimulationConfig(seed=42 + i + block_idx * 100),
            subject_id=sid,
            demonstrator_kernel=AsocialQLearningKernel(),
            demonstrator_params=QParams(alpha=0.3, beta=20.0),
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

# ── 5. Save to CSV ─────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "within_subject_data.csv"
save_dataset_to_csv(dataset, schema=SOCIAL_PRE_CHOICE_SCHEMA, path=csv_path)
print(f"Saved {len(dataset.subjects)} subjects x {len(task.blocks)} blocks to {csv_path}")

# ── 6. Fit each subject with condition-aware Stan ───────────────────────────
stan_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_BLOCK_CONDITION,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
)

adapter = SocialRlSelfRewardDemoMixtureStanAdapter()

param_labels = [
    "alpha_self",
    "alpha_other_outcome",
    "alpha_other_action",
    "w_imitation",
    "beta",
]

print(
    f"\n{'Subject':<10} {'Cond':<14} "
    + " ".join(f"{'True ' + p[:5]:>8} {'Post ' + p[:5]:>10}" for p in param_labels)
)
print("-" * (10 + 14 + 19 * len(param_labels)))

for subject in dataset.subjects:
    result = fit(
        stan_config,
        kernel,
        subject,
        SOCIAL_PRE_CHOICE_SCHEMA,
        layout=layout,
        adapter=adapter,
    )

    # Each param is shape (n_draws, C) for SUBJECT_BLOCK_CONDITION
    for c_idx, condition in enumerate(layout.conditions):
        true_p = TRUE_PARAMS[condition]
        true_vals = [
            true_p.alpha_self,
            true_p.alpha_other_outcome,
            true_p.alpha_other_action,
            true_p.w_imitation,
            true_p.beta,
        ]
        post_vals = [float(np.mean(result.posterior_samples[p][:, c_idx])) for p in param_labels]
        row = f"{subject.subject_id:<10} {condition:<14} "
        row += " ".join(
            f"{t:>8.3f} {po:>10.3f}" for t, po in zip(true_vals, post_vals, strict=True)
        )
        print(row)

print("\nDone.")
