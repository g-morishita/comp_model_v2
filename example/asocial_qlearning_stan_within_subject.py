"""Simulate asocial Q-learning agents on a stationary bandit with two
conditions (different reward probabilities), then recover condition-specific
parameters with Stan (NUTS) inference using SharedDeltaLayout.

Within-subject design: each subject experiences both conditions. The model
estimates a shared baseline parameter set plus additive deltas for the
non-baseline condition on the unconstrained scale.

Ground-truth:
  Condition "easy":   alpha=0.3, beta=2.0  (baseline)
  Condition "hard":   alpha=0.5, beta=1.0

Requires: cmdstanpy and a working CmdStan installation.
"""

from pathlib import Path

import numpy as np

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

# ── 1. Define task with two conditions ──────────────────────────────────────
N_ACTIONS = 2
N_TRIALS = 200
N_SUBJECTS = 5

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

# ── 2. Ground-truth parameters per condition ────────────────────────────────
TRUE_PARAMS = {
    "easy": QParams(alpha=0.3, beta=2.0),
    "hard": QParams(alpha=0.5, beta=1.0),
}

kernel = AsocialQLearningKernel()

# ── 3. Set up condition-aware layout ────────────────────────────────────────
layout = SharedDeltaLayout(
    kernel_spec=kernel.spec(),
    conditions=("easy", "hard"),
    baseline_condition="easy",
)

print(f"Layout parameter keys: {layout.parameter_keys()}")

# ── 4. Simulate dataset ────────────────────────────────────────────────────
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
        env = StationaryBanditEnvironment(
            n_actions=N_ACTIONS, reward_probs=REWARD_PROBS[condition]
        )
        sub = simulate_subject(
            task=TaskSpec(task_id="tmp", blocks=(block_spec,)),
            env=env,
            kernel=kernel,
            params=TRUE_PARAMS[condition],
            config=SimulationConfig(seed=42 + i + block_idx * 100),
            subject_id=sid,
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

# ── 5. Save to CSV ─────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "asocial_qlearning_within_subject_data.csv"
save_dataset_to_csv(dataset, schema=ASOCIAL_BANDIT_SCHEMA, path=csv_path)
print(f"Saved {len(dataset.subjects)} subjects × {len(task.blocks)} blocks to {csv_path}")

# ── 6. Fit single subject with condition-aware Stan ─────────────────────────
stan_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_BLOCK_CONDITION,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
)

adapter = AsocialQLearningStanAdapter()
sigmoid = get_transform("sigmoid")
softplus = get_transform("softplus")

print(f"\n{'Subject':<10} {'Cond':<8} {'True α':>8} {'Post. α':>10} "
      f"{'True β':>8} {'Post. β':>10}")
print("-" * 60)

for subject in dataset.subjects:
    result = fit(
        stan_config, kernel, subject, ASOCIAL_BANDIT_SCHEMA,
        layout=layout, adapter=adapter,
    )

    # alpha, beta are vectors of size C in posterior samples
    alpha_samples = result.posterior_samples["alpha"]  # (n_draws, C)
    beta_samples = result.posterior_samples["beta"]

    for c_idx, condition in enumerate(layout.conditions):
        true_p = TRUE_PARAMS[condition]
        post_alpha = float(np.mean(alpha_samples[:, c_idx]))
        post_beta = float(np.mean(beta_samples[:, c_idx]))
        print(
            f"{subject.subject_id:<10} {condition:<8} "
            f"{true_p.alpha:>8.3f} {post_alpha:>10.3f} "
            f"{true_p.beta:>8.3f} {post_beta:>10.3f}"
        )

print("\nDone.")
