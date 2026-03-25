"""Simulate asocial asymmetric RL agents on a stationary bandit with two
conditions (different reward probabilities), then recover condition-specific
parameters with per-subject MLE using SharedDeltaLayout.

Within-subject design: each subject experiences both conditions. The model
estimates a shared baseline parameter set plus additive deltas for the
non-baseline condition on the unconstrained scale.

Ground-truth:
  Condition "easy":   alpha_pos=0.5, alpha_neg=0.2, beta=3.0  (baseline)
  Condition "hard":   alpha_pos=0.3, alpha_neg=0.3, beta=2.0
"""

from pathlib import Path

from comp_model.data import Block, Dataset, SubjectData
from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.io import save_dataset_to_csv
from comp_model.models import SharedDeltaLayout
from comp_model.models.kernels import AsocialRlAsymmetricKernel, AsocialRlAsymmetricParams
from comp_model.models.kernels.transforms import get_transform
from comp_model.runtime import SimulationConfig, simulate_subject
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec

# ── 1. Define task with two conditions ──────────────────────────────────────
N_ACTIONS = 2
N_TRIALS = 100
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
    "easy": AsocialRlAsymmetricParams(alpha_pos=0.5, alpha_neg=0.2, beta=3.0),
    "hard": AsocialRlAsymmetricParams(alpha_pos=0.3, alpha_neg=0.3, beta=2.0),
}

kernel = AsocialRlAsymmetricKernel()

# ── 3. Set up condition-aware layout ────────────────────────────────────────
layout = SharedDeltaLayout(
    kernel_spec=kernel.spec(),
    conditions=("easy", "hard"),
    baseline_condition="easy",
)

print(f"Layout parameter keys: {layout.parameter_keys()}")
print(f"Number of params: {layout.n_params()}")

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
        env = StationaryBanditEnvironment(n_actions=N_ACTIONS, reward_probs=REWARD_PROBS[condition])
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
                schema_id=sub.blocks[0].schema_id,
                trials=sub.blocks[0].trials,
            )
        )
    subjects.append(SubjectData(subject_id=sid, blocks=tuple(blocks)))

dataset = Dataset(subjects=tuple(subjects))

# ── 5. Save to CSV ─────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "within_subject_data.csv"
save_dataset_to_csv(dataset, schema=ASOCIAL_BANDIT_SCHEMA, path=csv_path)
print(f"\nSaved {len(dataset.subjects)} subjects x {len(task.blocks)} blocks to {csv_path}")

# ── 6. Fit each subject with condition-aware MLE ───────────────────────────
mle_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_BLOCK_CONDITION,
    backend="mle",
    mle_config=MleOptimizerConfig(n_restarts=20, seed=0),
)

sigmoid = get_transform("sigmoid")
softplus = get_transform("softplus")

print(
    f"\n{'Subject':<10} {'Cond':<8} {'True a+':>8} {'Fit a+':>8} "
    f"{'True a-':>8} {'Fit a-':>8} {'True b':>8} {'Fit b':>8}"
)
print("-" * 78)

for subject in dataset.subjects:
    result = fit(mle_config, kernel, subject, ASOCIAL_BANDIT_SCHEMA, layout=layout)
    for condition in ("easy", "hard"):
        true_p = TRUE_PARAMS[condition]
        fit_p = result.params_by_condition[condition]
        fit_apos = sigmoid.forward(fit_p["alpha_pos"])
        fit_aneg = sigmoid.forward(fit_p["alpha_neg"])
        fit_beta = softplus.forward(fit_p["beta"])
        print(
            f"{subject.subject_id:<10} {condition:<8} "
            f"{true_p.alpha_pos:>8.3f} {fit_apos:>8.3f} "
            f"{true_p.alpha_neg:>8.3f} {fit_aneg:>8.3f} "
            f"{true_p.beta:>8.3f} {fit_beta:>8.3f}"
        )
    print(f"  -> LL={result.log_likelihood:.2f}, AIC={result.aic:.2f}, BIC={result.bic:.2f}")

print("\nDone.")
