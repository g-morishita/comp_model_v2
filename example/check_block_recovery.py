"""Check whether increasing blocks improves parameter recovery."""

import numpy as np

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.models.kernels import AsocialQLearningKernel, QParams
from comp_model.runtime import SimulationConfig, simulate_dataset
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec

N_ACTIONS = 2
REWARD_PROBS = (0.8, 0.2)
TRUE_ALPHA = 0.3
TRUE_BETA = 2.0
N_SUBJECTS = 20
TRIALS_PER_BLOCK = 50

kernel = AsocialQLearningKernel()
true_params = QParams(alpha=TRUE_ALPHA, beta=TRUE_BETA)

mle_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_SHARED,
    backend="mle",
    mle_config=MleOptimizerConfig(n_restarts=10, seed=0),
)

for n_blocks in [1, 2, 4, 8]:
    total_trials = n_blocks * TRIALS_PER_BLOCK
    task = TaskSpec(
        task_id="test",
        blocks=tuple(
            BlockSpec(
                condition="learning",
                n_trials=TRIALS_PER_BLOCK,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": N_ACTIONS},
            )
            for _ in range(n_blocks)
        ),
    )

    params_per_subject = {f"sub_{i:02d}": true_params for i in range(N_SUBJECTS)}
    dataset = simulate_dataset(
        task=task,
        env_factory=lambda: StationaryBanditEnvironment(
            n_actions=N_ACTIONS, reward_probs=REWARD_PROBS
        ),
        kernel=kernel,
        params_per_subject=params_per_subject,
        config=SimulationConfig(seed=42),
    )

    alphas, betas = [], []
    for subject in dataset.subjects:
        result = fit(mle_config, kernel, subject, ASOCIAL_BANDIT_SCHEMA)
        alphas.append(result.constrained_params["alpha"])
        betas.append(result.constrained_params["beta"])

    alphas, betas = np.array(alphas), np.array(betas)
    print(
        f"Blocks={n_blocks:>2}  Trials={total_trials:>4}  |  "
        f"α: mean={np.mean(alphas):.3f} std={np.std(alphas):.3f}  |  "
        f"β: mean={np.mean(betas):.3f} std={np.std(betas):.3f}"
    )

print(f"\nTrue values: alpha={TRUE_ALPHA}, beta={TRUE_BETA}")
