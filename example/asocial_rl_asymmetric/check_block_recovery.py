"""Check whether increasing blocks improves parameter recovery."""

import numpy as np

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.models.kernels import AsocialRlAsymmetricKernel, AsocialRlAsymmetricParams
from comp_model.runtime import SimulationConfig, simulate_dataset
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec

N_ACTIONS = 2
REWARD_PROBS = (0.8, 0.2)
TRUE_ALPHA_POS = 0.5
TRUE_ALPHA_NEG = 0.2
TRUE_BETA = 3.0
N_SUBJECTS = 20
TRIALS_PER_BLOCK = 50

kernel = AsocialRlAsymmetricKernel()
true_params = AsocialRlAsymmetricParams(
    alpha_pos=TRUE_ALPHA_POS, alpha_neg=TRUE_ALPHA_NEG, beta=TRUE_BETA
)

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

    apos_list, aneg_list, betas = [], [], []
    for subject in dataset.subjects:
        result = fit(mle_config, kernel, subject, ASOCIAL_BANDIT_SCHEMA)
        apos_list.append(result.constrained_params["alpha_pos"])
        aneg_list.append(result.constrained_params["alpha_neg"])
        betas.append(result.constrained_params["beta"])

    apos_arr = np.array(apos_list)
    aneg_arr = np.array(aneg_list)
    betas_arr = np.array(betas)
    print(
        f"Blocks={n_blocks:>2}  Trials={total_trials:>4}  |  "
        f"a+: mean={np.mean(apos_arr):.3f} std={np.std(apos_arr):.3f}  |  "
        f"a-: mean={np.mean(aneg_arr):.3f} std={np.std(aneg_arr):.3f}  |  "
        f"b: mean={np.mean(betas_arr):.3f} std={np.std(betas_arr):.3f}"
    )

print(f"\nTrue values: alpha_pos={TRUE_ALPHA_POS}, alpha_neg={TRUE_ALPHA_NEG}, beta={TRUE_BETA}")
