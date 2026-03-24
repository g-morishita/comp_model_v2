"""Check whether increasing trials improves parameter recovery for the mixture model."""

import numpy as np

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    QParams,
    SocialRlSelfRewardDemoMixtureKernel,
    SocialRlSelfRewardDemoMixtureParams,
)
from comp_model.runtime import SimulationConfig, simulate_dataset
from comp_model.tasks import SOCIAL_PRE_CHOICE_SCHEMA, BlockSpec, TaskSpec

N_ACTIONS = 2
REWARD_PROBS = (0.8, 0.2)
TRUE_ALPHA_SELF = 0.3
TRUE_ALPHA_OTHER_OUTCOME = 0.2
TRUE_ALPHA_OTHER_ACTION = 0.4
TRUE_W_IMITATION = 0.3
TRUE_BETA = 2.0
N_SUBJECTS = 20
TRIALS_PER_BLOCK = 50

kernel = SocialRlSelfRewardDemoMixtureKernel()
true_params = SocialRlSelfRewardDemoMixtureParams(
    alpha_self=TRUE_ALPHA_SELF,
    alpha_other_outcome=TRUE_ALPHA_OTHER_OUTCOME,
    alpha_other_action=TRUE_ALPHA_OTHER_ACTION,
    w_imitation=TRUE_W_IMITATION,
    beta=TRUE_BETA,
)

mle_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_SHARED,
    backend="mle",
    mle_config=MleOptimizerConfig(n_restarts=10, seed=0),
)

param_names = ("alpha_self", "alpha_other_outcome", "alpha_other_action", "w_imitation", "beta")
true_vals = (
    TRUE_ALPHA_SELF,
    TRUE_ALPHA_OTHER_OUTCOME,
    TRUE_ALPHA_OTHER_ACTION,
    TRUE_W_IMITATION,
    TRUE_BETA,
)

header = f"{'Blocks':>6}  {'Trials':>6}  " + "  ".join(
    f"{'mean_' + p[:4]:>10} {'std_' + p[:4]:>9}" for p in param_names
)
print(header)
print("-" * len(header))

for n_blocks in [1, 2, 4, 8]:
    total_trials = n_blocks * TRIALS_PER_BLOCK
    task = TaskSpec(
        task_id="test",
        blocks=tuple(
            BlockSpec(
                condition="social",
                n_trials=TRIALS_PER_BLOCK,
                schema=SOCIAL_PRE_CHOICE_SCHEMA,
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
        demonstrator_kernel=AsocialQLearningKernel(),
        demonstrator_params=QParams(alpha=0.0, beta=0.0),
    )

    recovered: dict[str, list[float]] = {p: [] for p in param_names}
    for subject in dataset.subjects:
        result = fit(mle_config, kernel, subject, SOCIAL_PRE_CHOICE_SCHEMA)
        for p in param_names:
            recovered[p].append(result.constrained_params[p])

    row = f"{n_blocks:>6}  {total_trials:>6}  "
    row += "  ".join(
        f"{np.mean(recovered[p]):>10.3f} {np.std(recovered[p]):>9.3f}" for p in param_names
    )
    print(row)

print(
    f"\nTrue values: alpha_self={TRUE_ALPHA_SELF}, "
    f"alpha_other_outcome={TRUE_ALPHA_OTHER_OUTCOME}, "
    f"alpha_other_action={TRUE_ALPHA_OTHER_ACTION}, "
    f"w_imitation={TRUE_W_IMITATION}, beta={TRUE_BETA}"
)
