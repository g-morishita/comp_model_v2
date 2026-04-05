"""Flat-population MLE example.

Model: social_rl_self_reward_demo_mixture_sticky
"""

import numpy as np
from scipy import stats

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import HierarchyStructure, InferenceConfig, fit
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    QParams,
    SocialRlSelfRewardDemoMixtureStickyKernel,
)
from comp_model.recovery import FlatParamDist, sample_true_params
from comp_model.runtime import SimulationConfig, simulate_dataset
from comp_model.tasks import SOCIAL_PRE_CHOICE_SCHEMA, BlockSpec, TaskSpec

# Define the task
N_ACTIONS = 2
N_TRIALS = 80
N_SUBJECTS = 6
REWARD_PROBS = (0.75, 0.25)

task = TaskSpec(
    task_id="social_rl_self_reward_demo_mixture_sticky_flat_population",
    blocks=(
        BlockSpec(
            condition="example",
            n_trials=N_TRIALS,
            schema=SOCIAL_PRE_CHOICE_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
    ),
)

# Define flat sampling distributions for each parameter
kernel = SocialRlSelfRewardDemoMixtureStickyKernel()
param_dists = (
    FlatParamDist(
        "alpha_self",
        stats.uniform(loc=0.1, scale=0.4),
    ),
    FlatParamDist(
        "alpha_other_outcome",
        stats.uniform(loc=0.25, scale=0.4),
    ),
    FlatParamDist(
        "alpha_other_action",
        stats.uniform(loc=0.15, scale=0.4),
    ),
    FlatParamDist(
        "w_imitation",
        stats.uniform(loc=0.45, scale=0.4),
    ),
    FlatParamDist(
        "beta",
        stats.uniform(loc=1.5, scale=2.5),
    ),
    FlatParamDist(
        "stickiness",
        stats.norm(loc=1.0, scale=0.35),
    ),
)

# Sample one parameter set per subject
true_table, params_per_subject, _ = sample_true_params(
    param_dists,
    kernel,
    N_SUBJECTS,
    np.random.default_rng(7),
)

# Define the demonstrator used in the social task
demonstrator_kernel = AsocialQLearningKernel()
demonstrator_params = QParams(alpha=0.30, beta=3.50)

# Simulate a dataset from those subject-specific parameters
dataset = simulate_dataset(
    task=task,
    env_factory=lambda: StationaryBanditEnvironment(
        n_actions=N_ACTIONS,
        reward_probs=REWARD_PROBS,
    ),
    kernel=kernel,
    params_per_subject=params_per_subject,
    config=SimulationConfig(seed=7),
    demonstrator_kernel=demonstrator_kernel,
    demonstrator_params=demonstrator_params,
)

# Fit each subject separately with MLE
inference_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_SHARED,
    backend="mle",
    mle_config=MleOptimizerConfig(n_restarts=8, max_iter=300, seed=7),
)

param_names = [
    "alpha_self",
    "alpha_other_outcome",
    "alpha_other_action",
    "w_imitation",
    "beta",
    "stickiness",
]
header_parts = [f"{'subject':<12}"]
for name in param_names:
    header_parts.append(f"{f'true {name}':>12}")
    header_parts.append(f"{f'fit {name}':>12}")
header_parts.append(f"{'LL':>10}")
header_parts.append(f"{'AIC':>10}")
header_parts.append(f"{'BIC':>10}")
header = "".join(header_parts)

print("Flat population sampling + MLE: social_rl_self_reward_demo_mixture_sticky")
print(header)
print("-" * len(header))

for subject in dataset.subjects:
    result = fit(inference_config, kernel, subject, SOCIAL_PRE_CHOICE_SCHEMA)
    row_parts = [f"{subject.subject_id:<12}"]
    for name in param_names:
        row_parts.append(f"{true_table[subject.subject_id][name]:>12.3f}")
        row_parts.append(f"{result.constrained_params[name]:>12.3f}")
    row_parts.append(f"{result.log_likelihood:>10.2f}")
    row_parts.append(f"{result.aic:>10.2f}")
    row_parts.append(f"{result.bic:>10.2f}")
    print("".join(row_parts))
