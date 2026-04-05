"""Single-subject MLE example.

Model: asocial_q_learning
"""

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import HierarchyStructure, InferenceConfig, fit
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    QParams,
)
from comp_model.runtime import SimulationConfig, simulate_subject
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec

# Define the task
N_ACTIONS = 2
N_TRIALS = 60
REWARD_PROBS = (0.75, 0.25)

task = TaskSpec(
    task_id="asocial_q_learning_manual_subject",
    blocks=(
        BlockSpec(
            condition="example",
            n_trials=N_TRIALS,
            schema=ASOCIAL_BANDIT_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
    ),
)

# Choose one readable parameter set for the focal model
kernel = AsocialQLearningKernel()
true_params = QParams(
    alpha=0.35,
    beta=2.5,
)

# Simulate one subject
subject = simulate_subject(
    task=task,
    env=StationaryBanditEnvironment(
        n_actions=N_ACTIONS,
        reward_probs=REWARD_PROBS,
    ),
    kernel=kernel,
    params=true_params,
    config=SimulationConfig(seed=7),
    subject_id="example_subject",
)

# Fit the model with MLE
inference_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_SHARED,
    backend="mle",
    mle_config=MleOptimizerConfig(n_restarts=8, max_iter=300, seed=7),
)

result = fit(inference_config, kernel, subject, ASOCIAL_BANDIT_SCHEMA)

# Compare the true parameters with the MLE estimate
param_names = ["alpha", "beta"]
true_values = {
    "alpha": 0.35,
    "beta": 2.5,
}

print("Manual subject + MLE: asocial_q_learning")
print(f"{'parameter':<24} {'true':>10} {'fit':>10}")
print("-" * 46)
for name in param_names:
    print(f"{name:<24} {true_values[name]:>10.3f} {result.constrained_params[name]:>10.3f}")

print()
print(f"log likelihood: {result.log_likelihood:.2f}")
print(f"AIC: {result.aic:.2f}")
print(f"BIC: {result.bic:.2f}")
