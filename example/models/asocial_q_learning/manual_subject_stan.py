"""Single-subject Stan example.

Model: asocial_q_learning
"""

from pathlib import Path

import numpy as np

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import HierarchyStructure, InferenceConfig, PriorSpec, fit
from comp_model.inference.bayes.stan import (
    AsocialQLearningStanAdapter,
    StanFitConfig,
)
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    QParams,
)
from comp_model.runtime import SimulationConfig, simulate_subject
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec

# Check that the Stan toolchain is available
try:
    import cmdstanpy
except ImportError as error:
    raise RuntimeError(
        "This example requires cmdstanpy. Install with `pip install .[stan]`."
    ) from error

try:
    cmdstan_path = cmdstanpy.cmdstan_path()
except (RuntimeError, ValueError) as error:
    raise RuntimeError(
        "This example requires a working CmdStan installation. "
        "Run `python -m cmdstanpy.install_cmdstan` first."
    ) from error

if not (Path(cmdstan_path) / "bin" / "diagnose").exists():
    raise RuntimeError(f"CmdStan at {cmdstan_path!r} is incomplete: missing `bin/diagnose`.")

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

# Set explicit subject-level priors
prior_specs = {
    "alpha": PriorSpec("normal", {"mu": -0.619039, "sigma": 1.0}),
    "beta": PriorSpec("normal", {"mu": 2.41435, "sigma": 1.0}),
}

# Fit the model with Stan
adapter = AsocialQLearningStanAdapter()
inference_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_SHARED,
    backend="stan",
    stan_config=StanFitConfig(
        n_warmup=250,
        n_samples=250,
        n_chains=2,
        seed=7,
        show_console=False,
    ),
    prior_specs=prior_specs,
)

result = fit(inference_config, kernel, subject, ASOCIAL_BANDIT_SCHEMA, adapter=adapter)

# Compare the true parameters with the posterior means
param_names = ["alpha", "beta"]
true_values = {
    "alpha": 0.35,
    "beta": 2.5,
}

print("Manual subject + Stan: asocial_q_learning")
print(f"{'parameter':<24} {'true':>10} {'posterior':>10}")
print("-" * 46)
for name in param_names:
    posterior_mean = float(np.mean(np.asarray(result.posterior_samples[name])))
    print(f"{name:<24} {true_values[name]:>10.3f} {posterior_mean:>10.3f}")

print()
print(f"Stan divergences: {result.diagnostics['n_divergences']}")
