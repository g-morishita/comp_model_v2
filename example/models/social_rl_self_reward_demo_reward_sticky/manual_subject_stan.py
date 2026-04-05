"""Single-subject Stan example.

Model: social_rl_self_reward_demo_reward_sticky
"""

from pathlib import Path

import numpy as np

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import HierarchyStructure, InferenceConfig, PriorSpec, fit
from comp_model.inference.bayes.stan import (
    SocialRlSelfRewardDemoRewardStickyStanAdapter,
    StanFitConfig,
)
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    QParams,
    SocialRlSelfRewardDemoRewardStickyKernel,
    SocialRlSelfRewardDemoRewardStickyParams,
)
from comp_model.runtime import SimulationConfig, simulate_subject
from comp_model.tasks import SOCIAL_PRE_CHOICE_SCHEMA, BlockSpec, TaskSpec

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
    task_id="social_rl_self_reward_demo_reward_sticky_manual_subject",
    blocks=(
        BlockSpec(
            condition="example",
            n_trials=N_TRIALS,
            schema=SOCIAL_PRE_CHOICE_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
    ),
)

# Choose one readable parameter set for the focal model
kernel = SocialRlSelfRewardDemoRewardStickyKernel()
true_params = SocialRlSelfRewardDemoRewardStickyParams(
    alpha_self=0.3,
    alpha_other=0.45,
    beta=2.5,
    stickiness=1.0,
)

# Define the demonstrator used in the social task
demonstrator_kernel = AsocialQLearningKernel()
demonstrator_params = QParams(alpha=0.30, beta=3.50)

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
    demonstrator_kernel=demonstrator_kernel,
    demonstrator_params=demonstrator_params,
)

# Set explicit subject-level priors
prior_specs = {
    "alpha_self": PriorSpec("normal", {"mu": -0.847298, "sigma": 1.0}),
    "alpha_other": PriorSpec("normal", {"mu": -0.200671, "sigma": 1.0}),
    "beta": PriorSpec("normal", {"mu": 2.41435, "sigma": 1.0}),
    "stickiness": PriorSpec("normal", {"mu": 1.0, "sigma": 1.0}),
}

# Fit the model with Stan
adapter = SocialRlSelfRewardDemoRewardStickyStanAdapter()
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

result = fit(inference_config, kernel, subject, SOCIAL_PRE_CHOICE_SCHEMA, adapter=adapter)

# Compare the true parameters with the posterior means
param_names = ["alpha_self", "alpha_other", "beta", "stickiness"]
true_values = {
    "alpha_self": 0.3,
    "alpha_other": 0.45,
    "beta": 2.5,
    "stickiness": 1.0,
}

print("Manual subject + Stan: social_rl_self_reward_demo_reward_sticky")
print(f"{'parameter':<24} {'true':>10} {'posterior':>10}")
print("-" * 46)
for name in param_names:
    posterior_mean = float(np.mean(np.asarray(result.posterior_samples[name])))
    print(f"{name:<24} {true_values[name]:>10.3f} {posterior_mean:>10.3f}")

print()
print(f"Stan divergences: {result.diagnostics['n_divergences']}")
