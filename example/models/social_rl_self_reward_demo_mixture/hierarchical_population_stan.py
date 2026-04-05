"""Hierarchical Stan example.

Model: social_rl_self_reward_demo_mixture
"""

from pathlib import Path

import numpy as np
from scipy import stats

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import HierarchyStructure, InferenceConfig, PriorSpec, fit
from comp_model.inference.bayes.stan import (
    SocialRlSelfRewardDemoMixtureStanAdapter,
    StanFitConfig,
)
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    QParams,
    SocialRlSelfRewardDemoMixtureKernel,
    get_transform,
)
from comp_model.recovery import HierarchicalParamDist, sample_true_params
from comp_model.runtime import SimulationConfig, simulate_dataset
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
N_TRIALS = 80
N_SUBJECTS = 8
REWARD_PROBS = (0.75, 0.25)

task = TaskSpec(
    task_id="social_rl_self_reward_demo_mixture_hierarchical_population",
    blocks=(
        BlockSpec(
            condition="example",
            n_trials=N_TRIALS,
            schema=SOCIAL_PRE_CHOICE_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
    ),
)

# Define hierarchical sampling distributions on the unconstrained scale
kernel = SocialRlSelfRewardDemoMixtureKernel()
param_dists = (
    HierarchicalParamDist(
        "alpha_self",
        mu_prior=stats.norm(loc=-0.847298, scale=0.5),
        sd_prior=stats.halfnorm(scale=0.3),
    ),
    HierarchicalParamDist(
        "alpha_other_outcome",
        mu_prior=stats.norm(loc=-0.200671, scale=0.5),
        sd_prior=stats.halfnorm(scale=0.3),
    ),
    HierarchicalParamDist(
        "alpha_other_action",
        mu_prior=stats.norm(loc=-0.619039, scale=0.5),
        sd_prior=stats.halfnorm(scale=0.3),
    ),
    HierarchicalParamDist(
        "w_imitation",
        mu_prior=stats.norm(loc=0.619039, scale=0.5),
        sd_prior=stats.halfnorm(scale=0.3),
    ),
    HierarchicalParamDist(
        "beta",
        mu_prior=stats.norm(loc=2.41435, scale=0.6),
        sd_prior=stats.halfnorm(scale=0.3),
    ),
)

# Sample one population and subject-level parameters
true_table, params_per_subject, pop_params = sample_true_params(
    param_dists,
    kernel,
    N_SUBJECTS,
    np.random.default_rng(7),
)

# Define the demonstrator used in the social task
demonstrator_kernel = AsocialQLearningKernel()
demonstrator_params = QParams(alpha=0.30, beta=3.50)

# Simulate a dataset from those sampled parameters
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

# Set explicit population and subject-level priors
prior_specs = {
    "alpha_self": PriorSpec("normal", {"mu": -0.847298, "sigma": 1.0}),
    "sd_alpha_self": PriorSpec("normal", {"mu": 0.0, "sigma": 0.5}),
    "alpha_other_outcome": PriorSpec("normal", {"mu": -0.200671, "sigma": 1.0}),
    "sd_alpha_other_outcome": PriorSpec("normal", {"mu": 0.0, "sigma": 0.5}),
    "alpha_other_action": PriorSpec("normal", {"mu": -0.619039, "sigma": 1.0}),
    "sd_alpha_other_action": PriorSpec("normal", {"mu": 0.0, "sigma": 0.5}),
    "w_imitation": PriorSpec("normal", {"mu": 0.619039, "sigma": 1.0}),
    "sd_w_imitation": PriorSpec("normal", {"mu": 0.0, "sigma": 0.5}),
    "beta": PriorSpec("normal", {"mu": 2.41435, "sigma": 1.0}),
    "sd_beta": PriorSpec("normal", {"mu": 0.0, "sigma": 0.5}),
}

# Fit the dataset with hierarchical Stan
adapter = SocialRlSelfRewardDemoMixtureStanAdapter()
inference_config = InferenceConfig(
    hierarchy=HierarchyStructure.STUDY_SUBJECT,
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

result = fit(inference_config, kernel, dataset, SOCIAL_PRE_CHOICE_SCHEMA, adapter=adapter)

# Compare true population means with posterior population means
param_names = ["alpha_self", "alpha_other_outcome", "alpha_other_action", "w_imitation", "beta"]
true_population_values = {}
for parameter in kernel.spec().parameter_specs:
    true_population_values[f"{parameter.name}_pop"] = get_transform(parameter.transform_id).forward(
        pop_params[f"mu_{parameter.name}_z"]
    )

print("Hierarchical population sampling + Stan: social_rl_self_reward_demo_mixture")
print(f"{'parameter':<24} {'true pop':>12} {'posterior':>12}")
print("-" * 52)
for name in param_names:
    population_name = f"{name}_pop"
    posterior_mean = float(np.mean(np.asarray(result.posterior_samples[population_name])))
    print(
        f"{population_name:<24} "
        f"{true_population_values[population_name]:>12.3f} "
        f"{posterior_mean:>12.3f}"
    )

print()
print("First four subjects")
subject_header_parts = [f"{'subject':<12}"]
for name in param_names:
    subject_header_parts.append(f"{f'true {name}':>12}")
    subject_header_parts.append(f"{f'post {name}':>12}")
subject_header = "".join(subject_header_parts)
print(subject_header)
print("-" * len(subject_header))
for subject_index, subject in enumerate(dataset.subjects[:4]):
    row_parts = [f"{subject.subject_id:<12}"]
    for name in param_names:
        posterior = np.asarray(result.posterior_samples[name])
        row_parts.append(f"{true_table[subject.subject_id][name]:>12.3f}")
        row_parts.append(f"{float(np.mean(posterior[:, subject_index])):>12.3f}")
    print("".join(row_parts))

print()
print(f"Stan divergences: {result.diagnostics['n_divergences']}")
