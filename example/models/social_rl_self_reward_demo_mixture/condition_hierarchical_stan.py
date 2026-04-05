"""Condition-aware hierarchical Stan example.

Model: social_rl_self_reward_demo_mixture

This example uses a shared-plus-delta parameterisation across two
conditions and fits the dataset with STUDY_SUBJECT_BLOCK_CONDITION.
"""

import os
from pathlib import Path

import numpy as np
from scipy import stats

from comp_model.data import Block, Dataset, SubjectData
from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import HierarchyStructure, InferenceConfig, PriorSpec, fit
from comp_model.inference.bayes.stan import (
    SocialRlSelfRewardDemoMixtureStanAdapter,
    StanFitConfig,
)
from comp_model.models import SharedDeltaLayout
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    QParams,
    SocialRlSelfRewardDemoMixtureKernel,
    get_transform,
)
from comp_model.recovery import HierarchicalParamDist, sample_true_params
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

diagnose_name = "diagnose.exe" if os.name == "nt" else "diagnose"
if not (Path(cmdstan_path) / "bin" / diagnose_name).exists():
    raise RuntimeError(f"CmdStan at {cmdstan_path!r} is incomplete: missing `bin/{diagnose_name}`.")

# Define a two-condition task
N_ACTIONS = 2
N_TRIALS = 80
N_SUBJECTS = 8
REWARD_PROBS = {
    "baseline": (0.75, 0.25),
    "shifted": (0.65, 0.35),
}

task = TaskSpec(
    task_id="social_rl_self_reward_demo_mixture_condition_hierarchical",
    blocks=(
        BlockSpec(
            condition="baseline",
            n_trials=N_TRIALS,
            schema=SOCIAL_PRE_CHOICE_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
        BlockSpec(
            condition="shifted",
            n_trials=N_TRIALS,
            schema=SOCIAL_PRE_CHOICE_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
    ),
)

# Define the shared-plus-delta layout
kernel = SocialRlSelfRewardDemoMixtureKernel()
layout = SharedDeltaLayout(
    kernel_spec=kernel.spec(),
    conditions=("baseline", "shifted"),
    baseline_condition="baseline",
)

print(f"Layout parameter keys: {layout.parameter_keys()}")

# Define hierarchical sampling distributions for shared and delta parameters
param_dists = (
    HierarchicalParamDist(
        "alpha_self",
        mu_prior=stats.norm(loc=-0.847298, scale=0.5),
        sd_prior=stats.halfnorm(scale=0.3),
    ),
    HierarchicalParamDist(
        "alpha_self__delta",
        mu_prior=stats.norm(loc=0.646627, scale=0.4),
        sd_prior=stats.halfnorm(scale=0.2),
    ),
    HierarchicalParamDist(
        "alpha_other_outcome",
        mu_prior=stats.norm(loc=-0.200671, scale=0.5),
        sd_prior=stats.halfnorm(scale=0.3),
    ),
    HierarchicalParamDist(
        "alpha_other_outcome__delta",
        mu_prior=stats.norm(loc=0.606136, scale=0.4),
        sd_prior=stats.halfnorm(scale=0.2),
    ),
    HierarchicalParamDist(
        "alpha_other_action",
        mu_prior=stats.norm(loc=-0.619039, scale=0.5),
        sd_prior=stats.halfnorm(scale=0.3),
    ),
    HierarchicalParamDist(
        "alpha_other_action__delta",
        mu_prior=stats.norm(loc=0.619039, scale=0.4),
        sd_prior=stats.halfnorm(scale=0.2),
    ),
    HierarchicalParamDist(
        "w_imitation",
        mu_prior=stats.norm(loc=0.619039, scale=0.5),
        sd_prior=stats.halfnorm(scale=0.3),
    ),
    HierarchicalParamDist(
        "w_imitation__delta",
        mu_prior=stats.norm(loc=0.767255, scale=0.4),
        sd_prior=stats.halfnorm(scale=0.2),
    ),
    HierarchicalParamDist(
        "beta",
        mu_prior=stats.norm(loc=2.41435, scale=0.5),
        sd_prior=stats.halfnorm(scale=0.3),
    ),
    HierarchicalParamDist(
        "beta__delta",
        mu_prior=stats.norm(loc=-0.855236, scale=0.4),
        sd_prior=stats.halfnorm(scale=0.2),
    ),
)

# Sample one population and subject-level parameters
true_table, params_per_subject, pop_params = sample_true_params(
    param_dists,
    kernel,
    N_SUBJECTS,
    np.random.default_rng(7),
    layout=layout,
)

# Define the demonstrator used in the social task
demonstrator_kernel = AsocialQLearningKernel()
demonstrator_params = QParams(alpha=0.30, beta=3.50)

# Simulate a two-condition dataset without helper wrappers
subjects = []
for subject_index in range(N_SUBJECTS):
    subject_id = f"sub_{subject_index:02d}"
    blocks = []
    for block_index, block_spec in enumerate(task.blocks):
        condition = block_spec.condition
        subject_data = simulate_subject(
            task=TaskSpec(task_id="tmp", blocks=(block_spec,)),
            env=StationaryBanditEnvironment(
                n_actions=N_ACTIONS,
                reward_probs=REWARD_PROBS[condition],
            ),
            kernel=kernel,
            params=params_per_subject[subject_id][condition],
            config=SimulationConfig(seed=7 + subject_index * 1000 + block_index),
            subject_id=subject_id,
            demonstrator_kernel=demonstrator_kernel,
            demonstrator_params=demonstrator_params,
        )
        blocks.append(
            Block(
                block_index=block_index,
                condition=subject_data.blocks[0].condition,
                schema_id=subject_data.blocks[0].schema_id,
                trials=subject_data.blocks[0].trials,
                metadata=subject_data.blocks[0].metadata,
            )
        )
    subjects.append(SubjectData(subject_id=subject_id, blocks=tuple(blocks)))

dataset = Dataset(subjects=tuple(subjects))

# Set explicit shared and delta priors
prior_specs = {
    "alpha_self": PriorSpec("normal", {"mu": -0.847298, "sigma": 1.0}),
    "sd_alpha_self": PriorSpec("normal", {"mu": 0.0, "sigma": 0.5}),
    "alpha_self_delta": PriorSpec("normal", {"mu": 0.646627, "sigma": 1.0}),
    "sd_alpha_self_delta": PriorSpec("normal", {"mu": 0.0, "sigma": 0.5}),
    "alpha_other_outcome": PriorSpec("normal", {"mu": -0.200671, "sigma": 1.0}),
    "sd_alpha_other_outcome": PriorSpec("normal", {"mu": 0.0, "sigma": 0.5}),
    "alpha_other_outcome_delta": PriorSpec("normal", {"mu": 0.606136, "sigma": 1.0}),
    "sd_alpha_other_outcome_delta": PriorSpec("normal", {"mu": 0.0, "sigma": 0.5}),
    "alpha_other_action": PriorSpec("normal", {"mu": -0.619039, "sigma": 1.0}),
    "sd_alpha_other_action": PriorSpec("normal", {"mu": 0.0, "sigma": 0.5}),
    "alpha_other_action_delta": PriorSpec("normal", {"mu": 0.619039, "sigma": 1.0}),
    "sd_alpha_other_action_delta": PriorSpec("normal", {"mu": 0.0, "sigma": 0.5}),
    "w_imitation": PriorSpec("normal", {"mu": 0.619039, "sigma": 1.0}),
    "sd_w_imitation": PriorSpec("normal", {"mu": 0.0, "sigma": 0.5}),
    "w_imitation_delta": PriorSpec("normal", {"mu": 0.767255, "sigma": 1.0}),
    "sd_w_imitation_delta": PriorSpec("normal", {"mu": 0.0, "sigma": 0.5}),
    "beta": PriorSpec("normal", {"mu": 2.41435, "sigma": 1.0}),
    "sd_beta": PriorSpec("normal", {"mu": 0.0, "sigma": 0.5}),
    "beta_delta": PriorSpec("normal", {"mu": -0.855236, "sigma": 1.0}),
    "sd_beta_delta": PriorSpec("normal", {"mu": 0.0, "sigma": 0.5}),
}

# Fit the two-condition dataset with hierarchical Stan
adapter = SocialRlSelfRewardDemoMixtureStanAdapter()
inference_config = InferenceConfig(
    hierarchy=HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
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

result = fit(
    inference_config,
    kernel,
    dataset,
    SOCIAL_PRE_CHOICE_SCHEMA,
    layout=layout,
    adapter=adapter,
)

# Compare true and posterior population values for each condition
param_names = ["alpha_self", "alpha_other_outcome", "alpha_other_action", "w_imitation", "beta"]
print()
print("Condition-aware hierarchical Stan: social_rl_self_reward_demo_mixture")
print(
    f"{'parameter':<24} {'baseline true':>14} {'baseline post':>14} "
    f"{'shifted true':>14} {'shifted post':>14}"
)
print("-" * 84)
baseline_index = layout.conditions.index("baseline")
shifted_index = layout.conditions.index("shifted")
for name in param_names:
    transform = next(
        get_transform(parameter.transform_id)
        for parameter in kernel.spec().parameter_specs
        if parameter.name == name
    )
    baseline_true = transform.forward(pop_params[f"mu_{name}_shared_z"])
    shifted_true = transform.forward(
        pop_params[f"mu_{name}_shared_z"] + pop_params[f"mu_{name}_delta_z__shifted"]
    )
    population_samples = np.asarray(result.posterior_samples[f"{name}_pop"])
    if population_samples.ndim != 2 or population_samples.shape[1] != len(layout.conditions):
        raise RuntimeError(
            f"Expected {name}_pop to have shape (draws, {len(layout.conditions)}) "
            "for a two-condition hierarchical fit."
        )
    baseline_post = float(np.mean(population_samples[:, baseline_index]))
    shifted_post = float(np.mean(population_samples[:, shifted_index]))
    print(
        f"{name:<24} {baseline_true:>14.3f} {baseline_post:>14.3f} "
        f"{shifted_true:>14.3f} {shifted_post:>14.3f}"
    )

print()
print(f"{'delta param':<24} {'true mu_z':>12} {'post mu_z':>12}")
print("-" * 50)
for name in param_names:
    true_delta = float(pop_params[f"mu_{name}_delta_z__shifted"])
    posterior_delta = float(np.mean(np.asarray(result.posterior_samples[f"mu_{name}_delta_z"])))
    delta_name = f"{name}_delta"
    print(f"{delta_name:<24} {true_delta:>12.3f} {posterior_delta:>12.3f}")

print()
print("First four subjects")
column_widths = {name: max(12, len(name) + 5) for name in param_names}
header_parts = [f"{'subject':<12}", f"{'condition':<12}"]
for name in param_names:
    width = column_widths[name]
    header_parts.append(f" {f'true {name}':>{width}}")
    header_parts.append(f" {f'post {name}':>{width}}")
header = "".join(header_parts)
print(header)
print("-" * len(header))
for subject_index, subject in enumerate(dataset.subjects[:4]):
    for condition_index, condition in enumerate(layout.conditions):
        row_parts = [f"{subject.subject_id:<12}", f"{condition:<12}"]
        for name in param_names:
            width = column_widths[name]
            true_key = f"{name}__{condition}"
            posterior = np.asarray(result.posterior_samples[name])
            row_parts.append(f" {true_table[subject.subject_id][true_key]:>{width}.3f}")
            posterior_mean = float(np.mean(posterior[:, subject_index, condition_index]))
            row_parts.append(f" {posterior_mean:>{width}.3f}")
        print("".join(row_parts))

print()
print(f"Stan divergences: {result.diagnostics['n_divergences']}")
