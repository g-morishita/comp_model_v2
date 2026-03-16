"""Simulate asocial Q-learning agents on a stationary bandit, then recover
parameters with hierarchical Stan (NUTS) inference.

Ground-truth: alpha=0.3, beta=2.0
Requires: cmdstanpy and a working CmdStan installation.
"""

from pathlib import Path

import numpy as np

from comp_model.data import Dataset
from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.bayes.stan import AsocialQLearningStanAdapter, StanFitConfig
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.io import save_dataset_to_csv
from comp_model.models.kernels import AsocialQLearningKernel, QParams
from comp_model.models.kernels.transforms import get_transform
from comp_model.runtime import SimulationConfig, simulate_dataset
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec

# ── 1. Define task ──────────────────────────────────────────────────────────
N_ACTIONS = 2
N_TRIALS = 100
N_SUBJECTS = 5

task = TaskSpec(
    task_id="asocial_bandit",
    blocks=(
        BlockSpec(
            condition="learning",
            n_trials=N_TRIALS,
            schema=ASOCIAL_BANDIT_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
    ),
)

# ── 2. Environment ─────────────────────────────────────────────────────────
REWARD_PROBS = (0.8, 0.2)

# ── 3. Ground-truth parameters ─────────────────────────────────────────────
TRUE_ALPHA = 0.3
TRUE_BETA = 2.0
kernel = AsocialQLearningKernel()
true_params = QParams(alpha=TRUE_ALPHA, beta=TRUE_BETA)

# ── 4. Simulate dataset ────────────────────────────────────────────────────
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

# ── 5. Save to CSV ─────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "asocial_qlearning_data.csv"
save_dataset_to_csv(dataset, schema=ASOCIAL_BANDIT_SCHEMA, path=csv_path)
print(f"Saved {len(dataset.subjects)} subjects to {csv_path}")

# ── 6. Fit with Stan ───────────────────────────────────────────────────────
stan_config = InferenceConfig(
    hierarchy=HierarchyStructure.STUDY_SUBJECT,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
)

adapter = AsocialQLearningStanAdapter()
result = fit(stan_config, kernel, dataset, ASOCIAL_BANDIT_SCHEMA, adapter=adapter)

# ── 7. Report results ──────────────────────────────────────────────────────
sigmoid = get_transform("sigmoid")
softplus = get_transform("softplus")

print(f"\nModel: {result.model_id}")
print(f"Hierarchy: {result.hierarchy}")

# Population-level means (unconstrained)
if "mu_alpha" in result.posterior_samples:
    mu_alpha_z = result.posterior_samples["mu_alpha"]
    mu_beta_z = result.posterior_samples["mu_beta"]
    print(f"\nPopulation posterior means (constrained):")
    print(f"  alpha: {np.mean(sigmoid.forward(mu_alpha_z)):.3f}  (true: {TRUE_ALPHA})")
    print(f"  beta:  {np.mean(softplus.forward(mu_beta_z)):.3f}  (true: {TRUE_BETA})")

# Per-subject
if result.subject_params:
    print(f"\n{'Subject':<12} {'Post. α':>10} {'Post. β':>10}")
    print("-" * 35)
    for sid, params_dict in result.subject_params.items():
        alpha_samples = sigmoid.forward(params_dict["alpha"])
        beta_samples = softplus.forward(params_dict["beta"])
        print(f"{sid:<12} {np.mean(alpha_samples):>10.3f} {np.mean(beta_samples):>10.3f}")

print(f"\nTrue values: alpha={TRUE_ALPHA}, beta={TRUE_BETA}")
print("Done.")
