"""Parameter recovery for the asymmetric RL kernel using hierarchical Stan.

Simulates data from the asymmetric RL model with a strong optimism bias
(alpha_pos >> alpha_neg), then fits a hierarchical Stan model and checks
how well the per-subject parameters are recovered.

Usage
-----
    uv run python example/model_recovery/stan_parameter_recovery.py
"""

from scipy import stats

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference.bayes.stan import AsocialRlAsymmetricStanAdapter, StanFitConfig
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.models.kernels import AsocialRlAsymmetricKernel
from comp_model.recovery import (
    ParamDist,
    RecoveryStudyConfig,
    compute_recovery_metrics,
    recovery_table,
    run_recovery,
)
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec

N_ACTIONS = 2
N_TRIALS = 200

task = TaskSpec(
    task_id="recovery_bandit",
    blocks=(
        BlockSpec(
            condition="default",
            n_trials=N_TRIALS,
            schema=ASOCIAL_BANDIT_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
    ),
)

kernel = AsocialRlAsymmetricKernel()
adapter = AsocialRlAsymmetricStanAdapter()

config = RecoveryStudyConfig(
    n_replications=5,
    n_subjects=20,
    param_dists=(
        # Optimism bias: strong positive RPE learning, weak negative
        ParamDist("alpha_pos", stats.uniform(0.4, 0.5)),  # alpha_pos in [0.4, 0.9]
        ParamDist("alpha_neg", stats.uniform(0.05, 0.25)),  # alpha_neg in [0.05, 0.3]
        ParamDist("beta", stats.uniform(1.0, 9.0)),
    ),
    task=task,
    env_factory=lambda: StationaryBanditEnvironment(n_actions=N_ACTIONS, reward_probs=(0.75, 0.25)),
    kernel=kernel,
    schema=ASOCIAL_BANDIT_SCHEMA,
    inference_config=InferenceConfig(
        hierarchy=HierarchyStructure.STUDY_SUBJECT,
        backend="stan",
        stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
    ),
    adapter=adapter,
    simulation_base_seed=42,
)

if __name__ == "__main__":
    print(
        f"Running parameter recovery: {config.n_replications} reps x "
        f"{config.n_subjects} subjects (asymmetric RL, Stan)"
    )
    result = run_recovery(config)
    metrics = compute_recovery_metrics(result)
    print("\nRecovery Metrics (hierarchical Stan):")
    print(recovery_table(metrics))
    print("\nDone.")
