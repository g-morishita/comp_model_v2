"""Parameter recovery for asocial Q-learning using per-subject Stan.

Fits each subject independently with no pooling (SUBJECT_SHARED).

Usage:
    uv run python example/asocial_qlearning_stan_per_subject_recovery.py
"""

from scipy import stats

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference.bayes.stan import AsocialQLearningStanAdapter, StanFitConfig
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.models.kernels import AsocialQLearningKernel
from comp_model.recovery import (
    ParamDist,
    ParameterRecoveryConfig,
    compute_parameter_recovery_metrics,
    parameter_recovery_table,
    run_parameter_recovery,
)
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec


def main() -> None:
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

    kernel = AsocialQLearningKernel()
    adapter = AsocialQLearningStanAdapter()

    config = ParameterRecoveryConfig(
        n_replications=10,
        n_subjects=20,
        param_dists=(
            ParamDist("alpha", stats.norm(-0.847, 0.5), scale="unconstrained"),
            ParamDist("beta", stats.norm(1.687, 0.5), scale="unconstrained"),
        ),
        task=task,
        env_factory=lambda: StationaryBanditEnvironment(
            n_actions=N_ACTIONS, reward_probs=(0.8, 0.2)
        ),
        kernel=kernel,
        schema=ASOCIAL_BANDIT_SCHEMA,
        inference_config=InferenceConfig(
            hierarchy=HierarchyStructure.SUBJECT_SHARED,
            backend="stan",
            stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
        ),
        adapter=adapter,
        simulation_base_seed=42,
    )

    print(f"Running {config.n_replications} reps x {config.n_subjects} subjects...")
    result = run_parameter_recovery(config)
    metrics = compute_parameter_recovery_metrics(result)
    print("\nRecovery Metrics (per-subject Stan):")
    print(parameter_recovery_table(metrics))
    print("\nDone.")


if __name__ == "__main__":
    main()
