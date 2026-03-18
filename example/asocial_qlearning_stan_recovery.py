"""Parameter recovery analysis for asocial Q-learning using Stan.

Runs two recovery studies sequentially:
  1. Per-subject (SUBJECT_SHARED) — no pooling, fits each subject independently.
  2. Hierarchical (STUDY_SUBJECT) — population-level priors on alpha and beta.

Usage:
    uv run python example/asocial_qlearning_stan_recovery.py
"""

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference.bayes.stan import AsocialQLearningStanAdapter, StanFitConfig
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.models.kernels import AsocialQLearningKernel
from comp_model.recovery import (
    ParamDist,
    RecoveryStudyConfig,
    compute_recovery_metrics,
    recovery_table,
    run_recovery,
)
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec


def main() -> None:
    # -- 1. Task setup ---------------------------------------------------------
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

    param_dists = (
        ParamDist("alpha", mu_unconstrained=-0.847, sd_unconstrained=0.5),
        ParamDist("beta", mu_unconstrained=1.687, sd_unconstrained=0.5),
    )

    def env_factory() -> StationaryBanditEnvironment:
        return StationaryBanditEnvironment(
            n_actions=N_ACTIONS, reward_probs=(0.8, 0.2)
        )

    stan_config = StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42)

    # -- 2. Per-subject recovery (SUBJECT_SHARED) ------------------------------
    config_shared = RecoveryStudyConfig(
        n_replications=10,
        n_subjects=20,
        param_dists=param_dists,
        task=task,
        env_factory=env_factory,
        kernel=kernel,
        schema=ASOCIAL_BANDIT_SCHEMA,
        inference_config=InferenceConfig(
            hierarchy=HierarchyStructure.SUBJECT_SHARED,
            backend="stan",
            stan_config=stan_config,
        ),
        adapter=adapter,
        simulation_base_seed=42,
    )

    print("=== Per-subject Stan (SUBJECT_SHARED) ===")
    print(f"Running {config_shared.n_replications} reps x {config_shared.n_subjects} subjects...")
    result_shared = run_recovery(config_shared)
    metrics_shared = compute_recovery_metrics(result_shared)
    print("\nRecovery Metrics (per-subject Stan):")
    print(recovery_table(metrics_shared))

    # -- 3. Hierarchical recovery (STUDY_SUBJECT) ------------------------------
    config_hier = RecoveryStudyConfig(
        n_replications=10,
        n_subjects=20,
        param_dists=param_dists,
        task=task,
        env_factory=env_factory,
        kernel=kernel,
        schema=ASOCIAL_BANDIT_SCHEMA,
        inference_config=InferenceConfig(
            hierarchy=HierarchyStructure.STUDY_SUBJECT,
            backend="stan",
            stan_config=stan_config,
        ),
        adapter=adapter,
        simulation_base_seed=42,
    )

    print("\n=== Hierarchical Stan (STUDY_SUBJECT) ===")
    print(f"Running {config_hier.n_replications} reps x {config_hier.n_subjects} subjects...")
    result_hier = run_recovery(config_hier)
    metrics_hier = compute_recovery_metrics(result_hier)
    print("\nRecovery Metrics (hierarchical Stan):")
    print(recovery_table(metrics_hier))

    print("\nDone.")


if __name__ == "__main__":
    main()
