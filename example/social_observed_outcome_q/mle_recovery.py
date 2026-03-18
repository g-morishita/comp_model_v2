"""Parameter recovery analysis for social observed-outcome Q-learning.

Demonstrates MLE-based parameter recovery for a social Q-learning kernel
on a bandit task with pre-choice demonstrator observation.  Runs multiple
replications with parallel per-subject fitting and reports standard
recovery metrics.

Usage:
    uv run python example/social_observed_outcome_q_mle_recovery.py
"""

from comp_model.environments import SocialBanditEnvironment, StationaryBanditEnvironment
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.models.kernels import SocialObservedOutcomeQKernel
from comp_model.recovery import (
    ParamDist,
    RecoveryStudyConfig,
    compute_recovery_metrics,
    recovery_table,
    run_recovery,
)
from comp_model.tasks import SOCIAL_PRE_CHOICE_SCHEMA, BlockSpec, TaskSpec


def main() -> None:
    # -- 1. Task setup ---------------------------------------------------------
    N_ACTIONS = 2
    N_TRIALS = 200

    task = TaskSpec(
        task_id="recovery_social_bandit",
        blocks=(
            BlockSpec(
                condition="social",
                n_trials=N_TRIALS,
                schema=SOCIAL_PRE_CHOICE_SCHEMA,
                metadata={"n_actions": N_ACTIONS},
            ),
        ),
    )

    # -- 2. Configure recovery study -------------------------------------------
    kernel = SocialObservedOutcomeQKernel()

    config = RecoveryStudyConfig(
        n_replications=100,
        n_subjects=20,
        param_dists=(
            ParamDist("alpha_self", mu_unconstrained=-0.847, sd_unconstrained=0.5),
            ParamDist("alpha_other", mu_unconstrained=-0.847, sd_unconstrained=0.5),
            ParamDist("beta", mu_unconstrained=1.687, sd_unconstrained=0.5),
        ),
        task=task,
        env_factory=lambda: SocialBanditEnvironment(
            inner=StationaryBanditEnvironment(
                n_actions=N_ACTIONS, reward_probs=(0.8, 0.2)
            ),
            demonstrator_policy=(0.5, 0.5),
        ),
        kernel=kernel,
        schema=SOCIAL_PRE_CHOICE_SCHEMA,
        inference_config=InferenceConfig(
            hierarchy=HierarchyStructure.SUBJECT_SHARED,
            backend="mle",
            mle_config=MleOptimizerConfig(n_restarts=10, seed=0),
        ),
        simulation_base_seed=42,
    )

    # -- 3. Run recovery -------------------------------------------------------
    print(f"Running {config.n_replications} replications x {config.n_subjects} subjects...")
    result = run_recovery(config)

    # -- 4. Compute and display metrics ----------------------------------------
    metrics = compute_recovery_metrics(result)
    print("\nRecovery Metrics:")
    print(recovery_table(metrics))

    print("\nDone.")


if __name__ == "__main__":
    main()
