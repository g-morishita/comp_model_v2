"""Parameter recovery analysis for social observed-outcome Q-learning.

Demonstrates MLE-based parameter recovery for a social Q-learning kernel
on a bandit task with pre-choice demonstrator observation.  Runs multiple
replications with parallel per-subject fitting and reports standard
recovery metrics.

Usage:
    uv run python example/social_rl_self_reward_demo_reward_mle_recovery.py
"""

from scipy import stats

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    QParams,
    SocialRlSelfRewardDemoRewardKernel,
)
from comp_model.recovery import (
    FlatParamDist,
    ParameterRecoveryConfig,
    compute_parameter_recovery_metrics,
    parameter_recovery_table,
    run_parameter_recovery,
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
    kernel = SocialRlSelfRewardDemoRewardKernel()

    config = ParameterRecoveryConfig(
        n_replications=100,
        n_subjects=20,
        param_dists=(
            FlatParamDist("alpha_self", stats.norm(-0.847, 0.5), scale="unconstrained"),
            FlatParamDist("alpha_other", stats.norm(-0.847, 0.5), scale="unconstrained"),
            FlatParamDist("beta", stats.norm(1.687, 0.5), scale="unconstrained"),
        ),
        task=task,
        env_factory=lambda: StationaryBanditEnvironment(
            n_actions=N_ACTIONS, reward_probs=(0.8, 0.2)
        ),
        demonstrator_kernel=AsocialQLearningKernel(),
        demonstrator_params=QParams(alpha=0.0, beta=0.0),
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
    result = run_parameter_recovery(config)

    # -- 4. Compute and display metrics ----------------------------------------
    metrics = compute_parameter_recovery_metrics(result)
    print("\nRecovery Metrics:")
    print(parameter_recovery_table(metrics))

    print("\nDone.")


if __name__ == "__main__":
    main()
