"""Parameter recovery analysis for the social self-reward + demo-mixture RL model.

Demonstrates MLE-based parameter recovery for
``SocialRlSelfRewardDemoMixtureKernel`` on a bandit task with pre-choice
demonstrator observation.  Runs multiple replications with parallel
per-subject fitting and reports standard recovery metrics.

Ground-truth parameter distributions (unconstrained scale):
  alpha_self          ~ sigmoid(Normal(-0.847, 0.5))
  alpha_other_outcome ~ sigmoid(Normal(-1.386, 0.5))
  alpha_other_action  ~ sigmoid(Normal(-0.405, 0.5))
  w_imitation         ~ sigmoid(Normal(-0.847, 0.5))
  beta                ~ softplus(Normal(1.687, 0.5))

Usage:
    uv run python example/social_rl_self_reward_demo_mixture/mle_recovery.py
"""

from scipy import stats

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    QParams,
    SocialRlSelfRewardDemoMixtureKernel,
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
    """Run MLE parameter recovery for the mixture social RL kernel."""
    # -- 1. Task setup ---------------------------------------------------------
    N_ACTIONS = 2
    N_TRIALS = 200

    task = TaskSpec(
        task_id="recovery_social_bandit_mixture",
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
    kernel = SocialRlSelfRewardDemoMixtureKernel()

    config = ParameterRecoveryConfig(
        n_replications=100,
        n_subjects=20,
        param_dists=(
            FlatParamDist("alpha_self", stats.norm(-0.847, 0.5), scale="unconstrained"),
            FlatParamDist("alpha_other_outcome", stats.norm(-1.386, 0.5), scale="unconstrained"),
            FlatParamDist("alpha_other_action", stats.norm(-0.405, 0.5), scale="unconstrained"),
            FlatParamDist("w_imitation", stats.norm(-0.847, 0.5), scale="unconstrained"),
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
