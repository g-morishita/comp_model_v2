"""Parameter recovery for the social self-reward + demo-mixture RL model using hierarchical Stan.

Recovers per-subject parameters (alpha_self, alpha_other_outcome,
alpha_other_action, w_imitation, beta) under a hierarchical
(STUDY_SUBJECT) model with population-level priors.  Uses a pre-choice
demonstrator observation schema with a uniform demonstrator policy.

Usage:
    uv run python example/social_rl_self_reward_demo_mixture/stan_recovery.py
"""

from scipy import stats

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference.bayes.stan import (
    SocialRlSelfRewardDemoMixtureStanAdapter,
    StanFitConfig,
)
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    QParams,
    SocialRlSelfRewardDemoMixtureKernel,
)
from comp_model.recovery import (
    ParamDist,
    RecoveryStudyConfig,
    compute_recovery_metrics,
    recovery_table,
    run_recovery,
)
from comp_model.tasks import SOCIAL_PRE_CHOICE_SCHEMA, BlockSpec, TaskSpec


def main() -> None:
    """Run hierarchical Stan parameter recovery for the mixture social RL kernel."""
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

    kernel = SocialRlSelfRewardDemoMixtureKernel()
    adapter = SocialRlSelfRewardDemoMixtureStanAdapter()

    config = RecoveryStudyConfig(
        n_replications=10,
        n_subjects=20,
        param_dists=(
            ParamDist("alpha_self", stats.norm(-0.847, 0.5), scale="unconstrained"),
            ParamDist("alpha_other_outcome", stats.norm(-1.386, 0.5), scale="unconstrained"),
            ParamDist("alpha_other_action", stats.norm(-0.405, 0.5), scale="unconstrained"),
            ParamDist("w_imitation", stats.norm(-0.847, 0.5), scale="unconstrained"),
            ParamDist("beta", stats.norm(1.687, 0.5), scale="unconstrained"),
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
            hierarchy=HierarchyStructure.STUDY_SUBJECT,
            backend="stan",
            stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
        ),
        adapter=adapter,
        simulation_base_seed=42,
    )

    print(f"Running {config.n_replications} reps x {config.n_subjects} subjects...")
    result = run_recovery(config)
    metrics = compute_recovery_metrics(result)
    print("\nRecovery Metrics (hierarchical Stan):")
    print(recovery_table(metrics))
    print("\nDone.")


if __name__ == "__main__":
    main()
