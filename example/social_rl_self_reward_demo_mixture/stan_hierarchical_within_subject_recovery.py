"""Bayesian hierarchical within-subject parameter recovery for the mixture model.

Two demonstrator conditions per subject:
  "strong_demo": demonstrator alpha=0.3, beta=20.0
  "weak_demo":   demonstrator alpha=0.3, beta=1.5

Two blocks per condition (4 blocks total), 60 trials per block, 64 subjects.
Subject parameters are sampled from uniform distributions via SharedDeltaLayout.
Fit with STUDY_SUBJECT_BLOCK_CONDITION hierarchy.

Usage:
    uv run python \
        example/social_rl_self_reward_demo_mixture/stan_hierarchical_within_subject_recovery.py
"""

from scipy import stats

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference.bayes.stan import (
    SocialRlSelfRewardDemoMixtureStanAdapter,
    StanFitConfig,
)
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.models import SharedDeltaLayout
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    QParams,
    SocialRlSelfRewardDemoMixtureKernel,
)
from comp_model.recovery import (
    ParamDist,
    RecoveryStudyConfig,
    compute_recovery_metrics,
    recovery_summary,
    recovery_table,
    run_recovery,
)
from comp_model.tasks import SOCIAL_PRE_CHOICE_SCHEMA, BlockSpec, TaskSpec


def main() -> None:
    """Run hierarchical within-subject Stan recovery for the mixture social RL kernel."""
    N_ACTIONS = 2
    N_TRIALS_PER_BLOCK = 60
    N_BLOCKS_PER_CONDITION = 2
    N_SUBJECTS = 64
    REWARD_PROBS = (0.8, 0.2)
    CONDITIONS = ("strong_demo", "weak_demo")

    task = TaskSpec(
        task_id="recovery_within_subject_mixture",
        blocks=tuple(
            BlockSpec(
                condition=condition,
                n_trials=N_TRIALS_PER_BLOCK,
                schema=SOCIAL_PRE_CHOICE_SCHEMA,
                metadata={"n_actions": N_ACTIONS},
            )
            for condition in CONDITIONS
            for _ in range(N_BLOCKS_PER_CONDITION)
        ),
    )

    kernel = SocialRlSelfRewardDemoMixtureKernel()
    layout = SharedDeltaLayout(
        kernel_spec=kernel.spec(),
        conditions=CONDITIONS,
        baseline_condition="strong_demo",
    )

    # Shared (baseline): uniform on constrained scale.
    # Delta (weak_demo - strong_demo): normal on unconstrained scale.
    param_dists = (
        ParamDist("alpha_self", stats.uniform(0, 1), scale="constrained"),
        ParamDist("alpha_other_outcome", stats.uniform(0, 1), scale="constrained"),
        ParamDist("alpha_other_action", stats.uniform(0, 1), scale="constrained"),
        ParamDist("w_imitation", stats.uniform(0, 1), scale="constrained"),
        ParamDist("beta", stats.uniform(0.5, 4.5), scale="constrained"),
        ParamDist("alpha_self__delta", stats.norm(0, 0.5), scale="unconstrained"),
        ParamDist("alpha_other_outcome__delta", stats.norm(0, 0.5), scale="unconstrained"),
        ParamDist("alpha_other_action__delta", stats.norm(0, 0.5), scale="unconstrained"),
        ParamDist("w_imitation__delta", stats.norm(0, 0.5), scale="unconstrained"),
        ParamDist("beta__delta", stats.norm(0, 0.3), scale="unconstrained"),
    )

    config = RecoveryStudyConfig(
        n_replications=1,
        n_subjects=N_SUBJECTS,
        param_dists=param_dists,
        task=task,
        env_factory=lambda: StationaryBanditEnvironment(
            n_actions=N_ACTIONS, reward_probs=REWARD_PROBS
        ),
        kernel=kernel,
        schema=SOCIAL_PRE_CHOICE_SCHEMA,
        layout=layout,
        demonstrator_kernel=AsocialQLearningKernel(),
        condition_demonstrator_params={
            "strong_demo": QParams(alpha=0.3, beta=20.0),
            "weak_demo": QParams(alpha=0.3, beta=1.5),
        },
        inference_config=InferenceConfig(
            hierarchy=HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
            backend="stan",
            stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
        ),
        adapter=SocialRlSelfRewardDemoMixtureStanAdapter(),
        simulation_base_seed=42,
    )

    print(f"Running {config.n_replications} rep x {config.n_subjects} subjects...")
    result = run_recovery(config)

    print("\nPer-subject true vs posterior mean:")
    print(recovery_summary(result))

    metrics = compute_recovery_metrics(result)
    print("\nRecovery metrics (STUDY_SUBJECT_BLOCK_CONDITION Stan):")
    print(recovery_table(metrics))
    print("\nDone.")


if __name__ == "__main__":
    main()
