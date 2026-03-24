"""Bayesian hierarchical within-subject parameter recovery for the mixture model.

Two demonstrator conditions per subject:
  "strong_demo": demonstrator alpha=0.3, beta=20.0
  "weak_demo":   demonstrator alpha=0.3, beta=1.5

Two blocks per condition, 60 trials per block, 64 subjects.
Parameters are sampled uniformly on the unconstrained scale.
Each condition is fit independently with STUDY_SUBJECT hierarchy.

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

N_ACTIONS = 2
N_TRIALS_PER_BLOCK = 60
N_BLOCKS_PER_CONDITION = 2
N_SUBJECTS = 64
REWARD_PROBS = (0.8, 0.2)

# Uniform on unconstrained scale:
#   sigmoid(-3, 3) → constrained (0.047, 0.953) for alpha params
#   softplus(-1, 5) → constrained (0.31, 5.0)   for beta
PARAM_DISTS = (
    ParamDist("alpha_self", stats.uniform(-3, 6), scale="unconstrained"),
    ParamDist("alpha_other_outcome", stats.uniform(-3, 6), scale="unconstrained"),
    ParamDist("alpha_other_action", stats.uniform(-3, 6), scale="unconstrained"),
    ParamDist("w_imitation", stats.uniform(-3, 6), scale="unconstrained"),
    ParamDist("beta", stats.uniform(-1, 6), scale="unconstrained"),
)

STAN_CONFIG = InferenceConfig(
    hierarchy=HierarchyStructure.STUDY_SUBJECT,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
)


def main() -> None:
    """Run per-condition hierarchical Stan recovery for the mixture social RL kernel."""
    kernel = SocialRlSelfRewardDemoMixtureKernel()
    adapter = SocialRlSelfRewardDemoMixtureStanAdapter()

    for condition, demo_params in (
        ("strong_demo", QParams(alpha=0.3, beta=20.0)),
        ("weak_demo", QParams(alpha=0.3, beta=1.5)),
    ):
        task = TaskSpec(
            task_id=f"recovery_{condition}",
            blocks=tuple(
                BlockSpec(
                    condition=condition,
                    n_trials=N_TRIALS_PER_BLOCK,
                    schema=SOCIAL_PRE_CHOICE_SCHEMA,
                    metadata={"n_actions": N_ACTIONS},
                )
                for _ in range(N_BLOCKS_PER_CONDITION)
            ),
        )

        config = RecoveryStudyConfig(
            n_replications=1,
            n_subjects=N_SUBJECTS,
            param_dists=PARAM_DISTS,
            task=task,
            env_factory=lambda: StationaryBanditEnvironment(
                n_actions=N_ACTIONS, reward_probs=REWARD_PROBS
            ),
            kernel=kernel,
            schema=SOCIAL_PRE_CHOICE_SCHEMA,
            demonstrator_kernel=AsocialQLearningKernel(),
            demonstrator_params=demo_params,
            inference_config=STAN_CONFIG,
            adapter=adapter,
            simulation_base_seed=42,
        )

        print(f"\n{'=' * 60}")
        print(f"Condition: {condition}  (demonstrator beta={demo_params.beta})")
        print(f"Running {config.n_replications} rep x {config.n_subjects} subjects...")
        result = run_recovery(config)

        print("\nPer-subject true vs posterior mean:")
        print(recovery_summary(result))

        metrics = compute_recovery_metrics(result)
        print("\nRecovery metrics:")
        print(recovery_table(metrics))

    print("\nDone.")


if __name__ == "__main__":
    main()
