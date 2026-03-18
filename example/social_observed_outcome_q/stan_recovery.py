"""Parameter recovery for social observed-outcome Q-learning using hierarchical Stan.

Recovers per-subject parameters (alpha_self, alpha_other, beta) under a
hierarchical (STUDY_SUBJECT) model with population-level priors.  Uses a
pre-choice demonstrator observation schema with a uniform demonstrator policy.

Usage:
    uv run python example/social_observed_outcome_q_stan_recovery.py
"""

from scipy import stats

from comp_model.environments import SocialBanditEnvironment, StationaryBanditEnvironment
from comp_model.inference.bayes.stan import SocialObservedOutcomeQStanAdapter, StanFitConfig
from comp_model.inference.config import HierarchyStructure, InferenceConfig
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

    kernel = SocialObservedOutcomeQKernel()
    adapter = SocialObservedOutcomeQStanAdapter()

    config = RecoveryStudyConfig(
        n_replications=10,
        n_subjects=20,
        param_dists=(
            ParamDist("alpha_self", stats.norm(-0.847, 0.5), scale="unconstrained"),
            ParamDist("alpha_other", stats.norm(-0.847, 0.5), scale="unconstrained"),
            ParamDist("beta", stats.norm(1.687, 0.5), scale="unconstrained"),
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
