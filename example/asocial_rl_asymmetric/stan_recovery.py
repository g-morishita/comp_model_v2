"""Parameter recovery for the asocial asymmetric RL kernel using hierarchical Stan.

Recovers per-subject alpha_pos, alpha_neg, and beta under a hierarchical
(STUDY_SUBJECT) model that places population-level priors on all parameters.

The table reports two kinds of rows:

Subject-level (N = n_replications x n_subjects):
    alpha_pos, alpha_neg, beta
    True value = each subject's sampled constrained parameter.
    Estimated  = posterior mean of that subject's parameter.

Population-level (N = n_replications):
    alpha_pos_pop, alpha_neg_pop, beta_pop
    True value  = empirical mean of constrained params across subjects in
                  that replication (varies per replication -> correlation valid).
    Estimated   = posterior mean of the Stan GQ scalar
                  ``{param}_pop = inv_logit(mu_{param}_z)`` etc.
    Coverage    = fraction of replications where the true empirical mean
                  falls inside the posterior HDI.

Usage:
    uv run python example/asocial_rl_asymmetric/stan_recovery.py
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

    kernel = AsocialRlAsymmetricKernel()
    adapter = AsocialRlAsymmetricStanAdapter()

    config = RecoveryStudyConfig(
        n_replications=10,
        n_subjects=20,
        param_dists=(
            ParamDist("alpha_pos", stats.uniform(0.0, 1.0), scale="constrained"),
            ParamDist("alpha_neg", stats.uniform(0.0, 1.0), scale="constrained"),
            ParamDist("beta", stats.uniform(0.5, 14.5), scale="constrained"),
        ),
        task=task,
        env_factory=lambda: StationaryBanditEnvironment(
            n_actions=N_ACTIONS, reward_probs=(0.8, 0.2)
        ),
        kernel=kernel,
        schema=ASOCIAL_BANDIT_SCHEMA,
        inference_config=InferenceConfig(
            hierarchy=HierarchyStructure.STUDY_SUBJECT,
            backend="stan",
            stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
        ),
        adapter=adapter,
        simulation_base_seed=42,
        max_workers=4,
    )

    print(f"Running {config.n_replications} reps x {config.n_subjects} subjects...")
    print(f"  subject-level  N per parameter : {config.n_replications * config.n_subjects}")
    print(f"  population-level N per parameter: {config.n_replications}")
    result = run_recovery(config)
    metrics = compute_recovery_metrics(result)
    print("\nRecovery Metrics (hierarchical Stan):")
    print(recovery_table(metrics))
    print("\nDone.")


if __name__ == "__main__":
    main()
