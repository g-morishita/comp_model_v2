"""Parameter recovery for the asocial asymmetric RL kernel using hierarchical Stan.

Recovers per-subject alpha_pos, alpha_neg, and beta under a hierarchical
(STUDY_SUBJECT) model that places population-level priors on all parameters.

The generative model draws subject parameters from Normal distributions on
the unconstrained (logit / softplus) scale, matching the Stan hierarchy
exactly.  This enables recovery of both individual-level and population-level
parameters (mu_z / sd_z).

Population-level recovery:
    Each replication contributes ONE (true, estimated) pair for mu_z / sd_z.
    Metrics are aggregated across replications (N = n_replications), so RMSE
    and coverage tell you how well the hierarchical model recovers the fixed
    population mean and SD.

Subject-level recovery:
    Each replication contributes n_subjects pairs.
    N = n_replications x n_subjects.

True population parameters (unconstrained scale):
    alpha_pos:  mu_z = 0.0  (-> alpha_pos ~ 0.50),  sd_z = 0.5
    alpha_neg:  mu_z = -0.847 (-> alpha_neg ~ 0.30), sd_z = 0.5
    beta:       mu_z = 1.5   (-> beta ~ 4.5),         sd_z = 0.5

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

    # Use scale="unconstrained" so that the true mu_z / sd_z are known and can
    # be compared to Stan's population-level estimates across replications.
    # The Stan hierarchy is: subject_z ~ Normal(mu_z, sd_z), so the generating
    # distribution must also be Normal on the unconstrained scale.
    config = RecoveryStudyConfig(
        n_replications=10,
        n_subjects=20,
        param_dists=(
            # alpha_pos: logit scale.  mu_z=0.0 -> mean alpha_pos ~ 0.50
            ParamDist("alpha_pos", stats.norm(0.0, 0.5), scale="unconstrained"),
            # alpha_neg: logit scale.  mu_z=-0.847 -> mean alpha_neg ~ 0.30
            ParamDist("alpha_neg", stats.norm(-0.847, 0.5), scale="unconstrained"),
            # beta: softplus scale.  mu_z=1.5 -> mean beta ~ 4.5
            ParamDist("beta", stats.norm(1.5, 0.5), scale="unconstrained"),
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
    print("  subject-level N per parameter :", config.n_replications * config.n_subjects)
    print("  population-level N per parameter:", config.n_replications)
    result = run_recovery(config)
    metrics = compute_recovery_metrics(result)
    print("\nRecovery Metrics (hierarchical Stan):")
    print(recovery_table(metrics))
    print()
    print("Note: mu_*/sd_* rows are population-level (N = n_replications).")
    print("      Correlation is NaN for population params because true values")
    print("      are fixed constants across replications.")
    print("\nDone.")


if __name__ == "__main__":
    main()
