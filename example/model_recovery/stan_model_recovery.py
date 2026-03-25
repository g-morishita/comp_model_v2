"""Model recovery analysis using WAIC: symmetric vs. asymmetric RL.

Fits both models using hierarchical Stan and selects the winner per
replication using WAIC (Widely Applicable Information Criterion).

WAIC is computed from the per-observation log-likelihoods in the Stan
generated quantities block and accounts for effective parameter count
via the variance of the log-likelihood draws.

Usage
-----
    uv run python example/model_recovery/stan_model_recovery.py
"""

from pathlib import Path

from scipy import stats

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference.bayes.stan import (
    AsocialQLearningStanAdapter,
    AsocialRlAsymmetricStanAdapter,
    StanFitConfig,
)
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.models.kernels import AsocialQLearningKernel, AsocialRlAsymmetricKernel
from comp_model.recovery import ParamDist
from comp_model.recovery.model import (
    CandidateModelSpec,
    GeneratingModelSpec,
    ModelRecoveryConfig,
    compute_confusion_matrix,
    compute_recovery_rates,
    model_recovery_confusion_table,
    model_recovery_rate_table,
    plot_confusion_matrix,
    plot_recovery_rates,
    run_model_recovery,
    save_confusion_matrix_csv,
    save_replication_csv,
)
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec

# ---------------------------------------------------------------------------
# Task setup
# ---------------------------------------------------------------------------

N_ACTIONS = 2
N_TRIALS = 200

task = TaskSpec(
    task_id="model_recovery_bandit",
    blocks=(
        BlockSpec(
            condition="default",
            n_trials=N_TRIALS,
            schema=ASOCIAL_BANDIT_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
    ),
)


def env_factory() -> StationaryBanditEnvironment:
    return StationaryBanditEnvironment(n_actions=N_ACTIONS, reward_probs=(0.75, 0.25))


# ---------------------------------------------------------------------------
# Stan inference config (shared across candidates)
# ---------------------------------------------------------------------------

STAN_CONFIG = InferenceConfig(
    hierarchy=HierarchyStructure.STUDY_SUBJECT,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=500, n_samples=500, n_chains=4, seed=42),
)

# ---------------------------------------------------------------------------
# Generating model specs
# ---------------------------------------------------------------------------

generating_models = (
    GeneratingModelSpec(
        name="Symmetric",
        kernel=AsocialQLearningKernel(),
        param_dists=(
            ParamDist("alpha", stats.uniform(0.1, 0.6)),  # alpha in [0.1, 0.7]
            ParamDist("beta", stats.uniform(1.0, 9.0)),
        ),
    ),
    GeneratingModelSpec(
        name="Asymmetric",
        kernel=AsocialRlAsymmetricKernel(),
        param_dists=(
            ParamDist("alpha_pos", stats.uniform(0.5, 0.4)),  # alpha_pos in [0.5, 0.9]
            ParamDist("alpha_neg", stats.uniform(0.05, 0.25)),  # alpha_neg in [0.05, 0.3]
            ParamDist("beta", stats.uniform(1.0, 9.0)),
        ),
    ),
)

# ---------------------------------------------------------------------------
# Candidate model specs
# ---------------------------------------------------------------------------

candidate_models = (
    CandidateModelSpec(
        name="Symmetric",
        kernel=AsocialQLearningKernel(),
        inference_config=STAN_CONFIG,
        adapter=AsocialQLearningStanAdapter(),
    ),
    CandidateModelSpec(
        name="Asymmetric",
        kernel=AsocialRlAsymmetricKernel(),
        inference_config=STAN_CONFIG,
        adapter=AsocialRlAsymmetricStanAdapter(),
    ),
)

# ---------------------------------------------------------------------------
# Recovery study configuration
# ---------------------------------------------------------------------------

config = ModelRecoveryConfig(
    generating_models=generating_models,
    candidate_models=candidate_models,
    n_replications=10,
    n_subjects=20,
    task=task,
    env_factory=env_factory,
    schema=ASOCIAL_BANDIT_SCHEMA,
    criterion="waic",
    simulation_base_seed=42,
)

OUTPUT_DIR = Path("output/stan_model_recovery")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        f"Running model recovery: {len(config.generating_models)} generating models x "
        f"{len(config.candidate_models)} candidates x {config.n_replications} replications "
        f"x {config.n_subjects} subjects"
    )
    print(f"Criterion: {config.criterion.upper()}\n")

    result = run_model_recovery(config)

    model_names = [spec.name for spec in config.generating_models]
    matrix = compute_confusion_matrix(result)
    rates = compute_recovery_rates(result)

    print("Confusion Matrix (rows = generating model, cols = selected model):")
    print(model_recovery_confusion_table(matrix, model_names))
    print()
    print("Recovery Rates:")
    print(model_recovery_rate_table(rates, result))

    # --- CSV export ---
    save_replication_csv(result, OUTPUT_DIR / "replications.csv")
    save_confusion_matrix_csv(result, OUTPUT_DIR / "confusion_matrix.csv")
    print(f"\nCSV saved to {OUTPUT_DIR}/")

    # --- Plots ---
    plot_confusion_matrix(result, save_path=OUTPUT_DIR / "confusion_matrix.png")
    plot_recovery_rates(result, save_path=OUTPUT_DIR / "recovery_rates.png")
    print(f"Plots saved to {OUTPUT_DIR}/")
