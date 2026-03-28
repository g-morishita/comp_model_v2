"""Model recovery analysis among social RL models using WAIC.

Compares four models that span the asocial-to-social learning spectrum:

- Model 1  : AsocialQLearning       — learns only from own reward
- Model 3  : SocialRlSelfRewardDemoReward — learns from own + demo reward
- Model 4c : SocialRlSelfRewardDemoActionMixture — own reward + demo action (mixture)
- Model 7c : SocialRlSelfRewardDemoMixture — own reward + demo reward + demo action (full mixture)

All models are simulated using hierarchical sampling (population mu/sd drawn
from priors, then per-subject parameters drawn from Normal(mu, sd)) and
fitted using the social pre-choice schema with two demonstrator conditions
(strong vs. weak). Hierarchical Stan inference with
STUDY_SUBJECT_BLOCK_CONDITION hierarchy is used for fitting. Model selection
is based on WAIC.

Usage
-----
    uv run python example/model_recovery/social_model_recovery.py
"""

from pathlib import Path

from scipy import stats

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference.bayes.stan import (
    AsocialQLearningStanAdapter,
    SocialRlSelfRewardDemoActionMixtureStanAdapter,
    SocialRlSelfRewardDemoMixtureStanAdapter,
    SocialRlSelfRewardDemoRewardStanAdapter,
    StanFitConfig,
)
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.models import SharedDeltaLayout
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    QParams,
    SocialRlSelfRewardDemoActionMixtureKernel,
    SocialRlSelfRewardDemoMixtureKernel,
    SocialRlSelfRewardDemoRewardKernel,
)
from comp_model.recovery import HierarchicalParamDist
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
from comp_model.tasks import SOCIAL_PRE_CHOICE_SCHEMA, BlockSpec, TaskSpec

# ---------------------------------------------------------------------------
# Task setup (matches param_recovery.py settings)
# ---------------------------------------------------------------------------

N_ACTIONS = 3
N_TRIALS_PER_BLOCK = 60
N_BLOCKS_PER_CONDITION = 2
N_SUBJECTS = 64
N_REPLICATIONS = 20
REWARD_PROBS = (0.25, 0.5, 0.75)
CONDITIONS = ("strong_demo", "weak_demo")

task = TaskSpec(
    task_id="social_model_recovery",
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


def env_factory() -> StationaryBanditEnvironment:
    """Create a fresh stationary bandit environment."""
    return StationaryBanditEnvironment(n_actions=N_ACTIONS, reward_probs=REWARD_PROBS)


# ---------------------------------------------------------------------------
# Demonstrator setup
# ---------------------------------------------------------------------------

demonstrator_kernel = AsocialQLearningKernel()
condition_demonstrator_params = {
    "strong_demo": QParams(alpha=0.3, beta=20.0),
    "weak_demo": QParams(alpha=0.3, beta=1.5),
}

# ---------------------------------------------------------------------------
# Stan inference config (shared across candidates)
# ---------------------------------------------------------------------------

STAN_CONFIG_CONDITION = InferenceConfig(
    hierarchy=HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
    backend="stan",
    stan_config=StanFitConfig(n_warmup=1000, n_samples=1000, n_chains=4, seed=42),
)

# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------

layout_model1 = SharedDeltaLayout(
    kernel_spec=AsocialQLearningKernel.spec(),
    conditions=CONDITIONS,
    baseline_condition="strong_demo",
)

layout_model3 = SharedDeltaLayout(
    kernel_spec=SocialRlSelfRewardDemoRewardKernel.spec(),
    conditions=CONDITIONS,
    baseline_condition="strong_demo",
)

layout_model4c = SharedDeltaLayout(
    kernel_spec=SocialRlSelfRewardDemoActionMixtureKernel.spec(),
    conditions=CONDITIONS,
    baseline_condition="strong_demo",
)

layout_model7c = SharedDeltaLayout(
    kernel_spec=SocialRlSelfRewardDemoMixtureKernel.spec(),
    conditions=CONDITIONS,
    baseline_condition="strong_demo",
)

# ---------------------------------------------------------------------------
# Shared prior configurations
# ---------------------------------------------------------------------------

ALPHA_PRIORS = {"mu_prior": stats.norm(0, 1), "sd_prior": stats.halfnorm(0, 1)}
BETA_PRIORS = {"mu_prior": stats.norm(1, 1), "sd_prior": stats.halfnorm(0, 1)}
DELTA_PRIORS = {"mu_prior": stats.norm(0, 0.5), "sd_prior": stats.halfnorm(0, 0.5)}

# ---------------------------------------------------------------------------
# Generating model specs
# ---------------------------------------------------------------------------

generating_models = (
    # Model 1: AsocialQLearning (alpha, beta)
    GeneratingModelSpec(
        name="M1_Asocial",
        kernel=AsocialQLearningKernel(),
        param_dists=(
            HierarchicalParamDist("alpha", **ALPHA_PRIORS),
            HierarchicalParamDist("beta", **BETA_PRIORS),
            HierarchicalParamDist("alpha__delta", **DELTA_PRIORS),
            HierarchicalParamDist("beta__delta", **DELTA_PRIORS),
        ),
        layout=layout_model1,
    ),
    # Model 3: SocialRlSelfRewardDemoReward (alpha_self, alpha_other, beta)
    GeneratingModelSpec(
        name="M3_DemoReward",
        kernel=SocialRlSelfRewardDemoRewardKernel(),
        param_dists=(
            HierarchicalParamDist("alpha_self", **ALPHA_PRIORS),
            HierarchicalParamDist("alpha_other", **ALPHA_PRIORS),
            HierarchicalParamDist("beta", **BETA_PRIORS),
            HierarchicalParamDist("alpha_self__delta", **DELTA_PRIORS),
            HierarchicalParamDist("alpha_other__delta", **DELTA_PRIORS),
            HierarchicalParamDist("beta__delta", **DELTA_PRIORS),
        ),
        layout=layout_model3,
    ),
    # Model 4c: SocialRlSelfRewardDemoActionMixture
    # (alpha_self, alpha_other_action, w_imitation, beta)
    GeneratingModelSpec(
        name="M4c_DemoAction",
        kernel=SocialRlSelfRewardDemoActionMixtureKernel(),
        param_dists=(
            HierarchicalParamDist("alpha_self", **ALPHA_PRIORS),
            HierarchicalParamDist("alpha_other_action", **ALPHA_PRIORS),
            HierarchicalParamDist("w_imitation", **ALPHA_PRIORS),
            HierarchicalParamDist("beta", **BETA_PRIORS),
            HierarchicalParamDist("alpha_self__delta", **DELTA_PRIORS),
            HierarchicalParamDist("alpha_other_action__delta", **DELTA_PRIORS),
            HierarchicalParamDist("w_imitation__delta", **DELTA_PRIORS),
            HierarchicalParamDist("beta__delta", **DELTA_PRIORS),
        ),
        layout=layout_model4c,
    ),
    # Model 7c: SocialRlSelfRewardDemoMixture
    # (alpha_self, alpha_other_outcome, alpha_other_action, w_imitation, beta)
    GeneratingModelSpec(
        name="M7c_FullMixture",
        kernel=SocialRlSelfRewardDemoMixtureKernel(),
        param_dists=(
            HierarchicalParamDist("alpha_self", **ALPHA_PRIORS),
            HierarchicalParamDist("alpha_other_outcome", **ALPHA_PRIORS),
            HierarchicalParamDist("alpha_other_action", **ALPHA_PRIORS),
            HierarchicalParamDist("w_imitation", **ALPHA_PRIORS),
            HierarchicalParamDist("beta", **BETA_PRIORS),
            HierarchicalParamDist("alpha_self__delta", **DELTA_PRIORS),
            HierarchicalParamDist("alpha_other_outcome__delta", **DELTA_PRIORS),
            HierarchicalParamDist("alpha_other_action__delta", **DELTA_PRIORS),
            HierarchicalParamDist("w_imitation__delta", **DELTA_PRIORS),
            HierarchicalParamDist("beta__delta", **DELTA_PRIORS),
        ),
        layout=layout_model7c,
    ),
)

# ---------------------------------------------------------------------------
# Candidate model specs
# ---------------------------------------------------------------------------

candidate_models = (
    CandidateModelSpec(
        name="M1_Asocial",
        kernel=AsocialQLearningKernel(),
        inference_config=STAN_CONFIG_CONDITION,
        adapter=AsocialQLearningStanAdapter(),
        layout=layout_model1,
    ),
    CandidateModelSpec(
        name="M3_DemoReward",
        kernel=SocialRlSelfRewardDemoRewardKernel(),
        inference_config=STAN_CONFIG_CONDITION,
        adapter=SocialRlSelfRewardDemoRewardStanAdapter(),
        layout=layout_model3,
    ),
    CandidateModelSpec(
        name="M4c_DemoAction",
        kernel=SocialRlSelfRewardDemoActionMixtureKernel(),
        inference_config=STAN_CONFIG_CONDITION,
        adapter=SocialRlSelfRewardDemoActionMixtureStanAdapter(),
        layout=layout_model4c,
    ),
    CandidateModelSpec(
        name="M7c_FullMixture",
        kernel=SocialRlSelfRewardDemoMixtureKernel(),
        inference_config=STAN_CONFIG_CONDITION,
        adapter=SocialRlSelfRewardDemoMixtureStanAdapter(),
        layout=layout_model7c,
    ),
)

# ---------------------------------------------------------------------------
# Recovery study configuration
# ---------------------------------------------------------------------------

config = ModelRecoveryConfig(
    generating_models=generating_models,
    candidate_models=candidate_models,
    n_replications=N_REPLICATIONS,
    n_subjects=N_SUBJECTS,
    task=task,
    env_factory=env_factory,
    schema=SOCIAL_PRE_CHOICE_SCHEMA,
    criterion="waic",
    demonstrator_kernel=demonstrator_kernel,
    condition_demonstrator_params=condition_demonstrator_params,
    simulation_base_seed=42,
)

OUTPUT_DIR = Path("output/social_model_recovery")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        f"Running model recovery: {len(config.generating_models)} generating models x "
        f"{len(config.candidate_models)} candidates x {config.n_replications} replications "
        f"x {config.n_subjects} subjects"
    )
    print(f"Criterion: {config.criterion.upper()}")
    print("Models: M1 (Asocial), M3 (DemoReward), M4c (DemoAction), M7c (FullMixture)\n")

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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_replication_csv(result, OUTPUT_DIR / "replications.csv")
    save_confusion_matrix_csv(result, OUTPUT_DIR / "confusion_matrix.csv")
    print(f"\nCSV saved to {OUTPUT_DIR}/")

    # --- Plots ---
    plot_confusion_matrix(result, save_path=OUTPUT_DIR / "confusion_matrix.png")
    plot_recovery_rates(result, save_path=OUTPUT_DIR / "recovery_rates.png")
    print(f"Plots saved to {OUTPUT_DIR}/")
