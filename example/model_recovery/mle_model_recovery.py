"""Model recovery analysis: asocial RL (symmetric) vs. asocial RL (asymmetric).

This script tests whether AIC-based model selection can correctly identify the
true generative model when data are simulated from either:

  - **AsocialRlKernel** (``AsocialQLearningKernel``): a single learning
    rate ``alpha`` applied to all prediction errors.
  - **AsocialRlAsymmetricKernel**: separate rates ``alpha_pos`` and
    ``alpha_neg`` for positive and negative prediction errors.

The asymmetric model is more complex (3 vs. 2 parameters) and nests the
symmetric model when ``alpha_pos == alpha_neg``.  The study asks:

  1. When data come from the **symmetric** model, does AIC prefer symmetric
     (penalising the extra free parameter in asymmetric)?
  2. When data come from the **asymmetric** model with *truly different* rates,
     does AIC recover that the asymmetric model is the true one?

Usage
-----
    uv run python example/model_recovery/mle_model_recovery.py
"""

from scipy import stats

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.models.kernels import AsocialQLearningKernel, AsocialRlAsymmetricKernel
from comp_model.recovery import ParamDist
from comp_model.recovery.model import (
    CandidateModelSpec,
    GeneratingModelSpec,
    ModelRecoveryConfig,
    confusion_matrix,
    confusion_matrix_table,
    recovery_rate_table,
    recovery_rates,
    run_model_recovery,
)
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec

# ---------------------------------------------------------------------------
# Task setup
# ---------------------------------------------------------------------------

N_ACTIONS = 2
N_TRIALS = 200  # trials per subject — more trials → easier to discriminate models

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
# Inference config (shared across candidates)
# ---------------------------------------------------------------------------

MLE_CONFIG = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_SHARED,
    backend="mle",
    mle_config=MleOptimizerConfig(n_restarts=5, seed=0),
)

# ---------------------------------------------------------------------------
# Generating model specs
#
# Symmetric: alpha ~ U(0.1, 0.7), beta ~ U(1, 10)
#   → a single moderate learning rate
#
# Asymmetric: alpha_pos ~ U(0.5, 0.9), alpha_neg ~ U(0.05, 0.3), beta ~ U(1, 10)
#   → strong positive RPE learning, weak negative RPE learning (optimism bias)
# ---------------------------------------------------------------------------

generating_models = (
    GeneratingModelSpec(
        name="Symmetric",
        kernel=AsocialQLearningKernel(),
        param_dists=(
            ParamDist("alpha", stats.uniform(0.1, 0.6)),  # alpha in [0.1, 0.7]
            ParamDist("beta", stats.uniform(1.0, 9.0)),  # beta  in [1, 10]
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
# Candidate model specs (the models we try to fit to each simulated dataset)
# ---------------------------------------------------------------------------

candidate_models = (
    CandidateModelSpec(
        name="Symmetric",
        kernel=AsocialQLearningKernel(),
        inference_config=MLE_CONFIG,
    ),
    CandidateModelSpec(
        name="Asymmetric",
        kernel=AsocialRlAsymmetricKernel(),
        inference_config=MLE_CONFIG,
    ),
)

# ---------------------------------------------------------------------------
# Recovery study configuration
# ---------------------------------------------------------------------------

config = ModelRecoveryConfig(
    generating_models=generating_models,
    candidate_models=candidate_models,
    n_replications=20,
    n_subjects=30,
    task=task,
    env_factory=env_factory,
    schema=ASOCIAL_BANDIT_SCHEMA,
    criterion="aic",
    simulation_base_seed=42,
)

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
    matrix = confusion_matrix(result)
    rates = recovery_rates(result)

    print("Confusion Matrix (rows = generating model, cols = selected model):")
    print(confusion_matrix_table(matrix, model_names))
    print()
    print("Recovery Rates:")
    print(recovery_rate_table(rates, result))
