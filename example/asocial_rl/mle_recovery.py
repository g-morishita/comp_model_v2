"""Parameter recovery analysis using the recovery module.

Demonstrates MLE-based parameter recovery for an asocial Q-learning kernel
on a stationary bandit task. Runs multiple replications with parallel
per-subject fitting and reports standard recovery metrics.

Usage:
    uv run python example/parameter_recovery.py
"""

import numpy as np
from scipy import stats

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.models.kernels import AsocialQLearningKernel
from comp_model.recovery import (
    ParamDist,
    ParameterRecoveryConfig,
    compute_parameter_recovery_metrics,
    parameter_recovery_table,
    run_parameter_recovery,
)
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec


def main() -> None:
    # -- 1. Task setup ---------------------------------------------------------
    N_ACTIONS = 2
    N_TRIALS = 1000

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

    # -- 2. Configure recovery study -------------------------------------------
    kernel = AsocialQLearningKernel()

    config = ParameterRecoveryConfig(
        n_replications=1,
        n_subjects=100,
        param_dists=(
            ParamDist("alpha", stats.uniform(0.0, 1.0)),
            ParamDist("beta", stats.uniform(0.1, 20.0)),
        ),
        task=task,
        env_factory=lambda: StationaryBanditEnvironment(
            n_actions=N_ACTIONS, reward_probs=(0.8, 0.2)
        ),
        kernel=kernel,
        schema=ASOCIAL_BANDIT_SCHEMA,
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
    metrics = compute_parameter_recovery_metrics(result, transforms={"beta": np.log})
    print("\nRecovery Metrics:")
    print(parameter_recovery_table(metrics))

    print("\nDone.")


if __name__ == "__main__":
    main()
