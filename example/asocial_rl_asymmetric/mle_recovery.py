"""Parameter recovery for the asocial asymmetric RL kernel using MLE.

Demonstrates MLE-based parameter recovery for an asocial asymmetric RL kernel
on a stationary bandit task.  Separate learning rates for positive (alpha_pos)
and negative (alpha_neg) prediction errors are recovered independently.

Usage:
    uv run python example/asocial_rl_asymmetric/mle_recovery.py
"""

import numpy as np
from scipy import stats

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
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

    kernel = AsocialRlAsymmetricKernel()

    config = RecoveryStudyConfig(
        n_replications=1,
        n_subjects=100,
        param_dists=(
            ParamDist("alpha_pos", stats.uniform(0.0, 1.0)),
            ParamDist("alpha_neg", stats.uniform(0.0, 1.0)),
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

    print(f"Running {config.n_replications} replications x {config.n_subjects} subjects...")
    result = run_recovery(config)

    metrics = compute_recovery_metrics(result, transforms={"beta": np.log})
    print("\nRecovery Metrics:")
    print(recovery_table(metrics))
    print("\nDone.")


if __name__ == "__main__":
    main()
