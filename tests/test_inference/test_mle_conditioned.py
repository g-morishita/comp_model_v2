"""Tests for conditioned MLE fitting."""

from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.inference.mle.optimize import MleOptimizerConfig, fit_mle_conditioned
from comp_model.models.condition.shared_delta import SharedDeltaLayout
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.runtime.engine import SimulationConfig, simulate_subject
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA
from comp_model.tasks.spec import BlockSpec, TaskSpec


def _two_condition_task() -> TaskSpec:
    """Create a two-block task with distinct conditions.

    Returns
    -------
    TaskSpec
        Two-condition task for conditioned MLE tests.
    """

    return TaskSpec(
        task_id="bandit-two-condition",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=6,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
            BlockSpec(
                condition="social",
                n_trials=6,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
        ),
    )


def test_fit_mle_conditioned_returns_params_by_condition() -> None:
    """Ensure conditioned fitting returns reconstructed parameters per condition.

    Returns
    -------
    None
        This test asserts conditioned fit metadata.
    """

    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    subject = simulate_subject(
        task=_two_condition_task(),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=params,
        config=SimulationConfig(seed=5),
        subject_id="s1",
    )
    layout = SharedDeltaLayout(
        kernel_spec=AsocialQLearningKernel.spec(),
        conditions=("baseline", "social"),
        baseline_condition="baseline",
    )

    result = fit_mle_conditioned(
        kernel,
        layout,
        subject,
        ASOCIAL_BANDIT_SCHEMA,
        MleOptimizerConfig(n_restarts=2, seed=0, max_iter=30),
    )

    assert result.params_by_condition is not None
    assert set(result.params_by_condition) == {"baseline", "social"}
