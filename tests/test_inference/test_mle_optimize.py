"""Tests for MLE optimization."""

from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.inference.mle.objective import log_likelihood_simple
from comp_model.inference.mle.optimize import MleOptimizerConfig, fit_mle_simple
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.runtime.engine import SimulationConfig, simulate_subject
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA
from comp_model.tasks.spec import BlockSpec, TaskSpec


def _task() -> TaskSpec:
    """Create a simple task for MLE smoke tests.

    Returns
    -------
    TaskSpec
        One-block asocial bandit task.
    """

    return TaskSpec(
        task_id="bandit",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=12,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
        ),
    )


def test_fit_mle_simple_returns_restart_metadata_and_finite_fit() -> None:
    """Ensure MLE fitting returns a finite fit result and restart traces.

    Returns
    -------
    None
        This test asserts smoke-test fit properties.
    """

    kernel = AsocialQLearningKernel()
    generating_params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    subject = simulate_subject(
        task=_task(),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=generating_params,
        config=SimulationConfig(seed=3),
        subject_id="s1",
    )

    result = fit_mle_simple(
        kernel,
        subject,
        ASOCIAL_BANDIT_SCHEMA,
        MleOptimizerConfig(n_restarts=3, seed=0, max_iter=50),
    )

    baseline_log_likelihood = log_likelihood_simple(
        kernel,
        subject,
        {"alpha": 0.0, "beta": 1.0},
        ASOCIAL_BANDIT_SCHEMA,
    )

    assert result.subject_id == "s1"
    assert len(result.all_candidates) == 3
    assert len(result.all_log_likelihoods) == 3
    assert result.log_likelihood >= baseline_log_likelihood
