"""Tests for conditioned MLE fitting."""

from types import SimpleNamespace

import numpy as np

from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.inference.mle.optimize import MleOptimizerConfig, fit_mle_conditioned
from comp_model.models.condition.shared_delta import SharedDeltaLayout
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.models.kernels.transforms import get_transform
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


def test_fit_mle_conditioned_uses_bounds_for_shared_terms_and_fallback_for_deltas(
    monkeypatch,
) -> None:
    """Conditioned MLE seeds shared terms from bounds and deltas from fallback z bounds."""

    captured_starts: list[np.ndarray] = []

    def fake_minimize(objective, start, method, tol, options):  # type: ignore[no-untyped-def]
        start_array = np.asarray(start, dtype=float)
        captured_starts.append(start_array.copy())
        return SimpleNamespace(x=start_array, fun=0.0, success=True)

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

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

    fit_mle_conditioned(
        kernel,
        layout,
        subject,
        ASOCIAL_BANDIT_SCHEMA,
        MleOptimizerConfig(n_restarts=2, seed=0, restart_upper_bound=2.0, max_iter=1),
    )

    assert len(captured_starts) == 2

    alpha_transform = get_transform("sigmoid")
    beta_transform = get_transform("softplus")

    default_start = captured_starts[0]
    assert default_start[0] == 0.0
    assert np.isclose(beta_transform.forward(default_start[1]), 1.0)
    assert default_start[2] == 0.0
    assert default_start[3] == 0.0

    random_start = captured_starts[1]
    random_alpha_shared = alpha_transform.forward(random_start[0])
    random_beta_shared = beta_transform.forward(random_start[1])
    assert 0.0 <= random_alpha_shared <= 1.0
    assert 0.0 <= random_beta_shared <= 2.0
    assert -2.0 <= random_start[2] <= 2.0
    assert -2.0 <= random_start[3] <= 2.0
