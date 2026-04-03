"""Tests for the asocial sticky Stan adapter."""

from pathlib import Path

from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.inference.bayes.stan.adapters.asocial_rl_sticky import (
    AsocialRlStickyStanAdapter,
)
from comp_model.inference.config import HierarchyStructure
from comp_model.models.condition.shared_delta import SharedDeltaLayout
from comp_model.models.kernels.asocial_rl_sticky import AsocialRlStickyKernel
from comp_model.runtime.engine import SimulationConfig, simulate_subject
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA
from comp_model.tasks.spec import BlockSpec, TaskSpec


def _task() -> TaskSpec:
    """Create a simple one-block task for sticky-adapter tests."""

    return TaskSpec(
        task_id="asocial-sticky-adapter",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=4,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
        ),
    )


def test_asocial_sticky_adapter_program_paths_exist() -> None:
    """Ensure the sticky adapter resolves existing Stan program files."""

    adapter = AsocialRlStickyStanAdapter()

    for hierarchy in HierarchyStructure:
        path = Path(adapter.stan_program_path(hierarchy))
        assert path.exists(), f"Missing Stan program: {path}"


def test_asocial_sticky_adapter_builds_subject_stan_data() -> None:
    """Ensure the sticky adapter exports Stan data for a simulated subject."""

    kernel = AsocialRlStickyKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0, "stickiness": 0.0})
    subject = simulate_subject(
        task=_task(),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=params,
        config=SimulationConfig(seed=23),
        subject_id="s1",
    )
    adapter = AsocialRlStickyStanAdapter()

    stan_data = adapter.build_stan_data(
        subject,
        ASOCIAL_BANDIT_SCHEMA,
        HierarchyStructure.SUBJECT_SHARED,
    )

    assert stan_data["A"] == 2
    assert stan_data["E"] == 8
    assert "alpha_prior_family" in stan_data
    assert "stickiness_prior_family" in stan_data


def test_asocial_sticky_adapter_adds_condition_data_for_subject_condition_fit() -> None:
    """Ensure the sticky adapter augments Stan data with condition indices."""

    task = TaskSpec(
        task_id="asocial-sticky-conditioned",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=2,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
            BlockSpec(
                condition="social",
                n_trials=2,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
        ),
    )
    kernel = AsocialRlStickyKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0, "stickiness": 0.0})
    subject = simulate_subject(
        task=task,
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=params,
        config=SimulationConfig(seed=29),
        subject_id="s1",
    )
    adapter = AsocialRlStickyStanAdapter()
    layout = SharedDeltaLayout(
        kernel_spec=kernel.spec(),
        conditions=("baseline", "social"),
        baseline_condition="baseline",
    )

    stan_data = adapter.build_stan_data(
        subject,
        ASOCIAL_BANDIT_SCHEMA,
        HierarchyStructure.SUBJECT_BLOCK_CONDITION,
        layout=layout,
    )

    assert stan_data["C"] == 2
    assert stan_data["baseline_cond"] == 1
    assert stan_data["step_condition"] == [1, 1, 1, 1, 2, 2, 2, 2]


def test_asocial_sticky_adapter_subject_param_names() -> None:
    """Ensure the sticky adapter reports the stickiness parameter."""

    adapter = AsocialRlStickyStanAdapter()
    assert adapter.subject_param_names() == ("alpha", "beta", "stickiness")
