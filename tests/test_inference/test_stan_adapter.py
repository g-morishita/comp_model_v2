"""Tests for the asocial Stan adapter."""

from pathlib import Path

import pytest

from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.inference.bayes.stan.adapters.asocial_q_learning import (
    AsocialQLearningStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.base import require_layout_for_condition_hierarchy
from comp_model.inference.config import HierarchyStructure
from comp_model.models.condition.shared_delta import SharedDeltaLayout
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.runtime.engine import SimulationConfig, simulate_subject
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA
from comp_model.tasks.spec import BlockSpec, TaskSpec


def _task() -> TaskSpec:
    """Create a simple one-block task for adapter tests.

    Returns
    -------
    TaskSpec
        One-block asocial bandit task.
    """

    return TaskSpec(
        task_id="adapter",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=4,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
        ),
    )


def test_asocial_adapter_program_paths_exist() -> None:
    """Ensure the adapter resolves existing Stan program files.

    Returns
    -------
    None
        This test asserts program path existence.
    """

    adapter = AsocialQLearningStanAdapter()

    subject_path = Path(adapter.stan_program_path(HierarchyStructure.SUBJECT_SHARED))
    hierarchy_path = Path(adapter.stan_program_path(HierarchyStructure.STUDY_SUBJECT))

    assert subject_path.exists()
    assert hierarchy_path.exists()


def test_asocial_adapter_builds_subject_stan_data() -> None:
    """Ensure the adapter exports Stan data for a simulated subject.

    Returns
    -------
    None
        This test asserts required Stan data keys.
    """

    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    subject = simulate_subject(
        task=_task(),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=params,
        config=SimulationConfig(seed=23),
        subject_id="s1",
    )
    adapter = AsocialQLearningStanAdapter()

    stan_data = adapter.build_stan_data(
        subject,
        ASOCIAL_BANDIT_SCHEMA,
        HierarchyStructure.SUBJECT_SHARED,
    )

    assert stan_data["A"] == 2
    assert stan_data["E"] == 8  # 4 trials x 2 steps (action + self-update)
    assert "alpha_prior_family" in stan_data


def test_asocial_adapter_adds_condition_data_for_subject_condition_fit() -> None:
    """Ensure the adapter augments Stan data with condition indices when requested.

    Returns
    -------
    None
        This test asserts conditioned Stan export fields.
    """

    task = TaskSpec(
        task_id="adapter-conditioned",
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
    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    subject = simulate_subject(
        task=task,
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=params,
        config=SimulationConfig(seed=29),
        subject_id="s1",
    )
    adapter = AsocialQLearningStanAdapter()
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
    # 2 trials x 2 steps per trial per condition = 4 steps per condition
    assert stan_data["step_condition"] == [1, 1, 1, 1, 2, 2, 2, 2]


@pytest.mark.parametrize(
    "hierarchy",
    [
        HierarchyStructure.SUBJECT_BLOCK_CONDITION,
        HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
    ],
)
def test_require_layout_raises_for_condition_hierarchy_without_layout(
    hierarchy: HierarchyStructure,
) -> None:
    """require_layout_for_condition_hierarchy raises ValueError when layout is None.

    Parameters
    ----------
    hierarchy
        Condition-aware hierarchy under test.

    Returns
    -------
    None
        This test asserts a ValueError is raised.
    """
    with pytest.raises(ValueError, match="requires a SharedDeltaLayout"):
        require_layout_for_condition_hierarchy(hierarchy, layout=None)


@pytest.mark.parametrize(
    "hierarchy",
    [
        HierarchyStructure.SUBJECT_SHARED,
        HierarchyStructure.STUDY_SUBJECT,
    ],
)
def test_require_layout_does_not_raise_for_non_condition_hierarchy(
    hierarchy: HierarchyStructure,
) -> None:
    """require_layout_for_condition_hierarchy is silent for non-condition hierarchies.

    Parameters
    ----------
    hierarchy
        Non-condition-aware hierarchy under test.

    Returns
    -------
    None
        This test asserts no exception is raised.
    """
    require_layout_for_condition_hierarchy(hierarchy, layout=None)  # must not raise


def test_build_stan_data_raises_for_condition_hierarchy_without_layout() -> None:
    """build_stan_data raises ValueError when a condition-aware hierarchy lacks a layout.

    Returns
    -------
    None
        This test asserts a ValueError is raised from the adapter.
    """
    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    subject = simulate_subject(
        task=_task(),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=params,
        config=SimulationConfig(seed=42),
        subject_id="s1",
    )
    adapter = AsocialQLearningStanAdapter()

    with pytest.raises(ValueError, match="requires a SharedDeltaLayout"):
        adapter.build_stan_data(
            subject,
            ASOCIAL_BANDIT_SCHEMA,
            HierarchyStructure.SUBJECT_BLOCK_CONDITION,
        )
