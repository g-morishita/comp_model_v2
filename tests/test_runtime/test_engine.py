"""Tests for the simulation engine."""

from comp_model.data.validation import validate_subject
from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.runtime.engine import SimulationConfig, simulate_dataset, simulate_subject
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA
from comp_model.tasks.spec import BlockSpec, TaskSpec


def _task_spec() -> TaskSpec:
    """Create a simple one-block bandit task.

    Returns
    -------
    TaskSpec
        Task specification for runtime tests.
    """

    return TaskSpec(
        task_id="bandit",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=3,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
        ),
    )


def test_simulate_subject_returns_valid_hierarchical_data() -> None:
    """Ensure subject simulation produces schema-valid hierarchical data.

    Returns
    -------
    None
        This test validates the simulated subject.
    """

    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})

    subject = simulate_subject(
        task=_task_spec(),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=params,
        config=SimulationConfig(seed=1),
        subject_id="s1",
    )

    validate_subject(subject, schema=ASOCIAL_BANDIT_SCHEMA)
    assert len(subject.blocks) == 1
    assert len(subject.blocks[0].trials) == 3


def test_simulate_dataset_creates_one_subject_per_parameter_set() -> None:
    """Ensure dataset simulation creates one subject per parameter entry.

    Returns
    -------
    None
        This test asserts the dataset size and subject identifiers.
    """

    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})

    dataset = simulate_dataset(
        task=_task_spec(),
        env_factory=lambda: StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params_per_subject={"s1": params, "s2": params},
        config=SimulationConfig(seed=1),
    )

    assert tuple(subject.subject_id for subject in dataset.subjects) == ("s1", "s2")
