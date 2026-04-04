"""Tests for the simulation engine."""

from dataclasses import dataclass
from typing import Any, cast

from comp_model.data.extractors import DecisionTrialView
from comp_model.data.validation import validate_subject
from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.models.kernels.base import ModelKernel, ModelKernelSpec
from comp_model.runtime.engine import SimulationConfig, simulate_dataset, simulate_subject
from comp_model.tasks.schemas import (
    ASOCIAL_BANDIT_SCHEMA,
    SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA,
)
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


def _no_self_outcome_task_spec() -> TaskSpec:
    """Create a two-trial social task without subject self-feedback."""

    return TaskSpec(
        task_id="social-no-self-outcome",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=2,
                schema=SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA,
                metadata={"n_actions": 2},
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class _DecisionMemoryParams:
    """Parameter placeholder for the deterministic decision-memory test kernel."""


@dataclass(frozen=True, slots=True)
class _DecisionMemoryKernel(ModelKernel[int, _DecisionMemoryParams]):
    """Deterministic kernel whose choice flips only if observe_decision runs."""

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        return ModelKernelSpec(model_id="decision_memory_test", parameter_specs=())

    def parse_params(self, raw: dict[str, float]) -> _DecisionMemoryParams:
        del raw
        return _DecisionMemoryParams()

    def initial_state(self, n_actions: int, params: _DecisionMemoryParams) -> int:
        del n_actions, params
        return 0

    def action_probabilities(
        self,
        state: int,
        view: DecisionTrialView,
        params: _DecisionMemoryParams,
    ) -> tuple[float, ...]:
        del view, params
        return (1.0, 0.0) if state % 2 == 0 else (0.0, 1.0)

    def observe_decision(
        self,
        state: int,
        view: DecisionTrialView,
        params: _DecisionMemoryParams,
    ) -> int:
        del params
        if view.actor_id == view.learner_id and view.action is not None:
            return state + 1
        return state

    def update(
        self,
        state: int,
        view: DecisionTrialView,
        params: _DecisionMemoryParams,
    ) -> int:
        del view, params
        return state


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


def test_simulate_subject_runs_decision_hook_without_self_updates() -> None:
    """Decision-time hooks should still run on schemas without subject self-updates."""

    kernel = _DecisionMemoryKernel()
    params = _DecisionMemoryParams()

    subject = simulate_subject(
        task=_no_self_outcome_task_spec(),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.5, 0.5)),
        kernel=kernel,
        params=params,
        demonstrator_kernel=cast("Any", _DecisionMemoryKernel()),
        demonstrator_params=params,
        config=SimulationConfig(seed=1),
        subject_id="s1",
    )

    validate_subject(subject, schema=SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA)
    subject_actions = tuple(
        int(trial.events[1].payload["action"]) for trial in subject.blocks[0].trials
    )
    assert subject_actions == (0, 1)
