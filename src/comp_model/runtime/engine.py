"""Simulation engine for generating event-based subject data.

The runtime executes task structure and environment dynamics while delegating
choice and learning to a model kernel. The resulting event traces are the same
objects later consumed by validation, replay, and Stan export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from comp_model.data.extractors import DecisionTrialView, replay_trial_steps
from comp_model.data.schema import Block, Dataset, Event, EventPhase, SubjectData, Trial

if TYPE_CHECKING:
    from collections.abc import Callable

    from comp_model.environments.base import Environment
    from comp_model.models.kernels.base import ModelKernel
    from comp_model.tasks.spec import TaskSpec


StateT = TypeVar("StateT")
ParamsT = TypeVar("ParamsT")


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Configuration for stochastic simulation runs.

    Attributes
    ----------
    seed
        Optional random seed for the subject-level RNG.

    Notes
    -----
    Dataset simulation offsets this base seed by subject index so each simulated
    subject receives an independent but reproducible random stream.
    """

    seed: int | None = None


def simulate_subject(
    *,
    task: TaskSpec,
    env: Environment,
    kernel: ModelKernel[StateT, ParamsT],
    params: ParamsT,
    demonstrator_kernel: ModelKernel[object, object] | None = None,
    demonstrator_params: object | None = None,
    config: SimulationConfig,
    subject_id: str = "sim_subject",
) -> SubjectData:
    """Simulate a single subject through an entire task.

    Parameters
    ----------
    task
        Task specification to execute.
    env
        Environment instance for the task.
    kernel
        Model kernel for the subject.
    params
        Parsed subject kernel parameters.
    demonstrator_kernel
        Optional kernel for the demonstrator agent.
    demonstrator_params
        Parsed demonstrator kernel parameters. Required when
        ``demonstrator_kernel`` is provided.
    config
        Simulation configuration.
    subject_id
        Identifier assigned to the simulated subject.

    Returns
    -------
    SubjectData
        Simulated hierarchical event trace for one subject.

    Notes
    -----
    The simulation loop steps through the schema position by position. At each
    DECISION step, the relevant agent's kernel supplies the action. After the
    full trial is assembled, ``replay_trial_steps`` drives ``next_state`` calls
    in schema order — ensuring that updates fire at their declared positions
    (e.g. a social update before the subject's own decision in pre-choice
    schemas).
    """

    rng = np.random.default_rng(config.seed)
    blocks: list[Block] = []

    n_actions = _infer_n_actions_from_task(task)
    states: dict[str, object] = {"subject": kernel.initial_state(n_actions, params)}
    if demonstrator_kernel is not None and demonstrator_params is not None:
        states["demonstrator"] = demonstrator_kernel.initial_state(n_actions, demonstrator_params)

    reset_policy = kernel.spec().state_reset_policy

    for block_index, block_spec in enumerate(task.blocks):
        env.reset(block_spec, rng=rng)
        schema = block_spec.schema

        if reset_policy == "per_block" and block_index > 0:
            states["subject"] = kernel.initial_state(n_actions, params)
            if demonstrator_kernel is not None and demonstrator_params is not None:
                states["demonstrator"] = demonstrator_kernel.initial_state(
                    n_actions, demonstrator_params
                )

        trials: list[Trial] = []

        for trial_index in range(block_spec.n_trials):
            trial_events: list[Event] = []

            for step_index, schema_step in enumerate(schema.steps):
                if schema_step.action_required:
                    actor = schema_step.actor_id
                    input_event = _find_actor_input(
                        trial_events,
                        schema_step.node_id,
                        actor,
                    )
                    if input_event is None:
                        raise ValueError(
                            f"Trial {trial_index}: no INPUT found for actor {actor!r} "
                            f"before DECISION at step {step_index}"
                        )

                    raw_observation = input_event.payload.get("observation")
                    observation: dict[str, Any] = {}
                    if isinstance(raw_observation, dict):
                        observation = cast("dict[str, Any]", raw_observation)
                    available_actions = tuple(input_event.payload["available_actions"])
                    partial_view = DecisionTrialView(
                        trial_index=trial_index,
                        available_actions=available_actions,
                        observation=observation,
                    )

                    if actor == "subject":
                        probabilities = kernel.action_probabilities(
                            cast("StateT", states["subject"]), partial_view, params
                        )
                    elif demonstrator_kernel is not None and demonstrator_params is not None:
                        probabilities = demonstrator_kernel.action_probabilities(
                            states["demonstrator"], partial_view, demonstrator_params
                        )
                    else:
                        raise ValueError(
                            f"Trial {trial_index}: DECISION step requires actor {actor!r} "
                            f"but no kernel provided for that actor"
                        )

                    action_index = int(
                        rng.choice(len(available_actions), p=np.array(probabilities))
                    )
                    action = available_actions[action_index]
                    events = env.step(action=action)
                else:
                    events = env.step(action=None)

                trial_events.extend(events)

            trial = Trial(trial_index=trial_index, events=tuple(trial_events))
            for event_type, learner_id, view in replay_trial_steps(trial, schema):
                if event_type == "update":
                    if learner_id == "subject":
                        states["subject"] = kernel.next_state(
                            cast("StateT", states["subject"]), view, params
                        )
                    elif demonstrator_kernel is not None and demonstrator_params is not None:
                        states["demonstrator"] = demonstrator_kernel.next_state(
                            states["demonstrator"], view, demonstrator_params
                        )
            trials.append(trial)

        blocks.append(
            Block(
                block_index=block_index,
                condition=block_spec.condition,
                trials=tuple(trials),
                metadata=block_spec.metadata,
            )
        )

    return SubjectData(subject_id=subject_id, blocks=tuple(blocks))


def simulate_dataset(
    *,
    task: TaskSpec,
    env_factory: Callable[[], Environment],
    kernel: ModelKernel[StateT, ParamsT],
    params_per_subject: dict[str, ParamsT],
    config: SimulationConfig,
) -> Dataset:
    """Simulate a dataset for multiple subjects.

    Parameters
    ----------
    task
        Task specification to execute.
    env_factory
        Factory returning a fresh environment per subject.
    kernel
        Model kernel used to choose and update.
    params_per_subject
        Parsed kernel parameters keyed by subject identifier.
    config
        Simulation configuration shared across subjects.

    Returns
    -------
    Dataset
        Simulated dataset across all requested subjects.

    Notes
    -----
    Subjects are simulated independently with fresh environments returned by
    ``env_factory``. Subject order follows the insertion order of
    ``params_per_subject``.
    """

    subjects: list[SubjectData] = []
    for offset, (subject_id, params) in enumerate(params_per_subject.items()):
        env = env_factory()
        subject_config = SimulationConfig(
            seed=None if config.seed is None else config.seed + offset,
        )
        subjects.append(
            simulate_subject(
                task=task,
                env=env,
                kernel=kernel,
                params=params,
                config=subject_config,
                subject_id=subject_id,
            )
        )
    return Dataset(subjects=tuple(subjects))


def _find_actor_input(events: list[Event], node_id: str, actor_id: str) -> Event | None:
    """Find the INPUT event for a given actor preceding a decision.

    Parameters
    ----------
    events
        Events accumulated so far for the current trial.
    node_id
        Node identifier of the decision point.
    actor_id
        Actor identifier whose INPUT event is sought.

    Returns
    -------
    Event | None
        Matching INPUT event, if any.

    Notes
    -----
    This helper searches only the events accumulated for the current in-progress
    trial, which matches the runtime contract that an action-required decision
    must have already been preceded by its actor's INPUT event.
    """

    for event in events:
        if (
            event.phase == EventPhase.INPUT
            and event.node_id == node_id
            and event.actor_id == actor_id
        ):
            return event
    return None


def _infer_n_actions_from_task(task: TaskSpec) -> int:
    """Infer the number of actions from task metadata.

    Parameters
    ----------
    task
        Task specification whose block metadata are inspected.

    Returns
    -------
    int
        Unique action count specified across blocks.

    Raises
    ------
    ValueError
        Raised when action counts are missing or inconsistent.

    Notes
    -----
    The runtime currently relies on ``BlockSpec.metadata['n_actions']`` rather
    than inferring action count from a model or environment. All blocks must
    agree on the same action count.
    """

    n_actions_values: set[int] = set()
    for block_spec in task.blocks:
        n_actions = block_spec.metadata.get("n_actions")
        if n_actions is not None:
            n_actions_values.add(int(n_actions))

    if len(n_actions_values) != 1:
        raise ValueError(
            "Cannot infer n_actions from task. Set metadata['n_actions'] on each BlockSpec, "
            "or all blocks must agree."
        )
    return n_actions_values.pop()
