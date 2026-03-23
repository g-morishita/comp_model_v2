"""Engine for running artificial participants through a task and producing synthetic data.

This module is the bridge between a computational model and a task design.
Given a model (kernel + parameters) and a task design (TaskSpec), it runs the
model through each trial and records everything that happens — choices, rewards,
learning updates — in exactly the same data format that real participant data
uses.

Why is this useful?
- Parameter recovery: simulate data with known parameters, then fit the model
  to the synthetic data to check that you can recover what you put in.
- Posterior predictive checks: simulate data from fitted parameters and compare
  the synthetic behaviour to what real participants did.
- Pilot work: explore what your model predicts before collecting real data.

The output is indistinguishable in format from real data, so all downstream
analysis code (validation, model fitting, visualisation) works on it without
modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

import numpy as np

from comp_model.data.extractors import DecisionTrialView
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
    """Settings that control how random choices are made during simulation.

    Attributes
    ----------
    seed
        An integer that pins the random-number generator to a specific
        sequence, making results exactly reproducible. Set to ``None``
        to use a different random sequence every time.

    Notes
    -----
    When simulating a whole group of participants (see
    :func:`simulate_dataset`), each simulated participant gets their own
    independent random stream. This is achieved by adding the participant's
    position in the list (0, 1, 2, …) to the base seed. So if you set
    ``seed=42``, participant 0 uses seed 42, participant 1 uses seed 43,
    and so on. This keeps participants independent while keeping everything
    reproducible.
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
    """Run one artificial participant through the entire task and record what happened.

    This function is the core simulation loop. It walks through every block and
    every trial, asking the model (``kernel``) what the participant would do,
    asking the task environment what reward they would receive, and recording
    everything as a structured event log.

    The result looks identical to real participant data: a log of choices,
    rewards, and learning updates, organised by block and trial.

    Parameters
    ----------
    task
        The experimental design — which blocks exist, how many trials each
        contains, and in what order events unfold within each trial.
    env
        The task environment that hands out rewards when the participant
        makes a choice (analogous to the slot machines or stimuli in the
        real experiment).
    kernel
        The computational model that decides how the participant chooses
        and learns. Think of it as the "brain" of the artificial participant.
    params
        The specific parameter values for this participant (e.g. their
        learning rate, inverse temperature).
    demonstrator_kernel
        If the task includes a second agent (a demonstrator) whose choices
        the participant can observe, supply that agent's model here.
    demonstrator_params
        Parameter values for the demonstrator. Required when
        ``demonstrator_kernel`` is provided.
    config
        Controls the random seed so simulations are reproducible.
    subject_id
        A name or identifier attached to the simulated participant's data.

    Returns
    -------
    SubjectData
        The full event log for one simulated participant, structured
        identically to real participant data.

    Notes
    -----
    The loop processes each trial in a single pass through the trial schema
    (the sequence of events defined for that block). Events are constructed
    and the model's internal state is updated in the same pass, one schema
    step at a time. This matters for social-learning tasks: if the schema
    says the demonstrator's learning update happens *before* the participant
    decides, that ordering is respected exactly.
    """

    rng = np.random.default_rng(config.seed)
    blocks: list[Block] = []

    n_actions = _infer_n_actions_from_task(task)

    # Give every agent a blank starting state (e.g. all Q-values set to their
    # initial values, no experience accumulated yet).
    states: dict[str, object] = {"subject": kernel.initial_state(n_actions, params)}
    if demonstrator_kernel is not None and demonstrator_params is not None:
        states["demonstrator"] = demonstrator_kernel.initial_state(n_actions, demonstrator_params)

    # Find out whether this model forgets everything between blocks or carries
    # its learned values from one block into the next.
    reset_policy = kernel.spec().state_reset_policy

    for block_index, block_spec in enumerate(task.blocks):
        # Prepare the reward environment for this block (e.g. set new reward
        # probabilities if they change between blocks).
        env.reset(block_spec, rng=rng)
        schema = block_spec.schema
        available_actions = tuple(range(int(block_spec.metadata["n_actions"])))

        # If the model is configured to reset between blocks, wipe its learned
        # state at the start of every block after the first.
        if reset_policy == "per_block" and block_index > 0:
            states["subject"] = kernel.initial_state(n_actions, params)
            if demonstrator_kernel is not None and demonstrator_params is not None:
                states["demonstrator"] = demonstrator_kernel.initial_state(
                    n_actions, demonstrator_params
                )

        trials: list[Trial] = []

        for trial_index in range(block_spec.n_trials):
            choices: dict[str, int] = {}
            rewards: dict[str, float] = {}
            trial_events: list[Event] = []

            # Walk through the schema step-by-step. The schema defines the
            # order of events in each trial (e.g. INPUT → DECISION → OUTCOME →
            # UPDATE). We construct each event and, where needed, update the
            # model's internal state, all in this single pass.
            for event_index, step in enumerate(schema.steps):
                if step.phase == EventPhase.INPUT:
                    # Record which actions were available at the start of this
                    # trial. No model update is needed here; this is purely
                    # bookkeeping so the data log is complete.
                    trial_events.append(
                        Event(
                            phase=EventPhase.INPUT,
                            event_index=event_index,
                            node_id=step.node_id,
                            actor_id=step.actor_id,
                            payload={"available_actions": available_actions},
                        )
                    )

                elif step.phase == EventPhase.DECISION:
                    actor = step.actor_id
                    view = DecisionTrialView(
                        trial_index=trial_index,
                        available_actions=available_actions,
                        actor_id=step.actor_id,
                        learner_id=step.learner_id,
                    )
                    # Ask the relevant agent's model for a probability over
                    # each available action, then randomly sample one action
                    # according to those probabilities (softmax sampling).
                    if actor == "subject":
                        probabilities = kernel.action_probabilities(
                            cast("StateT", states["subject"]), view, params
                        )
                    elif demonstrator_kernel is not None and demonstrator_params is not None:
                        probabilities = demonstrator_kernel.action_probabilities(
                            states["demonstrator"], view, demonstrator_params
                        )
                    else:
                        raise ValueError(
                            f"Trial {trial_index}: DECISION step for actor {actor!r} "
                            f"but no kernel provided for that actor"
                        )
                    action_index = int(
                        rng.choice(len(available_actions), p=np.array(probabilities))
                    )
                    action = available_actions[action_index]
                    choices[actor] = action
                    trial_events.append(
                        Event(
                            phase=EventPhase.DECISION,
                            event_index=event_index,
                            node_id=step.node_id,
                            actor_id=actor,
                            payload={"action": action},
                        )
                    )

                elif step.phase == EventPhase.OUTCOME:
                    # Ask the environment what reward the actor receives for
                    # the choice they just made, then record it.
                    actor = step.actor_id
                    reward = env.step(choices[actor])
                    rewards[actor] = reward
                    trial_events.append(
                        Event(
                            phase=EventPhase.OUTCOME,
                            event_index=event_index,
                            node_id=step.node_id,
                            actor_id=actor,
                            payload={"reward": reward},
                        )
                    )

                elif step.phase == EventPhase.UPDATE:
                    actor = step.actor_id
                    learner = step.learner_id

                    # First, record the update event (what choice was made and
                    # what reward was received — the information being learned from).
                    trial_events.append(
                        Event(
                            phase=EventPhase.UPDATE,
                            event_index=event_index,
                            node_id=step.node_id,
                            actor_id=actor,
                            payload={"choice": choices[actor], "reward": rewards[actor]},
                        )
                    )

                    # Then immediately call update to advance the learner's
                    # internal state (e.g. update Q-values). Doing this here,
                    # inside the schema loop, means the update fires at exactly
                    # the position the schema declares — crucially, a social
                    # update can happen *before* the subject's own decision in
                    # pre-choice schemas.
                    #
                    # actor_id and learner_id are always set so the kernel can
                    # tell self-update (actor == learner) from social update
                    # (actor != learner) without any special-cased fields.
                    # For self-updates both action and reward are always visible;
                    # for social updates visibility is gated by observable_fields.
                    obs = (
                        step.observable_fields
                        if actor != learner
                        else frozenset({"action", "reward"})
                    )
                    view = DecisionTrialView(
                        trial_index=trial_index,
                        available_actions=available_actions,
                        actor_id=actor,
                        learner_id=learner,
                        action=choices[actor] if "action" in obs else None,
                        reward=rewards[actor] if "reward" in obs else None,
                    )

                    if learner == "subject":
                        states["subject"] = kernel.update(
                            cast("StateT", states["subject"]), view, params
                        )
                    elif demonstrator_kernel is not None and demonstrator_params is not None:
                        states["demonstrator"] = demonstrator_kernel.update(
                            states["demonstrator"], view, demonstrator_params
                        )

            trials.append(Trial(trial_index=trial_index, events=tuple(trial_events)))

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
    demonstrator_kernel: ModelKernel[object, object] | None = None,
    demonstrator_params: object | None = None,
) -> Dataset:
    """Simulate a whole group of artificial participants and return the combined dataset.

    Calls :func:`simulate_subject` once per participant, giving each one a
    fresh, independent copy of the task environment and a unique random seed
    derived from ``config.seed``. The result is a dataset that can be used
    for parameter recovery or posterior predictive checks on a full sample.

    Parameters
    ----------
    task
        The experimental design shared by all participants.
    env_factory
        A function that creates a fresh environment instance. Called once
        per participant so that each participant starts with independent
        environment state (e.g. independent reward draws).
    kernel
        The computational model (learning and choice rule) applied to every
        participant.
    params_per_subject
        A mapping from participant identifier to that participant's
        parameter values. The order of entries determines simulation order.
    config
        Shared simulation settings, including the base random seed.
    demonstrator_kernel
        If the task involves a demonstrator agent, supply their model here.
        The same demonstrator model is used for every participant.
    demonstrator_params
        Parameter values for the demonstrator. Required when
        ``demonstrator_kernel`` is provided.

    Returns
    -------
    Dataset
        Simulated data for all participants, in the same order as
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
                demonstrator_kernel=demonstrator_kernel,
                demonstrator_params=demonstrator_params,
            )
        )
    return Dataset(subjects=tuple(subjects))


def _infer_n_actions_from_task(task: TaskSpec) -> int:
    """Read the number of response options from the task design.

    The model needs to know how many actions exist (e.g. two slot machines,
    three stimuli) before the task begins, so that it can initialise the right
    number of Q-values. This function reads that number from the block
    metadata rather than hard-coding it, keeping the engine general.

    Parameters
    ----------
    task
        The task design whose block metadata are inspected.

    Returns
    -------
    int
        The number of available actions, which must be the same in every block.

    Raises
    ------
    ValueError
        Raised when the action count is missing from block metadata, or when
        different blocks disagree on how many actions exist.

    Notes
    -----
    Every ``BlockSpec`` in the task must have ``metadata['n_actions']`` set,
    and all blocks must report the same value. Mismatches suggest a design
    error in the task specification.
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
