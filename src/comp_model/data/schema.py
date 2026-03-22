"""How experimental data is organized in this package.

Every piece of data — whether collected from a real participant or generated
by a simulation — is stored as nested containers that mirror the natural
structure of an experiment:

    Event  →  one thing that happened (e.g. a choice was made, a reward was shown)
    Trial  →  one complete round of the task (a sequence of events)
    Block  →  a group of trials that share the same condition or context
    SubjectData  →  everything one participant did across all blocks
    Dataset  →  all participants in a study

This hierarchy is the single source of truth for the whole package. Model
fitting, simulation, and data export all work from these objects rather than
from a separate flat spreadsheet-style representation. Keeping one
authoritative structure means that every analysis tool sees data in the same
format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from comp_model._defaults import empty_mapping

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


class EventPhase(StrEnum):
    """The four stages of a single decision cycle.

    Each trial is broken into ordered stages that reflect what is happening
    cognitively at that moment. Together they form a decision cycle:

    INPUT
        The participant sees the available options (e.g. which buttons they
        can press, what stimuli are on screen). This is the information the
        model uses to set up the choice.
    DECISION
        A choice is made — either by the participant or by a demonstrator
        being observed. The payload records which action was taken.
    OUTCOME
        The result of the choice is revealed (e.g. a reward is shown). This
        stage exists as a record of what happened but the reward value itself
        is forwarded into the UPDATE payload so the model does not need to
        piece it together from two separate events.
    UPDATE
        Learning happens here. The model uses this event to revise its
        internal beliefs or action values. Crucially, *where* this event
        appears in the trial sequence controls *when* learning fires relative
        to the rest of the trial — placing it before or after other events
        changes the order in which the model updates.

    These labels are intentionally generic. Whether an INPUT event refers to
    "social cues before a choice" or "stimulus onset" is encoded by the
    sequence of events and the actor labels, not by adding task-specific
    phase names.
    """

    INPUT = "input"
    DECISION = "decision"
    OUTCOME = "outcome"
    UPDATE = "update"


@dataclass(frozen=True, slots=True)
class Event:
    """The smallest unit of recorded data: one thing that happened.

    Every moment worth recording in a trial is stored as an Event. What the
    event actually contains depends on its phase:

    - An INPUT event might carry ``{"available_actions": [0, 1, 2]}``
      (the options the participant was shown).
    - A DECISION event carries ``{"action": 1}`` (the choice that was made).
    - An OUTCOME event carries ``{"reward": 1.0}`` (what the participant received).
    - An UPDATE event carries both ``{"choice": 1, "reward": 1.0}`` so the
      model has everything it needs in one place.

    Attributes
    ----------
    phase
        Which stage of the decision cycle this event belongs to
        (INPUT, DECISION, OUTCOME, or UPDATE).
    event_index
        The position of this event within the trial (0 = first event).
    node_id
        A label that groups the INPUT, DECISION, OUTCOME, and UPDATE events
        that all belong to the same choice point. For example, in a task
        where the participant first watches a demonstrator and then makes
        their own choice, the demonstrator's events and the participant's
        events each get their own ``node_id`` so it is always clear which
        events go together. Using a shared label avoids hard-coding
        domain-specific names like ``"social"`` or ``"self"``.
    actor_id
        Who produced this event. Use ``"subject"`` for the participant's
        own actions and ``"demonstrator"`` (or any other label) for another
        agent being observed. Defaults to ``"subject"``.
    payload
        The data carried by this event. Its contents depend on the phase
        (see examples above).
    """

    phase: EventPhase
    event_index: int
    node_id: str = "default"
    actor_id: str = "subject"
    payload: Mapping[str, Any] = field(default_factory=empty_mapping)


@dataclass(frozen=True, slots=True)
class Trial:
    """One complete round of the task.

    A Trial contains the full sequence of events that occurred during a single
    round: seeing the options, making a choice, receiving a reward, and
    updating. The events are stored in the order they happened so that
    replaying the trial faithfully reproduces the sequence the model would
    have experienced.

    Attributes
    ----------
    trial_index
        Which round this was within the current block (0 = first trial).
    events
        The ordered events that make up this trial.
    metadata
        Any extra information about this trial (e.g. reaction times,
        condition labels) that does not fit into the event structure.
    """

    trial_index: int
    events: tuple[Event, ...]
    metadata: Mapping[str, Any] = field(default_factory=empty_mapping)


@dataclass(frozen=True, slots=True)
class Block:
    """A group of consecutive trials that share the same experimental condition.

    In most learning experiments, participants complete multiple blocks where
    the task context changes between blocks (e.g. different reward
    probabilities, different partners, or a context switch). A Block groups
    the trials that belong to one such period.

    Blocks also serve as natural reset points: if a model is configured to
    reset its learned values at the start of each new context, that reset
    happens at the Block boundary.

    Attributes
    ----------
    block_index
        Which block this is for this participant (0 = first block).
    condition
        A label describing the experimental condition for this block
        (e.g. ``"high_volatility"`` or ``"social"``).
    trials
        The trials that make up this block, in order.
    metadata
        Any extra information about this block.
    """

    block_index: int
    condition: str
    trials: tuple[Trial, ...]
    metadata: Mapping[str, Any] = field(default_factory=empty_mapping)


@dataclass(frozen=True, slots=True)
class SubjectData:
    """Everything one participant did across the entire experiment.

    This container holds all the blocks (and therefore all the trials and
    events) for a single participant, in the order they were experienced.
    The ordering matters: when the model is fit to a participant's data, it
    replays the blocks in sequence, just as the participant lived through them.

    Attributes
    ----------
    subject_id
        A unique identifier for this participant within the dataset
        (e.g. ``"sub-01"``).
    blocks
        The blocks the participant completed, in chronological order.
    metadata
        Any extra participant-level information (e.g. age, group assignment).
    """

    subject_id: str
    blocks: tuple[Block, ...]
    metadata: Mapping[str, Any] = field(default_factory=empty_mapping)

    def iter_block_trials(self) -> Iterator[tuple[Block, Trial]]:
        """Step through every trial for this participant, one at a time.

        Yields each trial paired with the block it belongs to, in the order
        the participant experienced them. Having the block alongside the trial
        is useful when the model needs to know the current condition or when
        block boundaries trigger a state reset.

        Yields
        ------
        tuple[Block, Trial]
            A (block, trial) pair for every trial across all blocks, in
            chronological order.
        """

        for block in self.blocks:
            for trial in block.trials:
                yield block, trial

    @property
    def trials(self) -> tuple[Trial, ...]:
        """All of this participant's trials as a simple flat list.

        A convenience shortcut when you want to iterate over every trial
        without needing to know which block each trial came from. The trials
        are returned in the same chronological order as they were experienced.

        Returns
        -------
        tuple[Trial, ...]
            Every trial this participant completed, across all blocks, in order.
        """

        return tuple(trial for block in self.blocks for trial in block.trials)


@dataclass(frozen=True, slots=True)
class Dataset:
    """The whole study: every participant and all their data.

    A Dataset is the top-level container that holds data for all participants
    in a study. It is typically what you load at the start of an analysis and
    pass to fitting or simulation routines.

    The helper properties ``blocks`` and ``trials`` let you quickly access all
    blocks or all trials across every participant in a consistent, predictable
    order (participant by participant, then block by block within each
    participant). This consistent ordering ensures that indices stay aligned
    when exporting to analysis tools.

    Attributes
    ----------
    subjects
        All participants in the dataset, in a fixed order.
    metadata
        Any study-level information (e.g. study name, date collected,
        task version).
    """

    subjects: tuple[SubjectData, ...]
    metadata: Mapping[str, Any] = field(default_factory=empty_mapping)

    @property
    def blocks(self) -> tuple[Block, ...]:
        """All blocks in the dataset as a simple flat list.

        Returns every block from every participant, ordered participant by
        participant. Useful for quickly iterating over experimental conditions
        across the whole study.

        Returns
        -------
        tuple[Block, ...]
            Every block from all participants, in participant order.
        """

        return tuple(block for subject in self.subjects for block in subject.blocks)

    @property
    def trials(self) -> tuple[Trial, ...]:
        """All trials in the dataset as a simple flat list.

        Returns every trial from every participant and every block in a single
        sequence. The order goes participant by participant, and within each
        participant block by block — matching the order trials were experienced.

        Returns
        -------
        tuple[Trial, ...]
            Every trial from all participants, in participant-then-block order.
        """

        return tuple(
            trial for subject in self.subjects for block in subject.blocks for trial in block.trials
        )
