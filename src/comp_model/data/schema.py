"""Canonical hierarchical data structures for the modeling package."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


def _empty_mapping() -> Mapping[str, Any]:
    """Create an empty mapping with stable typing.

    Returns
    -------
    Mapping[str, Any]
        Empty metadata or payload mapping.
    """

    return {}


class EventPhase(StrEnum):
    """Ordered phase labels for fit-relevant trial events.

    Attributes
    ----------
    INPUT
        Information entering the modeled agent before a choice.
    DECISION
        A choice emitted by an actor.
    OUTCOME
        Scalar outcome associated with a decision point.
    UPDATE
        Explicit learning/update marker for the modeled agent.
    """

    INPUT = "input"
    DECISION = "decision"
    OUTCOME = "outcome"
    UPDATE = "update"


@dataclass(frozen=True, slots=True)
class Event:
    """Atomic ordered event within a trial.

    Attributes
    ----------
    phase
        Structural phase of the event.
    event_index
        Zero-based position within the containing trial.
    node_id
        Identifier linking events that belong to the same decision point.
    actor_id
        Identifier for the actor associated with the event.
    payload
        Phase-specific data carried by the event.
    """

    phase: EventPhase
    event_index: int
    node_id: str = "default"
    actor_id: str = "subject"
    payload: Mapping[str, Any] = field(default_factory=_empty_mapping)


@dataclass(frozen=True, slots=True)
class Trial:
    """Ordered event sequence for one trial.

    Attributes
    ----------
    trial_index
        Zero-based index within the containing block.
    events
        Ordered events emitted during the trial.
    metadata
        Optional trial-level metadata.
    """

    trial_index: int
    events: tuple[Event, ...]
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)


@dataclass(frozen=True, slots=True)
class Block:
    """Group of trials sharing a condition or other shared metadata.

    Attributes
    ----------
    block_index
        Zero-based block index within a subject.
    condition
        Condition label for the block.
    trials
        Ordered trials belonging to the block.
    metadata
        Optional block-level metadata.
    """

    block_index: int
    condition: str
    trials: tuple[Trial, ...]
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)


@dataclass(frozen=True, slots=True)
class SubjectData:
    """Hierarchical data for one subject.

    Attributes
    ----------
    subject_id
        Subject identifier unique within a dataset.
    blocks
        Ordered blocks completed by the subject.
    metadata
        Optional subject-level metadata.
    """

    subject_id: str
    blocks: tuple[Block, ...]
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def iter_block_trials(self) -> Iterator[tuple[Block, Trial]]:
        """Yield each trial together with its containing block.

        Yields
        ------
        tuple[Block, Trial]
            Each block-trial pair in hierarchical order.
        """

        for block in self.blocks:
            for trial in block.trials:
                yield block, trial

    @property
    def trials(self) -> tuple[Trial, ...]:
        """Flatten trials across all blocks.

        Returns
        -------
        tuple[Trial, ...]
            Trial records in block order.
        """

        return tuple(trial for block in self.blocks for trial in block.trials)


@dataclass(frozen=True, slots=True)
class Dataset:
    """Collection of subjects plus optional study-level metadata.

    Attributes
    ----------
    subjects
        Ordered subject records in the dataset.
    metadata
        Optional dataset-level metadata.
    """

    subjects: tuple[SubjectData, ...]
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    @property
    def blocks(self) -> tuple[Block, ...]:
        """Flatten blocks across all subjects.

        Returns
        -------
        tuple[Block, ...]
            Blocks in subject order.
        """

        return tuple(block for subject in self.subjects for block in subject.blocks)

    @property
    def trials(self) -> tuple[Trial, ...]:
        """Flatten trials across all subjects and blocks.

        Returns
        -------
        tuple[Trial, ...]
            Trials in dataset order.
        """

        return tuple(
            trial for subject in self.subjects for block in subject.blocks for trial in block.trials
        )
