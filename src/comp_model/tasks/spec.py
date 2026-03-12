"""Task-level design specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from comp_model._defaults import empty_mapping

if TYPE_CHECKING:
    from collections.abc import Mapping

    from comp_model.tasks.schemas import TrialSchema


@dataclass(frozen=True, slots=True)
class BlockSpec:
    """Design metadata for one task block.

    Attributes
    ----------
    condition
        Condition label for the block.
    n_trials
        Number of trials to run in the block.
    schema
        Trial schema executed by the environment.
    metadata
        Optional block-level task metadata.
    """

    condition: str
    n_trials: int
    schema: TrialSchema
    metadata: Mapping[str, Any] = field(default_factory=empty_mapping)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Immutable specification of a full task design.

    Attributes
    ----------
    task_id
        Stable task identifier.
    blocks
        Ordered block specifications for the task.
    """

    task_id: str
    blocks: tuple[BlockSpec, ...]

    @property
    def n_blocks(self) -> int:
        """Return the number of blocks in the task.

        Returns
        -------
        int
            Number of block specifications.
        """

        return len(self.blocks)

    @property
    def conditions(self) -> tuple[str, ...]:
        """Return condition labels in first-seen order without duplicates.

        Returns
        -------
        tuple[str, ...]
            Ordered unique condition labels.
        """

        seen: set[str] = set()
        ordered_conditions: list[str] = []
        for block in self.blocks:
            if block.condition not in seen:
                seen.add(block.condition)
                ordered_conditions.append(block.condition)
        return tuple(ordered_conditions)
