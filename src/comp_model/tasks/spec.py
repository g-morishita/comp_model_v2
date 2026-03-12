"""Task-level design specifications.

Task objects describe experimental structure independently of any agent or
inference backend. They own block order, condition labels, trial counts,
schemas, and task metadata consumed by environments.
"""

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

    Notes
    -----
    Environments read ``metadata`` to configure block-specific dynamics such as
    action count or reward contingencies. Kernels do not read ``BlockSpec``
    directly.
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

    Notes
    -----
    ``TaskSpec`` is the structural design layer. The runtime executes it, while
    fitted models only ever see extracted decision views derived from the event
    traces it generates.
    """

    task_id: str
    blocks: tuple[BlockSpec, ...]

    @property
    def n_blocks(self) -> int:
        """Return the number of blocks in the task.

        Returns
        -------
        int
            Number of block specifications in execution order.
        """

        return len(self.blocks)

    @property
    def conditions(self) -> tuple[str, ...]:
        """Return condition labels in first-seen order without duplicates.

        Returns
        -------
        tuple[str, ...]
            Ordered unique condition labels, preserving the first occurrence of
            each condition in ``blocks``.
        """

        seen: set[str] = set()
        ordered_conditions: list[str] = []
        for block in self.blocks:
            if block.condition not in seen:
                seen.add(block.condition)
                ordered_conditions.append(block.condition)
        return tuple(ordered_conditions)
