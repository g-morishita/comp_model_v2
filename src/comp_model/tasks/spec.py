"""Descriptions of the experimental design, independent of models and fitting.

This module is the "experiment design" layer. It lets you describe what
participants did — which blocks existed, how many trials each block contained,
in what order events unfolded within each trial, and what the reward
contingencies were — without saying anything about how participants behaved
or how models were fitted to their data.

Two classes carry this information:

- :class:`BlockSpec`: one block of the experiment.
- :class:`TaskSpec`: the full task as an ordered sequence of blocks.

Why keep design separate from models and fitting?
The same task design object can be reused with many different models and
fitting backends. Conversely, the same model can be applied to tasks with
different designs. Keeping them separate avoids accidental coupling and
makes it easy to swap components independently.

Note: computational models (kernels) never read ``BlockSpec`` or ``TaskSpec``
directly. They only ever see compact summaries of individual decision trials
(``DecisionTrialView`` objects produced during simulation or data replay).
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
    """The design specification for one block of the task.

    A block is a run of consecutive trials that share the same condition and
    the same within-trial event structure (schema). For example, in a
    two-phase bandit task you might have a "social observation" block followed
    by an "individual choice" block — each would be a separate ``BlockSpec``.

    Attributes
    ----------
    condition
        A label identifying the experimental condition for this block
        (e.g. ``"social"``, ``"individual"``, ``"test"``). Used to group
        and compare blocks in analysis.
    n_trials
        How many trials this block contains.
    schema
        The within-trial event sequence for this block — for example:
        INPUT → (demonstrator) DECISION → (demonstrator) OUTCOME →
        (demonstrator) UPDATE → (subject) DECISION → …
        See :class:`~comp_model.tasks.schemas.TrialSchema`.
    metadata
        A flexible dictionary for any extra block-level information the
        task environment needs, such as reward probabilities
        (``"reward_probs"``) or the number of available actions
        (``"n_actions"``). Computational models never read this directly;
        only the task environment uses it.
    """

    condition: str
    n_trials: int
    schema: TrialSchema
    metadata: Mapping[str, Any] = field(default_factory=empty_mapping)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """The complete experimental design: an ordered sequence of blocks.

    A ``TaskSpec`` is the top-level description of an entire task. It lists
    all blocks in the order they are presented to participants, and nothing
    more. It does not say anything about how participants behave (that is the
    kernel's job) or how models are fitted to data (that is the inference
    backend's job).

    The simulation engine (:func:`~comp_model.runtime.engine.simulate_subject`)
    reads this object to know what to simulate. Fitted models never read it
    directly — they only ever see the compact decision-trial summaries that the
    engine produces.

    Attributes
    ----------
    task_id
        A stable name or identifier for the task (e.g. ``"two_armed_bandit"``).
        Used as a label in output files and logs.
    blocks
        The blocks of the task in presentation order. Each element is a
        :class:`BlockSpec` describing that block's design.
    """

    task_id: str
    blocks: tuple[BlockSpec, ...]

    @property
    def n_blocks(self) -> int:
        """Return how many blocks the task contains.

        Returns
        -------
        int
            The total number of blocks, in the order they are presented.
        """

        return len(self.blocks)

    @property
    def conditions(self) -> tuple[str, ...]:
        """Return the unique condition labels in the order they first appear.

        Useful for iterating over conditions without repeating yourself
        when the same condition label appears in multiple blocks.

        Returns
        -------
        tuple[str, ...]
            Each unique condition label exactly once, ordered by first
            occurrence across ``blocks``.
        """

        seen: set[str] = set()
        ordered_conditions: list[str] = []
        for block in self.blocks:
            if block.condition not in seen:
                seen.add(block.condition)
                ordered_conditions.append(block.condition)
        return tuple(ordered_conditions)
