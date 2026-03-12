"""Environment protocol for executable task dynamics.

Environments execute the task layer one schema step at a time and emit events
that satisfy the current :class:`~comp_model.tasks.schemas.TrialSchema`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

    from comp_model.data.schema import Event
    from comp_model.tasks.spec import BlockSpec


class Environment(Protocol):
    """Protocol implemented by executable environments.

    Notes
    -----
    The runtime owns the outer loop over blocks, trials, and agent decisions.
    The environment owns only task-side dynamics: resetting block state and
    emitting one or more events for the next schema position.
    """

    @property
    def environment_id(self) -> str:
        """Return a stable identifier for the environment.

        Returns
        -------
        str
            Environment identifier.
        """

        ...

    def reset(self, block_spec: BlockSpec, *, rng: np.random.Generator) -> None:
        """Prepare the environment for a new block.

        Parameters
        ----------
        block_spec
            Block specification the environment should execute.
        rng
            Random number generator used for stochastic outcomes.

        Returns
        -------
        None
            This function resets the environment in-place.

        Notes
        -----
        ``reset`` is called once per block before any trials in that block are
        executed. It should bind the block specification and discard any prior
        within-block state.
        """

        ...

    def step(self, action: int | None = None) -> tuple[Event, ...]:
        """Advance the environment by one schema step.

        Parameters
        ----------
        action
            Optional externally supplied action for action-required steps.

        Returns
        -------
        tuple[Event, ...]
            Events emitted for the current schema step.

        Notes
        -----
        ``action`` is supplied only for schema steps whose
        ``action_required`` flag is true. The environment is responsible for
        converting that action into one or more canonical events.
        """

        ...
