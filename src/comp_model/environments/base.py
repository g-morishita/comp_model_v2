"""Environment protocol for executable task dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

    from comp_model.data.schema import Event
    from comp_model.tasks.spec import BlockSpec


class Environment(Protocol):
    """Protocol implemented by executable environments."""

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
        """

        ...
