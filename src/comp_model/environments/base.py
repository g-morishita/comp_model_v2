"""Environment protocol for executable task dynamics.

Environments are pure reward oracles. The simulation engine owns all event
construction and phase logic; the environment only generates stochastic
outcomes for a given action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

    from comp_model.tasks.spec import BlockSpec


class Environment(Protocol):
    """Protocol implemented by executable environments.

    Notes
    -----
    The runtime owns the outer loop over blocks, trials, and agent decisions.
    The environment owns only reward generation: resetting block state and
    returning a scalar reward for each action taken.
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

    def step(self, action: int) -> float:
        """Return the reward for the given action.

        Parameters
        ----------
        action
            Action taken by the acting agent at this OUTCOME step.

        Returns
        -------
        float
            Scalar reward for the action.

        Notes
        -----
        Called only at OUTCOME steps by the simulation engine. The environment
        has no knowledge of phases, events, or schemas — it is a pure reward
        oracle.
        """

        ...
