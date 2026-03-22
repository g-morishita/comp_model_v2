"""Concrete bandit environments.

This module provides:

- :class:`StationaryBanditEnvironment` — a simple k-armed bandit with fixed
  reward probabilities (schema-agnostic pure reward oracle).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from comp_model.tasks.spec import BlockSpec


@dataclass(slots=True)
class StationaryBanditEnvironment:
    """Simple k-armed bandit with fixed reward probabilities.

    Notes
    -----
    The environment is a pure reward oracle: ``step(action)`` samples a
    Bernoulli reward from ``reward_probs[action]`` and returns it. All event
    construction is handled by the simulation engine.
    """

    n_actions: int
    reward_probs: tuple[float, ...]
    _rng: np.random.Generator | None = field(default=None, init=False, repr=False)

    @property
    def environment_id(self) -> str:
        """Return the stable environment identifier.

        Returns
        -------
        str
            Environment identifier.
        """

        return "stationary_bandit"

    def reset(self, block_spec: BlockSpec, *, rng: np.random.Generator) -> None:
        """Reset the environment for a new block.

        Parameters
        ----------
        block_spec
            Block specification (unused beyond binding the RNG).
        rng
            Random number generator for stochastic rewards.

        Returns
        -------
        None
            This function resets the environment in-place.
        """

        self._rng = rng

    def step(self, action: int) -> float:
        """Sample and return a Bernoulli reward for the given action.

        Parameters
        ----------
        action
            Action index whose reward probability should be sampled.

        Returns
        -------
        float
            Bernoulli reward (0.0 or 1.0).
        """

        if self._rng is None:
            raise RuntimeError("Environment must be reset before stepping")
        return float(self._rng.random() < self.reward_probs[action])
