"""Concrete stationary bandit environment.

This environment executes the current trial schema one step at a time while
sampling Bernoulli rewards from fixed arm probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from comp_model.data.schema import Event, EventPhase

if TYPE_CHECKING:
    import numpy as np

    from comp_model.tasks.schemas import TrialSchema
    from comp_model.tasks.spec import BlockSpec


@dataclass(slots=True)
class StationaryBanditEnvironment:
    """Simple k-armed bandit with fixed reward probabilities.

    Notes
    -----
    The environment is intentionally schema-agnostic beyond the current step's
    declared phase. It can therefore execute any schema whose steps follow the
    INPUT/DECISION/OUTCOME/UPDATE contract expected by the runtime.
    """

    n_actions: int
    reward_probs: tuple[float, ...]
    _block_spec: BlockSpec | None = field(default=None, init=False, repr=False)
    _rng: np.random.Generator | None = field(default=None, init=False, repr=False)
    _step_index: int = field(default=0, init=False, repr=False)
    _trial_index: int = field(default=0, init=False, repr=False)
    _last_action: int | None = field(default=None, init=False, repr=False)

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
        """Reset internal counters and bind a block specification.

        Parameters
        ----------
        block_spec
            Block specification to execute.
        rng
            Random number generator for stochastic rewards.

        Returns
        -------
        None
            This function resets the environment in-place.

        Notes
        -----
        Reset clears step and trial counters, stores the block specification,
        and binds the random generator that will be used for stochastic outcome
        sampling throughout the block.
        """

        self._block_spec = block_spec
        self._rng = rng
        self._step_index = 0
        self._trial_index = 0
        self._last_action = None

    def step(self, action: int | None = None) -> tuple[Event, ...]:
        """Advance one schema step and emit the corresponding event.

        Parameters
        ----------
        action
            Optional chosen action for decision steps.

        Returns
        -------
        tuple[Event, ...]
            Events emitted for the current step.

        Notes
        -----
        The emitted event is determined purely by the current schema step:

        - INPUT emits legal actions and a simple observation payload,
        - DECISION records the externally supplied action,
        - OUTCOME samples a Bernoulli reward from ``reward_probs[action]``, and
        - UPDATE emits an explicit update marker.
        """

        if self._block_spec is None or self._rng is None:
            raise RuntimeError("Environment must be reset before stepping")

        schema = self._block_spec.schema
        schema_step = schema.steps[self._step_index]
        node_id = schema_step.node_id
        actor_id = schema_step.actor_id

        if schema_step.phase == EventPhase.INPUT:
            event = Event(
                phase=EventPhase.INPUT,
                event_index=self._step_index,
                node_id=node_id,
                actor_id=actor_id,
                payload={
                    "available_actions": tuple(range(self.n_actions)),
                    "observation": {"trial_index": self._trial_index},
                },
            )
            self._advance(schema)
            return (event,)

        if schema_step.phase == EventPhase.DECISION:
            if action is None:
                raise ValueError("Decision steps require an action")
            self._last_action = action
            event = Event(
                phase=EventPhase.DECISION,
                event_index=self._step_index,
                node_id=node_id,
                actor_id=actor_id,
                payload={"action": action},
            )
            self._advance(schema)
            return (event,)

        if schema_step.phase == EventPhase.OUTCOME:
            if self._last_action is None:
                raise ValueError("Outcome step requires a prior action")
            reward = float(self._rng.random() < self.reward_probs[self._last_action])
            event = Event(
                phase=EventPhase.OUTCOME,
                event_index=self._step_index,
                node_id=node_id,
                actor_id=actor_id,
                payload={"reward": reward},
            )
            self._advance(schema)
            return (event,)

        if schema_step.phase == EventPhase.UPDATE:
            event = Event(
                phase=EventPhase.UPDATE,
                event_index=self._step_index,
                node_id=node_id,
                actor_id=actor_id,
                payload={},
            )
            self._advance(schema)
            return (event,)

        raise ValueError(f"Unexpected schema step: {schema_step}")

    def _advance(self, schema: TrialSchema) -> None:
        """Advance internal step counters after emitting an event.

        Parameters
        ----------
        schema
            Trial schema whose length determines trial boundaries.

        Returns
        -------
        None
            This function mutates the environment in-place.

        Notes
        -----
        Advancing wraps the step counter to zero at the end of the schema,
        increments the trial counter, and clears the cached last action so the
        next trial must begin with a fresh decision sequence.
        """

        n_steps = len(schema.steps)
        self._step_index += 1
        if self._step_index >= n_steps:
            self._step_index = 0
            self._trial_index += 1
            self._last_action = None
