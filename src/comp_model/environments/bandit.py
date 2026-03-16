"""Concrete bandit environments.

This module provides:

- :class:`StationaryBanditEnvironment` — a simple k-armed bandit with fixed
  reward probabilities (schema-agnostic).
- :class:`SocialBanditEnvironment` — a wrapper that injects demonstrator
  observations into social schemas using a configurable demonstrator policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from comp_model.data.extractors import DecisionTrialView
from comp_model.data.schema import Event, EventPhase

if TYPE_CHECKING:
    from comp_model.models.kernels.base import ModelKernel
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


@dataclass(slots=True)
class SocialBanditEnvironment:
    """Bandit environment with a configurable demonstrator for social schemas.

    Wraps a :class:`StationaryBanditEnvironment` and intercepts non-subject
    INPUT steps (demonstrator observation events) to inject ``social_action``
    and ``social_reward`` into the observation payload.

    The demonstrator's behaviour is controlled by ``demonstrator_policy``:

    * **Fixed probabilities** (``tuple[float, ...]``): the demonstrator samples
      an action from this distribution on every trial.
    * **Fixed action sequence** (``Sequence[int]``): the demonstrator plays
      ``actions[t]`` on trial *t*.  Raises ``IndexError`` if the sequence is
      shorter than the number of trials.
    * **Learning agent** (``ModelKernel``): the demonstrator maintains its own
      latent state and uses the kernel's ``action_probabilities`` /
      ``next_state`` methods.  Requires ``demo_params`` to be set.
    """

    inner: StationaryBanditEnvironment
    demonstrator_policy: ModelKernel | tuple[float, ...] | Sequence[int]
    demo_params: Any = None

    _rng: np.random.Generator | None = field(default=None, init=False, repr=False)
    _demo_state: Any = field(default=None, init=False, repr=False)

    @property
    def environment_id(self) -> str:
        return "social_bandit"

    def _is_kernel_policy(self) -> bool:
        return hasattr(self.demonstrator_policy, "action_probabilities")

    def _is_sequence_policy(self) -> bool:
        if isinstance(self.demonstrator_policy, tuple):
            return len(self.demonstrator_policy) > 0 and all(
                isinstance(x, int) for x in self.demonstrator_policy
            )
        return isinstance(self.demonstrator_policy, (list, Sequence)) and not isinstance(
            self.demonstrator_policy, (str, bytes)
        )

    def _is_probability_policy(self) -> bool:
        return isinstance(self.demonstrator_policy, tuple) and (
            len(self.demonstrator_policy) == 0
            or any(isinstance(x, float) for x in self.demonstrator_policy)
        )

    def reset(self, block_spec: BlockSpec, *, rng: np.random.Generator) -> None:
        self._rng = rng
        self.inner.reset(block_spec, rng=rng)

        if self._is_kernel_policy():
            kernel = self.demonstrator_policy  # type: ignore[assignment]
            if self.demo_params is None:
                raise ValueError(
                    "demo_params is required when demonstrator_policy is a ModelKernel"
                )
            self._demo_state = kernel.initial_state(self.inner.n_actions, self.demo_params)

    def step(self, action: int | None = None) -> tuple[Event, ...]:
        assert self.inner._block_spec is not None
        schema = self.inner._block_spec.schema
        schema_step = schema.steps[self.inner._step_index]

        if schema_step.phase == EventPhase.INPUT and schema_step.actor_id != "subject":
            return self._demonstrator_step()

        return self.inner.step(action=action)

    def _demonstrator_step(self) -> tuple[Event, ...]:
        assert self._rng is not None
        n_actions = self.inner.n_actions
        trial_index = self.inner._trial_index
        available_actions = tuple(range(n_actions))

        # Generate demonstrator action
        if self._is_kernel_policy():
            demo_action = self._kernel_demo_action(available_actions, trial_index)
        elif self._is_probability_policy():
            probs = self.demonstrator_policy
            if not probs:
                probs = self.inner.reward_probs
            demo_action = int(self._rng.choice(n_actions, p=np.array(probs)))
        else:
            # Sequence policy
            seq = self.demonstrator_policy
            demo_action = int(seq[trial_index])  # type: ignore[index]

        # Sample reward
        demo_reward = float(self._rng.random() < self.inner.reward_probs[demo_action])

        # Update kernel demonstrator state
        if self._is_kernel_policy():
            self._kernel_demo_update(available_actions, trial_index, demo_action, demo_reward)

        # Advance inner env step counter and build patched event
        events = self.inner.step(action=None)
        original = events[0]
        patched = Event(
            phase=original.phase,
            event_index=original.event_index,
            node_id=original.node_id,
            actor_id=original.actor_id,
            payload={
                "available_actions": original.payload["available_actions"],
                "observation": {
                    "social_action": demo_action,
                    "social_reward": demo_reward,
                },
            },
        )
        return (patched,)

    def _kernel_demo_action(
        self, available_actions: tuple[int, ...], trial_index: int
    ) -> int:
        assert self._rng is not None
        kernel = self.demonstrator_policy  # type: ignore[assignment]
        partial_view = DecisionTrialView(
            trial_index=trial_index,
            available_actions=available_actions,
            choice=-1,
        )
        probs = kernel.action_probabilities(self._demo_state, partial_view, self.demo_params)
        action_index = int(self._rng.choice(len(available_actions), p=np.array(probs)))
        return available_actions[action_index]

    def _kernel_demo_update(
        self,
        available_actions: tuple[int, ...],
        trial_index: int,
        demo_action: int,
        demo_reward: float,
    ) -> None:
        kernel = self.demonstrator_policy  # type: ignore[assignment]
        complete_view = DecisionTrialView(
            trial_index=trial_index,
            available_actions=available_actions,
            choice=demo_action,
            reward=demo_reward,
        )
        self._demo_state = kernel.next_state(
            self._demo_state, complete_view, self.demo_params
        )
