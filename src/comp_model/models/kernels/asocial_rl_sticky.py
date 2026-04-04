"""Asocial RL kernel with a stickiness (perseveration) term.

This module extends standard asocial Q-learning with a choice-history bias that
encourages repeating the participant's own previous action. Learning still
comes only from the participant's own outcomes; stickiness only affects the
decision rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comp_model.models.kernels.base import ModelKernel, ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.probabilities import stable_softmax

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class AsocialRlStickyParams:
    """Free parameters for the asocial sticky RL kernel.

    Attributes
    ----------
    alpha
        Learning rate for own-outcome updates; in ``(0, 1)``.
    beta
        Inverse temperature for the softmax choice rule; positive.
    stickiness
        Additive logit bias toward repeating the participant's previous own
        choice. Positive values bias repeats; negative values bias switches.
    """

    alpha: float
    beta: float
    stickiness: float


@dataclass(slots=True)
class AsocialRlStickyState:
    """Latent state for the asocial sticky RL kernel.

    Attributes
    ----------
    q_values
        Per-action Q-values updated from the participant's own outcomes.
    last_self_action
        Most recent own choice. ``None`` until the first self-update is
        observed.
    """

    q_values: list[float]
    last_self_action: int | None = None


@dataclass(frozen=True)
class AsocialRlStickyKernel(ModelKernel[AsocialRlStickyState, AsocialRlStickyParams]):
    """Asocial Q-learning kernel with a perseveration term in the choice rule.

    The model uses standard Rescorla-Wagner updating for own outcomes and adds
    a separate choice-history influence:

    - ``q_values``: updated from the participant's own reward
    - ``last_self_action``: adds a ``stickiness`` bonus to the previously
      chosen action at the next decision

    Attributes
    ----------
    q_init
        Starting value assigned to all Q-values before any learning occurs.
        Defaults to ``0.5``.
    """

    q_init: float = 0.5

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return the kernel specification for the asocial sticky model."""

        return ModelKernelSpec(
            model_id="asocial_rl_sticky",
            parameter_specs=(
                ParameterSpec(
                    name="alpha",
                    transform_id="sigmoid",
                    description="learning rate",
                    bounds=(0.0, 1.0),
                ),
                ParameterSpec(
                    name="beta",
                    transform_id="softplus",
                    description="inverse temperature",
                    bounds=(0.0, None),
                ),
                ParameterSpec(
                    name="stickiness",
                    transform_id="identity",
                    description="logit bias toward repeating the previous own choice",
                ),
            ),
            requires_social=False,
        )

    def parse_params(self, raw: dict[str, float]) -> AsocialRlStickyParams:
        """Convert raw optimiser values into interpretable model parameters."""

        transforms = self._parameter_transforms()
        return AsocialRlStickyParams(
            alpha=transforms["alpha"].forward(raw["alpha"]),
            beta=transforms["beta"].forward(raw["beta"]),
            stickiness=transforms["stickiness"].forward(raw["stickiness"]),
        )

    def initial_state(self, n_actions: int, params: AsocialRlStickyParams) -> AsocialRlStickyState:
        """Create the initial latent state with neutral values and no prior choice."""

        del params
        return AsocialRlStickyState(q_values=[self.q_init] * n_actions, last_self_action=None)

    def action_probabilities(
        self,
        state: AsocialRlStickyState,
        view: DecisionTrialView,
        params: AsocialRlStickyParams,
    ) -> tuple[float, ...]:
        """Compute choice probabilities from Q-values plus a stickiness bonus."""

        logits: list[float] = []
        for action in view.available_actions:
            logit = params.beta * state.q_values[action]
            if action == state.last_self_action:
                logit += params.stickiness
            logits.append(logit)
        return stable_softmax(logits)

    def observe_decision(
        self,
        state: AsocialRlStickyState,
        view: DecisionTrialView,
        params: AsocialRlStickyParams,
    ) -> AsocialRlStickyState:
        """Store the participant's most recent own choice at decision time."""

        del params
        if view.actor_id != view.learner_id or view.action is None:
            return state

        return AsocialRlStickyState(
            q_values=list(state.q_values),
            last_self_action=view.action,
        )

    def update(
        self,
        state: AsocialRlStickyState,
        view: DecisionTrialView,
        params: AsocialRlStickyParams,
    ) -> AsocialRlStickyState:
        """Update own-action values and store the most recent own choice.

        Social UPDATE steps are ignored entirely so the kernel remains safe to
        fit against social-schema datasets for model-comparison purposes.
        """

        if view.actor_id != view.learner_id:
            return state

        updated_q_values = list(state.q_values)
        last_self_action = state.last_self_action

        if view.action is not None:
            last_self_action = view.action
            if view.reward is not None:
                updated_q_values[view.action] += params.alpha * (
                    view.reward - updated_q_values[view.action]
                )

        return AsocialRlStickyState(
            q_values=updated_q_values,
            last_self_action=last_self_action,
        )
