"""Sticky social RL demo-action kernel.

This module extends the demonstrator-action-only learner with a stickiness
(perseveration) term that biases the agent toward repeating its own previous
choice.

The model maintains one demonstrator-driven value system:

- ``v_tendency``: updated by demonstrator action frequency

At decision time those action tendencies are passed through a softmax after
adding a separate stickiness bonus to the most recently chosen own action:

    logit[a] = beta * v_tendency[a] + stickiness * I[a == last_self_action]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comp_model.models.kernels.base import ModelKernel, ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.probabilities import stable_softmax

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class SocialRlDemoActionStickyParams:
    """Free parameters for the sticky demo-action social RL kernel."""

    alpha_other_action: float
    beta: float
    stickiness: float


@dataclass(slots=True)
class SocialRlDemoActionStickyState:
    """Latent state for the sticky demo-action social RL kernel."""

    v_tendency: list[float]
    last_self_action: int | None = None


@dataclass(frozen=True)
class SocialRlDemoActionStickyKernel(
    ModelKernel[SocialRlDemoActionStickyState, SocialRlDemoActionStickyParams]
):
    """Sticky demonstrator-action-only social RL kernel."""

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return the kernel specification for the sticky demo-action model."""

        return ModelKernelSpec(
            model_id="social_rl_demo_action_sticky",
            parameter_specs=(
                ParameterSpec(
                    name="alpha_other_action",
                    transform_id="sigmoid",
                    description="learning rate for demonstrator action tendency",
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
            requires_social=True,
            required_social_fields=frozenset({"action"}),
        )

    def parse_params(self, raw: dict[str, float]) -> SocialRlDemoActionStickyParams:
        """Convert raw optimiser values into interpretable model parameters."""

        transforms = self._parameter_transforms()
        return SocialRlDemoActionStickyParams(
            alpha_other_action=transforms["alpha_other_action"].forward(raw["alpha_other_action"]),
            beta=transforms["beta"].forward(raw["beta"]),
            stickiness=transforms["stickiness"].forward(raw["stickiness"]),
        )

    def initial_state(
        self, n_actions: int, params: SocialRlDemoActionStickyParams
    ) -> SocialRlDemoActionStickyState:
        """Create the initial state with uniform tendencies and no prior choice."""

        del params
        return SocialRlDemoActionStickyState(
            v_tendency=[1.0 / n_actions] * n_actions,
            last_self_action=None,
        )

    def action_probabilities(
        self,
        state: SocialRlDemoActionStickyState,
        view: DecisionTrialView,
        params: SocialRlDemoActionStickyParams,
    ) -> tuple[float, ...]:
        """Compute choice probabilities from action tendencies plus stickiness."""

        logits: list[float] = []
        for action in view.available_actions:
            logit = params.beta * state.v_tendency[action]
            if state.last_self_action == action:
                logit += params.stickiness
            logits.append(logit)
        return stable_softmax(logits)

    def observe_decision(
        self,
        state: SocialRlDemoActionStickyState,
        view: DecisionTrialView,
        params: SocialRlDemoActionStickyParams,
    ) -> SocialRlDemoActionStickyState:
        """Store the participant's most recent own choice at decision time."""

        del params
        if view.actor_id != view.learner_id or view.action is None:
            return state

        return SocialRlDemoActionStickyState(
            v_tendency=list(state.v_tendency),
            last_self_action=view.action,
        )

    def update(
        self,
        state: SocialRlDemoActionStickyState,
        view: DecisionTrialView,
        params: SocialRlDemoActionStickyParams,
    ) -> SocialRlDemoActionStickyState:
        """Update demonstrator action tendencies and preserve own-choice history."""
        last_self_action = state.last_self_action

        if view.actor_id == view.learner_id:
            if view.action is not None:
                last_self_action = view.action
            return SocialRlDemoActionStickyState(
                v_tendency=list(state.v_tendency),
                last_self_action=last_self_action,
            )

        if view.action is not None:
            alpha = params.alpha_other_action
            updated_v_tendency = [value * (1 - alpha) for value in state.v_tendency]
            updated_v_tendency[view.action] += alpha
            return SocialRlDemoActionStickyState(
                v_tendency=updated_v_tendency,
                last_self_action=last_self_action,
            )

        return SocialRlDemoActionStickyState(
            v_tendency=list(state.v_tendency),
            last_self_action=last_self_action,
        )
