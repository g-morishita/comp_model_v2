"""Sticky social RL demo-mixture kernel.

This module extends the demonstrator-only mixture model with a stickiness
(perseveration) term that biases the agent toward repeating its own previous
choice.

The model maintains two demonstrator-driven value systems:

- ``v_outcome``: updated by demonstrator reward
- ``v_tendency``: updated by demonstrator action frequency

At decision time these systems are combined via ``w_imitation`` and passed
through a softmax. A separate ``stickiness`` term is then added to the logit
of the action most recently chosen by the subject:

    logit[a] = beta * (
        w_imitation * v_tendency[a] + (1 - w_imitation) * v_outcome[a]
    ) + stickiness * I[a == last_self_action]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comp_model.models.kernels.base import ModelKernel, ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.probabilities import stable_softmax

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class SocialRlDemoMixtureStickyParams:
    """Free parameters for the sticky demo-mixture social RL kernel."""

    alpha_other_outcome: float
    alpha_other_action: float
    w_imitation: float
    beta: float
    stickiness: float


@dataclass(slots=True)
class SocialRlDemoMixtureStickyState:
    """Latent state for the sticky demo-mixture social RL kernel."""

    v_outcome: list[float]
    v_tendency: list[float]
    last_self_action: int | None = None


@dataclass(frozen=True)
class SocialRlDemoMixtureStickyKernel(
    ModelKernel[SocialRlDemoMixtureStickyState, SocialRlDemoMixtureStickyParams]
):
    """Sticky social RL kernel combining demonstrator mixture learning and perseverance."""

    v_outcome_init: float = 0.5

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return the kernel specification for the sticky demo-mixture model."""

        return ModelKernelSpec(
            model_id="social_rl_demo_mixture_sticky",
            parameter_specs=(
                ParameterSpec(
                    name="alpha_other_outcome",
                    transform_id="sigmoid",
                    description="learning rate for demonstrator outcome",
                    bounds=(0.0, 1.0),
                ),
                ParameterSpec(
                    name="alpha_other_action",
                    transform_id="sigmoid",
                    description="learning rate for demonstrator action tendency",
                    bounds=(0.0, 1.0),
                ),
                ParameterSpec(
                    name="w_imitation",
                    transform_id="sigmoid",
                    description="mixing weight for imitation at decision time",
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
            required_social_fields=frozenset({"action", "reward"}),
        )

    def parse_params(self, raw: dict[str, float]) -> SocialRlDemoMixtureStickyParams:
        """Convert raw optimiser values into interpretable model parameters."""

        transforms = self._parameter_transforms()
        return SocialRlDemoMixtureStickyParams(
            alpha_other_outcome=transforms["alpha_other_outcome"].forward(
                raw["alpha_other_outcome"]
            ),
            alpha_other_action=transforms["alpha_other_action"].forward(raw["alpha_other_action"]),
            w_imitation=transforms["w_imitation"].forward(raw["w_imitation"]),
            beta=transforms["beta"].forward(raw["beta"]),
            stickiness=transforms["stickiness"].forward(raw["stickiness"]),
        )

    def initial_state(
        self,
        n_actions: int,
        params: SocialRlDemoMixtureStickyParams,
    ) -> SocialRlDemoMixtureStickyState:
        """Create the initial state with neutral values and no prior choice."""

        del params
        return SocialRlDemoMixtureStickyState(
            v_outcome=[self.v_outcome_init] * n_actions,
            v_tendency=[1.0 / n_actions] * n_actions,
            last_self_action=None,
        )

    def action_probabilities(
        self,
        state: SocialRlDemoMixtureStickyState,
        view: DecisionTrialView,
        params: SocialRlDemoMixtureStickyParams,
    ) -> tuple[float, ...]:
        """Compute choice probabilities from the mixture systems plus stickiness."""

        logits: list[float] = []
        for action in view.available_actions:
            combined_value = (
                params.w_imitation * state.v_tendency[action]
                + (1 - params.w_imitation) * state.v_outcome[action]
            )
            logit = params.beta * combined_value
            if state.last_self_action == action:
                logit += params.stickiness
            logits.append(logit)
        return stable_softmax(logits)

    def observe_decision(
        self,
        state: SocialRlDemoMixtureStickyState,
        view: DecisionTrialView,
        params: SocialRlDemoMixtureStickyParams,
    ) -> SocialRlDemoMixtureStickyState:
        """Store the participant's most recent own choice at decision time."""

        del params
        if view.actor_id != view.learner_id or view.action is None:
            return state

        return SocialRlDemoMixtureStickyState(
            v_outcome=list(state.v_outcome),
            v_tendency=list(state.v_tendency),
            last_self_action=view.action,
        )

    def update(
        self,
        state: SocialRlDemoMixtureStickyState,
        view: DecisionTrialView,
        params: SocialRlDemoMixtureStickyParams,
    ) -> SocialRlDemoMixtureStickyState:
        """Update demonstrator-driven values and preserve own-choice history."""

        updated_v_outcome = list(state.v_outcome)
        updated_v_tendency = list(state.v_tendency)
        last_self_action = state.last_self_action

        if view.actor_id == view.learner_id:
            if view.action is not None:
                last_self_action = view.action
            return SocialRlDemoMixtureStickyState(
                v_outcome=updated_v_outcome,
                v_tendency=updated_v_tendency,
                last_self_action=last_self_action,
            )

        if view.action is not None and view.reward is not None:
            updated_v_outcome[view.action] += params.alpha_other_outcome * (
                view.reward - updated_v_outcome[view.action]
            )
            for action in range(len(updated_v_tendency)):
                target = 1.0 if action == view.action else 0.0
                updated_v_tendency[action] += params.alpha_other_action * (
                    target - updated_v_tendency[action]
                )

        return SocialRlDemoMixtureStickyState(
            v_outcome=updated_v_outcome,
            v_tendency=updated_v_tendency,
            last_self_action=last_self_action,
        )
