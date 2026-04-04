"""Sticky social RL kernel that learns from own and demonstrator outcomes.

This module extends the self-reward + demonstrator-reward model with a
perseveration term that biases the participant toward repeating their previous
own choice. Q-values still update from both own and demonstrator rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comp_model.models.kernels.base import ModelKernel, ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.probabilities import stable_softmax

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class SocialRlSelfRewardDemoRewardStickyParams:
    """Free parameters for the sticky self-reward + demo-reward social RL kernel."""

    alpha_self: float
    alpha_other: float
    beta: float
    stickiness: float


@dataclass(slots=True)
class SocialRlSelfRewardDemoRewardStickyState:
    """Latent state for the sticky self-reward + demo-reward social RL kernel."""

    q_values: list[float]
    last_self_action: int | None = None


@dataclass(frozen=True)
class SocialRlSelfRewardDemoRewardStickyKernel(
    ModelKernel[
        SocialRlSelfRewardDemoRewardStickyState,
        SocialRlSelfRewardDemoRewardStickyParams,
    ]
):
    """Social Q-learning kernel with self learning, social learning, and stickiness.

    The model combines two influences at decision time:

    - learned ``q_values`` updated by both own and demonstrator rewards
    - a ``stickiness`` bonus for the participant's previous own choice

    Own rows update both the relevant Q-value and ``last_self_action`` when the
    participant actually made a choice. This preserves the choice-history state
    on no-feedback trials while still allowing timeout rows to pass through
    without crashing.
    """

    q_init: float = 0.5

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return the kernel specification for the sticky self+demo reward model."""

        return ModelKernelSpec(
            model_id="social_rl_self_reward_demo_reward_sticky",
            parameter_specs=(
                ParameterSpec(
                    name="alpha_self",
                    transform_id="sigmoid",
                    description="self learning rate",
                    bounds=(0.0, 1.0),
                ),
                ParameterSpec(
                    name="alpha_other",
                    transform_id="sigmoid",
                    description="social learning rate",
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

    def parse_params(
        self,
        raw: dict[str, float],
    ) -> SocialRlSelfRewardDemoRewardStickyParams:
        """Convert raw optimiser values into interpretable model parameters."""

        transforms = self._parameter_transforms()
        return SocialRlSelfRewardDemoRewardStickyParams(
            alpha_self=transforms["alpha_self"].forward(raw["alpha_self"]),
            alpha_other=transforms["alpha_other"].forward(raw["alpha_other"]),
            beta=transforms["beta"].forward(raw["beta"]),
            stickiness=transforms["stickiness"].forward(raw["stickiness"]),
        )

    def initial_state(
        self,
        n_actions: int,
        params: SocialRlSelfRewardDemoRewardStickyParams,
    ) -> SocialRlSelfRewardDemoRewardStickyState:
        """Create the initial latent state with neutral values and no prior choice."""

        del params
        return SocialRlSelfRewardDemoRewardStickyState(
            q_values=[self.q_init] * n_actions,
            last_self_action=None,
        )

    def action_probabilities(
        self,
        state: SocialRlSelfRewardDemoRewardStickyState,
        view: DecisionTrialView,
        params: SocialRlSelfRewardDemoRewardStickyParams,
    ) -> tuple[float, ...]:
        """Compute choice probabilities from Q-values plus a stickiness bonus."""

        logits: list[float] = []
        for action in view.available_actions:
            logit = params.beta * state.q_values[action]
            if action == state.last_self_action:
                logit += params.stickiness
            logits.append(logit)
        return stable_softmax(logits)

    def update(
        self,
        state: SocialRlSelfRewardDemoRewardStickyState,
        view: DecisionTrialView,
        params: SocialRlSelfRewardDemoRewardStickyParams,
    ) -> SocialRlSelfRewardDemoRewardStickyState:
        """Update learned values and preserve own-choice history.

        Self rows update ``q_values`` only when both an action and reward are
        present, but they still refresh ``last_self_action`` whenever the
        participant actually chose. Social rows update the chosen demonstrator
        action only when both the demonstrator action and reward are available.
        """

        updated_q_values = list(state.q_values)
        last_self_action = state.last_self_action

        if view.actor_id == view.learner_id:
            if view.action is not None:
                if view.reward is not None:
                    updated_q_values[view.action] += params.alpha_self * (
                        view.reward - updated_q_values[view.action]
                    )
                last_self_action = view.action
        elif view.action is not None and view.reward is not None:
            updated_q_values[view.action] += params.alpha_other * (
                view.reward - updated_q_values[view.action]
            )

        return SocialRlSelfRewardDemoRewardStickyState(
            q_values=updated_q_values,
            last_self_action=last_self_action,
        )
