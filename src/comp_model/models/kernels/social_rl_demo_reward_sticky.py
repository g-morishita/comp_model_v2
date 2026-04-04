"""Social RL kernel that learns from demonstrator outcomes with stickiness.

This module extends the pure demonstrator-reward model with a perseveration
term that biases the participant toward repeating their previous own choice.
The latent Q-values still update only from demonstrator outcomes; self-outcome
rows do not change learned values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comp_model.models.kernels.base import ModelKernel, ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.probabilities import stable_softmax

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class SocialRlDemoRewardStickyParams:
    """Free parameters for the sticky demonstrator-reward social RL kernel.

    Attributes
    ----------
    alpha_other
        Social learning rate controlling updates from demonstrator rewards.
    beta
        Inverse temperature scaling the softmax choice rule.
    stickiness
        Additive logit bias toward repeating the participant's previous own
        action. Positive values bias repeats; negative values bias switches.
    """

    alpha_other: float
    beta: float
    stickiness: float


@dataclass(slots=True)
class SocialRlDemoRewardStickyState:
    """Latent state for the sticky demonstrator-reward social RL kernel.

    Attributes
    ----------
    q_values
        Per-action Q-values updated only from demonstrator outcomes.
    last_self_action
        Most recent own choice. ``None`` until the first self action is
        observed.
    """

    q_values: list[float]
    last_self_action: int | None = None


@dataclass(frozen=True)
class SocialRlDemoRewardStickyKernel(
    ModelKernel[SocialRlDemoRewardStickyState, SocialRlDemoRewardStickyParams]
):
    """Demonstrator-reward social RL kernel with a perseveration term.

    The model combines two influences at decision time:

    - learned ``q_values`` from demonstrator outcome observations
    - a ``stickiness`` bonus for the participant's previous own choice

    Own outcome rows are ignored for value learning, but own decision rows
    still refresh ``last_self_action`` so no-feedback trials preserve the
    choice-history state.
    """

    q_init: float = 0.5

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return the kernel specification for the sticky demo-reward model."""

        return ModelKernelSpec(
            model_id="social_rl_demo_reward_sticky",
            parameter_specs=(
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

    def parse_params(self, raw: dict[str, float]) -> SocialRlDemoRewardStickyParams:
        """Convert raw optimiser values into interpretable model parameters."""

        transforms = self._parameter_transforms()
        return SocialRlDemoRewardStickyParams(
            alpha_other=transforms["alpha_other"].forward(raw["alpha_other"]),
            beta=transforms["beta"].forward(raw["beta"]),
            stickiness=transforms["stickiness"].forward(raw["stickiness"]),
        )

    def initial_state(
        self,
        n_actions: int,
        params: SocialRlDemoRewardStickyParams,
    ) -> SocialRlDemoRewardStickyState:
        """Create the initial latent state with neutral values and no prior choice."""

        del params
        return SocialRlDemoRewardStickyState(
            q_values=[self.q_init] * n_actions,
            last_self_action=None,
        )

    def action_probabilities(
        self,
        state: SocialRlDemoRewardStickyState,
        view: DecisionTrialView,
        params: SocialRlDemoRewardStickyParams,
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
        state: SocialRlDemoRewardStickyState,
        view: DecisionTrialView,
        params: SocialRlDemoRewardStickyParams,
    ) -> SocialRlDemoRewardStickyState:
        """Store the participant's most recent own choice at decision time."""

        del params
        if view.actor_id != view.learner_id or view.action is None:
            return state

        return SocialRlDemoRewardStickyState(
            q_values=list(state.q_values),
            last_self_action=view.action,
        )

    def update(
        self,
        state: SocialRlDemoRewardStickyState,
        view: DecisionTrialView,
        params: SocialRlDemoRewardStickyParams,
    ) -> SocialRlDemoRewardStickyState:
        """Update demonstrator-driven values and preserve own-choice history."""

        updated_q_values = list(state.q_values)
        last_self_action = state.last_self_action

        if view.actor_id == view.learner_id:
            if view.action is not None:
                last_self_action = view.action
            return SocialRlDemoRewardStickyState(
                q_values=updated_q_values,
                last_self_action=last_self_action,
            )

        if view.action is not None and view.reward is not None:
            updated_q_values[view.action] += params.alpha_other * (
                view.reward - updated_q_values[view.action]
            )

        return SocialRlDemoRewardStickyState(
            q_values=updated_q_values,
            last_self_action=last_self_action,
        )
