"""Social RL kernel that learns from demonstrator outcomes only.

This module implements the pure social-outcome-learning variant of the reward
model family. The participant maintains a single Q-value per action, but those
values are updated only when the participant observes a demonstrator's choice
and reward. The participant's own rewards do not change the latent values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comp_model.models.kernels.base import ModelKernel, ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.probabilities import stable_softmax

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class SocialRlDemoRewardParams:
    """Free parameters for the demonstrator-reward-only social RL kernel.

    Attributes
    ----------
    alpha_other
        Social learning rate — a number strictly between 0 and 1. Controls
        how strongly the participant updates from the demonstrator's reward.
    beta
        Inverse temperature — a positive number controlling how
        deterministically the participant acts on current Q-values.
    """

    alpha_other: float
    beta: float


@dataclass(slots=True)
class SocialRlDemoRewardState:
    """Latent state for the demonstrator-reward-only social RL kernel.

    Attributes
    ----------
    q_values
        One Q-value per action. Values start at ``0.5`` and change only in
        response to demonstrator outcome observations.
    """

    q_values: list[float]


@dataclass(frozen=True)
class SocialRlDemoRewardKernel(ModelKernel[SocialRlDemoRewardState, SocialRlDemoRewardParams]):
    """Q-learning model driven only by demonstrator outcome observations.

    The participant still chooses actions with a standard softmax over Q-values,
    but those Q-values are not updated by the participant's own rewards. Self
    UPDATE steps are no-ops; only demonstrator UPDATE steps change the latent
    values via ``alpha_other``.

    Attributes
    ----------
    q_init
        Starting value assigned to all Q-values before any learning occurs.
        Defaults to ``0.5``.
    """

    q_init: float = 0.5

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return the kernel specification for the demonstrator-reward model."""

        return ModelKernelSpec(
            model_id="social_rl_demo_reward",
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
            ),
            requires_social=True,
            required_social_fields=frozenset({"action", "reward"}),
        )

    def parse_params(self, raw: dict[str, float]) -> SocialRlDemoRewardParams:
        """Convert raw optimiser values into interpretable model parameters."""

        transforms = self._parameter_transforms()
        return SocialRlDemoRewardParams(
            alpha_other=transforms["alpha_other"].forward(raw["alpha_other"]),
            beta=transforms["beta"].forward(raw["beta"]),
        )

    def initial_state(
        self, n_actions: int, params: SocialRlDemoRewardParams
    ) -> SocialRlDemoRewardState:
        """Create the initial neutral Q-value state."""

        del params
        return SocialRlDemoRewardState(q_values=[self.q_init] * n_actions)

    def action_probabilities(
        self,
        state: SocialRlDemoRewardState,
        view: DecisionTrialView,
        params: SocialRlDemoRewardParams,
    ) -> tuple[float, ...]:
        """Compute choice probabilities from the current Q-values."""

        logits = [params.beta * state.q_values[action] for action in view.available_actions]
        return stable_softmax(logits)

    def update(
        self,
        state: SocialRlDemoRewardState,
        view: DecisionTrialView,
        params: SocialRlDemoRewardParams,
    ) -> SocialRlDemoRewardState:
        """Update Q-values from demonstrator rewards only."""

        if view.actor_id == view.learner_id:
            return state

        updated_q_values = list(state.q_values)
        if view.action is not None and view.reward is not None:
            updated_q_values[view.action] += params.alpha_other * (
                view.reward - updated_q_values[view.action]
            )
        return SocialRlDemoRewardState(q_values=updated_q_values)
