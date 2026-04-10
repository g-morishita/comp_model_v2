"""Social RL demo-action kernel.

This module implements the pure demonstrator-action-learning limit of the
action-mixture family. The agent maintains only an action-tendency system
updated from demonstrator action frequency; there is no reward-based value
tracker and no self-reward learning.

At decision time the current action tendencies are passed through a softmax:

    p(a) = softmax(beta * v_tendency[a])

This differs from :class:`SocialRlDemoMixtureKernel` by removing the reward
component entirely, and from
:class:`SocialRlSelfRewardDemoActionMixtureKernel` by removing the self-reward
component. Without a second value system, a mixture weight would be redundant,
so the identifiable model contract is just ``alpha_other_action`` and ``beta``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comp_model.models.kernels.base import ModelKernel, ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.probabilities import stable_softmax

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class SocialRlDemoActionParams:
    """Free parameters for the demo-action social RL kernel."""

    alpha_other_action: float
    beta: float


@dataclass(slots=True)
class SocialRlDemoActionState:
    """Latent state for the demo-action social RL kernel."""

    v_tendency: list[float]


@dataclass(frozen=True)
class SocialRlDemoActionKernel(ModelKernel[SocialRlDemoActionState, SocialRlDemoActionParams]):
    """Demonstrator-action-only social RL kernel."""

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return the kernel specification for the demo-action social RL model."""

        return ModelKernelSpec(
            model_id="social_rl_demo_action",
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
            ),
            requires_social=True,
            required_social_fields=frozenset({"action"}),
        )

    def parse_params(self, raw: dict[str, float]) -> SocialRlDemoActionParams:
        """Convert raw optimiser values into interpretable model parameters."""

        transforms = self._parameter_transforms()
        return SocialRlDemoActionParams(
            alpha_other_action=transforms["alpha_other_action"].forward(raw["alpha_other_action"]),
            beta=transforms["beta"].forward(raw["beta"]),
        )

    def initial_state(
        self, n_actions: int, params: SocialRlDemoActionParams
    ) -> SocialRlDemoActionState:
        """Create the initial action-tendency state."""

        del params
        return SocialRlDemoActionState(v_tendency=[1.0 / n_actions] * n_actions)

    def action_probabilities(
        self,
        state: SocialRlDemoActionState,
        view: DecisionTrialView,
        params: SocialRlDemoActionParams,
    ) -> tuple[float, ...]:
        """Compute choice probabilities from the current action tendencies."""

        logits = [params.beta * state.v_tendency[action] for action in view.available_actions]
        return stable_softmax(logits)

    def update(
        self,
        state: SocialRlDemoActionState,
        view: DecisionTrialView,
        params: SocialRlDemoActionParams,
    ) -> SocialRlDemoActionState:
        """Update action tendencies from demonstrator action observations only."""
        if view.actor_id == view.learner_id or view.action is None:
            return state

        alpha = params.alpha_other_action
        updated_v_tendency = [value * (1 - alpha) for value in state.v_tendency]
        updated_v_tendency[view.action] += alpha
        return SocialRlDemoActionState(v_tendency=updated_v_tendency)
