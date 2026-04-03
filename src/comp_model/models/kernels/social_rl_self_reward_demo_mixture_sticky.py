"""Sticky social RL mixture kernel.

This module extends the self-reward + demonstrator-mixture model with a
stickiness (perseveration) term that biases the agent to repeat its own
previous choice.

The model maintains two independent value systems:

- ``v_outcome``: updated by self reward and demonstrator reward
- ``v_tendency``: updated by demonstrator action frequency

At decision time the systems are combined via ``w_imitation`` and passed
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
from comp_model.models.kernels.transforms import get_transform

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class SocialRlSelfRewardDemoMixtureStickyParams:
    """Free parameters for the sticky mixture social RL kernel.

    Attributes
    ----------
    alpha_self
        Learning rate for self outcome updates; in (0, 1).
    alpha_other_outcome
        Learning rate for demonstrator outcome updates to ``v_outcome``; in (0, 1).
    alpha_other_action
        Learning rate for demonstrator action updates to ``v_tendency``; in (0, 1).
    w_imitation
        Mixing weight for imitation at decision time; in (0, 1).
    beta
        Inverse temperature for the softmax choice rule; positive.
    stickiness
        Perseveration term added to the logit of the subject's previous own
        choice. Positive values bias repeats; negative values bias switches.
    """

    alpha_self: float
    alpha_other_outcome: float
    alpha_other_action: float
    w_imitation: float
    beta: float
    stickiness: float


@dataclass(slots=True)
class SocialRlSelfRewardDemoMixtureStickyState:
    """Latent state for the sticky mixture social RL kernel.

    Attributes
    ----------
    v_outcome
        Outcome-based value estimates, updated by self and demonstrator rewards.
    v_tendency
        Action tendency estimates, updated by demonstrator action frequency.
    last_self_action
        The subject's most recent own choice. ``None`` until the first
        self-update is observed.
    """

    v_outcome: list[float]
    v_tendency: list[float]
    last_self_action: int | None = None


@dataclass(frozen=True)
class SocialRlSelfRewardDemoMixtureStickyKernel(
    ModelKernel[
        SocialRlSelfRewardDemoMixtureStickyState,
        SocialRlSelfRewardDemoMixtureStickyParams,
    ]
):
    """Sticky mixture social RL kernel.

    This model combines outcome-based and action-tendency systems exactly like
    :class:`SocialRlSelfRewardDemoMixtureKernel`, but adds a perseveration
    effect so that the subject's previous own action receives an additive logit
    bonus at the next decision.

    Attributes
    ----------
    v_outcome_init
        Starting value for outcome-tracker values. Defaults to 0.5.
    """

    v_outcome_init: float = 0.5

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return the kernel specification for the sticky mixture model."""

        return ModelKernelSpec(
            model_id="social_rl_self_reward_demo_mixture_sticky",
            parameter_specs=(
                ParameterSpec(
                    name="alpha_self",
                    transform_id="sigmoid",
                    description="learning rate for self outcome",
                    bounds=(0.0, 1.0),
                ),
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

    def parse_params(self, raw: dict[str, float]) -> SocialRlSelfRewardDemoMixtureStickyParams:
        """Convert raw optimiser values into interpretable model parameters."""

        transforms = {ps.name: get_transform(ps.transform_id) for ps in self.spec().parameter_specs}
        return SocialRlSelfRewardDemoMixtureStickyParams(
            alpha_self=transforms["alpha_self"].forward(raw["alpha_self"]),
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
        params: SocialRlSelfRewardDemoMixtureStickyParams,
    ) -> SocialRlSelfRewardDemoMixtureStickyState:
        """Create the agent's initial belief state."""

        del params
        return SocialRlSelfRewardDemoMixtureStickyState(
            v_outcome=[self.v_outcome_init] * n_actions,
            v_tendency=[1.0 / n_actions] * n_actions,
            last_self_action=None,
        )

    def action_probabilities(
        self,
        state: SocialRlSelfRewardDemoMixtureStickyState,
        view: DecisionTrialView,
        params: SocialRlSelfRewardDemoMixtureStickyParams,
    ) -> tuple[float, ...]:
        """Compute choice probabilities with an added perseveration term."""

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

    def update(
        self,
        state: SocialRlSelfRewardDemoMixtureStickyState,
        view: DecisionTrialView,
        params: SocialRlSelfRewardDemoMixtureStickyParams,
    ) -> SocialRlSelfRewardDemoMixtureStickyState:
        """Update latent state from self and social observations."""

        updated_v_outcome = list(state.v_outcome)
        updated_v_tendency = list(state.v_tendency)
        last_self_action = state.last_self_action

        if view.actor_id == view.learner_id:
            assert view.action is not None and view.reward is not None
            updated_v_outcome[view.action] += params.alpha_self * (
                view.reward - updated_v_outcome[view.action]
            )
            last_self_action = view.action
        else:
            if view.action is not None and view.reward is not None:
                updated_v_outcome[view.action] += params.alpha_other_outcome * (
                    view.reward - updated_v_outcome[view.action]
                )
                for action in range(len(updated_v_tendency)):
                    target = 1.0 if action == view.action else 0.0
                    updated_v_tendency[action] += params.alpha_other_action * (
                        target - updated_v_tendency[action]
                    )

        return SocialRlSelfRewardDemoMixtureStickyState(
            v_outcome=updated_v_outcome,
            v_tendency=updated_v_tendency,
            last_self_action=last_self_action,
        )
