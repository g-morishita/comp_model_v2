"""Sticky social RL self-reward + demo-action mixture kernel.

This module extends the self-reward + demonstrator-action mixture model with a
stickiness (perseveration) term that biases the agent to repeat its own
previous choice.

The model maintains two independent value systems:

- ``v_outcome``: updated by self reward only
- ``v_tendency``: updated by demonstrator action frequency

At decision time the systems are combined via ``w_imitation`` and passed
through a softmax. A separate ``stickiness`` term is then added to the logit
of the action most recently chosen by the subject.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comp_model.models.kernels.base import ModelKernel, ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.probabilities import stable_softmax

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class SocialRlSelfRewardDemoActionMixtureStickyParams:
    """Free parameters for the sticky self-reward + demo-action mixture kernel."""

    alpha_self: float
    alpha_other_action: float
    w_imitation: float
    beta: float
    stickiness: float


@dataclass(slots=True)
class SocialRlSelfRewardDemoActionMixtureStickyState:
    """Latent state for the sticky self-reward + demo-action mixture kernel."""

    v_outcome: list[float]
    v_tendency: list[float]
    last_self_action: int | None = None


@dataclass(frozen=True)
class SocialRlSelfRewardDemoActionMixtureStickyKernel(
    ModelKernel[
        SocialRlSelfRewardDemoActionMixtureStickyState,
        SocialRlSelfRewardDemoActionMixtureStickyParams,
    ]
):
    """Sticky social RL kernel combining outcome-, action-, and perseveration-based influences.

    Maintains the same two value systems as
    :class:`SocialRlSelfRewardDemoActionMixtureKernel` and adds a third choice
    influence:

    - ``v_outcome``: updated by self reward only
    - ``v_tendency``: updated by demonstrator action frequency
    - ``last_self_action``: adds a ``stickiness`` bonus to the previously
      self-chosen action at the next decision
    """

    v_outcome_init: float = 0.5

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return the kernel specification for the sticky action-mixture model."""

        return ModelKernelSpec(
            model_id="social_rl_self_reward_demo_action_mixture_sticky",
            parameter_specs=(
                ParameterSpec(
                    name="alpha_self",
                    transform_id="sigmoid",
                    description="learning rate for self outcome",
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
            required_social_fields=frozenset({"action"}),
        )

    def parse_params(
        self,
        raw: dict[str, float],
    ) -> SocialRlSelfRewardDemoActionMixtureStickyParams:
        """Convert raw optimiser values into interpretable model parameters."""

        transforms = self._parameter_transforms()
        return SocialRlSelfRewardDemoActionMixtureStickyParams(
            alpha_self=transforms["alpha_self"].forward(raw["alpha_self"]),
            alpha_other_action=transforms["alpha_other_action"].forward(raw["alpha_other_action"]),
            w_imitation=transforms["w_imitation"].forward(raw["w_imitation"]),
            beta=transforms["beta"].forward(raw["beta"]),
            stickiness=transforms["stickiness"].forward(raw["stickiness"]),
        )

    def initial_state(
        self,
        n_actions: int,
        params: SocialRlSelfRewardDemoActionMixtureStickyParams,
    ) -> SocialRlSelfRewardDemoActionMixtureStickyState:
        """Create the initial state with neutral values and no prior own choice."""

        del params
        return SocialRlSelfRewardDemoActionMixtureStickyState(
            v_outcome=[self.v_outcome_init] * n_actions,
            v_tendency=[1.0 / n_actions] * n_actions,
            last_self_action=None,
        )

    def action_probabilities(
        self,
        state: SocialRlSelfRewardDemoActionMixtureStickyState,
        view: DecisionTrialView,
        params: SocialRlSelfRewardDemoActionMixtureStickyParams,
    ) -> tuple[float, ...]:
        """Compute choice probabilities by combining both value systems and stickiness."""

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
        state: SocialRlSelfRewardDemoActionMixtureStickyState,
        view: DecisionTrialView,
        params: SocialRlSelfRewardDemoActionMixtureStickyParams,
    ) -> SocialRlSelfRewardDemoActionMixtureStickyState:
        """Update both value systems and the previous-own-choice state."""

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
            if view.action is not None:
                for action in range(len(updated_v_tendency)):
                    target = 1.0 if action == view.action else 0.0
                    updated_v_tendency[action] += params.alpha_other_action * (
                        target - updated_v_tendency[action]
                    )

        return SocialRlSelfRewardDemoActionMixtureStickyState(
            v_outcome=updated_v_outcome,
            v_tendency=updated_v_tendency,
            last_self_action=last_self_action,
        )
