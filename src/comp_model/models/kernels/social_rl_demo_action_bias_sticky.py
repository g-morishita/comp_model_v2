"""Sticky social RL demo-action bias kernel.

This kernel implements a pure action-bias social model with a separate
perseveration term. It does not learn latent action values from rewards.
Instead, choice is driven by two transient logit bonuses:

- ``demo_bias`` toward the most recently observed demonstrator action
- ``stickiness`` toward the participant's own previous choice

Because the demonstrator action is stored in the latent state, the same kernel
supports both demonstrator-first schemas (biasing the current trial) and
subject-first schemas (biasing the next trial).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comp_model.models.kernels.base import ModelKernel, ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.probabilities import stable_softmax

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class SocialRlDemoActionBiasStickyParams:
    """Free parameters for the sticky demo-action bias social RL kernel."""

    demo_bias: float
    stickiness: float


@dataclass(slots=True)
class SocialRlDemoActionBiasStickyState:
    """Latent state for the sticky demo-action bias social RL kernel."""

    last_demo_action: int | None = None
    last_self_action: int | None = None


@dataclass(frozen=True)
class SocialRlDemoActionBiasStickyKernel(
    ModelKernel[SocialRlDemoActionBiasStickyState, SocialRlDemoActionBiasStickyParams]
):
    """Pure demo-action bias model with own-choice stickiness.

    The kernel has no learned action values. Choice probabilities come only
    from additive logit bonuses:

    - ``demo_bias`` for the most recently observed demonstrator action
    - ``stickiness`` for the participant's previous own action
    """

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return the kernel specification for the sticky demo-action bias model."""

        return ModelKernelSpec(
            model_id="social_rl_demo_action_bias_sticky",
            parameter_specs=(
                ParameterSpec(
                    name="demo_bias",
                    transform_id="identity",
                    description="logit bonus toward the most recently observed demonstrator action",
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

    def parse_params(self, raw: dict[str, float]) -> SocialRlDemoActionBiasStickyParams:
        """Convert raw optimiser values into interpretable model parameters."""

        transforms = self._parameter_transforms()
        return SocialRlDemoActionBiasStickyParams(
            demo_bias=transforms["demo_bias"].forward(raw["demo_bias"]),
            stickiness=transforms["stickiness"].forward(raw["stickiness"]),
        )

    def initial_state(
        self,
        n_actions: int,
        params: SocialRlDemoActionBiasStickyParams,
    ) -> SocialRlDemoActionBiasStickyState:
        """Create the initial state with no stored demonstrator or self choice."""

        del n_actions, params
        return SocialRlDemoActionBiasStickyState()

    def action_probabilities(
        self,
        state: SocialRlDemoActionBiasStickyState,
        view: DecisionTrialView,
        params: SocialRlDemoActionBiasStickyParams,
    ) -> tuple[float, ...]:
        """Compute choice probabilities from demonstrator bias and stickiness."""

        logits: list[float] = []
        for action in view.available_actions:
            logit = 0.0
            if action == state.last_demo_action:
                logit += params.demo_bias
            if action == state.last_self_action:
                logit += params.stickiness
            logits.append(logit)
        return stable_softmax(logits)

    def observe_decision(
        self,
        state: SocialRlDemoActionBiasStickyState,
        view: DecisionTrialView,
        params: SocialRlDemoActionBiasStickyParams,
    ) -> SocialRlDemoActionBiasStickyState:
        """Store the participant's most recent own choice at decision time."""

        del params
        if view.actor_id != view.learner_id or view.action is None:
            return state

        return SocialRlDemoActionBiasStickyState(
            last_demo_action=state.last_demo_action,
            last_self_action=view.action,
        )

    def update(
        self,
        state: SocialRlDemoActionBiasStickyState,
        view: DecisionTrialView,
        params: SocialRlDemoActionBiasStickyParams,
    ) -> SocialRlDemoActionBiasStickyState:
        """Refresh the latest demonstrator and self actions without value learning."""

        del params
        last_demo_action = state.last_demo_action
        last_self_action = state.last_self_action

        if view.actor_id == view.learner_id:
            if view.action is not None:
                last_self_action = view.action
        else:
            if view.action is not None:
                last_demo_action = view.action

        return SocialRlDemoActionBiasStickyState(
            last_demo_action=last_demo_action,
            last_self_action=last_self_action,
        )
