"""Non-sticky social RL demo-action bias kernel.

This kernel implements a pure action-bias social model with no perseveration
term. It does not learn latent action values from rewards. Instead, choice is
driven only by a transient logit bonus toward the most recently observed
demonstrator action.

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
class SocialRlDemoActionBiasParams:
    """Free parameters for the non-sticky demo-action bias social RL kernel."""

    demo_bias: float


@dataclass(slots=True)
class SocialRlDemoActionBiasState:
    """Latent state for the non-sticky demo-action bias social RL kernel."""

    last_demo_action: int | None = None


@dataclass(frozen=True)
class SocialRlDemoActionBiasKernel(
    ModelKernel[SocialRlDemoActionBiasState, SocialRlDemoActionBiasParams]
):
    """Pure demo-action bias model without own-choice stickiness."""

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return the kernel specification for the non-sticky demo-action bias model."""

        return ModelKernelSpec(
            model_id="social_rl_demo_action_bias",
            parameter_specs=(
                ParameterSpec(
                    name="demo_bias",
                    transform_id="identity",
                    description="logit bonus toward the most recently observed demonstrator action",
                ),
            ),
            requires_social=True,
            required_social_fields=frozenset({"action"}),
        )

    def parse_params(self, raw: dict[str, float]) -> SocialRlDemoActionBiasParams:
        """Convert raw optimiser values into interpretable model parameters."""

        transforms = self._parameter_transforms()
        return SocialRlDemoActionBiasParams(
            demo_bias=transforms["demo_bias"].forward(raw["demo_bias"]),
        )

    def initial_state(
        self,
        n_actions: int,
        params: SocialRlDemoActionBiasParams,
    ) -> SocialRlDemoActionBiasState:
        """Create the initial state with no stored demonstrator choice."""

        del n_actions, params
        return SocialRlDemoActionBiasState()

    def action_probabilities(
        self,
        state: SocialRlDemoActionBiasState,
        view: DecisionTrialView,
        params: SocialRlDemoActionBiasParams,
    ) -> tuple[float, ...]:
        """Compute choice probabilities from demonstrator bias only."""

        logits = [
            params.demo_bias if action == state.last_demo_action else 0.0
            for action in view.available_actions
        ]
        return stable_softmax(logits)

    def update(
        self,
        state: SocialRlDemoActionBiasState,
        view: DecisionTrialView,
        params: SocialRlDemoActionBiasParams,
    ) -> SocialRlDemoActionBiasState:
        """Refresh the latest demonstrator action without any value learning."""

        del params
        if view.actor_id != view.learner_id and view.action is not None:
            return SocialRlDemoActionBiasState(last_demo_action=view.action)
        return SocialRlDemoActionBiasState(last_demo_action=state.last_demo_action)
