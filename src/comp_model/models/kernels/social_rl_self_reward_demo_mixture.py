from __future__ import annotations

from dataclasses import dataclass

from comp_model.models.kernels.base import ModelKernel, ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.transforms import get_transform


@dataclass(frozen=True, slots=True)
class SocialRlSelfRewardDemoMixtureParams:
    alpha_self: float
    alpha_other_outcome: float
    alpha_other_action: float
    weight_action: float
    beta: float


@dataclass(slots=True)
class SocialRlSelfRewardDemoMixtureState:
    q_reward: list[float]  # updated by self reward + demo reward (outcome tracker)
    q_action: list[float]  # updated by demo action frequency (action tracker)


class SocialRlSelfRewardDemoMixtureKernel(
    ModelKernel[SocialRlSelfRewardDemoMixtureState, SocialRlSelfRewardDemoMixtureParams]
):
    @classmethod
    def spec(cls) -> ModelKernelSpec:
        return ModelKernelSpec(
            model_id="social_rl_self_reward_demo_mixture",
            parameter_specs=(
                ParameterSpec(
                    name="alpha_self",
                    transform_id="sigmoid",
                    description="learning from self outcome",
                ),
                ParameterSpec(
                    name="alpha_other_outcome",
                    transform_id="sigmoid",
                    description="learning from other outcome",
                ),
                ParameterSpec(
                    name="alpha_other_action",
                    transform_id="sigmoid",
                    description="learning from other action",
                ),
                ParameterSpec(
                    name="weight_action",
                    transform_id="sigmoid",
                    description="weight of learning from other action",
                ),
                ParameterSpec(
                    name="beta",
                    transform_id="softplus",
                    description="inverse temperature",
                ),
            ),
            requires_social=True,
        )

    def parse_params(self, raw: dict[str, float]) -> SocialRlSelfRewardDemoMixtureParams:
        transforms = {ps.name: get_transform(ps.transform_id) for ps in self.spec().parameter_specs}
        return SocialRlSelfRewardDemoMixtureParams(
            alpha_self=transforms["alpha_self"].forward(raw["alpha_self"]),
            alpha_other_outcome=transforms["alpha_other_outcome"].forward(
                raw["alpha_other_outcome"]
            ),
            alpha_other_action=transforms["alpha_other_action"].forward(raw["alpha_other_action"]),
            weight_action=transforms["weight_action"].forward(raw["weight_action"]),
            beta=transforms["beta"].forward(raw["beta"]),
        )
