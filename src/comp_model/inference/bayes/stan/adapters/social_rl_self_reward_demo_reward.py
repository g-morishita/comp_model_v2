"""Stan adapter for the social self-reward + demo-reward RL kernel."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from comp_model.data.schema import Dataset, SubjectData
from comp_model.inference.bayes.stan.data_builder import (
    add_initial_value_data,
    add_prior_data,
    add_state_reset_data,
    dataset_to_step_data,
    subject_to_step_data,
)
from comp_model.inference.config import HierarchyStructure, PriorSpec
from comp_model.models.kernels.social_rl_self_reward_demo_reward import (
    SocialRlSelfRewardDemoRewardKernel,
)

if TYPE_CHECKING:
    from comp_model.models.condition.shared_delta import SharedDeltaLayout
    from comp_model.models.kernels.base import ModelKernelSpec
    from comp_model.tasks.schemas import TrialSchema


class SocialRlSelfRewardDemoRewardStanAdapter:
    """Stan adapter for the social self-reward + demo-reward RL kernel."""

    def kernel_spec(self) -> ModelKernelSpec:
        return SocialRlSelfRewardDemoRewardKernel.spec()

    def stan_program_path(self, hierarchy: HierarchyStructure) -> str:
        programs_dir = Path(__file__).resolve().parent.parent / "programs"
        filename = f"{self.kernel_spec().model_id}__{hierarchy.value}.stan"
        return str(programs_dir / filename)

    def build_stan_data(
        self,
        data: SubjectData | Dataset,
        schema: TrialSchema,
        hierarchy: HierarchyStructure,
        layout: SharedDeltaLayout | None = None,
        prior_specs: dict[str, PriorSpec] | None = None,
    ) -> dict[str, Any]:
        kspec = self.kernel_spec()

        condition_map: dict[str, int] | None = None
        if layout is not None and hierarchy in (
            HierarchyStructure.SUBJECT_BLOCK_CONDITION,
            HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
        ):
            condition_map = {cond: idx for idx, cond in enumerate(layout.conditions, start=1)}

        if isinstance(data, SubjectData):
            stan_data = subject_to_step_data(
                data,
                schema,
                kernel_spec=kspec,
                condition_map=condition_map,
                include_social=True,
            )
        else:
            stan_data = dataset_to_step_data(
                data,
                schema,
                kernel_spec=kspec,
                condition_map=condition_map,
                include_social=True,
            )

        add_prior_data(stan_data, kspec, prior_specs)
        add_state_reset_data(stan_data, kspec)
        add_initial_value_data(stan_data, 0.5)

        if layout is not None and condition_map is not None:
            stan_data["C"] = len(layout.conditions)
            stan_data["baseline_cond"] = condition_map[layout.baseline_condition]

        return stan_data

    def subject_param_names(self) -> tuple[str, ...]:
        return ("alpha_self", "alpha_other", "beta")

    def population_param_names(self, hierarchy: HierarchyStructure) -> tuple[str, ...]:
        if hierarchy == HierarchyStructure.SUBJECT_SHARED:
            return ()
        if hierarchy == HierarchyStructure.SUBJECT_BLOCK_CONDITION:
            return (
                "alpha_self_shared_z",
                "alpha_other_shared_z",
                "beta_shared_z",
                "alpha_self_delta_z",
                "alpha_other_delta_z",
                "beta_delta_z",
            )
        if hierarchy == HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION:
            return (
                "mu_alpha_self_shared_z",
                "sd_alpha_self_shared_z",
                "mu_alpha_other_shared_z",
                "sd_alpha_other_shared_z",
                "mu_beta_shared_z",
                "sd_beta_shared_z",
                "mu_alpha_self_delta_z",
                "sd_alpha_self_delta_z",
                "mu_alpha_other_delta_z",
                "sd_alpha_other_delta_z",
                "mu_beta_delta_z",
                "sd_beta_delta_z",
                "alpha_self_shared_z",
                "alpha_other_shared_z",
                "beta_shared_z",
                "alpha_self_delta_z",
                "alpha_other_delta_z",
                "beta_delta_z",
                "alpha_self_shared_pop",
                "alpha_other_shared_pop",
                "beta_shared_pop",
            )
        return (
            "mu_alpha_self_z",
            "sd_alpha_self_z",
            "mu_alpha_other_z",
            "sd_alpha_other_z",
            "mu_beta_z",
            "sd_beta_z",
            "alpha_self_pop",
            "alpha_other_pop",
            "beta_pop",
        )
