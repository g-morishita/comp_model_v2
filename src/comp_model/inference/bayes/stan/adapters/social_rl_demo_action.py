"""Stan adapter for the social demo-action RL kernel."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from comp_model.data.schema import Dataset, SubjectData
from comp_model.inference.bayes.stan.adapters.base import require_layout_for_condition_hierarchy
from comp_model.inference.bayes.stan.data_builder import (
    add_delta_prior_data,
    add_prior_data,
    add_sd_prior_data,
    add_state_reset_data,
    dataset_to_step_data,
    subject_to_step_data,
)
from comp_model.inference.config import HierarchyStructure, PriorSpec
from comp_model.models.kernels.social_rl_demo_action import SocialRlDemoActionKernel

if TYPE_CHECKING:
    from comp_model.models.condition.shared_delta import SharedDeltaLayout
    from comp_model.models.kernels.base import ModelKernelSpec
    from comp_model.tasks.schemas import TrialSchema


class SocialRlDemoActionStanAdapter:
    """Stan adapter for the social demo-action RL kernel."""

    def kernel_spec(self) -> ModelKernelSpec:
        """Return the kernel specification served by this adapter."""

        return SocialRlDemoActionKernel.spec()

    def stan_program_path(self, hierarchy: HierarchyStructure) -> str:
        """Return the Stan program path for the requested hierarchy."""

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
        """Build Stan data for the demo-action programs."""

        require_layout_for_condition_hierarchy(hierarchy, layout)
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
        if hierarchy == HierarchyStructure.STUDY_SUBJECT:
            add_sd_prior_data(stan_data, kspec, prior_specs)
        elif hierarchy == HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION:
            add_delta_prior_data(stan_data, kspec, prior_specs)
            add_sd_prior_data(stan_data, kspec, prior_specs, include_delta=True)
        add_state_reset_data(stan_data, kspec)

        if layout is not None and condition_map is not None:
            stan_data["C"] = len(layout.conditions)
            stan_data["baseline_cond"] = condition_map[layout.baseline_condition]

        return stan_data

    def subject_param_names(self) -> tuple[str, ...]:
        """Return subject-level parameter names extracted from Stan fits."""

        return ("alpha_other_action", "beta")

    def population_param_names(self, hierarchy: HierarchyStructure) -> tuple[str, ...]:
        """Return population-level parameter names for the hierarchy."""

        if hierarchy in (
            HierarchyStructure.SUBJECT_SHARED,
            HierarchyStructure.SUBJECT_BLOCK_CONDITION,
        ):
            return ()
        if hierarchy == HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION:
            return (
                "mu_alpha_other_action_shared_z",
                "sd_alpha_other_action_shared_z",
                "mu_beta_shared_z",
                "sd_beta_shared_z",
                "mu_alpha_other_action_delta_z",
                "sd_alpha_other_action_delta_z",
                "mu_beta_delta_z",
                "sd_beta_delta_z",
                "alpha_other_action_pop",
                "beta_pop",
                "alpha_other_action_shared_pop",
                "beta_shared_pop",
            )
        return (
            "mu_alpha_other_action_z",
            "sd_alpha_other_action_z",
            "mu_beta_z",
            "sd_beta_z",
            "alpha_other_action_pop",
            "beta_pop",
        )

    def extra_posterior_param_names(self, hierarchy: HierarchyStructure) -> tuple[str, ...]:
        """Return additional non-population posterior variable names."""

        if hierarchy in (
            HierarchyStructure.SUBJECT_BLOCK_CONDITION,
            HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
        ):
            return (
                "alpha_other_action_shared_z",
                "beta_shared_z",
                "alpha_other_action_delta_z",
                "beta_delta_z",
            )
        return ()
