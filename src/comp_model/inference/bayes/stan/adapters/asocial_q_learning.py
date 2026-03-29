"""Stan adapter for the asocial Q-learning kernel.

Adapters isolate Stan-specific choices such as program filenames, data export,
and which posterior variables should be read back from a completed fit.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from comp_model.data.schema import Dataset, SubjectData
from comp_model.inference.bayes.stan.adapters.base import require_layout_for_condition_hierarchy
from comp_model.inference.bayes.stan.data_builder import (
    add_initial_value_data,
    add_prior_data,
    add_sd_prior_data,
    add_state_reset_data,
    dataset_to_step_data,
    subject_to_step_data,
)
from comp_model.inference.config import HierarchyStructure, PriorSpec
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel

if TYPE_CHECKING:
    from comp_model.models.condition.shared_delta import SharedDeltaLayout
    from comp_model.models.kernels.base import ModelKernelSpec
    from comp_model.tasks.schemas import TrialSchema


class AsocialQLearningStanAdapter:
    """Stan adapter for the asocial Q-learning kernel.

    Notes
    -----
    The adapter currently supports the asocial Q-learning family only and uses a
    simple filename convention based on ``model_id`` and hierarchy name.
    """

    def kernel_spec(self) -> ModelKernelSpec:
        """Return the kernel specification served by this adapter.

        Returns
        -------
        ModelKernelSpec
            Static kernel metadata.
        """

        return AsocialQLearningKernel.spec()

    def stan_program_path(self, hierarchy: HierarchyStructure) -> str:
        """Return the Stan program path for the requested hierarchy.

        Parameters
        ----------
        hierarchy
            Hierarchy structure whose Stan program is requested.

        Returns
        -------
        str
            Absolute path to the Stan program file.

        Notes
        -----
        Program filenames follow
        ``{model_id}__{hierarchy.value}.stan`` inside the adapter's sibling
        ``programs`` directory.
        """

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
        """Build Stan data for the asocial Q-learning programs.

        Parameters
        ----------
        data
            Subject or dataset to export.
        schema
            Trial schema used for replay extraction.
        hierarchy
            Hierarchy structure targeted by the Stan program.
        layout
            Optional condition-aware parameter layout.
        prior_specs
            Optional mapping from parameter name to prior specification.

        Returns
        -------
        dict[str, Any]
            Stan-ready data dictionary.

        Notes
        -----
        The adapter exports step-stream data, prior hyperparameters, state-reset
        flags, and initial value data. Condition indices are added for
        condition-aware hierarchies when a layout is provided.
        """

        require_layout_for_condition_hierarchy(hierarchy, layout)
        kspec = self.kernel_spec()

        # Build condition map if needed
        condition_map: dict[str, int] | None = None
        if layout is not None and hierarchy in (
            HierarchyStructure.SUBJECT_BLOCK_CONDITION,
            HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
        ):
            condition_map = {cond: idx for idx, cond in enumerate(layout.conditions, start=1)}

        # Build step-stream data
        if isinstance(data, SubjectData):
            stan_data = subject_to_step_data(
                data, schema, kernel_spec=kspec, condition_map=condition_map
            )
        else:
            stan_data = dataset_to_step_data(
                data, schema, kernel_spec=kspec, condition_map=condition_map
            )

        # Add prior, reset, and initial value data
        add_prior_data(stan_data, kspec, prior_specs)
        if hierarchy == HierarchyStructure.STUDY_SUBJECT:
            add_sd_prior_data(stan_data, kspec, prior_specs)
        elif hierarchy == HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION:
            add_sd_prior_data(stan_data, kspec, prior_specs, include_delta=True)
        add_state_reset_data(stan_data, kspec)
        add_initial_value_data(stan_data, 0.5)

        # Add condition metadata for condition-aware hierarchies
        if layout is not None and condition_map is not None:
            stan_data["C"] = len(layout.conditions)
            stan_data["baseline_cond"] = condition_map[layout.baseline_condition]

        return stan_data

    def subject_param_names(self) -> tuple[str, ...]:
        """Return subject-level parameter names extracted from Stan fits.

        Returns
        -------
        tuple[str, ...]
            Subject-level parameter names expected in the Stan generated
            quantities or transformed parameters blocks.
        """

        return ("alpha", "beta")

    def population_param_names(self, hierarchy: HierarchyStructure) -> tuple[str, ...]:
        """Return population-level parameter names for the hierarchy.

        Parameters
        ----------
        hierarchy
            Hierarchy structure targeted by the Stan program.

        Returns
        -------
        tuple[str, ...]
            Population-level parameter names. Subject-shared fits have no
            separate population-level outputs.
        """

        if hierarchy == HierarchyStructure.SUBJECT_SHARED:
            return ()
        if hierarchy == HierarchyStructure.SUBJECT_BLOCK_CONDITION:
            return (
                "alpha_shared_z",
                "beta_shared_z",
                "alpha_delta_z",
                "beta_delta_z",
            )
        if hierarchy == HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION:
            return (
                "mu_alpha_shared_z",
                "sd_alpha_shared_z",
                "mu_beta_shared_z",
                "sd_beta_shared_z",
                "mu_alpha_delta_z",
                "sd_alpha_delta_z",
                "mu_beta_delta_z",
                "sd_beta_delta_z",
                "alpha_shared_z",
                "beta_shared_z",
                "alpha_delta_z",
                "beta_delta_z",
                "alpha_pop",
                "beta_pop",
                "alpha_shared_pop",
                "beta_shared_pop",
            )
        return ("mu_alpha_z", "sd_alpha_z", "mu_beta_z", "sd_beta_z", "alpha_pop", "beta_pop")
