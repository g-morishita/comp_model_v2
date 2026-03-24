"""Stan adapter for the asocial asymmetric RL kernel."""

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
from comp_model.models.kernels.asocial_rl_asymmetric import AsocialRlAsymmetricKernel

if TYPE_CHECKING:
    from comp_model.models.condition.shared_delta import SharedDeltaLayout
    from comp_model.models.kernels.base import ModelKernelSpec
    from comp_model.tasks.schemas import TrialSchema


class AsocialRlAsymmetricStanAdapter:
    """Stan adapter for the asocial asymmetric RL kernel.

    Notes
    -----
    Parameters ``alpha_pos`` and ``alpha_neg`` are both sigmoid-transformed.
    The update rule branches on the sign of the prediction error:
    positive RPEs use ``alpha_pos``, negative RPEs use ``alpha_neg``.
    """

    def kernel_spec(self) -> ModelKernelSpec:
        """Return the kernel specification served by this adapter.

        Returns
        -------
        ModelKernelSpec
            Static kernel metadata for the asocial asymmetric RL kernel.
        """
        return AsocialRlAsymmetricKernel.spec()

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
        Program filenames follow ``{model_id}__{hierarchy.value}.stan``
        inside the adapter's sibling ``programs`` directory.
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
        """Build Stan data for the asocial asymmetric RL programs.

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
            Stan-ready data dictionary passed directly to CmdStanPy.

        Notes
        -----
        Assembles step-stream data, prior hyperparameters, state-reset flags,
        and initial value data. Condition indices are added for
        condition-aware hierarchies (SUBJECT_BLOCK_CONDITION and
        STUDY_SUBJECT_BLOCK_CONDITION) when a layout is provided.
        """
        kspec = self.kernel_spec()

        condition_map: dict[str, int] | None = None
        if layout is not None and hierarchy in (
            HierarchyStructure.SUBJECT_BLOCK_CONDITION,
            HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
        ):
            condition_map = {cond: idx for idx, cond in enumerate(layout.conditions, start=1)}

        if isinstance(data, SubjectData):
            stan_data = subject_to_step_data(
                data, schema, kernel_spec=kspec, condition_map=condition_map
            )
        else:
            stan_data = dataset_to_step_data(
                data, schema, kernel_spec=kspec, condition_map=condition_map
            )

        add_prior_data(stan_data, kspec, prior_specs)
        add_state_reset_data(stan_data, kspec)
        add_initial_value_data(stan_data, kspec)

        if layout is not None and condition_map is not None:
            stan_data["C"] = len(layout.conditions)
            stan_data["baseline_cond"] = condition_map[layout.baseline_condition]

        return stan_data

    def subject_param_names(self) -> tuple[str, ...]:
        """Return subject-level parameter names extracted from Stan fits.

        Returns
        -------
        tuple[str, ...]
            Subject-level parameter names: ``alpha_pos``, ``alpha_neg``,
            and ``beta``.
        """
        return ("alpha_pos", "alpha_neg", "beta")

    def population_param_names(self, hierarchy: HierarchyStructure) -> tuple[str, ...]:
        """Return population-level parameter names for the hierarchy.

        Parameters
        ----------
        hierarchy
            Hierarchy structure targeted by the Stan program.

        Returns
        -------
        tuple[str, ...]
            Population-level parameter names. Returns an empty tuple for
            SUBJECT_SHARED fits. For SUBJECT_BLOCK_CONDITION returns shared
            and delta z-score parameters. For STUDY_SUBJECT returns
            group-level means, standard deviations, and population-scale
            parameters. For STUDY_SUBJECT_BLOCK_CONDITION returns the full
            set of study-level hyperparameters plus shared and delta
            z-scores.
        """
        if hierarchy == HierarchyStructure.SUBJECT_SHARED:
            return ()
        if hierarchy == HierarchyStructure.SUBJECT_BLOCK_CONDITION:
            return (
                "alpha_pos_shared_z",
                "alpha_neg_shared_z",
                "beta_shared_z",
                "alpha_pos_delta_z",
                "alpha_neg_delta_z",
                "beta_delta_z",
            )
        if hierarchy == HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION:
            return (
                "mu_alpha_pos_shared_z",
                "sd_alpha_pos_shared_z",
                "mu_alpha_neg_shared_z",
                "sd_alpha_neg_shared_z",
                "mu_beta_shared_z",
                "sd_beta_shared_z",
                "mu_alpha_pos_delta_z",
                "sd_alpha_pos_delta_z",
                "mu_alpha_neg_delta_z",
                "sd_alpha_neg_delta_z",
                "mu_beta_delta_z",
                "sd_beta_delta_z",
                "alpha_pos_shared_z",
                "alpha_neg_shared_z",
                "beta_shared_z",
                "alpha_pos_delta_z",
                "alpha_neg_delta_z",
                "beta_delta_z",
                "alpha_pos_shared_pop",
                "alpha_neg_shared_pop",
                "beta_shared_pop",
            )
        # STUDY_SUBJECT
        return (
            "mu_alpha_pos_z",
            "sd_alpha_pos_z",
            "mu_alpha_neg_z",
            "sd_alpha_neg_z",
            "mu_beta_z",
            "sd_beta_z",
            "alpha_pos_pop",
            "alpha_neg_pop",
            "beta_pop",
        )
