"""Stan adapter for the asocial Q-learning kernel.

Adapters isolate Stan-specific choices such as program filenames, data export,
and which posterior variables should be read back from a completed fit.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from comp_model.data.schema import Dataset, SubjectData
from comp_model.inference.bayes.stan.data_builder import (
    add_condition_data,
    add_prior_data,
    add_state_reset_data,
    dataset_to_stan_data,
    subject_to_stan_data,
)
from comp_model.inference.config import HierarchyStructure
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

        Returns
        -------
        dict[str, Any]
            Stan-ready data dictionary.

        Notes
        -----
        The adapter always exports replay data, prior hyperparameters, and the
        integer reset-policy flag. Condition indices are added only for
        condition-aware hierarchies and only when fitting a single subject.
        """

        if isinstance(data, SubjectData):
            stan_data = subject_to_stan_data(data, schema)
        else:
            stan_data = dataset_to_stan_data(data, schema)

        add_prior_data(stan_data, self.kernel_spec())
        add_state_reset_data(stan_data, self.kernel_spec())

        if layout is not None and hierarchy in (
            HierarchyStructure.SUBJECT_BLOCK_CONDITION,
            HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
        ):
            if isinstance(data, SubjectData):
                add_condition_data(stan_data, data, layout)
            else:
                raise ValueError("Condition-aware Stan export is currently single-subject only")

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
        return ("mu_alpha", "sd_alpha", "mu_beta", "sd_beta", "alpha_pop", "beta_pop")
