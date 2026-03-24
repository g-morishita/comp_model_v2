"""Protocols for Stan model adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from comp_model.inference.config import HierarchyStructure

if TYPE_CHECKING:
    from comp_model.data.schema import Dataset, SubjectData
    from comp_model.inference.config import PriorSpec
    from comp_model.models.condition.shared_delta import SharedDeltaLayout
    from comp_model.models.kernels.base import ModelKernelSpec
    from comp_model.tasks.schemas import TrialSchema

_CONDITION_HIERARCHIES = (
    HierarchyStructure.SUBJECT_BLOCK_CONDITION,
    HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
)


def require_layout_for_condition_hierarchy(
    hierarchy: HierarchyStructure,
    layout: SharedDeltaLayout | None,
) -> None:
    """Raise if a condition-aware hierarchy is requested without a layout.

    Parameters
    ----------
    hierarchy
        Hierarchy structure targeted by the Stan program.
    layout
        Condition-aware parameter layout supplied by the caller.

    Raises
    ------
    ValueError
        If ``hierarchy`` is ``SUBJECT_BLOCK_CONDITION`` or
        ``STUDY_SUBJECT_BLOCK_CONDITION`` and ``layout`` is ``None``.
    """
    if hierarchy in _CONDITION_HIERARCHIES and layout is None:
        raise ValueError(
            f"hierarchy={hierarchy.value!r} requires a SharedDeltaLayout "
            "but layout=None was passed."
        )


class StanAdapter(Protocol):
    """Protocol implemented by Stan data and program adapters."""

    def kernel_spec(self) -> ModelKernelSpec:
        """Return the kernel specification served by the adapter.

        Returns
        -------
        ModelKernelSpec
            Kernel metadata used by the Stan backend.
        """

        ...

    def stan_program_path(self, hierarchy: HierarchyStructure) -> str:
        """Return the Stan program path for a hierarchy structure.

        Parameters
        ----------
        hierarchy
            Hierarchy structure whose Stan program is requested.

        Returns
        -------
        str
            Filesystem path to the Stan program.
        """

        ...

    def build_stan_data(
        self,
        data: SubjectData | Dataset,
        schema: TrialSchema,
        hierarchy: HierarchyStructure,
        layout: SharedDeltaLayout | None = None,
        prior_specs: dict[str, PriorSpec] | None = None,
    ) -> dict[str, Any]:
        """Build Stan data for a subject or dataset.

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
        """

        ...

    def subject_param_names(self) -> tuple[str, ...]:
        """Return subject-level Stan parameter names.

        Returns
        -------
        tuple[str, ...]
            Subject-level parameter names extracted from fit results.
        """

        ...

    def population_param_names(self, hierarchy: HierarchyStructure) -> tuple[str, ...]:
        """Return population-level Stan parameter names for a hierarchy.

        Parameters
        ----------
        hierarchy
            Hierarchy structure targeted by the Stan program.

        Returns
        -------
        tuple[str, ...]
            Population-level parameter names extracted from fit results.
        """

        ...
