"""Kernel + schema compatibility checks.

This module validates that a model kernel's social-information requirements
are satisfiable by a given trial schema.  The check prevents silent data
corruption when, for example, a social kernel is paired with an asocial
schema (producing all-zero social fields) or with an action-only schema
when the kernel needs demonstrator rewards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from comp_model.models.kernels.base import ModelKernel
    from comp_model.tasks.schemas import TrialSchema


def check_kernel_schema_compatibility(
    kernel: ModelKernel[Any, Any],
    schema: TrialSchema,
) -> None:
    """Raise ``ValueError`` if *kernel* is incompatible with *schema*.

    Compatibility rules
    -------------------
    * An asocial kernel (``requires_social=False``) is always compatible
      with any schema.  When paired with a social schema the kernel simply
      ignores the social UPDATE steps.
    * A social kernel (``requires_social=True``) requires the schema to
      contain at least one social UPDATE step directed at the subject.
    * A social kernel whose ``required_social_fields`` is non-empty
      additionally requires that the schema's social observable fields
      are a superset of those required fields.

    Parameters
    ----------
    kernel
        Model kernel to validate.
    schema
        Trial schema that will be used with *kernel*.

    Raises
    ------
    ValueError
        If the kernel's social requirements cannot be met by the schema.
    """

    spec = kernel.spec()

    if not spec.requires_social:
        return

    provided = schema.social_observable_fields

    if not provided:
        raise ValueError(
            f"Kernel {spec.model_id!r} requires social information "
            f"(requires_social=True), but schema {schema.schema_id!r} "
            f"has no social UPDATE steps directed at the subject."
        )

    missing = spec.required_social_fields - provided
    if missing:
        raise ValueError(
            f"Kernel {spec.model_id!r} requires social fields "
            f"{sorted(spec.required_social_fields)}, but schema "
            f"{schema.schema_id!r} only provides {sorted(provided)}. "
            f"Missing: {sorted(missing)}."
        )
