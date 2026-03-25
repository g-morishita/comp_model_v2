"""Kernel + schema compatibility checks.

This module validates that a model kernel's social-information requirements
are satisfiable by a given trial schema.  The check prevents silent data
corruption when, for example, a social kernel is paired with an asocial
schema (producing all-zero social fields) or with an action-only schema
when the kernel needs demonstrator rewards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comp_model.data.schema import EventPhase

if TYPE_CHECKING:
    from comp_model.models.kernels.base import ModelKernel, ModelKernelSpec
    from comp_model.tasks.schemas import TrialSchema


def check_kernel_schema_compatibility(
    kernel: ModelKernel[Any, Any],
    schema: TrialSchema,
) -> None:
    """Raise ``ValueError`` if *kernel* is incompatible with *schema*.

    Convenience wrapper around :func:`check_spec_schema_compatibility` that
    extracts the kernel's spec automatically.

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

    check_spec_schema_compatibility(kernel.spec(), schema)


def check_spec_schema_compatibility(
    spec: ModelKernelSpec,
    schema: TrialSchema,
) -> None:
    """Raise ``ValueError`` if *spec* is incompatible with *schema*.

    Compatibility rules
    -------------------
    * An asocial kernel (``requires_social=False``) is always compatible
      with any schema.  When paired with a social schema the kernel simply
      ignores the social UPDATE steps.
    * A social kernel (``requires_social=True``) requires the schema to
      contain at least one social UPDATE step directed at the subject.
    * A social kernel whose ``required_social_fields`` is non-empty
      additionally requires that **every** social UPDATE step directed at
      the subject provides all of those fields.  This per-step check
      prevents the union of fields across steps from masking a step that
      lacks a required field.

    Parameters
    ----------
    spec
        Kernel specification to validate.
    schema
        Trial schema that will be used with the kernel.

    Raises
    ------
    ValueError
        If the kernel's social requirements cannot be met by the schema.
    """

    if not spec.requires_social:
        return

    # Collect per-step observable fields for social UPDATE steps directed
    # at the subject.
    social_steps: list[tuple[int, frozenset[str]]] = []
    for i, step in enumerate(schema.steps):
        if (
            step.phase == EventPhase.UPDATE
            and step.actor_id != step.learner_id
            and step.learner_id == "subject"
        ):
            social_steps.append((i, step.observable_fields))

    if not social_steps:
        raise ValueError(
            f"Kernel {spec.model_id!r} requires social information "
            f"(requires_social=True), but schema {schema.schema_id!r} "
            f"has no social UPDATE steps directed at the subject."
        )

    if spec.required_social_fields:
        for step_index, step_fields in social_steps:
            missing = spec.required_social_fields - step_fields
            if missing:
                raise ValueError(
                    f"Kernel {spec.model_id!r} requires social fields "
                    f"{sorted(spec.required_social_fields)}, but schema "
                    f"{schema.schema_id!r} step {step_index} only provides "
                    f"{sorted(step_fields)}. Missing: {sorted(missing)}."
                )
