"""Shared-plus-delta parameter layout for within-subject conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comp_model.models.kernels.base import ModelKernelSpec


@dataclass(frozen=True, slots=True)
class SharedDeltaLayout:
    """Shared-plus-delta unconstrained parameter layout across conditions.

    Attributes
    ----------
    kernel_spec
        Kernel specification whose parameters are being laid out.
    conditions
        Ordered condition labels included in the layout.
    baseline_condition
        Condition that receives only shared parameters and no delta term.

    Notes
    -----
    This layout only changes parameter values across conditions. It does not reset
    latent kernel state on condition changes. When a kernel uses
    ``state_reset_policy="continuous"``, state learned in one condition carries into
    the next. Use ``state_reset_policy="per_block"`` (the default) to start each
    block with a blank slate.

    Shared parameters are represented once, and each non-baseline condition gets
    an additive delta on the unconstrained scale.
    """

    kernel_spec: ModelKernelSpec
    conditions: tuple[str, ...]
    baseline_condition: str

    def __post_init__(self) -> None:
        """Validate baseline membership and minimum condition count.

        Returns
        -------
        None
            This method raises on invalid layout definitions.
        """

        if self.baseline_condition not in self.conditions:
            raise ValueError(
                f"baseline_condition {self.baseline_condition!r} must be one of {self.conditions}"
            )
        if len(self.conditions) < 2:
            raise ValueError("SharedDeltaLayout requires at least 2 conditions")

    def parameter_keys(self) -> tuple[str, ...]:
        """Return the unconstrained parameter keys in layout order.

        Returns
        -------
        tuple[str, ...]
            Shared keys followed by non-baseline delta keys, both in kernel
            parameter order.
        """

        keys: list[str] = []
        for parameter in self.kernel_spec.parameter_specs:
            keys.append(f"{parameter.name}__shared_z")
        for condition in self.conditions:
            if condition == self.baseline_condition:
                continue
            for parameter in self.kernel_spec.parameter_specs:
                keys.append(f"{parameter.name}__delta_z__{condition}")
        return tuple(keys)

    def default_params_z(self) -> dict[str, float]:
        """Return zero initialization across all unconstrained layout keys.

        Returns
        -------
        dict[str, float]
            Zero-valued unconstrained parameter dictionary. A zero vector means
            baseline and non-baseline conditions initially coincide.
        """

        return {key: 0.0 for key in self.parameter_keys()}

    def n_params(self) -> int:
        """Return the number of unconstrained parameters in the layout.

        Returns
        -------
        int
            Number of shared and delta parameter keys.
        """

        return len(self.parameter_keys())

    def reconstruct(self, raw: dict[str, float], condition: str) -> dict[str, float]:
        """Reconstruct kernel-ready unconstrained parameters for one condition.

        Parameters
        ----------
        raw
            Unconstrained layout parameters keyed by layout-specific names.
        condition
            Condition label to reconstruct.

        Returns
        -------
        dict[str, float]
            Unconstrained kernel parameters keyed by parameter name.

        Notes
        -----
        For the baseline condition, reconstruction returns only the shared
        terms. For any other condition, reconstruction adds that condition's
        delta term to each shared parameter on the unconstrained scale.
        """

        if condition not in self.conditions:
            raise ValueError(f"Unknown condition {condition!r}")

        reconstructed: dict[str, float] = {}
        for parameter in self.kernel_spec.parameter_specs:
            value = raw[f"{parameter.name}__shared_z"]
            if condition != self.baseline_condition:
                value += raw[f"{parameter.name}__delta_z__{condition}"]
            reconstructed[parameter.name] = value
        return reconstructed

    def reconstruct_all(self, raw: dict[str, float]) -> dict[str, dict[str, float]]:
        """Reconstruct unconstrained parameters for all conditions.

        Parameters
        ----------
        raw
            Unconstrained layout parameters keyed by layout-specific names.

        Returns
        -------
        dict[str, dict[str, float]]
            Reconstructed unconstrained parameter dictionaries for every
            condition in layout order.
        """

        return {condition: self.reconstruct(raw, condition) for condition in self.conditions}
