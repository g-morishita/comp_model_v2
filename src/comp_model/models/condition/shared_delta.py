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
            Shared keys followed by non-baseline delta keys.
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
            Zero-valued unconstrained parameter dictionary.
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
            Reconstructed unconstrained parameter dictionaries per condition.
        """

        return {condition: self.reconstruct(raw, condition) for condition in self.conditions}
