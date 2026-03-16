"""Prior family registry for Stan code generation.

Maps ``PriorSpec.family`` strings to integer IDs and positional parameter
orderings consumed by the shared ``prior_lpdf`` Stan function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class PriorFamilyDef:
    """Metadata for one prior distribution family.

    Attributes
    ----------
    family_id
        Integer identifier passed to the Stan ``prior_lpdf`` function.
    param_keys
        Ordered mapping from ``PriorSpec.kwargs`` keys to positional
        parameters ``(p1, p2, p3)`` in the Stan function signature.
    """

    family_id: int
    param_keys: tuple[str, ...]


PRIOR_FAMILIES: dict[str, PriorFamilyDef] = {
    "normal": PriorFamilyDef(family_id=1, param_keys=("mu", "sigma")),
    "cauchy": PriorFamilyDef(family_id=2, param_keys=("mu", "sigma")),
    "student_t": PriorFamilyDef(family_id=3, param_keys=("mu", "sigma", "df")),
    "uniform": PriorFamilyDef(family_id=4, param_keys=("lower", "upper")),
}

_MAX_PRIOR_PARAMS = 3


def prior_spec_to_stan_data(
    family: str,
    kwargs: Mapping[str, float],
) -> tuple[int, float, float, float]:
    """Convert a prior specification to Stan data values.

    Parameters
    ----------
    family
        Prior family name (e.g. ``"normal"``).
    kwargs
        Hyperparameters for the prior family.

    Returns
    -------
    tuple[int, float, float, float]
        ``(family_id, p1, p2, p3)`` suitable for Stan data export.

    Raises
    ------
    ValueError
        Raised when the prior family is not registered.
    """

    if family not in PRIOR_FAMILIES:
        raise ValueError(
            f"Unknown prior family {family!r}. "
            f"Available: {sorted(PRIOR_FAMILIES)}"
        )

    definition = PRIOR_FAMILIES[family]
    params = [0.0] * _MAX_PRIOR_PARAMS
    for index, key in enumerate(definition.param_keys):
        params[index] = float(kwargs.get(key, 0.0))

    return (definition.family_id, params[0], params[1], params[2])
