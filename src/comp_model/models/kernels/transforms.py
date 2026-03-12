"""Parameter transform registry shared across inference backends.

Every free parameter is optimized on an unconstrained scale and mapped into its
support through this registry. The same transform metadata is reused by Python
MLE code and Stan code generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scipy import special

if TYPE_CHECKING:
    from collections.abc import Callable


_PROBABILITY_LOWER_BOUND = math.nextafter(0.0, 1.0)
_PROBABILITY_UPPER_BOUND = math.nextafter(1.0, 0.0)
_POSITIVE_LOWER_BOUND = math.nextafter(0.0, 1.0)
_SOFTPLUS_STABLE_SWITCH = math.log(2.0)


def _sigmoid(value: float) -> float:
    """Map an unconstrained value to the unit interval.

    Parameters
    ----------
    value
        Unconstrained scalar.

    Returns
    -------
    float
        Sigmoid-transformed scalar in `(0, 1)`.

    Notes
    -----
    SciPy's numerically stable logistic implementation is used so the forward
    transform matches the inverse and Stan's ``inv_logit`` behavior closely.
    """

    return float(special.expit(value))


def _inverse_sigmoid(value: float) -> float:
    """Map a probability back to the real line.

    Parameters
    ----------
    value
        Probability in `(0, 1)`.

    Returns
    -------
    float
        Logit-transformed scalar.

    Notes
    -----
    Inputs are clamped away from exactly ``0`` and ``1`` so restart generation
    and inverse-round-trip tests stay finite at closed-domain boundaries.
    """

    clamped_value = min(max(value, _PROBABILITY_LOWER_BOUND), _PROBABILITY_UPPER_BOUND)
    return float(special.logit(clamped_value))


def _softplus(value: float) -> float:
    """Map an unconstrained value to a positive scalar.

    Parameters
    ----------
    value
        Unconstrained scalar.

    Returns
    -------
    float
        Positive softplus-transformed scalar.

    Notes
    -----
    Large positive values bypass ``exp`` and return the input directly, matching
    the asymptotic behavior of softplus while avoiding overflow.
    """

    if value > 20.0:
        return value
    return math.log1p(math.exp(value))


def _inverse_softplus(value: float) -> float:
    """Map a positive scalar back to the real line.

    Parameters
    ----------
    value
        Positive scalar.

    Returns
    -------
    float
        Inverse softplus-transformed scalar.

    Notes
    -----
    Inputs are clamped above zero and evaluated with two stable branches so
    extremely small or moderately large positive values both remain finite.
    """

    clamped_value = max(value, _POSITIVE_LOWER_BOUND)
    if clamped_value < _SOFTPLUS_STABLE_SWITCH:
        return math.log(math.expm1(clamped_value))
    return clamped_value + math.log1p(-math.exp(-clamped_value))


@dataclass(frozen=True, slots=True)
class Transform:
    """Forward and inverse parameter transform metadata.

    Attributes
    ----------
    forward
        Function mapping an unconstrained scalar to the constrained space.
    inverse
        Function mapping a constrained scalar to the unconstrained space.
    stan_expression
        Stan template string using `{x}` as the placeholder variable.

    Notes
    -----
    ``stan_expression`` keeps the Stan backend tied to the same conceptual
    transform as the Python backend without having Stan call back into Python.
    """

    forward: Callable[[float], float]
    inverse: Callable[[float], float]
    stan_expression: str


TRANSFORM_REGISTRY: dict[str, Transform] = {
    "sigmoid": Transform(
        forward=_sigmoid,
        inverse=_inverse_sigmoid,
        stan_expression="inv_logit({x})",
    ),
    "exp": Transform(
        forward=math.exp,
        inverse=math.log,
        stan_expression="exp({x})",
    ),
    "softplus": Transform(
        forward=_softplus,
        inverse=_inverse_softplus,
        stan_expression="log1p_exp({x})",
    ),
    "identity": Transform(
        forward=lambda value: value,
        inverse=lambda value: value,
        stan_expression="{x}",
    ),
}


def get_transform(transform_id: str) -> Transform:
    """Look up a named transform from the registry.

    Parameters
    ----------
    transform_id
        Identifier of the requested transform.

    Returns
    -------
    Transform
        Registered transform metadata.

    Raises
    ------
    ValueError
        Raised when the requested transform is not registered.

    Notes
    -----
    All backends should resolve transforms through this function instead of
    assuming hard-coded transform logic at individual call sites.
    """

    if transform_id not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform {transform_id!r}. Available: {sorted(TRANSFORM_REGISTRY)}"
        )
    return TRANSFORM_REGISTRY[transform_id]
