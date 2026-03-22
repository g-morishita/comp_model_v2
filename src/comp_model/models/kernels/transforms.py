"""Functions that map model parameters between the optimiser's scale and their natural scale.

When fitting a model, numerical optimisers work most reliably with parameters
that can take any real value (from -infinity to +infinity). But model parameters
often have natural constraints — for example, a learning rate must lie between 0
and 1, and an inverse temperature must be positive.

This module provides a registry of mathematical transforms that convert between
the two scales:

- The *forward* direction takes a raw unconstrained number from the optimiser
  and maps it onto the parameter's natural scale (e.g. squeezes a real number
  into (0, 1) for a learning rate).
- The *inverse* direction takes a value on the natural scale and maps it back
  to an unconstrained number (needed to set starting values for the optimiser).

The same transform definitions are used by both the Python fitting code and the
Stan code generator, so both backends are guaranteed to use identical
transformations.
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
    """Squeeze any real number into the open interval (0, 1).

    Used to transform learning rate parameters (alpha) from the optimiser's
    unconstrained scale onto their natural scale. For example, an optimiser
    value of 0 maps to 0.5; large positive values approach 1; large negative
    values approach 0.

    Parameters
    ----------
    value
        Any real number (the optimiser's internal representation of the
        parameter).

    Returns
    -------
    float
        The corresponding learning rate in (0, 1), never exactly 0 or 1.
    """

    return float(special.expit(value))


def _inverse_sigmoid(value: float) -> float:
    """Convert a probability back to an unconstrained real number.

    This is the reverse of ``_sigmoid``. Used to convert a human-readable
    parameter value (e.g. a starting learning rate of 0.3) into the
    unconstrained number that the optimiser expects as its starting point.
    Values exactly at 0 or 1 are nudged slightly inward to keep the output
    finite.

    Parameters
    ----------
    value
        A probability in (0, 1), such as a learning rate.

    Returns
    -------
    float
        The corresponding unconstrained real number.
    """

    clamped_value = min(max(value, _PROBABILITY_LOWER_BOUND), _PROBABILITY_UPPER_BOUND)
    return float(special.logit(clamped_value))


def _softplus(value: float) -> float:
    """Map any real number to a strictly positive value.

    Used to transform the inverse temperature parameter (beta) from the
    optimiser's unconstrained scale to its natural scale. The softplus function
    is a smooth, always-positive alternative to simply exponentiating the value;
    it grows linearly for large inputs (avoiding overflow) and approaches zero
    for very negative inputs (but never reaches it).

    Parameters
    ----------
    value
        Any real number (the optimiser's internal representation of beta).

    Returns
    -------
    float
        A strictly positive number representing the inverse temperature.
    """

    if value > 20.0:
        return value
    return math.log1p(math.exp(value))


def _inverse_softplus(value: float) -> float:
    """Convert a positive parameter value back to an unconstrained real number.

    This is the reverse of ``_softplus``. Used to convert a human-readable
    beta value (e.g. a starting inverse temperature of 2.0) into the
    unconstrained number the optimiser expects. Values at or below zero are
    clamped to a tiny positive number to keep the output finite.

    Parameters
    ----------
    value
        A positive number, such as an inverse temperature value.

    Returns
    -------
    float
        The corresponding unconstrained real number.
    """

    clamped_value = max(value, _POSITIVE_LOWER_BOUND)
    if clamped_value < _SOFTPLUS_STABLE_SWITCH:
        return math.log(math.expm1(clamped_value))
    return clamped_value + math.log1p(-math.exp(-clamped_value))


@dataclass(frozen=True, slots=True)
class Transform:
    """A paired forward-and-inverse parameter transformation.

    Bundles the two directions of a transformation together with the equivalent
    Stan expression so that Python fitting code and Stan-based Bayesian fitting
    always use the identical mathematical operation.

    Attributes
    ----------
    forward
        Function that maps an unconstrained optimiser value to the parameter's
        natural scale (e.g. real number to learning rate in (0, 1)).
    inverse
        Function that maps a value on the natural scale back to the
        unconstrained scale (e.g. learning rate to a real number). Used when
        setting optimiser starting points from interpretable parameter values.
    stan_expression
        The equivalent Stan code snippet, written as a template with ``{x}``
        as a placeholder for the unconstrained variable name. Keeps the Stan
        model consistent with the Python model without requiring Stan to call
        back into Python.
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
    """Retrieve a named transform from the registry by its identifier.

    Centralising lookups here means both the Python fitting code and the Stan
    code generator always retrieve the same transform object, preventing the
    two backends from silently diverging.

    Available transforms:

    - ``"sigmoid"``: maps any real number to (0, 1). Used for learning rates.
    - ``"softplus"``: maps any real number to (0, +inf). Used for inverse
      temperature (beta).
    - ``"exp"``: maps any real number to (0, +inf) via exponentiation.
    - ``"identity"``: no transformation — the parameter is already on the
      correct scale.

    Parameters
    ----------
    transform_id
        The short name of the desired transform (e.g. ``"sigmoid"``).

    Returns
    -------
    Transform
        The corresponding ``Transform`` object with forward and inverse
        functions and a Stan expression.

    Raises
    ------
    ValueError
        Raised with a helpful message when the requested name is not in the
        registry.
    """

    if transform_id not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform {transform_id!r}. Available: {sorted(TRANSFORM_REGISTRY)}"
        )
    return TRANSFORM_REGISTRY[transform_id]
