"""Parameter transform registry shared across inference backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


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
    """

    if value >= 0.0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


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
    """

    return math.log(value / (1.0 - value))


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
    """

    return math.log(math.expm1(value))


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
    """

    if transform_id not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform {transform_id!r}. Available: {sorted(TRANSFORM_REGISTRY)}"
        )
    return TRANSFORM_REGISTRY[transform_id]
