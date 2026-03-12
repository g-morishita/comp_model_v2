"""Tests for parameter transforms."""

import math

import pytest

from comp_model.models.kernels.transforms import TRANSFORM_REGISTRY, get_transform


@pytest.mark.parametrize(
    ("transform_id", "value"),
    (
        ("sigmoid", 0.37),
        ("exp", 1.8),
        ("softplus", 1.2),
        ("identity", -0.5),
    ),
)
def test_transform_inverse_round_trip(transform_id: str, value: float) -> None:
    """Ensure each transform round-trips through inverse and forward.

    Parameters
    ----------
    transform_id
        Name of the transform under test.
    value
        Value in the transform's constrained domain.

    Returns
    -------
    None
        This test asserts inverse/forward consistency.
    """

    transform = get_transform(transform_id)

    recovered = transform.forward(transform.inverse(value))

    assert math.isclose(recovered, value, rel_tol=1e-9, abs_tol=1e-9)


def test_get_transform_rejects_unknown_identifier() -> None:
    """Ensure lookup errors are descriptive for unknown transforms.

    Returns
    -------
    None
        This test raises on an invalid transform identifier.
    """

    with pytest.raises(ValueError, match="Unknown transform"):
        get_transform("missing")


def test_transform_registry_contains_expected_entries() -> None:
    """Ensure the transform registry exposes the planned transform set.

    Returns
    -------
    None
        This test asserts registry membership only.
    """

    assert set(TRANSFORM_REGISTRY) == {"sigmoid", "exp", "softplus", "identity"}
