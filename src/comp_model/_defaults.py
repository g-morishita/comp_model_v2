"""Shared default factories used across immutable data structures."""

from __future__ import annotations

from typing import Any


def empty_mapping() -> dict[str, Any]:
    """Create an empty mapping with explicit typing.

    Returns
    -------
    dict[str, Any]
        Empty payload or metadata mapping.
    """

    return {}
