"""Shared probability utilities for model kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def stable_softmax(logits: Sequence[float]) -> tuple[float, ...]:
    """Compute a numerically stable softmax over one-dimensional logits.

    Parameters
    ----------
    logits
        One-dimensional sequence of logit values.

    Returns
    -------
    tuple[float, ...]
        Softmax probabilities summing to one.

    Raises
    ------
    ValueError
        Raised when ``logits`` is empty.
    """

    logits_array = np.asarray(tuple(logits), dtype=float)
    if logits_array.size == 0:
        raise ValueError("stable_softmax requires at least one logit")

    centered_logits = logits_array - np.max(logits_array)
    exp_logits = np.exp(centered_logits)
    probabilities = exp_logits / exp_logits.sum()
    probabilities = np.clip(probabilities, 1e-15, None)
    probabilities /= probabilities.sum()
    return tuple(float(value) for value in probabilities)
