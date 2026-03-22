"""Shared probability utilities used by all model kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def stable_softmax(logits: Sequence[float]) -> tuple[float, ...]:
    """Convert a list of raw scores into a proper probability distribution.

    The softmax function takes one score per option and returns one probability
    per option such that all probabilities are positive and sum to exactly 1.
    An option with a higher score gets a higher probability. The transformation
    is non-linear: doubling a score does not double its probability.

    "Stable" refers to two numerical precautions that prevent computational
    problems:

    1. Before exponentiating, the largest score is subtracted from all scores.
       This does not change the resulting probabilities (the ratio cancels) but
       prevents the exponential from producing infinitely large numbers when
       scores are large.

    2. After computing the probabilities, any value that rounds down to
       exactly 0 is replaced with a tiny positive number (1e-15). This matters
       because model fitting requires computing the logarithm of each choice
       probability, and log(0) is negative infinity, which breaks optimisation.

    Parameters
    ----------
    logits
        The raw scores, one per option. In this codebase these are Q-values
        multiplied by the inverse temperature (beta * Q[action]).

    Returns
    -------
    tuple[float, ...]
        Probabilities in the same order as the input scores, all positive and
        summing to 1.

    Raises
    ------
    ValueError
        Raised when the input list is empty.
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
