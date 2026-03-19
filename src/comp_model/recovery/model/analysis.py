"""Analysis utilities for model recovery results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comp_model.recovery.model.runner import ModelRecoveryResult


def confusion_matrix(result: ModelRecoveryResult) -> dict[str, dict[str, int]]:
    """Build a confusion matrix from model recovery results.

    Parameters
    ----------
    result
        Completed model recovery study.

    Returns
    -------
    dict[str, dict[str, int]]
        Nested mapping ``{generating_model: {selected_model: count}}``.
        All generating and candidate model names appear as keys, with zero
        counts for unobserved combinations.
    """

    gen_names = [spec.name for spec in result.config.generating_models]
    cand_names = [spec.name for spec in result.config.candidate_models]

    matrix: dict[str, dict[str, int]] = {g: {c: 0 for c in cand_names} for g in gen_names}

    for rep in result.replications:
        gen = rep.generating_model
        sel = rep.selected_model
        if gen in matrix and sel in matrix[gen]:
            matrix[gen][sel] += 1

    return matrix


def recovery_rates(result: ModelRecoveryResult) -> dict[str, float]:
    """Compute per-generating-model recovery rates.

    Parameters
    ----------
    result
        Completed model recovery study.

    Returns
    -------
    dict[str, float]
        Mapping from generating model name to the fraction of replications
        in which the correct model was selected.  Returns ``float("nan")``
        for any generating model with zero replications.
    """

    counts: dict[str, int] = {spec.name: 0 for spec in result.config.generating_models}
    totals: dict[str, int] = {spec.name: 0 for spec in result.config.generating_models}

    for rep in result.replications:
        gen = rep.generating_model
        if gen not in totals:
            continue
        totals[gen] += 1
        if rep.selected_model == gen:
            counts[gen] += 1

    return {
        name: counts[name] / totals[name] if totals[name] > 0 else float("nan") for name in totals
    }
