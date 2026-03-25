"""Formatted text output for model recovery results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comp_model.recovery.model.result import ModelRecoveryResult


def model_recovery_confusion_table(
    matrix: dict[str, dict[str, int]],
    generating_names: list[str],
    candidate_names: list[str] | None = None,
) -> str:
    """Format a model recovery confusion matrix as a human-readable table.

    Rows represent the generating model; columns represent the selected
    (winning) candidate model.

    Parameters
    ----------
    matrix
        Nested counts from :func:`~comp_model.recovery.model.analysis.compute_confusion_matrix`.
    generating_names
        Ordered list of generating model names used for row headers.
    candidate_names
        Ordered list of candidate model names used for column headers.
        When ``None``, defaults to *generating_names* (the common case where
        both sets are identical).

    Returns
    -------
    str
        Formatted table string suitable for printing.
    """
    if candidate_names is None:
        candidate_names = generating_names

    all_names = generating_names + candidate_names
    col_width = max(len(name) for name in all_names) + 2
    label_width = col_width

    gen_label = "Generating \\ Selected"
    header = f"{gen_label:<{label_width}}"
    for name in candidate_names:
        header += f"{name:>{col_width}}"
    lines = [header, "-" * len(header)]

    for gen in generating_names:
        row = f"{gen:<{label_width}}"
        for sel in candidate_names:
            count = matrix.get(gen, {}).get(sel, 0)
            row += f"{count:>{col_width}}"
        lines.append(row)

    return "\n".join(lines)


def model_recovery_rate_table(
    rates: dict[str, float],
    result: ModelRecoveryResult,
) -> str:
    """Format per-model recovery rates as a human-readable table.

    Parameters
    ----------
    rates
        Recovery rates from :func:`~comp_model.recovery.model.analysis.compute_recovery_rates`.
    result
        The completed model recovery result (used to count replications).

    Returns
    -------
    str
        Formatted table string.
    """

    totals: dict[str, int] = {spec.name: 0 for spec in result.config.generating_models}
    corrects: dict[str, int] = {spec.name: 0 for spec in result.config.generating_models}

    for rep in result.replications:
        gen = rep.generating_model
        if gen in totals:
            totals[gen] += 1
            if rep.selected_model == gen:
                corrects[gen] += 1

    name_width = max(len(spec.name) for spec in result.config.generating_models) + 2
    header = f"{'Model':<{name_width}} {'N':>6} {'Correct':>8} {'Rate':>8}"
    lines = [header, "-" * len(header)]

    for spec in result.config.generating_models:
        name = spec.name
        n = totals[name]
        n_correct = corrects[name]
        rate = rates[name]
        rate_str = f"{rate:.3f}" if rate == rate else "  nan"
        lines.append(f"{name:<{name_width}} {n:>6} {n_correct:>8} {rate_str:>8}")

    return "\n".join(lines)
