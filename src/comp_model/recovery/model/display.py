"""Formatted text output for model recovery results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comp_model.recovery.model.runner import ModelRecoveryResult


def confusion_matrix_table(
    matrix: dict[str, dict[str, int]],
    model_names: list[str],
) -> str:
    """Format a model recovery confusion matrix as a human-readable table.

    Rows represent the generating model; columns represent the selected
    (winning) model.  Diagonal cells indicate correct recovery.

    Parameters
    ----------
    matrix
        Nested counts from :func:`~comp_model.recovery.model.analysis.confusion_matrix`.
    model_names
        Ordered list of model names used for both row and column headers.

    Returns
    -------
    str
        Formatted table string suitable for printing.
    """

    col_width = max(len(name) for name in model_names) + 2
    label_width = col_width

    gen_label = "Generating \\ Selected"
    header = f"{gen_label:<{label_width}}"
    for name in model_names:
        header += f"{name:>{col_width}}"
    lines = [header, "-" * len(header)]

    for gen in model_names:
        row = f"{gen:<{label_width}}"
        for sel in model_names:
            count = matrix.get(gen, {}).get(sel, 0)
            row += f"{count:>{col_width}}"
        lines.append(row)

    return "\n".join(lines)


def recovery_rate_table(
    rates: dict[str, float],
    result: ModelRecoveryResult,
) -> str:
    """Format per-model recovery rates as a human-readable table.

    Parameters
    ----------
    rates
        Recovery rates from :func:`~comp_model.recovery.model.analysis.recovery_rates`.
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
