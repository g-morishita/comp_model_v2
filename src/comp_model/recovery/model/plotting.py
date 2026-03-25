"""Visualisation utilities for model recovery results."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportMissingImports=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from comp_model.recovery.model.result import ModelRecoveryResult


def _save_fig(fig: Any, save_path: Path | None) -> None:
    """Save *fig* to *save_path* if provided, creating parent dirs as needed."""
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def _empty_figure(message: str) -> Any:
    """Return a 1x1 figure with a centred text *message*."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    return fig


def plot_confusion_matrix(
    result: ModelRecoveryResult,
    save_path: Path | None = None,
) -> Any:
    """Plot a heatmap of the model recovery confusion matrix.

    Rows represent the generating model; columns represent the selected
    (winning) model.  Cell annotations show both the count and the
    row-normalised proportion.

    Parameters
    ----------
    result
        Completed model recovery result.
    save_path
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure with a single heatmap axes.
    """
    import matplotlib.pyplot as plt

    from comp_model.recovery.model.analysis import compute_confusion_matrix

    matrix = compute_confusion_matrix(result)
    gen_names = [spec.name for spec in result.config.generating_models]
    cand_names = [spec.name for spec in result.config.candidate_models]

    if not gen_names or not cand_names:
        fig = _empty_figure("No model recovery data")
        _save_fig(fig, save_path)
        return fig

    # Build numeric array
    n_gen = len(gen_names)
    n_cand = len(cand_names)
    arr = np.zeros((n_gen, n_cand), dtype=float)
    for i, g in enumerate(gen_names):
        for j, c in enumerate(cand_names):
            arr[i, j] = matrix.get(g, {}).get(c, 0)

    fig, ax = plt.subplots(1, 1, figsize=(max(5, n_cand * 1.5), max(4, n_gen * 1.2)))
    im = ax.imshow(arr, cmap="Blues", aspect="auto")

    ax.set_xticks(np.arange(n_cand))
    ax.set_yticks(np.arange(n_gen))
    ax.set_xticklabels(cand_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(gen_names, fontsize=9)
    ax.set_xlabel("Selected model")
    ax.set_ylabel("Generating model")
    ax.set_title("Model Recovery Confusion Matrix")

    # Annotate cells
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid division by zero
    proportions = arr / row_sums

    thresh = arr.max() / 2.0
    for i in range(n_gen):
        for j in range(n_cand):
            count = int(arr[i, j])
            prop = proportions[i, j]
            colour = "white" if arr[i, j] > thresh else "black"
            ax.text(
                j,
                i,
                f"{count}\n({prop:.2f})",
                ha="center",
                va="center",
                color=colour,
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_recovery_rates(
    result: ModelRecoveryResult,
    save_path: Path | None = None,
) -> Any:
    """Plot per-model recovery rates as a bar chart.

    Each bar represents the fraction of replications in which the correct
    generating model was selected.

    Parameters
    ----------
    result
        Completed model recovery result.
    save_path
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure with a single bar-chart axes.
    """
    import matplotlib.pyplot as plt

    from comp_model.recovery.model.analysis import compute_recovery_rates

    rates = compute_recovery_rates(result)

    if not rates:
        fig = _empty_figure("No recovery rate data")
        _save_fig(fig, save_path)
        return fig

    names = list(rates.keys())
    values = [rates[n] for n in names]
    # Replace NaN with 0 for display
    values_display = [v if v == v else 0.0 for v in values]

    x = np.arange(len(names))

    fig, ax = plt.subplots(1, 1, figsize=(max(5, len(names) * 1.5), 4))
    bars = ax.bar(x, values_display, color="steelblue", edgecolor="white")

    # Annotate bars
    for bar, val, raw_val in zip(bars, values_display, values, strict=True):
        label = f"{val:.2f}" if raw_val == raw_val else "NaN"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Recovery Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Recovery Rates")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig
