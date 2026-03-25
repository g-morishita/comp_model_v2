"""Scatter-plot and coverage visualisation for parameter recovery results."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportMissingImports=false

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from comp_model.recovery.parameter.result import ParameterRecoveryResult


def plot_subject_scatter(
    result: ParameterRecoveryResult,
    params: list[str] | None = None,
    save_path: Path | None = None,
) -> Any:
    """Plot true vs estimated scatter for subject-level recovery.

    One subplot per parameter.  An identity line and Pearson *r* annotation
    are shown in each panel.

    Parameters
    ----------
    result
        Completed parameter recovery result.
    params
        Parameter names to include.  ``None`` includes all parameters found
        in the subject-level records.
    save_path
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure with one scatter panel per parameter.
    """
    import matplotlib.pyplot as plt

    pairs: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for replication in result.replications:
        for record in replication.subject_level.records:
            key = (
                f"{record.param_name}__{record.condition}"
                if record.condition
                else record.param_name
            )
            if params is not None and key not in params:
                continue
            pairs[key][0].append(record.true_value)
            pairs[key][1].append(record.estimated_value)

    param_keys = list(pairs)
    n_params = len(param_keys)
    n_cols = min(n_params, 3)
    n_rows = math.ceil(n_params / n_cols) if n_cols else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for idx, key in enumerate(param_keys):
        ax = axes[idx // n_cols][idx % n_cols]
        true_arr = np.array(pairs[key][0])
        est_arr = np.array(pairs[key][1])

        ax.scatter(true_arr, est_arr, alpha=0.3, s=8, edgecolors="none")

        lo = min(true_arr.min(), est_arr.min())
        hi = max(true_arr.max(), est_arr.max())
        margin = (hi - lo) * 0.05
        ax.plot(
            [lo - margin, hi + margin], [lo - margin, hi + margin], "k--", linewidth=0.8, alpha=0.5
        )
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)

        if len(true_arr) > 1 and np.std(true_arr) > 0 and np.std(est_arr) > 0:
            r = float(np.corrcoef(true_arr, est_arr)[0, 1])
            ax.set_title(f"{key}\nr = {r:.3f}", fontsize=10)
        else:
            ax.set_title(key, fontsize=10)

        ax.set_xlabel("True")
        ax.set_ylabel("Estimated")
        ax.set_aspect("equal")

    # Hide unused axes
    for idx in range(n_params, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_population_scatter(
    result: ParameterRecoveryResult,
    params: list[str] | None = None,
    save_path: Path | None = None,
) -> Any:
    """Plot true vs estimated scatter for population-level recovery.

    One subplot per population parameter.  Each point represents one
    replication.

    Parameters
    ----------
    result
        Completed parameter recovery result.
    params
        Population parameter names to include.  ``None`` includes all.
    save_path
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure with one scatter panel per population parameter.
    """
    import matplotlib.pyplot as plt

    pairs: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for replication in result.replications:
        if replication.population_level is None:
            continue
        for record in replication.population_level.records:
            if params is not None and record.param_name not in params:
                continue
            pairs[record.param_name][0].append(record.true_value)
            pairs[record.param_name][1].append(record.estimated_value)

    param_keys = list(pairs)
    n_params = len(param_keys)
    if n_params == 0:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.text(0.5, 0.5, "No population-level data", ha="center", va="center")
        return fig

    n_cols = min(n_params, 3)
    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for idx, key in enumerate(param_keys):
        ax = axes[idx // n_cols][idx % n_cols]
        true_arr = np.array(pairs[key][0])
        est_arr = np.array(pairs[key][1])

        ax.scatter(true_arr, est_arr, alpha=0.7, s=20)

        lo = min(true_arr.min(), est_arr.min())
        hi = max(true_arr.max(), est_arr.max())
        margin = (hi - lo) * 0.05 if hi > lo else 0.1
        ax.plot(
            [lo - margin, hi + margin], [lo - margin, hi + margin], "k--", linewidth=0.8, alpha=0.5
        )

        if len(true_arr) > 1 and np.std(true_arr) > 0 and np.std(est_arr) > 0:
            r = float(np.corrcoef(true_arr, est_arr)[0, 1])
            ax.set_title(f"{key}\nr = {r:.3f} (N={len(true_arr)})", fontsize=10)
        else:
            ax.set_title(f"{key} (N={len(true_arr)})", fontsize=10)

        ax.set_xlabel("True")
        ax.set_ylabel("Estimated")

    for idx in range(n_params, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_coverage(
    result: ParameterRecoveryResult,
    save_path: Path | None = None,
) -> Any:
    """Plot coverage calibration bar chart for Bayesian recovery.

    Shows the fraction of true values falling inside the 90% and 95% HDI
    for each subject-level parameter.

    Parameters
    ----------
    result
        Completed parameter recovery result with posterior draws.
    save_path
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure with grouped bars for 90% and 95% coverage.
    """
    import matplotlib.pyplot as plt

    from comp_model.recovery.parameter.metrics import compute_parameter_recovery_metrics

    metrics = compute_parameter_recovery_metrics(result)

    param_names: list[str] = []
    cov90_vals: list[float] = []
    cov95_vals: list[float] = []

    for name, m in metrics.per_parameter.items():
        if m.coverage_90 is not None and m.coverage_95 is not None:
            param_names.append(name)
            cov90_vals.append(m.coverage_90)
            cov95_vals.append(m.coverage_95)

    if not param_names:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No coverage data (MLE?)", ha="center", va="center")
        return fig

    x = np.arange(len(param_names))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(max(6, len(param_names) * 1.2), 5))
    ax.bar(x - width / 2, cov90_vals, width, label="90% HDI", color="steelblue")
    ax.bar(x + width / 2, cov95_vals, width, label="95% HDI", color="coral")

    ax.axhline(0.90, color="steelblue", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(0.95, color="coral", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("HDI Coverage Calibration")
    ax.legend()

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
