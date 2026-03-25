"""Formatted output for recovery analysis results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comp_model.recovery.parameter.metrics import RecoveryMetrics
    from comp_model.recovery.parameter.runner import RecoveryResult


def recovery_table(metrics: RecoveryMetrics) -> str:
    """Format recovery metrics as a human-readable table.

    Parameters
    ----------
    metrics
        Computed recovery metrics.

    Returns
    -------
    str
        Formatted table string.
    """

    has_coverage = any(m.coverage_95 is not None for m in metrics.per_parameter.values())

    header = f"{'Parameter':<25} {'r':>6} {'RMSE':>8} {'Bias':>8} {'MAE':>8}"
    if has_coverage:
        header += f" {'Cov90':>7} {'Cov95':>7}"
    header += f" {'N':>6}"

    lines = [header, "-" * len(header)]

    for name, m in metrics.per_parameter.items():
        mae = m.mean_absolute_error
        line = f"{name:<25} {m.correlation:>6.3f} {m.rmse:>8.4f} {m.bias:>8.4f} {mae:>8.4f}"
        if has_coverage:
            cov90 = f"{m.coverage_90:.3f}" if m.coverage_90 is not None else "   N/A"
            cov95 = f"{m.coverage_95:.3f}" if m.coverage_95 is not None else "   N/A"
            line += f" {cov90:>7} {cov95:>7}"
        line += f" {m.n_observations:>6}"
        lines.append(line)

    return "\n".join(lines)


def recovery_summary(result: RecoveryResult) -> str:
    """Format per-subject true vs estimated values across replications.

    Parameters
    ----------
    result
        Completed recovery study result.

    Returns
    -------
    str
        Formatted summary table string.
    """

    if not result.replications:
        return "No replications to summarize."

    first = result.replications[0]
    if not first.subject_estimates:
        return "No subject estimates."

    param_names = sorted(first.true_params[first.subject_estimates[0].subject_id])

    header_parts = [f"{'Rep':>4}", f"{'Subject':<10}"]
    for name in param_names:
        header_parts.append(f"{'True ' + name:>15}")
        header_parts.append(f"{'Est ' + name:>15}")
    header = " ".join(header_parts)

    lines = [header, "-" * len(header)]

    for replication in result.replications:
        for subject_est in replication.subject_estimates:
            sid = subject_est.subject_id
            true_p = replication.true_params[sid]
            parts = [f"{replication.replication_index:>4}", f"{sid:<10}"]
            for name in param_names:
                true_val = true_p.get(name, float("nan"))
                est_val = subject_est.point_estimates.get(name, float("nan"))
                parts.append(f"{true_val:>15.4f}")
                parts.append(f"{est_val:>15.4f}")
            lines.append(" ".join(parts))

    return "\n".join(lines)
