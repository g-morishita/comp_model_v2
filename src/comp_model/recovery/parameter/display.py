"""Formatted output for recovery analysis results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comp_model.recovery.parameter.metrics import ParameterRecoveryMetricsTable
    from comp_model.recovery.parameter.result import ParameterRecoveryResult


def parameter_recovery_table(metrics: ParameterRecoveryMetricsTable) -> str:
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


def parameter_recovery_summary(result: ParameterRecoveryResult) -> str:
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
    if not first.subject_level.records:
        return "No subject estimates."

    # Derive display keys from the first replication's records
    seen_keys: list[str] = []
    seen_set: set[str] = set()
    for record in first.subject_level.records:
        key = f"{record.param_name}__{record.condition}" if record.condition else record.param_name
        if key not in seen_set:
            seen_keys.append(key)
            seen_set.add(key)

    header_parts = [f"{'Rep':>4}", f"{'Subject':<10}"]
    for name in sorted(seen_keys):
        header_parts.append(f"{'True ' + name:>15}")
        header_parts.append(f"{'Est ' + name:>15}")
    header = " ".join(header_parts)

    lines = [header, "-" * len(header)]

    for replication in result.replications:
        # Group records by subject
        subject_records: dict[str, dict[str, tuple[float, float]]] = {}
        for record in replication.subject_level.records:
            key = (
                f"{record.param_name}__{record.condition}"
                if record.condition
                else record.param_name
            )
            subject_records.setdefault(record.subject_id, {})[key] = (
                record.true_value,
                record.estimated_value,
            )

        for sid, param_map in subject_records.items():
            parts = [f"{replication.replication_index:>4}", f"{sid:<10}"]
            for name in sorted(seen_keys):
                true_val, est_val = param_map.get(name, (float("nan"), float("nan")))
                parts.append(f"{true_val:>15.4f}")
                parts.append(f"{est_val:>15.4f}")
            lines.append(" ".join(parts))

    return "\n".join(lines)
