"""CSV export for parameter recovery results."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from comp_model.recovery.parameter.result import ParameterRecoveryResult


def save_subject_csv(result: ParameterRecoveryResult, path: Path) -> None:
    """Write subject-level recovery data to a CSV file.

    Each row contains one (replication, subject, parameter, condition) data
    point with the true and estimated values.

    Parameters
    ----------
    result
        Completed parameter recovery result.
    path
        Output file path.  Parent directories are created if needed.

    Notes
    -----
    Columns: ``replication``, ``subject_id``, ``param_name``, ``condition``,
    ``true_value``, ``estimated_value``.  The ``condition`` column is empty
    for non-condition-aware fits.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "replication",
                "subject_id",
                "param_name",
                "condition",
                "true_value",
                "estimated_value",
            ]
        )
        for replication in result.replications:
            for record in replication.subject_level.records:
                writer.writerow(
                    [
                        replication.replication_index,
                        record.subject_id,
                        record.param_name,
                        record.condition or "",
                        f"{record.true_value:.6f}",
                        f"{record.estimated_value:.6f}",
                    ]
                )


def save_population_csv(result: ParameterRecoveryResult, path: Path) -> None:
    """Write population-level recovery data to a CSV file.

    Each row contains one (replication, parameter) data point with the
    true and estimated population-level values.

    Parameters
    ----------
    result
        Completed parameter recovery result.
    path
        Output file path.  Parent directories are created if needed.

    Notes
    -----
    Columns: ``replication``, ``param_name``, ``true_value``,
    ``estimated_value``.  Only replications with population-level results
    are included.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "replication",
                "param_name",
                "true_value",
                "estimated_value",
            ]
        )
        for replication in result.replications:
            if replication.population_level is None:
                continue
            for record in replication.population_level.records:
                writer.writerow(
                    [
                        replication.replication_index,
                        record.param_name,
                        f"{record.true_value:.6f}",
                        f"{record.estimated_value:.6f}",
                    ]
                )
