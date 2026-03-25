"""CSV export for model recovery results."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from comp_model.recovery.model.result import ModelRecoveryResult


def save_replication_csv(result: ModelRecoveryResult, path: Path) -> None:
    """Write per-replication model recovery data to a CSV file.

    Each row contains one replication with the generating model, selected
    (winning) model, and whether the selection was correct.

    Parameters
    ----------
    result
        Completed model recovery result.
    path
        Output file path.  Parent directories are created if needed.

    Notes
    -----
    Columns: ``replication``, ``generating_model``, ``selected_model``,
    ``correct``, ``winner_score``, ``delta_to_second``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "replication",
                "generating_model",
                "selected_model",
                "correct",
                "winner_score",
                "delta_to_second",
            ]
        )
        for rep in result.replications:
            writer.writerow(
                [
                    rep.replication_index,
                    rep.generating_model,
                    rep.selected_model,
                    int(rep.selected_model == rep.generating_model),
                    f"{rep.winner_score:.6f}",
                    f"{rep.delta_to_second:.6f}" if rep.delta_to_second is not None else "",
                ]
            )


def save_confusion_matrix_csv(result: ModelRecoveryResult, path: Path) -> None:
    """Write the confusion matrix to a CSV file.

    Rows represent the generating model; columns represent the selected
    (winning) model.  Cell values are counts.

    Parameters
    ----------
    result
        Completed model recovery result.
    path
        Output file path.  Parent directories are created if needed.

    Notes
    -----
    The first column is ``generating_model``; subsequent columns are named
    after each candidate model.
    """
    from comp_model.recovery.model.analysis import compute_confusion_matrix

    matrix = compute_confusion_matrix(result)
    cand_names = [spec.name for spec in result.config.candidate_models]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generating_model", *cand_names])
        for gen_name in matrix:
            writer.writerow([gen_name, *(matrix[gen_name][c] for c in cand_names)])
