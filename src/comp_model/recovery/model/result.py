"""Result dataclasses for model recovery studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comp_model.recovery.model.config import ModelRecoveryConfig


@dataclass(frozen=True, slots=True)
class ReplicationResult:
    """Outcome of one simulate-then-fit cycle for a single generating model.

    Attributes
    ----------
    replication_index
        Zero-based index of this replication within the study.
    generating_model
        Name of the generating model that produced the simulated data.
    candidate_scores
        Score for each candidate model (higher = better for all criteria).
    selected_model
        Name of the candidate model that achieved the highest score.
    winner_score
        Score of the selected model.
    second_best_model
        Name of the runner-up candidate, or ``None`` if only one candidate.
    delta_to_second
        ``winner_score - second_score``, or ``None`` if only one candidate.
    """

    replication_index: int
    generating_model: str
    candidate_scores: dict[str, float]
    selected_model: str
    winner_score: float
    second_best_model: str | None
    delta_to_second: float | None


@dataclass(frozen=True, slots=True)
class ModelRecoveryResult:
    """Complete results from a model recovery study.

    Attributes
    ----------
    config
        Study configuration used.
    replications
        Results from all replications across all generating models,
        in ``(generating_model, replication_index)`` order.
    """

    config: ModelRecoveryConfig
    replications: tuple[ReplicationResult, ...]
