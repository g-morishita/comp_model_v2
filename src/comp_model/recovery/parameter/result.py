"""Structured result types for parameter recovery studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np  # noqa: TC002 - used at runtime in field types

if TYPE_CHECKING:
    from comp_model.recovery.parameter.config import ParameterRecoveryConfig


@dataclass(frozen=True, slots=True)
class SubjectRecord:
    """One (subject, parameter, [condition]) recovery data point.

    Attributes
    ----------
    subject_id
        Subject identifier.
    param_name
        Name of the recovered parameter.
    condition
        Condition label, or ``None`` for non-condition-aware fits.
    true_value
        Ground-truth parameter value used for simulation.
    estimated_value
        Point estimate recovered by the fitting procedure.
    posterior_draws
        Full posterior draws for Bayesian fits, or ``None`` for MLE.
    """

    subject_id: str
    param_name: str
    condition: str | None
    true_value: float
    estimated_value: float
    posterior_draws: np.ndarray | None


@dataclass(frozen=True, slots=True)
class PopulationRecord:
    """One population parameter's recovery data for one replication.

    Attributes
    ----------
    param_name
        Population parameter name (e.g. ``alpha_pop``, ``mu_alpha_z``).
    true_value
        Ground-truth population parameter value.
    estimated_value
        Posterior mean of the population parameter.
    posterior_draws
        Full posterior draws, or ``None`` if unavailable.
    """

    param_name: str
    true_value: float
    estimated_value: float
    posterior_draws: np.ndarray | None


@dataclass(frozen=True, slots=True)
class SubjectLevelResult:
    """Subject-level recovery data for one replication.

    Attributes
    ----------
    records
        Tuple of subject-level recovery records.
    """

    records: tuple[SubjectRecord, ...]


@dataclass(frozen=True, slots=True)
class PopulationLevelResult:
    """Population-level recovery data for one replication.

    Attributes
    ----------
    records
        Tuple of population-level recovery records.
    """

    records: tuple[PopulationRecord, ...]


@dataclass(frozen=True, slots=True)
class ReplicationResult:
    """All recovery data from one simulate-fit cycle.

    Attributes
    ----------
    replication_index
        Zero-based index of this replication.
    subject_level
        Subject-level recovery records.
    population_level
        Population-level recovery records, or ``None`` for MLE fits.
    """

    replication_index: int
    subject_level: SubjectLevelResult
    population_level: PopulationLevelResult | None


@dataclass(frozen=True, slots=True)
class ParameterRecoveryResult:
    """Complete results from a parameter recovery study.

    Attributes
    ----------
    config
        Study configuration used for the recovery.
    replications
        Results from each simulate-fit replication.
    """

    config: ParameterRecoveryConfig
    replications: tuple[ReplicationResult, ...]
