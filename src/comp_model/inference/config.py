"""Shared inference configuration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from comp_model.inference.mle.optimize import MleOptimizerConfig

if TYPE_CHECKING:
    from comp_model.inference.bayes.stan import StanFitConfig


class HierarchyStructure(StrEnum):
    """Supported pooling structures for inference backends.

    Notes
    -----
    The enum names describe the intended parameter sharing pattern rather than
    the mechanics of any one backend implementation.
    """

    SUBJECT_SHARED = "subject_shared"
    SUBJECT_BLOCK_CONDITION = "subject_block_condition_hierarchy"
    STUDY_SUBJECT = "study_subject_hierarchy"
    STUDY_SUBJECT_BLOCK_CONDITION = "study_subject_block_condition_hierarchy"


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Configuration for dispatching model fits.

    Attributes
    ----------
    hierarchy
        Pooling structure used by the requested backend.
    backend
        Inference backend identifier.
    sampler
        Sampler or optimizer identifier within the backend.
    mle_config
        Configuration for MLE optimization.
    stan_config
        Optional backend-specific Stan configuration.

    Notes
    -----
    ``InferenceConfig`` selects the backend entry point only. Kernel semantics,
    task semantics, and condition layouts are supplied separately so they can be
    shared across MLE and Stan backends.
    """

    hierarchy: HierarchyStructure
    backend: str = "stan"
    sampler: str = "nuts"
    mle_config: MleOptimizerConfig = field(default_factory=MleOptimizerConfig)
    stan_config: StanFitConfig | None = None
