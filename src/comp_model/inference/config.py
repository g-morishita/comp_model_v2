"""Shared inference configuration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from comp_model.inference.mle.optimize import MleOptimizerConfig

if TYPE_CHECKING:
    from comp_model.inference.bayes.stan import StanFitConfig


class HierarchyStructure(StrEnum):
    """Supported pooling structures for inference backends."""

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
    """

    hierarchy: HierarchyStructure
    backend: str = "stan"
    sampler: str = "nuts"
    mle_config: MleOptimizerConfig = field(default_factory=MleOptimizerConfig)
    stan_config: StanFitConfig | None = None
