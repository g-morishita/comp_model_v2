"""Shared inference configuration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from comp_model.inference.mle.optimize import MleOptimizerConfig

if TYPE_CHECKING:
    from collections.abc import Mapping

    from comp_model.inference.bayes.stan import StanFitConfig


@dataclass(frozen=True, slots=True)
class PriorSpec:
    """Prior metadata for a model parameter.

    Attributes
    ----------
    family
        Prior family identifier, such as ``"normal"``.
    kwargs
        Hyperparameters for the prior family.
    parameterization
        Scale on which the prior is defined.
    """

    family: str
    kwargs: Mapping[str, float]
    parameterization: str = "unconstrained"


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
    prior_specs
        Optional mapping from parameter name to prior specification.
        Only used by the Bayesian backend. Parameters without an entry
        fall back to ``Normal(0, 2)`` on the unconstrained scale.

        Condition-delta mean priors for conditioned study hierarchies can
        be configured with the ``_delta`` suffix (e.g.,
        ``"alpha_delta"``). When omitted, these priors default to
        ``Normal(0, 1)`` on the unconstrained scale so the existing
        zero-centered delta regularisation is preserved.

        SD priors for hierarchical models can be configured with the
        ``sd_`` prefix (e.g., ``"sd_alpha"``).  When omitted, SD priors
        default to ``Normal(0, 1)``.  For condition-hierarchy delta SD
        priors, use the ``_delta`` suffix (e.g., ``"sd_alpha_delta"``);
        these fall back to the base SD prior, then to ``Normal(0, 1)``.

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
    prior_specs: dict[str, PriorSpec] | None = None
