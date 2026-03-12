"""Bayesian fit result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from comp_model.inference.config import HierarchyStructure


@dataclass(frozen=True, slots=True)
class BayesFitResult:
    """Posterior samples and diagnostics returned by a Bayesian fit.

    Attributes
    ----------
    model_id
        Identifier of the fitted model.
    hierarchy
        Hierarchy structure used for the fit.
    posterior_samples
        Posterior draws keyed by parameter name.
    log_lik
        Trialwise log-likelihood draws.
    subject_params
        Optional per-subject posterior draws for hierarchical models.
    diagnostics
        Backend diagnostics and summaries.

    Notes
    -----
    The container is backend-agnostic at the API level, but the current Stan
    backend fills ``posterior_samples`` and ``log_lik`` directly from CmdStanPy
    variables extracted from a completed fit.
    """

    model_id: str
    hierarchy: HierarchyStructure
    posterior_samples: dict[str, np.ndarray]
    log_lik: np.ndarray
    subject_params: dict[str, dict[str, np.ndarray]] | None
    diagnostics: dict[str, Any]
