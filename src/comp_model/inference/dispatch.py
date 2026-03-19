"""Unified inference dispatch entry point.

The dispatch layer chooses between backend implementations while preserving one
shared kernel/task/data interface.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, cast

from comp_model.data.schema import Dataset, SubjectData
from comp_model.inference.mle.optimize import MleFitResult, fit_mle_conditioned, fit_mle_simple

if TYPE_CHECKING:
    from comp_model.inference.bayes.result import BayesFitResult
    from comp_model.inference.config import InferenceConfig
    from comp_model.models.condition.shared_delta import SharedDeltaLayout
    from comp_model.models.kernels.base import ModelKernel
    from comp_model.tasks.schemas import TrialSchema


def fit(
    config: InferenceConfig,
    kernel: ModelKernel[object, object],
    data: SubjectData | Dataset,
    schema: TrialSchema,
    layout: SharedDeltaLayout | None = None,
    adapter: object | None = None,
) -> MleFitResult | BayesFitResult:
    """Dispatch a fit request to the configured inference backend.

    Parameters
    ----------
    config
        Inference configuration describing the requested backend.
    kernel
        Model kernel being fit.
    data
        Subject-level or dataset-level data.
    schema
        Trial schema used for replay extraction.
    layout
        Optional condition-aware parameter layout.
    adapter
        Optional backend-specific adapter for Stan inference.

    Returns
    -------
    MleFitResult | object
        Fit result for the selected backend.

    Notes
    -----
    Dispatch enforces the current scope of each backend:

    - MLE is single-subject only,
    - conditioned MLE is selected when ``layout`` is provided, and
    - the Stan backend requires an explicit adapter that knows how to build data
      and select a Stan program for the requested hierarchy.
    """

    if config.backend == "mle":
        if not isinstance(data, SubjectData):
            raise ValueError("MLE currently supports single-subject fitting only")
        if layout is not None:
            return fit_mle_conditioned(kernel, layout, data, schema, config.mle_config)
        return fit_mle_simple(kernel, data, schema, config.mle_config)

    if config.backend == "stan":
        if adapter is None:
            raise ValueError("Stan backend requires a StanAdapter")

        stan_backend = cast(
            "Any",
            importlib.import_module("comp_model.inference.bayes.stan.backend"),
        )
        fit_stan = stan_backend.fit_stan

        return fit_stan(
            adapter, data, schema, config.hierarchy, layout, config.stan_config, config.prior_specs
        )

    raise ValueError(f"Unknown backend: {config.backend!r}")
