"""Bayesian inference backends and results."""

from comp_model.inference.bayes.result import BayesFitResult
from comp_model.inference.bayes.stan import (
    DEFAULT_STAN_FIT_CONFIG,
    StanFitConfig,
    fit_stan,
)

__all__ = (
    "DEFAULT_STAN_FIT_CONFIG",
    "BayesFitResult",
    "StanFitConfig",
    "fit_stan",
)
