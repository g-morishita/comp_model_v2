"""CmdStanPy-backed Bayesian fitting.

The Stan backend compiles the program selected by an adapter, delegates data
construction to the adapter, and translates a completed CmdStanPy fit into the
package's backend-agnostic result container.
"""

from __future__ import annotations

import importlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from comp_model.inference.bayes.result import BayesFitResult

if TYPE_CHECKING:
    from comp_model.data.schema import Dataset, SubjectData
    from comp_model.inference.bayes.stan.adapters.base import StanAdapter
    from comp_model.inference.config import HierarchyStructure, PriorSpec
    from comp_model.models.condition.shared_delta import SharedDeltaLayout
    from comp_model.tasks.schemas import TrialSchema


@dataclass(frozen=True, slots=True)
class StanFitConfig:
    """Configuration for Stan NUTS sampling.

    Attributes
    ----------
    n_warmup
        Number of warmup iterations per chain.
    n_samples
        Number of post-warmup samples per chain.
    n_chains
        Number of sampling chains.
    seed
        Optional random seed.
    adapt_delta
        Stan NUTS target acceptance rate.
    max_treedepth
        Maximum NUTS tree depth.
    show_console
        If ``True`` (default), display CmdStan's raw text progress on
        the console.  Set to ``False`` to suppress it (e.g. in parallel
        runs where output would interleave).
    show_progress
        If ``True``, display a ``tqdm`` progress bar per chain instead
        of (or in addition to) raw console text.  Useful in parallel
        contexts: set ``show_console=False, show_progress=True`` to get
        tqdm bars that handle concurrency without interleaving.
    refresh
        How often (in iterations) CmdStan reports progress.  Lower
        values give more granular updates but add overhead.  ``None``
        uses CmdStan's default.

    Notes
    -----
    These fields are passed through to ``CmdStanModel.sample`` with matching
    names where possible.
    """

    n_warmup: int = 1000
    n_samples: int = 1000
    n_chains: int = 4
    seed: int | None = None
    adapt_delta: float = 0.8
    max_treedepth: int = 10
    show_console: bool = True
    show_progress: bool = False
    refresh: int | None = None


DEFAULT_STAN_FIT_CONFIG = StanFitConfig()


def fit_stan(
    adapter: StanAdapter,
    data: SubjectData | Dataset,
    schema: TrialSchema,
    hierarchy: HierarchyStructure,
    layout: SharedDeltaLayout | None = None,
    config: StanFitConfig | None = None,
    prior_specs: dict[str, PriorSpec] | None = None,
) -> BayesFitResult:
    """Fit a model with Stan using the supplied adapter and data.

    Parameters
    ----------
    adapter
        Stan adapter that provides data and program paths.
    data
        Subject or dataset to fit.
    schema
        Trial schema used for replay extraction.
    hierarchy
        Hierarchy structure targeted by the Stan program.
    layout
        Optional condition-aware parameter layout.
    config
        Optional Stan sampling configuration.

    Returns
    -------
    BayesFitResult
        Posterior samples and diagnostics from the Stan fit.

    Notes
    -----
    The backend imports CmdStanPy lazily so the package can be imported without
    Stan installed. The adapter determines both the Stan program path and the
    exact data dictionary to pass into sampling.
    """

    from comp_model.data.compatibility import check_spec_schema_compatibility

    check_spec_schema_compatibility(adapter.kernel_spec(), schema)

    resolved_config = config if config is not None else DEFAULT_STAN_FIT_CONFIG
    cmdstanpy = cast("Any", importlib.import_module("cmdstanpy"))
    stan_file = adapter.stan_program_path(hierarchy)
    functions_dir = str(Path(stan_file).parent / "functions")
    model = cmdstanpy.CmdStanModel(
        stan_file=stan_file,
        stanc_options={"include-paths": [functions_dir]},
    )

    with tempfile.TemporaryDirectory(prefix="comp_model_stan_") as tmpdir:
        stan_fit = model.sample(
            data=adapter.build_stan_data(data, schema, hierarchy, layout, prior_specs),
            iter_warmup=resolved_config.n_warmup,
            iter_sampling=resolved_config.n_samples,
            chains=resolved_config.n_chains,
            seed=resolved_config.seed,
            adapt_delta=resolved_config.adapt_delta,
            max_treedepth=resolved_config.max_treedepth,
            show_console=resolved_config.show_console,
            show_progress=resolved_config.show_progress,
            refresh=resolved_config.refresh,
            output_dir=tmpdir,
        )

        posterior_samples = {}
        for parameter_name in adapter.subject_param_names():
            posterior_samples[parameter_name] = np.asarray(stan_fit.stan_variable(parameter_name))
        for parameter_name in adapter.population_param_names(hierarchy):
            posterior_samples[parameter_name] = np.asarray(stan_fit.stan_variable(parameter_name))

        log_lik = np.asarray(stan_fit.stan_variable("log_lik"))
        diagnostics = {
            "n_divergences": int(stan_fit.diagnose().count("divergent")),
            "summary": stan_fit.summary(),
        }

    return BayesFitResult(
        model_id=adapter.kernel_spec().model_id,
        hierarchy=hierarchy,
        posterior_samples=posterior_samples,
        log_lik=log_lik,
        subject_params=None,
        diagnostics=diagnostics,
    )
