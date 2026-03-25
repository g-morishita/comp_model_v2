"""Configuration dataclasses for model recovery analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from comp_model.environments.base import Environment
    from comp_model.inference.config import InferenceConfig
    from comp_model.models.kernels.base import ModelKernel
    from comp_model.recovery.parameter.config import ParamDist
    from comp_model.tasks.schemas import TrialSchema
    from comp_model.tasks.spec import TaskSpec


@dataclass(frozen=True, slots=True)
class GeneratingModelSpec:
    """Specification for a data-generating model in a model recovery study.

    Attributes
    ----------
    name
        Display name identifying the generating model.
    kernel
        Model kernel used to simulate data.
    param_dists
        Population distributions from which true parameters are sampled.
    """

    name: str
    kernel: ModelKernel[Any, Any]
    param_dists: tuple[ParamDist, ...]


@dataclass(frozen=True, slots=True)
class CandidateModelSpec:
    """Specification for a candidate model fitted during model recovery.

    Attributes
    ----------
    name
        Display name identifying the candidate model.
    kernel
        Model kernel to fit.
    inference_config
        Inference configuration. The ``backend`` field determines which
        scoring criteria are valid (``"mle"`` for AIC/BIC/log_likelihood,
        ``"stan"`` for waic/loo).
    adapter
        Optional Stan adapter required when ``inference_config.backend == "stan"``.
    """

    name: str
    kernel: ModelKernel[Any, Any]
    inference_config: InferenceConfig
    adapter: object | None = None


@dataclass(frozen=True, slots=True)
class ModelRecoveryConfig:
    """Full specification for a model recovery study.

    Attributes
    ----------
    generating_models
        Models from which synthetic datasets will be simulated.
    candidate_models
        Models fitted to every simulated dataset.
    n_replications
        Number of simulate-then-fit cycles per generating model.
    n_subjects
        Number of subjects per replication.
    task
        Task specification used for simulation.
    env_factory
        Factory returning a fresh environment per subject.
    schema
        Trial schema used for replay extraction.
    criterion
        Model selection criterion.  MLE criteria (``"aic"``, ``"bic"``,
        ``"log_likelihood"``) require candidates with ``backend="mle"``.
        Bayesian criteria (``"waic"``, ``"loo"``) require ``backend="stan"``.
    simulation_base_seed
        Base seed for reproducible simulation.
    max_workers
        Maximum parallel workers when fitting candidate models.
        ``None`` selects automatically.
    """

    generating_models: tuple[GeneratingModelSpec, ...]
    candidate_models: tuple[CandidateModelSpec, ...]
    n_replications: int
    n_subjects: int
    task: TaskSpec
    env_factory: Callable[[], Environment]
    schema: TrialSchema
    criterion: Literal["aic", "bic", "log_likelihood", "waic", "loo"] = "aic"
    simulation_base_seed: int = 42
    max_workers: int | None = None
