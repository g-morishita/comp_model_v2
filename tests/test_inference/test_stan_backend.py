"""Tests for Stan backend configuration and error routing."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.inference.bayes.stan.backend import StanFitConfig, fit_stan
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.dispatch import fit
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.runtime.engine import SimulationConfig, simulate_subject
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA
from comp_model.tasks.spec import BlockSpec, TaskSpec

if TYPE_CHECKING:
    import pytest

    from comp_model.data.schema import Dataset, SubjectData
    from comp_model.inference.config import PriorSpec
    from comp_model.models.condition.shared_delta import SharedDeltaLayout
    from comp_model.models.kernels.base import ModelKernelSpec
    from comp_model.tasks.schemas import TrialSchema


def _subject() -> SubjectData:
    """Create a simulated subject for Stan backend dispatch tests.

    Returns
    -------
    SubjectData
        Simulated subject data.
    """

    kernel = AsocialQLearningKernel()
    return simulate_subject(
        task=TaskSpec(
            task_id="stan-dispatch",
            blocks=(
                BlockSpec(
                    condition="baseline",
                    n_trials=4,
                    schema=ASOCIAL_BANDIT_SCHEMA,
                    metadata={"n_actions": 2},
                ),
            ),
        ),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=kernel.parse_params({"alpha": 0.0, "beta": 1.0}),
        config=SimulationConfig(seed=31),
        subject_id="s1",
    )


def test_stan_fit_config_defaults_match_expected_sampling_values() -> None:
    """Ensure Stan fit config defaults match the implementation plan.

    Returns
    -------
    None
        This test asserts default sampler settings.
    """

    config = StanFitConfig()

    assert config.n_warmup == 1000
    assert config.n_samples == 1000
    assert config.n_chains == 4


def test_dispatch_rejects_stan_backend_without_adapter() -> None:
    """Ensure Stan dispatch requires an adapter before sampling.

    Returns
    -------
    None
        This test raises on missing Stan adapter input.
    """

    kernel = AsocialQLearningKernel()
    subject = _subject()

    try:
        fit(
            InferenceConfig(hierarchy=HierarchyStructure.SUBJECT_SHARED, backend="stan"),
            cast("Any", kernel),
            subject,
            ASOCIAL_BANDIT_SCHEMA,
        )
    except ValueError as exc:
        assert "StanAdapter" in str(exc)
    else:
        raise AssertionError("Expected Stan dispatch to require an adapter")


def test_fit_stan_extracts_extra_posterior_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure Stan backend extracts configured extra posterior variables."""

    observed: dict[str, Any] = {}

    class FakeStanFit:
        """Minimal CmdStanPy fit stub for posterior extraction tests."""

        def __init__(self) -> None:
            self.requests: list[str] = []
            self.variables = {
                "alpha": np.asarray([0.1, 0.2]),
                "alpha_pop": np.asarray([0.3, 0.4]),
                "alpha_shared_z": np.asarray([0.5, 0.6]),
                "log_lik": np.asarray([[0.0], [0.0]]),
            }

        def stan_variable(self, name: str) -> np.ndarray:
            self.requests.append(name)
            return self.variables[name]

        def diagnose(self) -> str:
            return "diagnose: ok"

        def summary(self) -> dict[str, str]:
            return {"status": "ok"}

    class FakeCmdStanModel:
        """Minimal CmdStanModel stub that returns a prebuilt fake fit."""

        def __init__(self, *, stan_file: str, stanc_options: dict[str, object]) -> None:
            observed["stan_file"] = stan_file
            observed["stanc_options"] = stanc_options

        def sample(self, **kwargs: object) -> FakeStanFit:
            observed["sample_kwargs"] = kwargs
            observed["fit"] = FakeStanFit()
            return observed["fit"]  # type: ignore[return-value]

    class FakeStanAdapter:
        """Minimal adapter stub for backend extraction tests."""

        def kernel_spec(self) -> ModelKernelSpec:
            return AsocialQLearningKernel.spec()

        def stan_program_path(self, hierarchy: HierarchyStructure) -> str:
            return __file__

        def build_stan_data(
            self,
            data: SubjectData | Dataset,
            schema: TrialSchema,
            hierarchy: HierarchyStructure,
            layout: SharedDeltaLayout | None = None,
            prior_specs: dict[str, PriorSpec] | None = None,
        ) -> dict[str, object]:
            return {"n_subjects": 1}

        def subject_param_names(self) -> tuple[str, ...]:
            return ("alpha",)

        def population_param_names(self, hierarchy: HierarchyStructure) -> tuple[str, ...]:
            return ("alpha_pop",)

        def extra_posterior_param_names(self, hierarchy: HierarchyStructure) -> tuple[str, ...]:
            return ("alpha_shared_z", "alpha_pop")

    def fake_import_module(name: str) -> types.SimpleNamespace:
        assert name == "cmdstanpy"
        return types.SimpleNamespace(CmdStanModel=FakeCmdStanModel)

    monkeypatch.setattr(
        "comp_model.inference.bayes.stan.backend.importlib.import_module",
        fake_import_module,
    )

    result = fit_stan(
        FakeStanAdapter(),
        _subject(),
        ASOCIAL_BANDIT_SCHEMA,
        HierarchyStructure.SUBJECT_BLOCK_CONDITION,
        config=StanFitConfig(
            n_warmup=5,
            n_samples=7,
            n_chains=2,
            show_console=False,
            show_progress=False,
        ),
    )

    fit_requests = cast("FakeStanFit", observed["fit"]).requests
    sample_kwargs = cast("dict[str, object]", observed["sample_kwargs"])
    assert fit_requests == ["alpha", "alpha_pop", "alpha_shared_z", "log_lik"]
    assert set(result.posterior_samples) == {"alpha", "alpha_pop", "alpha_shared_z"}
    assert np.array_equal(result.posterior_samples["alpha_shared_z"], np.asarray([0.5, 0.6]))
    assert np.array_equal(result.log_lik, np.asarray([[0.0], [0.0]]))
    assert sample_kwargs["data"] == {"n_subjects": 1}
    assert sample_kwargs["iter_warmup"] == 5
    assert sample_kwargs["iter_sampling"] == 7
    assert sample_kwargs["chains"] == 2
    assert sample_kwargs["seed"] is None
    assert sample_kwargs["adapt_delta"] == 0.8
    assert sample_kwargs["max_treedepth"] == 10
    assert sample_kwargs["show_console"] is False
    assert sample_kwargs["show_progress"] is False
    assert sample_kwargs["refresh"] is None
    assert isinstance(sample_kwargs["output_dir"], str)
