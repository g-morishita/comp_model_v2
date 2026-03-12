"""Tests for Stan backend configuration and error routing."""

from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.inference.bayes.stan.backend import StanFitConfig
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.dispatch import fit
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.runtime.engine import SimulationConfig, simulate_subject
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA
from comp_model.tasks.spec import BlockSpec, TaskSpec


def _subject() -> object:
    """Create a simulated subject for Stan backend dispatch tests.

    Returns
    -------
    object
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
            kernel,
            subject,
            ASOCIAL_BANDIT_SCHEMA,
        )
    except ValueError as exc:
        assert "StanAdapter" in str(exc)
    else:
        raise AssertionError("Expected Stan dispatch to require an adapter")
