"""Tests for Stan data export builders."""

from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.inference.bayes.stan.data_builder import (
    add_condition_data,
    add_prior_data,
    dataset_to_stan_data,
    subject_to_stan_data,
)
from comp_model.models.condition.shared_delta import SharedDeltaLayout
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.runtime.engine import SimulationConfig, simulate_dataset, simulate_subject
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA
from comp_model.tasks.spec import BlockSpec, TaskSpec


def _task() -> TaskSpec:
    """Create a simple one-block task for Stan export tests.

    Returns
    -------
    TaskSpec
        One-block asocial bandit task.
    """

    return TaskSpec(
        task_id="stan-export",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=4,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
        ),
    )


def test_subject_to_stan_data_exports_expected_shapes() -> None:
    """Ensure subject export produces expected trial counts and indexing.

    Returns
    -------
    None
        This test asserts Stan array structure.
    """

    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    subject = simulate_subject(
        task=_task(),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=params,
        config=SimulationConfig(seed=13),
        subject_id="s1",
    )

    stan_data = subject_to_stan_data(subject, ASOCIAL_BANDIT_SCHEMA)

    assert stan_data["A"] == 2
    assert stan_data["T"] == 4
    assert stan_data["B"] == 1
    assert stan_data["block_start"] == [1]
    assert all(choice in (1, 2) for choice in stan_data["choice"])


def test_dataset_to_stan_data_adds_subject_indices() -> None:
    """Ensure dataset export includes hierarchical subject indexing.

    Returns
    -------
    None
        This test asserts hierarchical export fields.
    """

    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    dataset = simulate_dataset(
        task=_task(),
        env_factory=lambda: StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params_per_subject={"s1": params, "s2": params},
        config=SimulationConfig(seed=17),
    )

    stan_data = dataset_to_stan_data(dataset, ASOCIAL_BANDIT_SCHEMA)

    assert stan_data["N"] == 2
    assert len(stan_data["subj"]) == stan_data["T"]
    assert set(stan_data["subj"]) == {1, 2}


def test_condition_and_prior_data_are_added() -> None:
    """Ensure condition and prior metadata extend the Stan data dictionary.

    Returns
    -------
    None
        This test asserts condition and prior exports.
    """

    task = TaskSpec(
        task_id="stan-conditioned",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=2,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
            BlockSpec(
                condition="social",
                n_trials=2,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
        ),
    )
    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    subject = simulate_subject(
        task=task,
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=params,
        config=SimulationConfig(seed=19),
        subject_id="s1",
    )
    stan_data = subject_to_stan_data(subject, ASOCIAL_BANDIT_SCHEMA)
    layout = SharedDeltaLayout(
        kernel_spec=kernel.spec(),
        conditions=("baseline", "social"),
        baseline_condition="baseline",
    )

    add_condition_data(stan_data, subject, layout)
    add_prior_data(stan_data, kernel.spec())

    assert stan_data["C"] == 2
    assert stan_data["baseline_cond"] == 1
    assert stan_data["cond"] == [1, 1, 2, 2]
    assert "alpha_prior_mu" in stan_data
    assert "beta_prior_sigma" in stan_data
