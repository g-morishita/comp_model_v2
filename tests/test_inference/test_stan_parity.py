"""Parity tests between Python replay and Stan exports."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from comp_model.data.extractors import replay_trial_steps
from comp_model.data.schema import Block, Event, EventPhase, SubjectData, Trial
from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.inference.bayes.stan.data_builder import subject_to_step_data
from comp_model.inference.mle.objective import log_likelihood_simple
from comp_model.models.condition.shared_delta import SharedDeltaLayout
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.models.kernels.transforms import get_transform
from comp_model.runtime.engine import SimulationConfig, simulate_subject
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA
from comp_model.tasks.spec import BlockSpec, TaskSpec

if TYPE_CHECKING:
    from comp_model.models.kernels.base import ModelKernelSpec


_STAN_PARITY_ATOL = 5e-6


class PerBlockAsocialQLearningKernel(AsocialQLearningKernel):
    """Asocial Q-learning kernel variant that resets latent state per block."""

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return the asocial kernel specification with per-block resets.

        Returns
        -------
        ModelKernelSpec
            Kernel specification with ``state_reset_policy="per_block"``.
        """

        return replace(super().spec(), state_reset_policy="per_block")


def _task() -> TaskSpec:
    """Create a small one-block task for parity tests.

    Returns
    -------
    TaskSpec
        One-block task with four asocial trials.
    """

    return TaskSpec(
        task_id="parity",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=4,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
        ),
    )


def _condition_task() -> TaskSpec:
    """Create a two-condition task for structural parity tests.

    Returns
    -------
    TaskSpec
        Two-block task with one decision per trial.
    """

    return TaskSpec(
        task_id="parity-conditioned",
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


def _subject() -> SubjectData:
    """Simulate a single subject for parity tests.

    Returns
    -------
    object
        Simulated subject data.
    """

    kernel = AsocialQLearningKernel()
    return simulate_subject(
        task=_task(),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=kernel.parse_params({"alpha": 0.0, "beta": 1.0}),
        config=SimulationConfig(seed=37),
        subject_id="s1",
    )


def _two_block_manual_subject() -> SubjectData:
    """Create a two-block subject with a reset-sensitive likelihood trace.

    Returns
    -------
    SubjectData
        Manual subject data whose second block depends on reset behavior.
    """

    def _trial(trial_index: int, action: int, reward: float) -> Trial:
        """Create a minimal asocial trial for parity tests.

        Parameters
        ----------
        trial_index
            Trial index within the current block.
        action
            Chosen action value.
        reward
            Observed reward.

        Returns
        -------
        Trial
            Event-based trial matching ``ASOCIAL_BANDIT_SCHEMA``.
        """

        return Trial(
            trial_index=trial_index,
            events=(
                Event(
                    phase=EventPhase.INPUT,
                    event_index=0,
                    node_id="main",
                    payload={"available_actions": (0, 1)},
                ),
                Event(
                    phase=EventPhase.DECISION,
                    event_index=1,
                    node_id="main",
                    payload={"action": action},
                ),
                Event(
                    phase=EventPhase.OUTCOME,
                    event_index=2,
                    node_id="main",
                    payload={"reward": reward},
                ),
                Event(
                    phase=EventPhase.UPDATE,
                    event_index=3,
                    node_id="main",
                    payload={"choice": action, "reward": reward},
                ),
            ),
        )

    return SubjectData(
        subject_id="reset-subject",
        blocks=(
            Block(
                block_index=0,
                condition="baseline",
                schema_id="asocial_bandit",
                trials=(_trial(trial_index=0, action=0, reward=1.0),),
            ),
            Block(
                block_index=1,
                condition="baseline",
                schema_id="asocial_bandit",
                trials=(_trial(trial_index=0, action=1, reward=0.0),),
            ),
        ),
    )


def _python_trial_log_likelihoods(subject: SubjectData, alpha: float, beta: float) -> list[float]:
    """Compute Python replay log-likelihood contributions per trial.

    Parameters
    ----------
    subject
        Subject data to replay.
    alpha
        Constrained learning rate.
    beta
        Constrained inverse temperature.

    Returns
    -------
    list[float]
        Per-trial log-likelihood values.
    """

    kernel = AsocialQLearningKernel()
    raw_params = {
        "alpha": get_transform("sigmoid").inverse(alpha),
        "beta": get_transform("softplus").inverse(beta),
    }
    params = kernel.parse_params(raw_params)
    state = kernel.initial_state(2, params)
    trial_log_likelihoods: list[float] = []

    for block in subject.blocks:
        for trial in block.trials:
            trial_log_likelihood = 0.0
            for event_type, learner_id, view in replay_trial_steps(trial, ASOCIAL_BANDIT_SCHEMA):
                if event_type == EventPhase.DECISION and learner_id == "subject":
                    probabilities = kernel.action_probabilities(state, view, params)
                    choice_index = view.action
                    trial_log_likelihood += float(np.log(probabilities[choice_index]))
                elif event_type == "update" and learner_id == "subject":
                    state = kernel.update(state, view, params)
            trial_log_likelihoods.append(trial_log_likelihood)

    return trial_log_likelihoods


def _cmdstanpy() -> Any:
    """Import CmdStanPy or skip the Stan-marked tests.

    Returns
    -------
    Any
        Imported ``cmdstanpy`` module.
    """

    cmdstanpy = pytest.importorskip("cmdstanpy")
    try:
        cmdstanpy.cmdstan_path()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CmdStan is not installed: {exc}")
    return cmdstanpy


def _fixed_param_model(filename: str) -> Any:
    """Compile a fixed-parameter Stan test program.

    Parameters
    ----------
    filename
        Test Stan filename under ``tests/test_inference/stan``.

    Returns
    -------
    Any
        Compiled ``CmdStanModel`` instance.
    """

    cmdstanpy = _cmdstanpy()
    stan_file = Path(__file__).parent / "stan" / filename
    return cmdstanpy.CmdStanModel(stan_file=str(stan_file))


def _subject_step_data(subject: SubjectData) -> dict[str, Any]:
    """Build step-based Stan data for parity tests.

    Parameters
    ----------
    subject
        Subject data to export.

    Returns
    -------
    dict[str, Any]
        Step-stream Stan data dict.
    """

    kernel_spec = AsocialQLearningKernel.spec()
    return subject_to_step_data(subject, ASOCIAL_BANDIT_SCHEMA, kernel_spec=kernel_spec)


@pytest.mark.stan
def test_log_likelihood_parity() -> None:
    """Ensure Python and Stan agree on total log-likelihood.

    Returns
    -------
    None
        This test asserts near-exact total log-likelihood parity.
    """

    subject = _subject()
    stan_data = _subject_step_data(subject)
    alpha = 0.5
    beta = 1.25
    model = _fixed_param_model("q_learning_loglik_fixed_params.stan")
    fit = model.sample(
        data={**stan_data, "alpha": alpha, "beta": beta, "reset_on_block": 0, "q_init": 0.5},
        fixed_param=True,
        iter_sampling=1,
        iter_warmup=1,
        chains=1,
        seed=1,
    )
    stan_log_lik = np.asarray(fit.stan_variable("log_lik"))[0]

    kernel = AsocialQLearningKernel()
    raw_params = {
        "alpha": get_transform("sigmoid").inverse(alpha),
        "beta": get_transform("softplus").inverse(beta),
    }
    python_log_lik = log_likelihood_simple(kernel, subject, raw_params, ASOCIAL_BANDIT_SCHEMA)

    assert abs(python_log_lik - float(stan_log_lik.sum())) < _STAN_PARITY_ATOL


@pytest.mark.stan
def test_trialwise_parity() -> None:
    """Ensure Python and Stan agree trial by trial on log-likelihood.

    Returns
    -------
    None
        This test asserts near-exact trialwise parity.
    """

    subject = _subject()
    stan_data = _subject_step_data(subject)
    alpha = 0.5
    beta = 1.25
    model = _fixed_param_model("q_learning_loglik_fixed_params.stan")
    fit = model.sample(
        data={**stan_data, "alpha": alpha, "beta": beta, "reset_on_block": 0, "q_init": 0.5},
        fixed_param=True,
        iter_sampling=1,
        iter_warmup=1,
        chains=1,
        seed=2,
    )
    stan_log_lik = np.asarray(fit.stan_variable("log_lik"))[0]
    python_log_lik = _python_trial_log_likelihoods(subject, alpha, beta)

    assert np.max(np.abs(np.asarray(python_log_lik) - stan_log_lik)) < _STAN_PARITY_ATOL


@pytest.mark.stan
def test_block_reset_parity() -> None:
    """Ensure Python and Stan agree when kernels reset state at block boundaries.

    Returns
    -------
    None
        This test asserts total log-likelihood parity under per-block resets.
    """

    subject = _two_block_manual_subject()
    stan_data = _subject_step_data(subject)
    alpha = 0.5
    beta = 2.0
    model = _fixed_param_model("q_learning_loglik_fixed_params.stan")
    fit = model.sample(
        data={**stan_data, "alpha": alpha, "beta": beta, "reset_on_block": 1, "q_init": 0.5},
        fixed_param=True,
        iter_sampling=1,
        iter_warmup=1,
        chains=1,
        seed=4,
    )
    stan_log_lik = np.asarray(fit.stan_variable("log_lik"))[0]

    kernel = PerBlockAsocialQLearningKernel()
    raw_params = {
        "alpha": get_transform("sigmoid").inverse(alpha),
        "beta": get_transform("softplus").inverse(beta),
    }
    python_log_lik = log_likelihood_simple(kernel, subject, raw_params, ASOCIAL_BANDIT_SCHEMA)

    assert abs(python_log_lik - float(stan_log_lik.sum())) < _STAN_PARITY_ATOL


@pytest.mark.stan
def test_condition_reconstruction_parity() -> None:
    """Ensure Python and Stan reconstruct condition parameters identically.

    Returns
    -------
    None
        This test asserts exact reconstruction parity.
    """

    layout = SharedDeltaLayout(
        kernel_spec=AsocialQLearningKernel.spec(),
        conditions=("baseline", "social"),
        baseline_condition="baseline",
    )
    raw = {
        "alpha__shared_z": 0.1,
        "beta__shared_z": 1.2,
        "alpha__delta_z__social": 0.3,
        "beta__delta_z__social": -0.5,
    }
    model = _fixed_param_model("condition_reconstruction.stan")
    fit = model.sample(
        data={
            "alpha_shared_z": raw["alpha__shared_z"],
            "beta_shared_z": raw["beta__shared_z"],
            "alpha_delta_z_social": raw["alpha__delta_z__social"],
            "beta_delta_z_social": raw["beta__delta_z__social"],
        },
        fixed_param=True,
        iter_sampling=1,
        iter_warmup=1,
        chains=1,
        seed=3,
    )

    reconstructed = layout.reconstruct_all(raw)
    assert reconstructed["baseline"]["alpha"] == pytest.approx(
        float(np.asarray(fit.stan_variable("alpha_baseline"))[0])
    )
    assert reconstructed["baseline"]["beta"] == pytest.approx(
        float(np.asarray(fit.stan_variable("beta_baseline"))[0])
    )
    assert reconstructed["social"]["alpha"] == pytest.approx(
        float(np.asarray(fit.stan_variable("alpha_social"))[0])
    )
    assert reconstructed["social"]["beta"] == pytest.approx(
        float(np.asarray(fit.stan_variable("beta_social"))[0])
    )


def test_simulation_fit_structural_parity() -> None:
    """Ensure simulation traces and Stan export agree structurally.

    Returns
    -------
    None
        This test asserts choices, rewards, and step counts survive export.
    """

    kernel = AsocialQLearningKernel()
    subject = simulate_subject(
        task=_condition_task(),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=kernel.parse_params({"alpha": 0.0, "beta": 1.0}),
        config=SimulationConfig(seed=41),
        subject_id="s1",
    )
    stan_data = _subject_step_data(subject)

    # Each trial produces 2 steps (action + self-update); collect in replay order.
    n_trials = sum(len(block.trials) for block in subject.blocks)
    assert stan_data["E"] == n_trials * 2
    assert stan_data["D"] == n_trials
    # Action steps (even positions) carry the 1-based choice; update steps (odd) carry reward.
    choices_from_actions = [c for c in stan_data["step_choice"] if c > 0]
    rewards_from_updates = [
        r
        for c, r in zip(stan_data["step_update_action"], stan_data["step_reward"], strict=True)
        if c > 0
    ]
    assert len(choices_from_actions) == n_trials
    assert all(c in (1, 2) for c in choices_from_actions)
    assert len(rewards_from_updates) == n_trials
