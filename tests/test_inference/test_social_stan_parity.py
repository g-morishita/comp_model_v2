"""Parity tests between Python social kernel replay and Stan exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from comp_model.data.extractors import replay_trial_steps
from comp_model.data.schema import Block, Event, EventPhase, SubjectData, Trial
from comp_model.inference.bayes.stan.data_builder import subject_to_step_data
from comp_model.models.kernels.social_rl_self_reward_demo_reward import (
    SocialQLearningKernel,
)
from comp_model.models.kernels.transforms import get_transform
from comp_model.tasks.schemas import SOCIAL_POST_OUTCOME_SCHEMA

_STAN_PARITY_ATOL = 5e-6


def _social_trial(
    trial_index: int,
    action: int,
    reward: float,
    social_action: int,
    social_reward: float,
) -> Trial:
    """Create a minimal social trial for parity tests.

    Parameters
    ----------
    trial_index
        Trial index within the current block.
    action
        Chosen action value.
    reward
        Observed reward.
    social_action
        Demonstrator's chosen action.
    social_reward
        Demonstrator's observed reward.

    Returns
    -------
    Trial
        Event-based trial matching ``SOCIAL_POST_OUTCOME_SCHEMA``.
    """

    # Schema: INPUT(subj) DECISION(subj) OUTCOME(subj) UPDATE(subj→subj)
    #         INPUT(demo) DECISION(demo) OUTCOME(demo) UPDATE(demo→demo) UPDATE(demo→subj)
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
                phase=EventPhase.DECISION, event_index=1, node_id="main", payload={"action": action}
            ),
            Event(
                phase=EventPhase.OUTCOME, event_index=2, node_id="main", payload={"reward": reward}
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=3,
                node_id="main",
                payload={"choice": action, "reward": reward},
            ),
            Event(
                phase=EventPhase.INPUT,
                event_index=4,
                node_id="main",
                actor_id="demonstrator",
                payload={"available_actions": (0, 1)},
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=5,
                node_id="main",
                actor_id="demonstrator",
                payload={"action": social_action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=6,
                node_id="main",
                actor_id="demonstrator",
                payload={"reward": social_reward},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=7,
                node_id="main",
                actor_id="demonstrator",
                payload={"choice": social_action, "reward": social_reward},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=8,
                node_id="main",
                actor_id="demonstrator",
                payload={"choice": social_action, "reward": social_reward},
            ),
        ),
    )


def _social_subject() -> SubjectData:
    """Create a social subject with multiple trials for parity testing.

    Returns
    -------
    SubjectData
        Subject with social trial data spanning one block.
    """

    return SubjectData(
        subject_id="parity-social",
        blocks=(
            Block(
                block_index=0,
                condition="baseline",
                trials=(
                    _social_trial(0, action=0, reward=1.0, social_action=1, social_reward=0.0),
                    _social_trial(1, action=1, reward=0.0, social_action=0, social_reward=1.0),
                    _social_trial(2, action=0, reward=0.5, social_action=1, social_reward=0.5),
                    _social_trial(3, action=1, reward=1.0, social_action=0, social_reward=0.0),
                ),
            ),
        ),
    )


def _python_social_log_likelihoods(
    subject: SubjectData,
    alpha_self: float,
    alpha_other: float,
    beta: float,
) -> list[float]:
    """Compute Python replay log-likelihood contributions per trial.

    Parameters
    ----------
    subject
        Subject data to replay.
    alpha_self
        Constrained self learning rate.
    alpha_other
        Constrained social learning rate.
    beta
        Constrained inverse temperature.

    Returns
    -------
    list[float]
        Per-trial log-likelihood values.
    """

    kernel = SocialQLearningKernel()
    raw_params = {
        "alpha_self": get_transform("sigmoid").inverse(alpha_self),
        "alpha_other": get_transform("sigmoid").inverse(alpha_other),
        "beta": get_transform("softplus").inverse(beta),
    }
    params = kernel.parse_params(raw_params)
    state = kernel.initial_state(2, params)
    trial_log_likelihoods: list[float] = []

    for block in subject.blocks:
        for trial in block.trials:
            trial_log_likelihood = 0.0
            for event_type, learner_id, view in replay_trial_steps(
                trial, SOCIAL_POST_OUTCOME_SCHEMA
            ):
                if event_type == EventPhase.DECISION and learner_id == "subject":
                    probabilities = kernel.action_probabilities(state, view, params)
                    choice_index = view.choice
                    trial_log_likelihood += float(np.log(probabilities[choice_index]))
                elif event_type == "update" and learner_id == "subject":
                    state = kernel.next_state(state, view, params)
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


def _social_step_data(subject: SubjectData) -> dict[str, Any]:
    """Build step-based Stan data with social fields for parity tests.

    Parameters
    ----------
    subject
        Subject data to export.

    Returns
    -------
    dict[str, Any]
        Step-stream Stan data dict with social fields.
    """

    kernel_spec = SocialQLearningKernel.spec()
    return subject_to_step_data(
        subject,
        SOCIAL_POST_OUTCOME_SCHEMA,
        kernel_spec=kernel_spec,
        include_social=True,
    )


_SOCIAL_PARITY_ALPHA_SELF = 0.4
_SOCIAL_PARITY_ALPHA_OTHER = 0.3
_SOCIAL_PARITY_BETA = 1.5


@pytest.fixture(scope="module")
def social_stan_log_lik() -> np.ndarray:
    """Compile and run the social Stan model once for all parity tests.

    The Stan model has no ``parameters`` block, so ``fixed_param=True``
    evaluates only the ``generated_quantities`` block.  Running the
    external process once per module (rather than once per test) avoids
    intermittent CI failures caused by repeated subprocess invocations
    under resource pressure.  ``show_console=False`` suppresses CmdStan
    stderr noise that some CmdStanPy versions mis-interpret as a sampling
    error when the model has no free parameters.

    Returns
    -------
    np.ndarray
        Per-trial log-likelihood array of shape ``(D,)``.
    """

    subject = _social_subject()
    stan_data = _social_step_data(subject)
    model = _fixed_param_model("social_q_learning_loglik_fixed_params.stan")
    fit = model.sample(
        data={
            **stan_data,
            "alpha_self": _SOCIAL_PARITY_ALPHA_SELF,
            "alpha_other": _SOCIAL_PARITY_ALPHA_OTHER,
            "beta": _SOCIAL_PARITY_BETA,
            "reset_on_block": 0,
            "q_init": 0.5,
        },
        fixed_param=True,
        iter_sampling=1,
        iter_warmup=1,
        chains=1,
        seed=1,
        show_console=False,
    )
    return np.asarray(fit.stan_variable("log_lik"))[0]


@pytest.mark.stan
def test_social_log_likelihood_parity(social_stan_log_lik: np.ndarray) -> None:
    """Ensure Python and Stan agree on total social log-likelihood.

    Returns
    -------
    None
        This test asserts near-exact total log-likelihood parity.
    """

    subject = _social_subject()
    python_log_lik = _python_social_log_likelihoods(
        subject,
        _SOCIAL_PARITY_ALPHA_SELF,
        _SOCIAL_PARITY_ALPHA_OTHER,
        _SOCIAL_PARITY_BETA,
    )

    assert abs(sum(python_log_lik) - float(social_stan_log_lik.sum())) < _STAN_PARITY_ATOL


@pytest.mark.stan
def test_social_trialwise_parity(social_stan_log_lik: np.ndarray) -> None:
    """Ensure Python and Stan agree trial by trial on social log-likelihood.

    Returns
    -------
    None
        This test asserts near-exact trialwise parity.
    """

    subject = _social_subject()
    python_log_lik = _python_social_log_likelihoods(
        subject,
        _SOCIAL_PARITY_ALPHA_SELF,
        _SOCIAL_PARITY_ALPHA_OTHER,
        _SOCIAL_PARITY_BETA,
    )

    assert np.max(np.abs(np.asarray(python_log_lik) - social_stan_log_lik)) < _STAN_PARITY_ATOL
