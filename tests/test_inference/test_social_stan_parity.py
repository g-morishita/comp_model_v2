"""Parity tests between Python social kernel replay and Stan exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from comp_model.data.extractors import extract_decision_views
from comp_model.data.schema import Block, Event, EventPhase, SubjectData, Trial
from comp_model.inference.bayes.stan.data_builder import subject_to_step_data
from comp_model.models.kernels.social_observed_outcome_q import (
    SocialObservedOutcomeQKernel,
)
from comp_model.models.kernels.transforms import get_transform
from comp_model.tasks.schemas import SOCIAL_PRE_CHOICE_SCHEMA

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
        Event-based trial matching ``SOCIAL_PRE_CHOICE_SCHEMA``.
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
                phase=EventPhase.INPUT,
                event_index=1,
                node_id="main",
                actor_id="demonstrator",
                payload={
                    "available_actions": (0, 1),
                    "observation": {
                        "social_action": social_action,
                        "social_reward": social_reward,
                    },
                },
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=2,
                node_id="main",
                payload={"action": action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=3,
                node_id="main",
                payload={"reward": reward},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=4,
                node_id="main",
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

    kernel = SocialObservedOutcomeQKernel()
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
            for view in extract_decision_views(trial, SOCIAL_PRE_CHOICE_SCHEMA):
                probabilities = kernel.action_probabilities(state, view, params)
                choice_index = view.choice
                trial_log_likelihood += float(np.log(probabilities[choice_index]))
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

    kernel_spec = SocialObservedOutcomeQKernel.spec()
    return subject_to_step_data(
        subject,
        SOCIAL_PRE_CHOICE_SCHEMA,
        kernel_spec=kernel_spec,
        include_social=True,
    )


@pytest.mark.stan
def test_social_log_likelihood_parity() -> None:
    """Ensure Python and Stan agree on total social log-likelihood.

    Returns
    -------
    None
        This test asserts near-exact total log-likelihood parity.
    """

    subject = _social_subject()
    stan_data = _social_step_data(subject)
    alpha_self = 0.4
    alpha_other = 0.3
    beta = 1.5
    model = _fixed_param_model("social_q_learning_loglik_fixed_params.stan")
    fit = model.sample(
        data={
            **stan_data,
            "alpha_self": alpha_self,
            "alpha_other": alpha_other,
            "beta": beta,
            "reset_on_block": 0,
            "q_init": 0.5,
        },
        fixed_param=True,
        iter_sampling=1,
        iter_warmup=1,
        chains=1,
        seed=1,
    )
    stan_log_lik = np.asarray(fit.stan_variable("log_lik"))[0]

    python_log_lik = _python_social_log_likelihoods(subject, alpha_self, alpha_other, beta)

    assert abs(sum(python_log_lik) - float(stan_log_lik.sum())) < _STAN_PARITY_ATOL


@pytest.mark.stan
def test_social_trialwise_parity() -> None:
    """Ensure Python and Stan agree trial by trial on social log-likelihood.

    Returns
    -------
    None
        This test asserts near-exact trialwise parity.
    """

    subject = _social_subject()
    stan_data = _social_step_data(subject)
    alpha_self = 0.4
    alpha_other = 0.3
    beta = 1.5
    model = _fixed_param_model("social_q_learning_loglik_fixed_params.stan")
    fit = model.sample(
        data={
            **stan_data,
            "alpha_self": alpha_self,
            "alpha_other": alpha_other,
            "beta": beta,
            "reset_on_block": 0,
            "q_init": 0.5,
        },
        fixed_param=True,
        iter_sampling=1,
        iter_warmup=1,
        chains=1,
        seed=2,
    )
    stan_log_lik = np.asarray(fit.stan_variable("log_lik"))[0]
    python_log_lik = _python_social_log_likelihoods(subject, alpha_self, alpha_other, beta)

    assert np.max(np.abs(np.asarray(python_log_lik) - stan_log_lik)) < _STAN_PARITY_ATOL
