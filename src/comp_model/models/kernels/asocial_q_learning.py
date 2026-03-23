"""Asocial Q-learning kernel.

This module implements the classic Q-learning (Rescorla-Wagner) model for a
single agent learning on their own, with no social information.

The core idea is simple: the agent keeps an internal estimate of how good each
option is (called a Q-value). After each trial the estimate for the chosen
option is nudged toward the reward that was actually received. Over many trials
the estimates converge toward the true reward probabilities of each option.

Two free parameters control the agent's behaviour:

- ``alpha`` (learning rate): how much weight the agent gives to new evidence.
  A high alpha means recent outcomes dominate; a low alpha means the agent
  updates slowly and relies heavily on accumulated past experience.
- ``beta`` (inverse temperature): how deterministically the agent exploits its
  best option. A high beta means the agent almost always picks the option with
  the highest Q-value; a low beta means choices are near-random regardless of
  Q-values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comp_model.models.kernels.base import InitSpec, ModelKernel, ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.probabilities import stable_softmax
from comp_model.models.kernels.transforms import get_transform

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class QParams:
    """The two free parameters that define an asocial Q-learning agent.

    Attributes
    ----------
    alpha
        Learning rate — a number strictly between 0 and 1. Controls how quickly
        the agent updates its beliefs after each outcome. A value near 1 means
        the agent almost replaces its old estimate with the new reward; a value
        near 0 means the agent changes its estimate very little each trial.
    beta
        Inverse temperature — a positive number. Controls how deterministically
        the agent acts on its beliefs. A large beta means the agent reliably
        picks the option it currently thinks is best; a small beta means choices
        are more random and exploratory.
    """

    alpha: float
    beta: float


@dataclass(slots=True)
class QState:
    """The agent's current internal beliefs — one value per available option.

    Each Q-value is the agent's running estimate of how rewarding a particular
    option is. Q-values are updated trial by trial as the agent gains
    experience. They start at 0.5 (reflecting genuine uncertainty) and drift
    toward 0 or 1 as evidence accumulates.

    Attributes
    ----------
    q_values
        List of Q-values, one per option, indexed by the option's integer code.
        For example, if there are two options, ``q_values[0]`` is the current
        estimate for option 0 and ``q_values[1]`` for option 1.
    """

    q_values: list[float]


class AsocialQLearningKernel(ModelKernel[QState, QParams]):
    """Q-learning model for a participant who learns only from their own experience.

    This is the standard asocial reinforcement learning model. The agent
    observes the options available on each trial, makes a choice, receives a
    reward, and updates only the Q-value for the option they actually chose.
    Options not chosen on a given trial are left unchanged.

    The same kernel works with any task design that produces a compatible trial
    record — it does not depend on the specifics of how a particular experiment
    was structured.
    """

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return a description of this model's parameters for the fitting machinery.

        Returns
        -------
        ModelKernelSpec
            A record declaring this model's two free parameters (``alpha`` and
            ``beta``), how each is constrained (alpha to (0, 1); beta to
            positive values), and sensible starting values for numerical
            optimisation.
        """

        return ModelKernelSpec(
            model_id="asocial_q_learning",
            parameter_specs=(
                ParameterSpec(
                    name="alpha",
                    transform_id="sigmoid",
                    description="learning rate",
                    mle_init=InitSpec(
                        strategy="fixed",
                        kwargs={},
                        default_unconstrained=0.0,
                    ),
                ),
                ParameterSpec(
                    name="beta",
                    transform_id="softplus",
                    description="inverse temperature",
                    mle_init=InitSpec(
                        strategy="fixed",
                        kwargs={},
                        default_unconstrained=1.0,
                    ),
                ),
            ),
            requires_social=False,
            state_reset_policy="per_subject",
        )

    def parse_params(self, raw: dict[str, float]) -> QParams:
        """Convert raw optimiser values into interpretable model parameters.

        During fitting, the optimiser works with unconstrained real-valued
        numbers (which can range from -infinity to +infinity). This method
        applies the appropriate mathematical transformations to map those raw
        numbers back onto their meaningful scales: alpha is squeezed into (0, 1)
        and beta is mapped to a positive value.

        Parameters
        ----------
        raw
            Dictionary of unconstrained parameter values produced by the
            optimiser, keyed by parameter name.

        Returns
        -------
        QParams
            Parameter object with alpha and beta on their natural scales,
            ready for use in ``action_probabilities`` and ``update``.
        """

        return QParams(
            alpha=get_transform("sigmoid").forward(raw["alpha"]),
            beta=get_transform("softplus").forward(raw["beta"]),
        )

    def initial_state(self, n_actions: int, params: QParams) -> QState:
        """Create the agent's belief state at the very start of the task.

        Before the first trial the agent has no experience, so all options are
        assumed equally rewarding. Each Q-value is initialised to 0.5,
        representing neutral uncertainty on a 0-1 reward scale.

        Parameters
        ----------
        n_actions
            The number of options available in the task.
        params
            The agent's fitted parameters (not used for initialisation, but
            required by the shared interface).

        Returns
        -------
        QState
            A state with all Q-values set to 0.5.
        """

        del params
        return QState(q_values=[self.spec().initial_value] * n_actions)

    def action_probabilities(
        self,
        state: QState,
        view: DecisionTrialView,
        params: QParams,
    ) -> tuple[float, ...]:
        """Compute the model's predicted probability of each available choice.

        This is the softmax decision rule. The agent's Q-value for each
        available option is multiplied by beta (the inverse temperature) and
        then passed through the softmax function, which converts the scaled
        Q-values into a proper probability distribution that sums to 1. A
        higher Q-value produces a higher choice probability; beta controls the
        steepness of that relationship.

        Only the options that are actually available on the current trial are
        scored — unavailable options are ignored.

        Parameters
        ----------
        state
            The agent's current Q-values (beliefs about each option).
        view
            The trial record, including which options were available.
        params
            The agent's fitted parameters (alpha and beta).

        Returns
        -------
        tuple[float, ...]
            Predicted choice probabilities in the same order as
            ``view.available_actions``.
        """

        logits = [params.beta * state.q_values[action] for action in view.available_actions]
        return stable_softmax(logits)

    def update(
        self,
        state: QState,
        view: DecisionTrialView,
        params: QParams,
    ) -> QState:
        """Update beliefs after observing the outcome of a trial.

        This implements the Rescorla-Wagner (delta rule) update. The agent
        compares the reward they actually received to the reward they expected
        (their current Q-value). The difference — called the prediction error —
        is multiplied by alpha and added to the Q-value:

            Q[chosen] = Q[chosen] + alpha * (reward_received - Q[chosen])

        If the reward was better than expected, the Q-value goes up. If it was
        worse, it goes down. Only the Q-value of the chosen option is changed;
        all other options remain exactly as they were.

        If no reward is recorded for this trial (e.g. some task designs omit
        feedback on certain trials), the Q-values are returned unchanged.

        Parameters
        ----------
        state
            The agent's Q-values before this trial's outcome.
        view
            The trial record, including the chosen option and the reward
            received.
        params
            The agent's fitted parameters (alpha controls the update size).

        Returns
        -------
        QState
            Updated Q-values to carry forward into the next trial.
        """

        updated_q_values = list(state.q_values)
        if view.reward is not None:
            assert view.choice is not None
            chosen_action = view.choice
            updated_q_values[chosen_action] += params.alpha * (
                view.reward - updated_q_values[chosen_action]
            )
        return QState(q_values=updated_q_values)
