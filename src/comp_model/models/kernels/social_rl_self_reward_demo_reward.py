"""Social reinforcement learning kernel — learning from own outcomes and from a demonstrator.

This module extends the basic Q-learning model to capture social learning.
A participant in a social learning experiment can update their beliefs in two
distinct ways on each trial:

1. From their own experience: after receiving a reward, they update their
   estimate of the option they personally chose (controlled by ``alpha_self``).
2. From the demonstrator's experience: after observing the demonstrator's
   choice and reward, they update their estimate of the option the demonstrator
   chose (controlled by ``alpha_other``).

The two updates are independent — both can occur in the same trial, or either
one can occur alone, depending on what information was available.

This model is useful for quantifying how much weight a participant gives to
social information relative to their own direct experience. A participant with
a large ``alpha_other`` relative to ``alpha_self`` is a strong social learner;
one with ``alpha_other`` near zero largely ignores the demonstrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comp_model.models.kernels.base import ModelKernel, ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.probabilities import stable_softmax

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class SocialRlSelfRewardDemoRewardParams:
    """The three free parameters that define a social Q-learning agent.

    Attributes
    ----------
    alpha_self
        Self learning rate — a number strictly between 0 and 1. Controls how
        much the participant updates their beliefs from their own reward on each
        trial. Identical in meaning to the ``alpha`` parameter in the asocial
        Q-learning model.
    alpha_other
        Social learning rate — a number strictly between 0 and 1. Controls how
        much the participant updates their beliefs from the demonstrator's
        reward. A value near 1 means the participant treats the demonstrator's
        outcome almost as strongly as their own; a value near 0 means the
        participant nearly ignores what the demonstrator experienced.
    beta
        Inverse temperature — a positive number. Controls how deterministically
        the participant acts on their current beliefs when choosing between
        options. Identical in meaning to the ``beta`` parameter in the asocial
        model.
    """

    alpha_self: float
    alpha_other: float
    beta: float


@dataclass(slots=True)
class SocialRlSelfRewardDemoRewardState:
    """The agent's current internal beliefs — one value per available option.

    Structurally identical to the asocial ``QState``: a list of Q-values, one
    per option, representing the agent's current estimate of each option's
    reward. The difference is that these values can be updated by both the
    agent's own experience and by observed demonstrator experience.

    Attributes
    ----------
    q_values
        List of Q-values, one per option, indexed by the option's integer code.
        All values start at 0.5 and shift trial by trial as the agent
        accumulates self-generated and socially observed evidence.
    """

    q_values: list[float]


@dataclass(frozen=True)
class SocialRlSelfRewardDemoRewardKernel(
    ModelKernel[SocialRlSelfRewardDemoRewardState, SocialRlSelfRewardDemoRewardParams]
):
    """Q-learning model for a participant who learns from both personal and social experience.

    On each trial this model can update the participant's beliefs in two ways:

    - Personal update: if the participant received a reward, the Q-value for
      the option they chose is nudged toward that reward using ``alpha_self``.
    - Social update: if the demonstrator's choice and reward are available, the
      Q-value for the option the demonstrator chose is nudged toward the
      demonstrator's reward using ``alpha_other``.

    The two updates are applied independently. Both can fire on the same trial
    (e.g. the participant both acted themselves and observed the demonstrator),
    or either one can fire alone.

    Note: the demonstrator's choice matters only to indicate *which* Q-value
    gets the social update. It is the demonstrator's *reward* that drives
    learning, not the fact that the demonstrator happened to pick a particular
    option.

    Whether the participant observes the demonstrator before or after their own
    choice is determined by the experimental design (the trial schema), not by
    this model. The model simply uses whatever social information is present in
    the trial record.

    Attributes
    ----------
    q_init
        Starting value assigned to all Q-values before any learning occurs.
        Defaults to 0.5, representing neutral uncertainty on a 0-1 reward scale.
    """

    q_init: float = 0.5

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return a description of this model's parameters for the fitting machinery.

        Returns
        -------
        ModelKernelSpec
            A record declaring three free parameters (``alpha_self``,
            ``alpha_other``, ``beta``), their constraints, and their starting
            values for numerical optimisation. Also flags that this model
            requires social information (demonstrator action and reward) to be
            present in the trial data.
        """

        return ModelKernelSpec(
            model_id="social_rl_self_reward_demo_reward",
            parameter_specs=(
                ParameterSpec(
                    name="alpha_self",
                    transform_id="sigmoid",
                    description="self learning rate",
                    bounds=(0.0, 1.0),
                ),
                ParameterSpec(
                    name="alpha_other",
                    transform_id="sigmoid",
                    description="social learning rate",
                    bounds=(0.0, 1.0),
                ),
                ParameterSpec(
                    name="beta",
                    transform_id="softplus",
                    description="inverse temperature",
                    bounds=(0.0, None),
                ),
            ),
            requires_social=True,
            required_social_fields=frozenset({"action", "reward"}),
        )

    def parse_params(self, raw: dict[str, float]) -> SocialRlSelfRewardDemoRewardParams:
        """Convert raw optimiser values into interpretable model parameters.

        The optimiser works internally with unconstrained numbers. This method
        applies the appropriate transformations to produce alpha_self and
        alpha_other in (0, 1) and beta as a positive value.

        Parameters
        ----------
        raw
            Dictionary of unconstrained parameter values produced by the
            optimiser, keyed by parameter name.

        Returns
        -------
        SocialRlSelfRewardDemoRewardParams
            Parameter object with all three parameters on their natural scales.
        """

        transforms = self._parameter_transforms()
        return SocialRlSelfRewardDemoRewardParams(
            alpha_self=transforms["alpha_self"].forward(raw["alpha_self"]),
            alpha_other=transforms["alpha_other"].forward(raw["alpha_other"]),
            beta=transforms["beta"].forward(raw["beta"]),
        )

    def initial_state(
        self, n_actions: int, params: SocialRlSelfRewardDemoRewardParams
    ) -> SocialRlSelfRewardDemoRewardState:
        """Create the agent's belief state at the very start of the task.

        All Q-values are initialised to 0.5, reflecting neutral uncertainty
        about every option before any personal or social experience has been
        accumulated.

        Parameters
        ----------
        n_actions
            The number of options available in the task.
        params
            The agent's fitted parameters (not used for initialisation, but
            required by the shared interface).

        Returns
        -------
        SocialRlSelfRewardDemoRewardState
            A state with all Q-values set to 0.5.
        """

        del params
        return SocialRlSelfRewardDemoRewardState(q_values=[self.q_init] * n_actions)

    def action_probabilities(
        self,
        state: SocialRlSelfRewardDemoRewardState,
        view: DecisionTrialView,
        params: SocialRlSelfRewardDemoRewardParams,
    ) -> tuple[float, ...]:
        """Compute the model's predicted probability of each available choice.

        Identical in form to the asocial version: Q-values for the available
        options are scaled by beta and converted to probabilities via softmax.
        The Q-values here, however, reflect both self-generated and socially
        observed experience (depending on what has occurred in prior trials).

        Parameters
        ----------
        state
            The agent's current Q-values (combining personal and social
            learning history).
        view
            The trial record, including which options were available.
        params
            The agent's fitted parameters.

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
        state: SocialRlSelfRewardDemoRewardState,
        view: DecisionTrialView,
        params: SocialRlSelfRewardDemoRewardParams,
    ) -> SocialRlSelfRewardDemoRewardState:
        """Update beliefs after observing the outcomes of a trial.

        Two independent updates may occur:

        Personal update — if the participant received a reward on this trial,
        the Q-value for their chosen option is updated using the Rescorla-Wagner
        rule scaled by ``alpha_self``:

            Q[own_choice] = Q[own_choice] + alpha_self * (own_reward - Q[own_choice])

        Social update — if the demonstrator's choice and reward are both
        available, the Q-value for the demonstrator's chosen option is updated
        using the same rule scaled by ``alpha_other``:

            Q[demo_choice] = Q[demo_choice] + alpha_other * (demo_reward - Q[demo_choice])

        Either, both, or neither update can occur on any given trial, depending
        on whether the corresponding information was present in the experimental
        record.

        Parameters
        ----------
        state
            The agent's Q-values before this trial's outcomes.
        view
            The trial record, including the participant's choice and reward
            (if present) and the demonstrator's choice and reward (if present).
        params
            The agent's fitted parameters (alpha_self and alpha_other control
            the respective update sizes).

        Returns
        -------
        SocialRlSelfRewardDemoRewardState
            Updated Q-values to carry forward into the next trial.
        """

        updated_q_values = list(state.q_values)
        if view.actor_id == view.learner_id:
            # Self-update: the learner is learning from their own experience.
            assert view.action is not None and view.reward is not None
            updated_q_values[view.action] += params.alpha_self * (
                view.reward - updated_q_values[view.action]
            )
        else:
            # Social update: the learner is learning from observing another agent.
            if view.action is not None and view.reward is not None:
                updated_q_values[view.action] += params.alpha_other * (
                    view.reward - updated_q_values[view.action]
                )
        return SocialRlSelfRewardDemoRewardState(q_values=updated_q_values)
