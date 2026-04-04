"""Sticky social RL mixture kernel.

This module extends the self-reward + demonstrator-mixture model with a
stickiness (perseveration) term that biases the agent to repeat its own
previous choice.

The model maintains two independent value systems:

- ``v_outcome``: updated by self reward and demonstrator reward
- ``v_tendency``: updated by demonstrator action frequency

At decision time the systems are combined via ``w_imitation`` and passed
through a softmax. A separate ``stickiness`` term is then added to the logit
of the action most recently chosen by the subject:

    logit[a] = beta * (
        w_imitation * v_tendency[a] + (1 - w_imitation) * v_outcome[a]
    ) + stickiness * I[a == last_self_action]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comp_model.models.kernels.base import ModelKernel, ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.probabilities import stable_softmax

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class SocialRlSelfRewardDemoMixtureStickyParams:
    """Free parameters for the sticky mixture social RL kernel.

    Attributes
    ----------
    alpha_self
        Learning rate for self outcome updates; in (0, 1).
    alpha_other_outcome
        Learning rate for demonstrator outcome updates to ``v_outcome``; in (0, 1).
    alpha_other_action
        Learning rate for demonstrator action updates to ``v_tendency``; in (0, 1).
    w_imitation
        Mixing weight for imitation at decision time; in (0, 1).
    beta
        Inverse temperature for the softmax choice rule; positive.
    stickiness
        Perseveration term added to the logit of the subject's previous own
        choice. Positive values bias repeats; negative values bias switches.
    """

    alpha_self: float
    alpha_other_outcome: float
    alpha_other_action: float
    w_imitation: float
    beta: float
    stickiness: float


@dataclass(slots=True)
class SocialRlSelfRewardDemoMixtureStickyState:
    """Latent state for the sticky mixture social RL kernel.

    Attributes
    ----------
    v_outcome
        Outcome-based value estimates, updated by self and demonstrator rewards.
    v_tendency
        Action tendency estimates, updated by demonstrator action frequency.
    last_self_action
        The subject's most recent own choice. ``None`` until the first
        self-update is observed.
    """

    v_outcome: list[float]
    v_tendency: list[float]
    last_self_action: int | None = None


@dataclass(frozen=True)
class SocialRlSelfRewardDemoMixtureStickyKernel(
    ModelKernel[
        SocialRlSelfRewardDemoMixtureStickyState,
        SocialRlSelfRewardDemoMixtureStickyParams,
    ]
):
    """Sticky social RL kernel combining outcome-, action-, and perseveration-based influences.

    Maintains the same two value systems as
    :class:`SocialRlSelfRewardDemoMixtureKernel` and adds a third choice
    influence:

    - ``v_outcome``: updated by self reward and demonstrator reward
    - ``v_tendency``: updated by demonstrator action frequency
    - ``last_self_action``: adds a ``stickiness`` bonus to the previously
      self-chosen action at the next decision

    Attributes
    ----------
    v_outcome_init
        Starting value for outcome-tracker values. Defaults to 0.5.
    """

    v_outcome_init: float = 0.5

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return the kernel specification for the sticky mixture model.

        Returns
        -------
        ModelKernelSpec
            Specification declaring six free parameters, their constraints,
            and that this model requires social information.
        """

        return ModelKernelSpec(
            model_id="social_rl_self_reward_demo_mixture_sticky",
            parameter_specs=(
                ParameterSpec(
                    name="alpha_self",
                    transform_id="sigmoid",
                    description="learning rate for self outcome",
                    bounds=(0.0, 1.0),
                ),
                ParameterSpec(
                    name="alpha_other_outcome",
                    transform_id="sigmoid",
                    description="learning rate for demonstrator outcome",
                    bounds=(0.0, 1.0),
                ),
                ParameterSpec(
                    name="alpha_other_action",
                    transform_id="sigmoid",
                    description="learning rate for demonstrator action tendency",
                    bounds=(0.0, 1.0),
                ),
                ParameterSpec(
                    name="w_imitation",
                    transform_id="sigmoid",
                    description="mixing weight for imitation at decision time",
                    bounds=(0.0, 1.0),
                ),
                ParameterSpec(
                    name="beta",
                    transform_id="softplus",
                    description="inverse temperature",
                    bounds=(0.0, None),
                ),
                ParameterSpec(
                    name="stickiness",
                    transform_id="identity",
                    description="logit bias toward repeating the previous own choice",
                ),
            ),
            requires_social=True,
            required_social_fields=frozenset({"action", "reward"}),
        )

    def parse_params(self, raw: dict[str, float]) -> SocialRlSelfRewardDemoMixtureStickyParams:
        """Convert raw optimiser values into interpretable model parameters.

        Parameters
        ----------
        raw
            Dictionary of unconstrained parameter values keyed by parameter name.

        Returns
        -------
        SocialRlSelfRewardDemoMixtureStickyParams
            Parameter object with all values mapped onto their natural scales.
        """

        transforms = self._parameter_transforms()
        return SocialRlSelfRewardDemoMixtureStickyParams(
            alpha_self=transforms["alpha_self"].forward(raw["alpha_self"]),
            alpha_other_outcome=transforms["alpha_other_outcome"].forward(
                raw["alpha_other_outcome"]
            ),
            alpha_other_action=transforms["alpha_other_action"].forward(raw["alpha_other_action"]),
            w_imitation=transforms["w_imitation"].forward(raw["w_imitation"]),
            beta=transforms["beta"].forward(raw["beta"]),
            stickiness=transforms["stickiness"].forward(raw["stickiness"]),
        )

    def initial_state(
        self,
        n_actions: int,
        params: SocialRlSelfRewardDemoMixtureStickyParams,
    ) -> SocialRlSelfRewardDemoMixtureStickyState:
        """Create the agent's belief state at the very start of the task.

        ``v_outcome`` is initialised to ``v_outcome_init`` (neutral reward
        expectation). ``v_tendency`` is initialised to a uniform distribution
        ``1 / n_actions`` (no prior tendency toward any action).
        ``last_self_action`` starts as ``None`` until the first self-generated
        update is observed.

        Parameters
        ----------
        n_actions
            Number of available actions.
        params
            Parsed kernel parameters (unused for initialisation).

        Returns
        -------
        SocialRlSelfRewardDemoMixtureStickyState
            Initial state with neutral outcome values, uniform action
            tendencies, and no previous self action.
        """

        del params
        return SocialRlSelfRewardDemoMixtureStickyState(
            v_outcome=[self.v_outcome_init] * n_actions,
            v_tendency=[1.0 / n_actions] * n_actions,
            last_self_action=None,
        )

    def action_probabilities(
        self,
        state: SocialRlSelfRewardDemoMixtureStickyState,
        view: DecisionTrialView,
        params: SocialRlSelfRewardDemoMixtureStickyParams,
    ) -> tuple[float, ...]:
        """Compute choice probabilities by combining both value systems and stickiness.

        The outcome and action-tendency systems are mixed via ``w_imitation``
        before applying the softmax. If an action matches
        ``state.last_self_action``, its logit receives an additive
        ``stickiness`` bonus:

            combined[a] = w_imitation * v_tendency[a] + (1 - w_imitation) * v_outcome[a]
            logit[a] = beta * combined[a] + stickiness * I[a == last_self_action]

        Parameters
        ----------
        state
            Current latent state containing ``v_outcome``, ``v_tendency``,
            and the most recent self-chosen action if one exists.
        view
            Trial observation including available actions.
        params
            Parsed kernel parameters.

        Returns
        -------
        tuple[float, ...]
            Probability of each action in ``view.available_actions``.
        """

        logits: list[float] = []
        for action in view.available_actions:
            combined_value = (
                params.w_imitation * state.v_tendency[action]
                + (1 - params.w_imitation) * state.v_outcome[action]
            )
            logit = params.beta * combined_value
            if state.last_self_action == action:
                logit += params.stickiness
            logits.append(logit)
        return stable_softmax(logits)

    def observe_decision(
        self,
        state: SocialRlSelfRewardDemoMixtureStickyState,
        view: DecisionTrialView,
        params: SocialRlSelfRewardDemoMixtureStickyParams,
    ) -> SocialRlSelfRewardDemoMixtureStickyState:
        """Store the participant's most recent own choice at decision time."""

        del params
        if view.actor_id != view.learner_id or view.action is None:
            return state

        return SocialRlSelfRewardDemoMixtureStickyState(
            v_outcome=list(state.v_outcome),
            v_tendency=list(state.v_tendency),
            last_self_action=view.action,
        )

    def update(
        self,
        state: SocialRlSelfRewardDemoMixtureStickyState,
        view: DecisionTrialView,
        params: SocialRlSelfRewardDemoMixtureStickyParams,
    ) -> SocialRlSelfRewardDemoMixtureStickyState:
        """Update both value systems and stickiness state from the trial observation.

        Self UPDATE (``actor_id == learner_id``):
            ``v_outcome[action] += alpha_self * (reward - v_outcome[action])``
            ``last_self_action = action``

        Social UPDATE (``actor_id != learner_id``):
            ``v_outcome[action] += alpha_other_outcome * (reward - v_outcome[action])``
            ``v_tendency[chosen]   += alpha_other_action * (1 - v_tendency[chosen])``
            ``v_tendency[unchosen] += alpha_other_action * (0 - v_tendency[unchosen])``

        Social updates do not modify ``last_self_action``; the stickiness
        memory reflects only the subject's own most recent action.

        Parameters
        ----------
        state
            Current latent state.
        view
            Trial observation.
        params
            Parsed kernel parameters.

        Returns
        -------
        SocialRlSelfRewardDemoMixtureStickyState
            Updated state with new ``v_outcome``, ``v_tendency``, and
            ``last_self_action`` values.
        """

        updated_v_outcome = list(state.v_outcome)
        updated_v_tendency = list(state.v_tendency)
        last_self_action = state.last_self_action

        if view.actor_id == view.learner_id:
            assert view.action is not None and view.reward is not None
            updated_v_outcome[view.action] += params.alpha_self * (
                view.reward - updated_v_outcome[view.action]
            )
            last_self_action = view.action
        else:
            if view.action is not None and view.reward is not None:
                updated_v_outcome[view.action] += params.alpha_other_outcome * (
                    view.reward - updated_v_outcome[view.action]
                )
                for action in range(len(updated_v_tendency)):
                    target = 1.0 if action == view.action else 0.0
                    updated_v_tendency[action] += params.alpha_other_action * (
                        target - updated_v_tendency[action]
                    )

        return SocialRlSelfRewardDemoMixtureStickyState(
            v_outcome=updated_v_outcome,
            v_tendency=updated_v_tendency,
            last_self_action=last_self_action,
        )
