"""Social RL self-reward + demo-action mixture kernel.

This module implements a mixture model that maintains two independent value
systems updated by different learning signals:

- ``v_outcome``: updated by the subject's own reward (self-reward learning)
- ``v_tendency``: updated by demonstrator action frequency (action tendency tracker)

At decision time the two systems are combined via a mixing weight ``w_imitation``:

    combined[a] = w_imitation * v_tendency[a] + (1 - w_imitation) * v_outcome[a]

This kernel differs from :class:`SocialRlSelfRewardDemoMixtureKernel` in that it
does **not** learn from the demonstrator's reward — the social UPDATE path only
updates ``v_tendency``, leaving ``v_outcome`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comp_model.models.kernels.base import ModelKernel, ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.probabilities import stable_softmax
from comp_model.models.kernels.transforms import get_transform

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class SocialRlSelfRewardDemoActionMixtureParams:
    """Free parameters for the self-reward + demo-action mixture kernel.

    Attributes
    ----------
    alpha_self
        Learning rate for the subject's own outcome updates to ``v_outcome``; in (0, 1).
    alpha_other_action
        Learning rate for demonstrator action updates to ``v_tendency``; in (0, 1).
    w_imitation
        Mixing weight for imitation at decision time; in (0, 1).
        ``w_imitation=0`` means decisions are driven purely by ``v_outcome``;
        ``w_imitation=1`` means decisions are driven purely by ``v_tendency``.
    beta
        Inverse temperature for the softmax choice rule; positive.
    """

    alpha_self: float
    alpha_other_action: float
    w_imitation: float
    beta: float


@dataclass(slots=True)
class SocialRlSelfRewardDemoActionMixtureState:
    """Latent state for the self-reward + demo-action mixture kernel.

    Attributes
    ----------
    v_outcome
        Outcome-based value estimates, updated by the subject's own rewards only.
    v_tendency
        Action tendency estimates, updated by demonstrator action frequency.
    """

    v_outcome: list[float]
    v_tendency: list[float]


@dataclass(frozen=True)
class SocialRlSelfRewardDemoActionMixtureKernel(
    ModelKernel[SocialRlSelfRewardDemoActionMixtureState, SocialRlSelfRewardDemoActionMixtureParams]
):
    """Self-reward + demo-action mixture kernel.

    Maintains two independent value systems combined at decision time:

    - ``v_outcome``: updated by the subject's own reward only
    - ``v_tendency``: updated by demonstrator action frequency

    This kernel does **not** learn from the demonstrator's reward.
    Social UPDATE steps only update ``v_tendency``.

    Attributes
    ----------
    v_outcome_init
        Starting value for outcome-tracker values. Defaults to 0.5,
        representing neutral uncertainty on a 0-1 reward scale.
    """

    v_outcome_init: float = 0.5

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return the kernel specification.

        Returns
        -------
        ModelKernelSpec
            Specification declaring four free parameters, their constraints,
            and that this model requires social information (action only).
        """
        return ModelKernelSpec(
            model_id="social_rl_self_reward_demo_action_mixture",
            parameter_specs=(
                ParameterSpec(
                    name="alpha_self",
                    transform_id="sigmoid",
                    description="learning rate for self outcome",
                ),
                ParameterSpec(
                    name="alpha_other_action",
                    transform_id="sigmoid",
                    description="learning rate for demonstrator action tendency",
                ),
                ParameterSpec(
                    name="w_imitation",
                    transform_id="sigmoid",
                    description="mixing weight for imitation at decision time",
                ),
                ParameterSpec(
                    name="beta",
                    transform_id="softplus",
                    description="inverse temperature",
                ),
            ),
            requires_social=True,
            required_social_fields=frozenset({"action"}),
        )

    def parse_params(self, raw: dict[str, float]) -> SocialRlSelfRewardDemoActionMixtureParams:
        """Convert raw optimiser values into interpretable model parameters.

        Parameters
        ----------
        raw
            Dictionary of unconstrained parameter values keyed by parameter name.

        Returns
        -------
        SocialRlSelfRewardDemoActionMixtureParams
            Parameter object with all four parameters on their natural scales.
        """
        transforms = {ps.name: get_transform(ps.transform_id) for ps in self.spec().parameter_specs}
        return SocialRlSelfRewardDemoActionMixtureParams(
            alpha_self=transforms["alpha_self"].forward(raw["alpha_self"]),
            alpha_other_action=transforms["alpha_other_action"].forward(raw["alpha_other_action"]),
            w_imitation=transforms["w_imitation"].forward(raw["w_imitation"]),
            beta=transforms["beta"].forward(raw["beta"]),
        )

    def initial_state(
        self, n_actions: int, params: SocialRlSelfRewardDemoActionMixtureParams
    ) -> SocialRlSelfRewardDemoActionMixtureState:
        """Create the agent's belief state at the very start of the task.

        ``v_outcome`` is initialised to ``v_outcome_init`` (neutral reward
        expectation). ``v_tendency`` is initialised to a uniform distribution
        ``1 / n_actions`` (no prior preference for any action).

        Parameters
        ----------
        n_actions
            Number of available actions.
        params
            Parsed kernel parameters (unused for initialisation).

        Returns
        -------
        SocialRlSelfRewardDemoActionMixtureState
            Initial state with neutral outcome values and uniform action tendencies.
        """
        del params
        return SocialRlSelfRewardDemoActionMixtureState(
            v_outcome=[self.v_outcome_init] * n_actions,
            v_tendency=[1.0 / n_actions] * n_actions,
        )

    def action_probabilities(
        self,
        state: SocialRlSelfRewardDemoActionMixtureState,
        view: DecisionTrialView,
        params: SocialRlSelfRewardDemoActionMixtureParams,
    ) -> tuple[float, ...]:
        """Compute choice probabilities by combining both value systems.

        The two systems are mixed via ``w_imitation`` before applying the softmax:

            combined[a] = w_imitation * v_tendency[a] + (1 - w_imitation) * v_outcome[a]

        Parameters
        ----------
        state
            Current latent state containing ``v_outcome`` and ``v_tendency``.
        view
            Trial observation including available actions.
        params
            Parsed kernel parameters.

        Returns
        -------
        tuple[float, ...]
            Probability of each action in ``view.available_actions``.
        """
        combined = [
            params.w_imitation * state.v_tendency[a] + (1 - params.w_imitation) * state.v_outcome[a]
            for a in view.available_actions
        ]
        return stable_softmax([params.beta * v for v in combined])

    def update(
        self,
        state: SocialRlSelfRewardDemoActionMixtureState,
        view: DecisionTrialView,
        params: SocialRlSelfRewardDemoActionMixtureParams,
    ) -> SocialRlSelfRewardDemoActionMixtureState:
        """Update value systems from the trial observation.

        Self UPDATE (``actor_id == learner_id``):
            ``v_outcome[action] += alpha_self * (reward - v_outcome[action])``

        Social UPDATE (``actor_id != learner_id``):
            ``v_tendency[chosen]   += alpha_other_action * (1 - v_tendency[chosen])``
            ``v_tendency[unchosen] += alpha_other_action * (0 - v_tendency[unchosen])``

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
        SocialRlSelfRewardDemoActionMixtureState
            Updated state with new ``v_outcome`` and/or ``v_tendency`` values.
        """
        updated_v_outcome = list(state.v_outcome)
        updated_v_tendency = list(state.v_tendency)

        if view.actor_id == view.learner_id:
            assert view.action is not None and view.reward is not None
            updated_v_outcome[view.action] += params.alpha_self * (
                view.reward - updated_v_outcome[view.action]
            )
        else:
            if view.action is not None:
                for a in view.available_actions:
                    target = 1.0 if a == view.action else 0.0
                    updated_v_tendency[a] += params.alpha_other_action * (
                        target - updated_v_tendency[a]
                    )

        return SocialRlSelfRewardDemoActionMixtureState(
            v_outcome=updated_v_outcome,
            v_tendency=updated_v_tendency,
        )
