"""Social RL kernel updating from self reward and demonstrator reward.

This kernel extends the asocial RL update with a second learning rate for
observed demonstrator outcomes. The demonstrator's action is used only as an
index to identify which Q-value to update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comp_model.models.kernels.base import ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.probabilities import stable_softmax
from comp_model.models.kernels.transforms import get_transform

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class SocialRlSelfRewardDemoRewardParams:
    """Parsed parameters for the social self-reward + demo-reward kernel.

    Attributes
    ----------
    alpha_self
        Learning rate for self-generated outcomes.
    alpha_other
        Learning rate for demonstrator outcomes.
    beta
        Inverse temperature for choice stochasticity.
    """

    alpha_self: float
    alpha_other: float
    beta: float


@dataclass(slots=True)
class SocialRlSelfRewardDemoRewardState:
    """Latent state for a social self-reward + demo-reward agent.

    Attributes
    ----------
    q_values
        Per-action Q-values indexed by action value.
    """

    q_values: list[float]


class SocialRlSelfRewardDemoRewardKernel:
    """Social RL kernel that updates from self reward and demonstrator reward.

    Notes
    -----
    The demonstrator's action is used only as an index identifying which
    Q-value to update with the demonstrator's reward. The active learning
    signals are the self reward and the demonstrator reward; the demonstrator
    action carries no independent learning signal.

    Whether demonstrator information appeared before the subject's choice or
    after the outcome is a schema concern handled by extraction.
    """

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return static metadata for this kernel.

        Returns
        -------
        ModelKernelSpec
            Static kernel specification declaring separate self and social
            learning rates plus a shared inverse temperature.
        """

        return ModelKernelSpec(
            model_id="social_rl_self_reward_demo_reward",
            parameter_specs=(
                ParameterSpec(
                    name="alpha_self",
                    transform_id="sigmoid",
                    description="self learning rate",
                ),
                ParameterSpec(
                    name="alpha_other",
                    transform_id="sigmoid",
                    description="social learning rate",
                ),
                ParameterSpec(
                    name="beta",
                    transform_id="softplus",
                    description="inverse temperature",
                ),
            ),
            requires_social=True,
            state_reset_policy="per_subject",
        )

    def parse_params(self, raw: dict[str, float]) -> SocialRlSelfRewardDemoRewardParams:
        """Transform unconstrained parameters into typed kernel parameters.

        Parameters
        ----------
        raw
            Unconstrained parameter values keyed by parameter name.

        Returns
        -------
        SocialRlSelfRewardDemoRewardParams
            Typed parameter object after applying the shared transform registry.
        """

        return SocialRlSelfRewardDemoRewardParams(
            alpha_self=get_transform("sigmoid").forward(raw["alpha_self"]),
            alpha_other=get_transform("sigmoid").forward(raw["alpha_other"]),
            beta=get_transform("softplus").forward(raw["beta"]),
        )

    def initial_state(
        self, n_actions: int, params: SocialRlSelfRewardDemoRewardParams
    ) -> SocialRlSelfRewardDemoRewardState:
        """Construct the initial latent state.

        Parameters
        ----------
        n_actions
            Number of legal actions in the task.
        params
            Parsed kernel parameters.

        Returns
        -------
        SocialRlSelfRewardDemoRewardState
            Initial Q-values set to ``0.5``.
        """

        del params
        return SocialRlSelfRewardDemoRewardState(q_values=[self.spec().initial_value] * n_actions)

    def action_probabilities(
        self,
        state: SocialRlSelfRewardDemoRewardState,
        view: DecisionTrialView,
        params: SocialRlSelfRewardDemoRewardParams,
    ) -> tuple[float, ...]:
        """Compute softmax action probabilities for the current state.

        Parameters
        ----------
        state
            Current latent state.
        view
            Extracted decision record.
        params
            Parsed kernel parameters.

        Returns
        -------
        tuple[float, ...]
            Probabilities aligned with ``view.available_actions``.
        """

        logits = [params.beta * state.q_values[action] for action in view.available_actions]
        return stable_softmax(logits)

    def next_state(
        self,
        state: SocialRlSelfRewardDemoRewardState,
        view: DecisionTrialView,
        params: SocialRlSelfRewardDemoRewardParams,
    ) -> SocialRlSelfRewardDemoRewardState:
        """Update Q-values from self reward and demonstrator reward.

        Parameters
        ----------
        state
            Current latent state.
        view
            Extracted decision record.
        params
            Parsed kernel parameters.

        Returns
        -------
        SocialRlSelfRewardDemoRewardState
            Updated latent state.

        Notes
        -----
        The subject's chosen action is updated with ``alpha_self`` when a
        reward is present. If demonstrator action and reward are both present,
        the observed action is updated with ``alpha_other`` using the
        demonstrator reward as the learning signal.
        """

        updated_q_values = list(state.q_values)
        if view.reward is not None:
            assert view.choice is not None
            updated_q_values[view.choice] += params.alpha_self * (
                view.reward - updated_q_values[view.choice]
            )
        if view.social_action is not None and view.social_reward is not None:
            updated_q_values[view.social_action] += params.alpha_other * (
                view.social_reward - updated_q_values[view.social_action]
            )
        return SocialRlSelfRewardDemoRewardState(q_values=updated_q_values)
