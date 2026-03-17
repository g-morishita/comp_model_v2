"""Social Q-learning kernel with observed demonstrator outcomes.

This kernel extends the asocial Q-learning rule with a second learning rate for
observed demonstrator outcomes.
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
class SocialQParams:
    """Parsed parameters for the social observed-outcome kernel.

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
class SocialQState:
    """Latent Q-values for a socially informed learning agent.

    Attributes
    ----------
    q_values
        Per-action Q-values indexed by action value.
    """

    q_values: list[float]


class SocialObservedOutcomeQKernel:
    """Q-learning kernel that updates from self and demonstrator outcomes.

    Notes
    -----
    The kernel still consumes only flat decision views. Whether demonstrator
    information appeared before the subject's choice or after the outcome is a
    schema concern handled by extraction.
    """

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return static metadata for the social kernel.

        Returns
        -------
        ModelKernelSpec
            Static kernel specification declaring separate self and social
            learning rates plus a shared inverse temperature.
        """

        return ModelKernelSpec(
            model_id="social_observed_outcome_q",
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

    def parse_params(self, raw: dict[str, float]) -> SocialQParams:
        """Transform unconstrained parameters into typed social parameters.

        Parameters
        ----------
        raw
            Unconstrained parameter values keyed by parameter name.

        Returns
        -------
        SocialQParams
            Typed parameter object after applying the shared transform registry.
        """

        return SocialQParams(
            alpha_self=get_transform("sigmoid").forward(raw["alpha_self"]),
            alpha_other=get_transform("sigmoid").forward(raw["alpha_other"]),
            beta=get_transform("softplus").forward(raw["beta"]),
        )

    def initial_state(self, n_actions: int, params: SocialQParams) -> SocialQState:
        """Construct the initial latent Q-state.

        Parameters
        ----------
        n_actions
            Number of legal actions in the task.
        params
            Parsed kernel parameters.

        Returns
        -------
        SocialQState
            Initial Q-values set to ``0.5``.
        """

        del params
        return SocialQState(q_values=[self.spec().initial_value] * n_actions)

    def action_probabilities(
        self,
        state: SocialQState,
        view: DecisionTrialView,
        params: SocialQParams,
    ) -> tuple[float, ...]:
        """Compute softmax action probabilities for the current state.

        Parameters
        ----------
        state
            Current latent Q-values.
        view
            Extracted decision record.
        params
            Parsed kernel parameters.

        Returns
        -------
        tuple[float, ...]
            Probabilities aligned with ``view.available_actions``.

        Notes
        -----
        Choice probabilities depend only on the agent's current latent Q-values
        and the legal actions in ``view``. Social information influences choice
        indirectly through past updates.
        """

        logits = [params.beta * state.q_values[action] for action in view.available_actions]
        return stable_softmax(logits)

    def next_state(
        self,
        state: SocialQState,
        view: DecisionTrialView,
        params: SocialQParams,
    ) -> SocialQState:
        """Update Q-values from self and social outcomes.

        Parameters
        ----------
        state
            Current latent Q-values.
        view
            Extracted decision record.
        params
            Parsed kernel parameters.

        Returns
        -------
        SocialQState
            Updated latent state.

        Notes
        -----
        The subject's chosen action is updated with ``alpha_self`` when a reward
        is present. If demonstrator action and reward are both present, the
        observed action is updated separately with ``alpha_other``.
        """

        updated_q_values = list(state.q_values)
        if view.reward is not None:
            updated_q_values[view.choice] += params.alpha_self * (
                view.reward - updated_q_values[view.choice]
            )
        if view.social_action is not None and view.social_reward is not None:
            updated_q_values[view.social_action] += params.alpha_other * (
                view.social_reward - updated_q_values[view.social_action]
            )
        return SocialQState(q_values=updated_q_values)
