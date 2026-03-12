"""Asocial Q-learning kernel.

This kernel implements a standard Rescorla-Wagner update with softmax action
selection over latent action values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from comp_model.models.kernels.base import InitSpec, ModelKernelSpec, ParameterSpec, PriorSpec
from comp_model.models.kernels.probabilities import stable_softmax
from comp_model.models.kernels.transforms import get_transform

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class QParams:
    """Parsed parameters for the asocial Q-learning kernel.

    Attributes
    ----------
    alpha
        Learning rate in `(0, 1)`.
    beta
        Inverse temperature in `(0, +inf)`.
    """

    alpha: float
    beta: float


@dataclass(slots=True)
class QState:
    """Latent Q-values for an asocial learning agent.

    Attributes
    ----------
    q_values
        Per-action Q-values indexed by action value.
    """

    q_values: list[float]


class AsocialQLearningKernel:
    """Standard softmax Q-learning kernel for asocial bandit tasks.

    Notes
    -----
    The kernel is schema-agnostic. It relies only on the extracted
    :class:`~comp_model.data.extractors.DecisionTrialView`, so the same kernel
    can be replayed against any task schema that yields compatible asocial
    decision views.
    """

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return static metadata for the asocial kernel.

        Returns
        -------
        ModelKernelSpec
            Static kernel specification declaring two parameters:
            ``alpha`` on the unit interval and ``beta`` on the positive real
            line.
        """

        return ModelKernelSpec(
            model_id="asocial_q_learning",
            parameter_specs=(
                ParameterSpec(
                    name="alpha",
                    transform_id="sigmoid",
                    description="learning rate",
                    prior=PriorSpec(family="normal", kwargs={"mu": 0.0, "sigma": 1.5}),
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
                    prior=PriorSpec(family="normal", kwargs={"mu": 0.0, "sigma": 2.0}),
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
        """Transform unconstrained parameters into typed kernel parameters.

        Parameters
        ----------
        raw
            Unconstrained parameter values keyed by parameter name.

        Returns
        -------
        QParams
            Typed parameter object after applying the shared transform registry.
        """

        return QParams(
            alpha=get_transform("sigmoid").forward(raw["alpha"]),
            beta=get_transform("softplus").forward(raw["beta"]),
        )

    def initial_state(self, n_actions: int, params: QParams) -> QState:
        """Construct the initial latent Q-state.

        Parameters
        ----------
        n_actions
            Number of legal actions in the task.
        params
            Parsed kernel parameters.

        Returns
        -------
        QState
            Initial Q-values set to ``0.5``.
        """

        del params
        return QState(q_values=[0.5] * n_actions)

    def action_probabilities(
        self,
        state: QState,
        view: DecisionTrialView,
        params: QParams,
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
        The kernel scores only the currently legal actions, multiplies their
        latent values by ``beta``, and normalizes them with a numerically stable
        softmax. The returned tuple is aligned exactly with the order of
        ``view.available_actions``.
        """

        logits = [params.beta * state.q_values[action] for action in view.available_actions]
        return stable_softmax(logits)

    def next_state(
        self,
        state: QState,
        view: DecisionTrialView,
        params: QParams,
    ) -> QState:
        """Update the chosen action's Q-value from reward prediction error.

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
        QState
            Updated latent state.

        Notes
        -----
        When a reward is present, the chosen action is updated according to

        ``Q[a] <- Q[a] + alpha * (reward - Q[a])``.

        Unchosen actions are left unchanged.
        """

        updated_q_values = list(state.q_values)
        if view.reward is not None:
            chosen_action = view.choice
            updated_q_values[chosen_action] += params.alpha * (
                view.reward - updated_q_values[chosen_action]
            )
        return QState(q_values=updated_q_values)
