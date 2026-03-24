"""Asocial RL kernel with asymmetric learning rates.

This kernel extends the standard Rescorla-Wagner update with separate learning
rates for positive and negative prediction errors, allowing the model to learn
differently from better-than-expected and worse-than-expected outcomes.
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
class AsocialRlAsymmetricParams:
    """Parsed parameters for the asocial asymmetric RL kernel.

    Attributes
    ----------
    alpha_pos
        Learning rate for positive prediction errors (reward > Q), in ``(0, 1)``.
    alpha_neg
        Learning rate for negative prediction errors (reward < Q), in ``(0, 1)``.
    beta
        Inverse temperature in ``(0, +inf)``.
    """

    alpha_pos: float
    alpha_neg: float
    beta: float


@dataclass(slots=True)
class AsocialRlAsymmetricState:
    """Latent state for an asocial asymmetric RL agent.

    Attributes
    ----------
    q_values
        Per-action Q-values indexed by action value.
    """

    q_values: list[float]


class AsocialRlAsymmetricKernel(ModelKernel[AsocialRlAsymmetricState, AsocialRlAsymmetricParams]):
    """Asocial RL kernel with separate learning rates for positive and negative RPEs.

    The update rule is:

    .. math::

        \\delta = r - Q[a]

        Q[a] \\leftarrow Q[a] + \\begin{cases}
            \\alpha^+ \\cdot \\delta & \\text{if } \\delta \\ge 0 \\\\
            \\alpha^- \\cdot \\delta & \\text{if } \\delta < 0
        \\end{cases}

    where :math:`\\delta` is the reward prediction error, and :math:`\\alpha^+`
    and :math:`\\alpha^-` are separate learning rates.

    Notes
    -----
    When :math:`\\alpha^+ = \\alpha^-` this model reduces to the standard
    symmetric :class:`~comp_model.models.kernels.AsocialRlKernel`.
    """

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return static metadata for the asocial asymmetric RL kernel.

        Returns
        -------
        ModelKernelSpec
            Static kernel specification declaring three parameters:
            ``alpha_pos`` and ``alpha_neg`` on the unit interval, and ``beta``
            on the positive real line.
        """

        return ModelKernelSpec(
            model_id="asocial_rl_asymmetric",
            parameter_specs=(
                ParameterSpec(
                    name="alpha_pos",
                    transform_id="sigmoid",
                    description="learning rate for positive prediction errors",
                    mle_init=InitSpec(
                        strategy="fixed",
                        kwargs={},
                        default_unconstrained=0.0,
                    ),
                ),
                ParameterSpec(
                    name="alpha_neg",
                    transform_id="sigmoid",
                    description="learning rate for negative prediction errors",
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
        )

    def parse_params(self, raw: dict[str, float]) -> AsocialRlAsymmetricParams:
        """Transform unconstrained parameters into typed kernel parameters.

        Parameters
        ----------
        raw
            Unconstrained parameter values keyed by parameter name.

        Returns
        -------
        AsocialRlAsymmetricParams
            Typed parameter object after applying the shared transform registry.
        """

        transforms = {ps.name: get_transform(ps.transform_id) for ps in self.spec().parameter_specs}
        return AsocialRlAsymmetricParams(
            alpha_pos=transforms["alpha_pos"].forward(raw["alpha_pos"]),
            alpha_neg=transforms["alpha_neg"].forward(raw["alpha_neg"]),
            beta=transforms["beta"].forward(raw["beta"]),
        )

    def initial_state(
        self, n_actions: int, params: AsocialRlAsymmetricParams
    ) -> AsocialRlAsymmetricState:
        """Construct the initial latent state.

        Parameters
        ----------
        n_actions
            Number of legal actions in the task.
        params
            Parsed kernel parameters.

        Returns
        -------
        AsocialRlAsymmetricState
            Initial Q-values set to ``0.5``.
        """

        del params
        return AsocialRlAsymmetricState(q_values=[0.5] * n_actions)

    def action_probabilities(
        self,
        state: AsocialRlAsymmetricState,
        view: DecisionTrialView,
        params: AsocialRlAsymmetricParams,
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

    def update(
        self,
        state: AsocialRlAsymmetricState,
        view: DecisionTrialView,
        params: AsocialRlAsymmetricParams,
    ) -> AsocialRlAsymmetricState:
        """Update the chosen action's Q-value using the signed prediction error.

        Positive prediction errors (:math:`r > Q[a]`) use ``alpha_pos``;
        negative prediction errors (:math:`r < Q[a]`) use ``alpha_neg``.

        Social UPDATE steps (where ``view.actor_id != view.learner_id``) are
        silently skipped, allowing this kernel to be fitted against data
        collected under a social schema for model comparison purposes.

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
        AsocialRlAsymmetricState
            Updated latent state.
        """

        # Asocial model: ignore social UPDATE steps (e.g. when fitted to social
        # schema data for model comparison). Only self-updates have the learner
        # acting as their own actor.
        if view.actor_id != view.learner_id:
            return state

        updated_q_values = list(state.q_values)
        if view.reward is not None:
            assert view.action is not None
            chosen_action = view.action
            delta = view.reward - updated_q_values[chosen_action]
            alpha = params.alpha_pos if delta >= 0 else params.alpha_neg
            updated_q_values[chosen_action] += alpha * delta
        return AsocialRlAsymmetricState(q_values=updated_q_values)
