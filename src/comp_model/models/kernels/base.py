"""Backend-agnostic kernel metadata and protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Mapping

    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class PriorSpec:
    """Prior metadata for a model parameter.

    Attributes
    ----------
    family
        Prior family identifier, such as ``"normal"``.
    kwargs
        Hyperparameters for the prior family.
    parameterization
        Scale on which the prior is defined.
    """

    family: str
    kwargs: Mapping[str, float]
    parameterization: str = "unconstrained"


@dataclass(frozen=True, slots=True)
class InitSpec:
    """Initialization metadata for MLE optimization.

    Attributes
    ----------
    strategy
        Initialization strategy identifier.
    kwargs
        Strategy-specific keyword arguments.
    default_unconstrained
        Default unconstrained starting value.
    """

    strategy: str
    kwargs: Mapping[str, float]
    default_unconstrained: float = 0.0


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Metadata for one free model parameter.

    Attributes
    ----------
    name
        Parameter name exposed to inference code.
    transform_id
        Identifier in the transform registry.
    description
        Human-readable description of the parameter.
    prior
        Optional prior metadata for Bayesian inference.
    mle_init
        Optional initialization metadata for MLE.
    """

    name: str
    transform_id: str
    description: str = ""
    prior: PriorSpec | None = None
    mle_init: InitSpec | None = None


@dataclass(frozen=True, slots=True)
class ModelKernelSpec:
    """Static metadata describing a model kernel.

    Attributes
    ----------
    model_id
        Stable identifier for the kernel.
    parameter_specs
        Ordered parameter metadata for the kernel.
    requires_social
        Whether the kernel reads social fields from decision views.
    n_actions
        Optional fixed action count. ``None`` means infer from data.
    state_reset_policy
        Policy for resetting kernel state, either ``"per_subject"`` or ``"per_block"``.
    description
        Human-readable model description.
    """

    model_id: str
    parameter_specs: tuple[ParameterSpec, ...]
    requires_social: bool = False
    n_actions: int | None = None
    state_reset_policy: str = "per_subject"
    description: str = ""


StateT = TypeVar("StateT")
ParamsT = TypeVar("ParamsT")


class ModelKernel(Protocol, Generic[StateT, ParamsT]):
    """Protocol shared by all backend-agnostic model kernels."""

    @classmethod
    def spec(cls) -> ModelKernelSpec:
        """Return static kernel metadata.

        Returns
        -------
        ModelKernelSpec
            Kernel specification used by inference code.
        """

        ...

    def parse_params(self, raw: dict[str, float]) -> ParamsT:
        """Convert unconstrained parameter values into typed parameters.

        Parameters
        ----------
        raw
            Raw unconstrained parameter values keyed by parameter name.

        Returns
        -------
        ParamsT
            Parsed parameter object for the kernel.
        """

        ...

    def initial_state(self, n_actions: int, params: ParamsT) -> StateT:
        """Construct the initial latent state.

        Parameters
        ----------
        n_actions
            Number of legal actions in the task.
        params
            Parsed kernel parameters.

        Returns
        -------
        StateT
            Initial latent state.
        """

        ...

    def action_probabilities(
        self,
        state: StateT,
        view: DecisionTrialView,
        params: ParamsT,
    ) -> tuple[float, ...]:
        """Return action probabilities for the current decision view.

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

        ...

    def next_state(
        self,
        state: StateT,
        view: DecisionTrialView,
        params: ParamsT,
    ) -> StateT:
        """Update the latent state after the decision outcome.

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
        StateT
            Updated latent state.
        """

        ...
