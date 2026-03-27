"""Definitions of what a computational model ("kernel") must be able to do.

A "kernel" is the core of a computational model: it encodes the learning
rule and the choice rule for a single participant. Every kernel must answer
exactly three questions:

1. What is the participant's internal state at the very start of the task
   (e.g. Q-values all equal to 0.5)?
2. Given the participant's current internal state, what is the probability
   of each available action on this trial?
3. After the participant chooses and receives an outcome, what is their
   updated internal state?

This module defines the shared interface (``ModelKernel``) that every kernel
must implement, plus supporting metadata classes that describe a kernel's
parameters and how the fitting machinery should handle them.

Kernels are deliberately kept separate from task structure and fitting
details. A kernel only ever sees a compact summary of one decision trial
(a ``DecisionTrialView``). It never inspects raw events, trial schemas, or
any fitting-backend specifics (e.g. Stan code). This separation means you
can swap fitting backends without changing the model, and define new models
without touching the infrastructure.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

if TYPE_CHECKING:
    from comp_model.data.extractors import DecisionTrialView


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Description of one free parameter in the model.

    Each free parameter — for example a learning rate or an inverse
    temperature — has a ``ParameterSpec`` that tells the fitting machinery
    everything it needs to know: what the parameter is called, how to
    transform raw (unconstrained) values into meaningful (constrained) values,
    and which constrained values are considered valid for the model.

    Attributes
    ----------
    name
        The parameter's name as it appears in results tables and in code
        (e.g. ``"alpha"`` for a learning rate).
    transform_id
        A short label that points to the mathematical function used to map
        unconstrained numbers onto the parameter's valid range. Two common
        examples:

        - ``"sigmoid"``: maps any real number to the interval (0, 1),
          suitable for rates and probabilities.
        - ``"softplus"``: maps any real number to positive values,
          suitable for parameters that must be greater than zero.

        The same label is used both when parsing parameters in Python and
        when generating the corresponding Stan code, so changing it updates
        both simultaneously.
    description
        A plain-English explanation of what the parameter represents, used
        in documentation and model summaries.
    bounds
        Optional lower and upper bounds on the parameter's constrained scale.
        Use ``None`` for an open side, for example ``(0.0, None)`` for a
        strictly positive parameter or ``(0.0, 1.0)`` for a probability-like
        parameter. These are model properties, not optimizer settings.
    """

    name: str
    transform_id: str
    description: str = ""
    bounds: tuple[float | None, float | None] | None = None

    def __post_init__(self) -> None:
        """Validate optional constrained bounds."""

        if self.bounds is None:
            return
        lower, upper = self.bounds
        if lower is None and upper is None:
            raise ValueError("bounds must specify at least one side or be set to None")
        if lower is not None and not math.isfinite(lower):
            raise ValueError("lower bound must be finite")
        if upper is not None and not math.isfinite(upper):
            raise ValueError("upper bound must be finite")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("lower bound must be smaller than upper bound")


@dataclass(frozen=True, slots=True)
class ModelKernelSpec:
    """The identity card for a computational model.

    This class holds all the static, descriptive information about a kernel
    that the fitting and simulation infrastructure needs to discover: what the
    model is called, what parameters it has, whether it needs social
    information, and how it manages memory across task blocks.

    Think of it as the "spec sheet" read by inference code before any data
    are ever touched. The kernel itself (the actual learning and choice
    equations) is defined separately; this class just describes it.

    Attributes
    ----------
    model_id
        A stable, unique name for the model (e.g. ``"rescorla_wagner"``).
        Used as a key in result files and Stan code.
    parameter_specs
        An ordered list of :class:`ParameterSpec` objects, one per free
        parameter. The order matters — it determines the order of columns
        in parameter tables and of entries in Stan parameter blocks.
    requires_social
        Set to ``True`` if this model uses information about another
        agent's choices or outcomes (i.e. it is a social-learning model).
        This tells the simulation engine and inference code to expect and
        provide social fields.
    required_social_fields
        The set of demonstrator-outcome fields that the kernel consumes
        during social UPDATE steps.  Valid entries are ``"action"`` and
        ``"reward"``.  Only meaningful when ``requires_social`` is
        ``True``.  Used by compatibility checks to verify that the trial
        schema exposes enough information for the kernel to learn from.
        An empty set (the default) means the kernel does not need any
        social information.
    n_actions
        The number of response options the model expects. If ``None``,
        this is inferred from the data. Override with a fixed integer if
        the model has a hard-coded action count.
    state_reset_policy
        Controls how the model's internal state (e.g. Q-values) is
        managed across task blocks:

        - ``"per_block"`` *(default)*: state is reset to initial values
          at the start of each new block; each block begins with a blank
          slate.
        - ``"continuous"``: state accumulates across all blocks; learning
          carries over from one block to the next (e.g. multi-condition
          within-subject designs).
    description
        A plain-English summary of the model, used in reports and logs.
    """

    model_id: str
    parameter_specs: tuple[ParameterSpec, ...]
    requires_social: bool = False
    required_social_fields: frozenset[str] = field(default_factory=lambda: frozenset[str]())
    n_actions: int | None = None
    state_reset_policy: Literal["per_block", "continuous"] = "per_block"
    description: str = ""


StateT = TypeVar("StateT")
ParamsT = TypeVar("ParamsT")


class ModelKernel(ABC, Generic[StateT, ParamsT]):
    """The interface that every computational model must implement.

    A ``ModelKernel`` is the heart of a computational model. It encodes two
    psychological mechanisms:

    - **Learning rule**: how the participant updates their internal
      representation of the world (e.g. Q-values) after each outcome.
    - **Choice rule**: how the participant translates their current
      representation into a probability distribution over available actions.

    Every kernel must answer three questions (the three core methods):

    1. :meth:`initial_state` — what is the participant's starting state?
    2. :meth:`action_probabilities` — given the current state, how likely is
       each action?
    3. :meth:`update` — given the outcome, what is the updated state?

    Two additional methods are needed by the fitting and simulation machinery:

    - :meth:`spec` — returns the model's identity card
      (:class:`ModelKernelSpec`).
    - :meth:`parse_params` — converts raw numerical values from the optimiser
      into the structured parameter object the model expects.

    Kernels are deliberately kept narrow: they only ever see a compact summary
    of one decision trial (:class:`~comp_model.data.extractors.DecisionTrialView`).
    They never inspect raw event logs, trial schemas, or fitting-backend
    details. This keeps the model definitions clean and portable.
    """

    @classmethod
    @abstractmethod
    def spec(cls) -> ModelKernelSpec:
        """Return the identity card (static metadata) for this model.

        Returns
        -------
        ModelKernelSpec
            Describes the model's name, free parameters, social requirements,
            and state reset policy. Read by simulation and inference code
            before any data are processed.
        """

        ...

    @abstractmethod
    def parse_params(self, raw: dict[str, float]) -> ParamsT:
        """Convert raw numbers from the optimiser into this model's parameter object.

        The optimiser works with plain floating-point numbers, often on an
        unconstrained scale. This method applies the appropriate transforms
        (e.g. sigmoid for learning rate, softplus for inverse temperature)
        and packages the result into the typed parameter structure the model
        uses internally.

        Parameters
        ----------
        raw
            A dictionary mapping parameter names to their current
            unconstrained values (as supplied by the optimiser or sampler).

        Returns
        -------
        ParamsT
            A typed parameter object ready to be passed to
            :meth:`initial_state`, :meth:`action_probabilities`, and
            :meth:`update`.
        """

        ...

    @abstractmethod
    def initial_state(self, n_actions: int, params: ParamsT) -> StateT:
        """Create the participant's blank starting state before any learning.

        Called once at the beginning of the task (and again at the start of
        each block if ``state_reset_policy == "per_block"``). Typically
        initialises Q-values or other internal quantities to their prior
        values.

        Parameters
        ----------
        n_actions
            How many response options exist in the task (e.g. 2 for a
            two-armed bandit).
        params
            The participant's parameter values, which may influence the
            starting state (e.g. an initial-value parameter).

        Returns
        -------
        StateT
            The initial internal state, ready for the first trial.
        """

        ...

    @abstractmethod
    def action_probabilities(
        self,
        state: StateT,
        view: DecisionTrialView,
        params: ParamsT,
    ) -> tuple[float, ...]:
        """Compute the probability of each available action on this trial.

        This is the choice rule. It translates the participant's current
        internal state (e.g. Q-values) into a probability distribution over
        actions, typically using a softmax function governed by the inverse
        temperature parameter.

        Parameters
        ----------
        state
            The participant's current internal state (e.g. Q-values).
        view
            A compact summary of the current decision trial, including which
            actions are available and the trial index.
        params
            The participant's parameter values.

        Returns
        -------
        tuple[float, ...]
            A probability for each action in ``view.available_actions``,
            in the same order. Probabilities sum to 1. Only available actions
            appear; unavailable actions are excluded entirely.
        """

        ...

    @abstractmethod
    def update(
        self,
        state: StateT,
        view: DecisionTrialView,
        params: ParamsT,
    ) -> StateT:
        """Update the participant's internal state after observing an outcome.

        This is the learning rule. It incorporates the new information from
        the trial (e.g. reward received, or a demonstrator's choice) and
        returns the updated state that will be used on the next trial.

        Parameters
        ----------
        state
            The participant's internal state before this update.
        view
            A compact summary of the trial outcome, containing the choice
            made, the reward received, and (for social-learning models) the
            demonstrator's choice and reward if available.
        params
            The participant's parameter values.

        Returns
        -------
        StateT
            The updated internal state, incorporating whatever information
            was present in ``view``.
        """

        ...
