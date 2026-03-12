"""Schema-driven extraction from event traces to model-facing decision views.

This module bridges the canonical event hierarchy and backend-agnostic model
kernels. Extraction is positional and schema-specific, while the resulting
decision views are flat and order-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from comp_model._defaults import empty_mapping
from comp_model.data.schema import EventPhase, Trial

if TYPE_CHECKING:
    from collections.abc import Mapping

    from comp_model.tasks.schemas import TrialSchema


@dataclass(frozen=True, slots=True)
class DecisionTrialView:
    """Flat per-decision record consumed by model kernels.

    Attributes
    ----------
    trial_index
        Index of the source trial within its block.
    available_actions
        Legal actions at the decision point.
    choice
        Chosen action value.
    reward
        Observed reward, if present.
    observation
        Subject-facing observation payload.
    social_action
        Observed demonstrator action, if present.
    social_reward
        Observed demonstrator reward, if present.
    metadata
        Additional extractor metadata.

    Notes
    -----
    Kernels consume :class:`DecisionTrialView` rather than :class:`Event`,
    :class:`Trial`, or :class:`TrialSchema`. This allows the same kernel to work
    with different event orders as long as the extractor produces the same flat
    fields.
    """

    trial_index: int
    available_actions: tuple[int, ...]
    choice: int
    reward: float | None = None
    observation: Mapping[str, Any] = field(default_factory=empty_mapping)
    social_action: int | None = None
    social_reward: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=empty_mapping)


def extract_decision_views(trial: Trial, schema: TrialSchema) -> tuple[DecisionTrialView, ...]:
    """Extract flat decision records from a schema-validated trial.

    Parameters
    ----------
    trial
        Trial whose events should be extracted.
    schema
        Schema defining the positional meaning of each event.

    Returns
    -------
    tuple[DecisionTrialView, ...]
        One flat record for each decision step in the schema.

    Notes
    -----
    The extraction algorithm follows the implementation plan directly:

    1. validate the trial against ``schema``,
    2. iterate over each DECISION step declared by the schema,
    3. scan the whole trial positionally to find the matching subject INPUT,
       any demonstrator INPUT, and the OUTCOME linked by ``node_id``, and
    4. emit one order-agnostic :class:`DecisionTrialView` for that decision.

    ``node_id`` is only used for structural linking between a decision and its
    outcome. Social information is detected from the schema's ``actor_id``
    rather than from special node-name conventions.

    Raises
    ------
    ValueError
        Raised when the schema is violated or when no subject INPUT can be
        found for a declared decision step.
    """

    schema.validate_trial(trial)

    events = trial.events
    steps = schema.steps
    views: list[DecisionTrialView] = []

    for decision_step_index in schema.decision_step_indices:
        decision_step = steps[decision_step_index]
        decision_event = events[decision_step_index]
        decision_actor = decision_step.actor_id
        decision_node = decision_step.node_id

        available_actions: tuple[int, ...] | None = None
        observation: dict[str, Any] = {}
        social_action: int | None = None
        social_reward: float | None = None
        reward: float | None = None

        for event, step in zip(events, steps, strict=True):
            if step.phase == EventPhase.INPUT:
                if step.actor_id == decision_actor:
                    if step.node_id == decision_node:
                        available_actions = tuple(event.payload["available_actions"])
                        raw_observation = event.payload.get("observation")
                        if isinstance(raw_observation, dict):
                            observation = dict(raw_observation)
                        elif raw_observation is not None:
                            observation = {"value": raw_observation}
                else:
                    social_payload = event.payload.get("observation", {})
                    if isinstance(social_payload, dict):
                        if "social_action" in social_payload:
                            social_action = int(social_payload["social_action"])
                        if "social_reward" in social_payload:
                            social_reward = float(social_payload["social_reward"])
            elif step.phase == EventPhase.OUTCOME and step.node_id == decision_node:
                reward = float(event.payload["reward"])

        if available_actions is None:
            raise ValueError(
                f"Trial {trial.trial_index}: no subject INPUT event found for decision at "
                f"step {decision_step_index}"
            )

        views.append(
            DecisionTrialView(
                trial_index=trial.trial_index,
                available_actions=available_actions,
                choice=int(decision_event.payload["action"]),
                reward=reward,
                observation=observation,
                social_action=social_action,
                social_reward=social_reward,
            )
        )

    return tuple(views)
