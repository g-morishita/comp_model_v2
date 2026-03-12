"""Schema-driven extraction from event traces to model-facing decision views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from comp_model.data.schema import EventPhase, Trial

if TYPE_CHECKING:
    from collections.abc import Mapping

    from comp_model.tasks.schemas import TrialSchema


def _empty_metadata() -> Mapping[str, Any]:
    """Create an empty metadata mapping with explicit typing.

    Returns
    -------
    Mapping[str, Any]
        Empty metadata mapping.
    """

    return {}


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
    """

    trial_index: int
    available_actions: tuple[int, ...]
    choice: int
    reward: float | None = None
    observation: Mapping[str, Any] = field(default_factory=_empty_metadata)
    social_action: int | None = None
    social_reward: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)


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
