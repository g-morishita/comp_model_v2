"""Conversion between canonical trial events and row-shaped CSV views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from comp_model.data import Event, EventPhase, Trial, replay_trial_steps
from comp_model.io.csv.parsing import _format_available_actions

if TYPE_CHECKING:
    from collections.abc import Mapping

    from comp_model.tasks import TrialSchema


@dataclass(frozen=True, slots=True)
class _CombinedTrialView:
    """Merged per-trial data for CSV row converters.

    Aggregates the subject's choice, optional self reward, and observed social
    information from every replay step in one trial. This is an internal helper
    for flat CSV export, not a general decision view, because it intentionally
    collapses multiple replay events into one row-shaped record.
    """

    trial_index: int
    available_actions: tuple[int, ...]
    choice: int | None
    reward: float | None
    social_action: int | None
    social_reward: float | None
    observation: dict[str, Any]


def _extract_single_view(trial: Trial, schema: TrialSchema) -> _CombinedTrialView:
    """Collapse one trial into the row-shaped view used by built-in converters.

    Parameters
    ----------
    trial
        Canonical trial to flatten.
    schema
        Schema used to validate and extract the trial.

    Returns
    -------
    _CombinedTrialView
        Merged per-trial record combining the subject's choice, optional self
        reward, observation, and any observed social information.

    Raises
    ------
    ValueError
        Raised when the schema does not yield exactly one subject action step.
    """

    choice: int | None = None
    available_actions: tuple[int, ...] = ()
    reward: float | None = None
    social_action: int | None = None
    social_reward: float | None = None
    observation: dict[str, Any] = {}

    for event, step in zip(trial.events, schema.steps, strict=True):
        if step.phase == EventPhase.INPUT and step.actor_id == "subject":
            available_actions = tuple(event.payload["available_actions"])
            observation = dict(event.payload.get("observation", {}))
        elif step.phase == EventPhase.DECISION and step.actor_id == "subject":
            raw_action = event.payload["action"]
            choice = None if raw_action is None else int(raw_action)

    for event_type, learner_id, view in replay_trial_steps(trial, schema):
        if event_type == EventPhase.DECISION and learner_id == "subject":
            # The flat CSV row stores only the subject-facing decision state.
            choice = view.action
            available_actions = view.available_actions
            observation = dict(view.observation)
        elif event_type == EventPhase.UPDATE:
            if learner_id == "subject" and view.available_actions:
                available_actions = view.available_actions
            # Subject-owned reward can appear in self-updates and in
            # demonstrator-facing updates for bidirectional schemas.
            if view.actor_id == "subject" and view.reward is not None:
                reward = view.reward
            if view.actor_id == "demonstrator":
                # Demonstrator reward must come from the actor's own update
                # when the subject-facing social view intentionally hides it.
                if view.action is not None:
                    social_action = view.action
                if view.reward is not None:
                    social_reward = view.reward

    return _CombinedTrialView(
        trial_index=trial.trial_index,
        available_actions=available_actions,
        choice=choice,
        reward=reward,
        observation=observation,
        social_action=social_action,
        social_reward=social_reward,
    )


def _build_common_row(
    *,
    subject_id: str,
    block_index: int,
    condition: str,
    schema_id: str,
    trial_index: int,
    available_actions: tuple[int, ...],
    choice: int | None,
    reward: float | None,
) -> dict[str, str]:
    """Build the shared CSV columns for one trial row.

    Parameters
    ----------
    subject_id
        Subject identifier for the containing subject.
    block_index
        Block index for the containing block.
    condition
        Condition label for the containing block.
    schema_id
        Schema identifier for the containing block.
    trial_index
        Trial index within the block.
    available_actions
        Legal action values for the trial.
    choice
        Chosen action value, or ``None`` for timeout-style subject rows.
    reward
        Observed reward, or ``None`` for schemas with no subject outcome.

    Returns
    -------
    dict[str, str]
        Shared CSV row columns as strings.
    """

    return {
        "subject_id": subject_id,
        "block_index": str(block_index),
        "condition": condition,
        "schema_id": schema_id,
        "trial_index": str(trial_index),
        "available_actions": _format_available_actions(available_actions),
        "choice": "" if choice is None else str(choice),
        "reward": "" if reward is None else str(reward),
    }


def _build_trial_from_schema(
    *,
    schema: TrialSchema,
    trial_index: int,
    available_actions: tuple[int, ...],
    choice: int | None,
    reward: float | None,
    demonstrator_observation: Mapping[str, Any] | None = None,
) -> Trial:
    """Build one canonical trial using the declared schema order.

    Parameters
    ----------
    schema
        Schema whose positional steps define event order.
    trial_index
        Trial index assigned to the rebuilt trial.
    available_actions
        Legal actions for subject and demonstrator input events.
    choice
        Chosen action value, or ``None`` for timeout-style subject rows.
    reward
        Observed reward value, or ``None`` when the schema omits the subject's
        own outcome entirely or the subject timed out.
    demonstrator_observation
        Demonstrator observation payload for non-subject input events, if any.

    Returns
    -------
    Trial
        Canonical trial rebuilt in schema order.

    Raises
    ------
    ValueError
        Raised when the schema expects a demonstrator input but no observation
        payload was supplied.
    """

    demonstrator_choice: int | None = None
    demonstrator_reward: float | None = None
    if demonstrator_observation is not None:
        demonstrator_choice = demonstrator_observation.get("social_action")
        demonstrator_reward = demonstrator_observation.get("social_reward")

    events: list[Event] = []
    for step_index, step in enumerate(schema.steps):
        # Reconstruction follows the schema verbatim: the row supplies values,
        # while the schema decides which event types actually appear.
        if step.phase == EventPhase.INPUT:
            payload: dict[str, Any] = {"available_actions": available_actions}
        elif step.phase == EventPhase.DECISION:
            if step.actor_id == "subject":
                payload = {"action": choice}
            else:
                if demonstrator_choice is None:
                    raise ValueError(f"Schema {schema.schema_id!r} requires demonstrator choice")
                payload = {"action": demonstrator_choice}
        elif step.phase == EventPhase.OUTCOME:
            if step.actor_id == "subject":
                payload = {"reward": reward}
            else:
                if demonstrator_reward is None:
                    raise ValueError(f"Schema {schema.schema_id!r} requires demonstrator reward")
                payload = {"reward": demonstrator_reward}
        elif step.phase == EventPhase.UPDATE:
            # Payload carries the actor's choice and reward for replay.
            if step.actor_id == "subject":
                payload = {"choice": choice, "reward": reward}
            else:
                if demonstrator_choice is None or demonstrator_reward is None:
                    raise ValueError(f"Schema {schema.schema_id!r} requires demonstrator data")
                payload = {"choice": demonstrator_choice, "reward": demonstrator_reward}
        else:
            raise ValueError(f"Unsupported event phase {step.phase!r}")

        events.append(
            Event(
                phase=step.phase,
                event_index=step_index,
                node_id=step.node_id,
                actor_id=step.actor_id,
                payload=payload,
            )
        )
    return Trial(trial_index=trial_index, events=tuple(events))
