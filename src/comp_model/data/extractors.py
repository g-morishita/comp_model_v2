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
    from collections.abc import Iterator, Mapping

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
        Chosen action value, or ``None`` when the view is produced before the
        subject's decision (e.g. a social-only update step).
    reward
        Observed reward, if present.
    observation
        Subject-facing observation payload.
    social_action
        Observed demonstrator action, if present.
    social_reward
        Observed demonstrator reward, if present (only when ``"reward"`` is in
        the corresponding INPUT step's ``observable_fields``).
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
    choice: int | None = None
    reward: float | None = None
    observation: Mapping[str, Any] = field(default_factory=empty_mapping)
    social_action: int | None = None
    social_reward: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=empty_mapping)


def replay_trial_steps(
    trial: Trial,
    schema: TrialSchema,
) -> Iterator[tuple[str, str, DecisionTrialView]]:
    """Yield schema-ordered replay steps for engine and MLE use.

    Steps through the schema positionally, emitting one item per DECISION or
    UPDATE step. The view attached to each item contains only information that
    has been accumulated up to that point in the schema, respecting
    ``observable_fields`` on non-subject INPUT steps.

    Parameters
    ----------
    trial
        Trial whose events should be replayed.
    schema
        Schema defining the positional meaning of each event.

    Yields
    ------
    event_type : str
        ``"action"`` — caller should evaluate ``action_probabilities`` and
        accumulate log-probability. Only emitted for subject DECISION steps.
        ``"update"`` — caller should call ``next_state``.
    learner_id : str
        Which agent's state should be updated. Always ``"subject"`` for
        ``"action"`` steps. Taken from the UPDATE step's ``learner_id`` for
        ``"update"`` steps.
    view : DecisionTrialView
        Partial view built from context accumulated so far.
        For ``"action"`` steps: ``available_actions`` and ``choice`` are set;
        ``reward`` is always ``None`` (outcome not yet seen).
        For ``"update"`` steps: contains whatever has been observed up to this
        point — social info filtered by ``observable_fields``, own reward only
        if OUTCOME has already been seen for the subject.

    Raises
    ------
    ValueError
        If the trial fails schema validation.
    """

    schema.validate_trial(trial)

    events = trial.events
    steps = schema.steps

    # Accumulated context per actor_id
    available_actions: dict[str, tuple[int, ...]] = {}
    observation: dict[str, dict[str, Any]] = {}
    choices: dict[str, int] = {}
    rewards: dict[str, float] = {}
    social_action: int | None = None
    social_reward: float | None = None

    for _step_index, (event, step) in enumerate(zip(events, steps, strict=True)):
        if step.phase == EventPhase.INPUT:
            if step.actor_id == "subject":
                available_actions["subject"] = tuple(event.payload["available_actions"])
                raw_obs = event.payload.get("observation")
                if isinstance(raw_obs, dict):
                    observation["subject"] = dict(raw_obs)
                elif raw_obs is not None:
                    observation["subject"] = {"value": raw_obs}
            else:
                # Non-subject INPUT: apply observable_fields filter
                obs_payload = event.payload.get("observation", {})
                if isinstance(obs_payload, dict):
                    if "action" in step.observable_fields and "social_action" in obs_payload:
                        social_action = int(obs_payload["social_action"])
                    if "reward" in step.observable_fields and "social_reward" in obs_payload:
                        social_reward = float(obs_payload["social_reward"])

        elif step.phase == EventPhase.DECISION:
            choices[step.actor_id] = int(event.payload["action"])
            if step.actor_id == "subject" and step.action_required:
                # Emit an "action" step so the caller can evaluate action probs
                yield (
                    "action",
                    "subject",
                    DecisionTrialView(
                        trial_index=trial.trial_index,
                        available_actions=available_actions.get("subject", ()),
                        choice=choices["subject"],
                        reward=None,
                        observation=observation.get("subject", {}),
                        social_action=social_action,
                        social_reward=social_reward,
                    ),
                )
                # Clear social info after the action step so subsequent UPDATE
                # steps (e.g. subject self-update) don't re-apply social learning.
                social_action = None
                social_reward = None

        elif step.phase == EventPhase.OUTCOME:
            rewards[step.actor_id] = float(event.payload["reward"])

        elif step.phase == EventPhase.UPDATE:
            learner = step.learner_id
            yield (
                "update",
                learner,
                DecisionTrialView(
                    trial_index=trial.trial_index,
                    available_actions=available_actions.get(learner, ()),
                    choice=choices.get(learner),
                    reward=rewards.get(learner),
                    observation=observation.get(learner, {}),
                    social_action=social_action,
                    social_reward=social_reward,
                ),
            )
            # Clear reward after it is consumed so a subsequent UPDATE for the
            # same learner (e.g. social-update after self-update) does not
            # re-apply the self-learning signal.
            rewards.pop(learner, None)
