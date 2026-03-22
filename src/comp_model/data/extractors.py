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
        the corresponding UPDATE step's ``observable_fields``).
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
) -> Iterator[tuple[EventPhase, str, DecisionTrialView]]:
    """Yield schema-ordered replay steps for engine and MLE use.

    Steps through the schema positionally, emitting one item per DECISION
    (subject only) or UPDATE step.  UPDATE event payloads carry the actor's
    choice and reward directly, so no accumulation of choices or rewards is
    needed beyond tracking INPUT events for ``available_actions``.

    Parameters
    ----------
    trial
        Trial whose events should be replayed.
    schema
        Schema defining the positional meaning of each event.

    Yields
    ------
    event_phase : EventPhase
        ``EventPhase.DECISION`` — caller should evaluate
        ``action_probabilities`` and accumulate log-probability. Only emitted
        for subject DECISION steps.
        ``EventPhase.UPDATE`` — caller should call ``next_state``.
    learner_id : str
        Which agent's state should be updated.
    view : DecisionTrialView
        View built from the event payload and accumulated INPUT context.

    Raises
    ------
    ValueError
        If the trial fails schema validation.
    """

    schema.validate_trial(trial)

    events = trial.events
    steps = schema.steps

    actor_inputs: dict[str, Any] = {}  # most recent INPUT event per actor

    for event, step in zip(events, steps, strict=True):
        if step.phase == EventPhase.INPUT:
            actor_inputs[step.actor_id] = event

        elif step.phase == EventPhase.DECISION:
            if step.actor_id == "subject":
                input_event = actor_inputs["subject"]
                yield (
                    EventPhase.DECISION,
                    "subject",
                    DecisionTrialView(
                        trial_index=trial.trial_index,
                        available_actions=tuple(input_event.payload["available_actions"]),
                        choice=int(event.payload["action"]),
                        reward=None,
                        observation=input_event.payload.get("observation", {}),
                        social_action=None,
                        social_reward=None,
                    ),
                )

        elif step.phase == EventPhase.OUTCOME:
            pass  # reward flows into the following UPDATE event payload

        elif step.phase == EventPhase.UPDATE:
            learner = step.learner_id
            actor = step.actor_id
            actor_choice = int(event.payload["choice"])
            actor_reward = float(event.payload["reward"])

            learner_input = actor_inputs.get(learner)
            available_actions: tuple[int, ...] = (
                tuple(learner_input.payload["available_actions"]) if learner_input else ()
            )

            if actor == learner:
                view = DecisionTrialView(
                    trial_index=trial.trial_index,
                    available_actions=available_actions,
                    choice=actor_choice,
                    reward=actor_reward,
                    social_action=None,
                    social_reward=None,
                )
            else:
                obs = step.observable_fields
                view = DecisionTrialView(
                    trial_index=trial.trial_index,
                    available_actions=available_actions,
                    choice=None,
                    reward=None,
                    social_action=actor_choice if "action" in obs else None,
                    social_reward=actor_reward if "reward" in obs else None,
                )

            yield (EventPhase.UPDATE, learner, view)
