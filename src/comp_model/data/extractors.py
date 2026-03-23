"""Translating raw event logs into flat summaries that learning models can use.

Raw experimental data is stored as a sequence of events (see schema.py).
Learning models, however, need a much simpler picture: for each decision
moment, just tell me who acted, who is learning, what was chosen, and what
reward was received.

This module does that translation. Its main job is to walk through a trial's
events and produce ``DecisionTrialView`` objects — clean, flat summaries of
each decision moment. Models never see raw events; they only ever see these
summaries.

The main function is ``replay_trial_steps``, which replays a trial step by
step and emits two kinds of signals to the caller:

- A **DECISION** signal: "the participant just made a choice — evaluate how
  probable that choice was under the current model."
- An **UPDATE** signal: "a learning step should happen now — advance the
  model's internal state using this observation."

Each view carries ``actor_id`` (who acted) and ``learner_id`` (who is
learning from it). A kernel compares these two fields to decide whether to
apply a self-update (``actor_id == learner_id``) or a social update
(``actor_id != learner_id``).

This separation keeps the model fitting code clean: the caller just needs to
respond to these two signals without worrying about which event in the raw
log triggered them.
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
    """A clean summary of one decision moment, ready for the learning model to use.

    Rather than handing the model a sequence of raw events, we package
    everything it needs for one decision into this flat record. This means
    the same model code works regardless of how the task's event sequence is
    structured — the extractor takes care of the bookkeeping.

    A view is always from the perspective of a specific observer
    (``learner_id``). Two fields together answer "who did what":

    - ``actor_id``: the agent who made the choice and received the reward.
    - ``learner_id``: the agent who is learning from this observation.

    When ``actor_id == learner_id`` the learner is updating from their own
    experience (self-update). When they differ the learner is updating from
    watching someone else (social update). Kernels use this comparison to
    select the appropriate learning rate.

    Attributes
    ----------
    trial_index
        Which trial within the current block this decision came from.
    available_actions
        The options that were available to the *learner* at this decision
        point (e.g. ``(0, 1, 2)`` for a three-armed bandit).
    actor_id
        Identifier of the agent who performed the action (e.g.
        ``"subject"`` or ``"demonstrator"``).
    learner_id
        Identifier of the agent who is learning from this observation
        (e.g. ``"subject"``).
    action
        The action that was taken (e.g. ``1``). For social-update steps
        where the schema marks the actor's action as *not* observable,
        this is ``None``.
    reward
        The reward received for the action. ``None`` if the reward is not
        observable to the learner (controlled by ``observable_fields`` on
        the schema step).
    observation
        Any additional information the learner saw alongside the options
        (e.g. stimulus features).
    metadata
        Any additional bookkeeping information attached by the extractor.
    """

    trial_index: int
    available_actions: tuple[int, ...]
    actor_id: str = ""
    learner_id: str = ""
    action: int | None = None
    reward: float | None = None
    observation: Mapping[str, Any] = field(default_factory=empty_mapping)
    metadata: Mapping[str, Any] = field(default_factory=empty_mapping)


def replay_trial_steps(
    trial: Trial,
    schema: TrialSchema,
) -> Iterator[tuple[EventPhase, str, DecisionTrialView]]:
    """Walk through a trial event-by-event and emit signals the model should respond to.

    This function is the bridge between raw data and model fitting. It reads a
    trial's events in order and, at the right moments, yields a signal telling
    the caller what the model should do next. There are two kinds of signals:

    - **DECISION** (``EventPhase.DECISION``): the participant just made a
      choice. The caller should ask the model "how probable was this action?"
      and record that probability (e.g. for computing log-likelihood).
      Only emitted for the participant's own choices, not a demonstrator's.

    - **UPDATE** (``EventPhase.UPDATE``): a learning step should happen now.
      The caller should advance the model's internal state using the view.
      The view's ``actor_id`` and ``learner_id`` tell the kernel whether
      this is a self-update or a social update. The UPDATE payload already
      contains both the action and the reward, so there is no need to
      remember information from earlier events.

    Design note — why carry action and reward in the UPDATE payload?
    In principle you could reconstruct them by looking back at earlier events.
    Instead, the schema pre-packages them in the UPDATE event so this function
    can emit a complete view without accumulating state across events.
    The only thing that does need to be tracked across events is the most
    recent INPUT event per actor, because that is where ``available_actions``
    lives — and we store that in ``actor_inputs`` below.

    Parameters
    ----------
    trial
        The trial to replay.
    schema
        Defines the expected structure of this trial — which events appear in
        which positions and what each one means.

    Yields
    ------
    event_phase : EventPhase
        ``EventPhase.DECISION`` — the model should evaluate action
        probabilities for this choice.
        ``EventPhase.UPDATE`` — the model should update its internal state.
    learner_id : str
        The identifier of the agent whose state should be updated or
        evaluated (e.g. ``"subject"``).
    view : DecisionTrialView
        A complete summary of the decision moment, ready for the model.

    Raises
    ------
    ValueError
        If the trial's events do not match the expected schema structure.
    """

    schema.validate_trial(trial)

    events = trial.events
    steps = schema.steps

    # Most recent INPUT event per actor — used to recover available_actions.
    actor_inputs: dict[str, Any] = {}

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
                        actor_id=step.actor_id,
                        learner_id=step.learner_id,
                        action=int(event.payload["action"]),
                        reward=None,
                        observation=input_event.payload.get("observation", {}),
                    ),
                )

        elif step.phase == EventPhase.OUTCOME:
            # Reward flows directly into the UPDATE event payload below.
            pass

        elif step.phase == EventPhase.UPDATE:
            learner = step.learner_id
            actor = step.actor_id
            actor_action = int(event.payload["choice"])
            actor_reward = float(event.payload["reward"])

            learner_input = actor_inputs.get(learner)
            available_actions: tuple[int, ...] = (
                tuple(learner_input.payload["available_actions"]) if learner_input else ()
            )

            # For self-updates both fields are always visible.
            # For social updates visibility is gated by observable_fields.
            obs = step.observable_fields if actor != learner else frozenset({"action", "reward"})
            view = DecisionTrialView(
                trial_index=trial.trial_index,
                available_actions=available_actions,
                actor_id=actor,
                learner_id=learner,
                action=actor_action if "action" in obs else None,
                reward=actor_reward if "reward" in obs else None,
            )

            yield (EventPhase.UPDATE, learner, view)
