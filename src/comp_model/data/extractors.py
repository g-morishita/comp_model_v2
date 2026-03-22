"""Translating raw event logs into flat summaries that learning models can use.

Raw experimental data is stored as a sequence of events (see schema.py).
Learning models, however, need a much simpler picture: for each decision
moment, just tell me what options were available, what was chosen, and what
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
  model's internal state using this choice and reward."

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

    Attributes
    ----------
    trial_index
        Which trial within the current block this decision came from.
    available_actions
        The options that were available at this decision point (e.g.
        ``(0, 1, 2)`` for a three-armed bandit).
    choice
        The action that was chosen (e.g. ``1``). This is ``None`` when the
        view represents a social-observation update — i.e. the participant is
        learning from watching a demonstrator, not from their own choice.
    reward
        The reward the participant received, if this was their own decision.
        ``None`` for social-observation update steps.
    observation
        Any additional information the participant saw alongside the options
        (e.g. stimulus features).
    social_action
        The action the demonstrator chose, if one was observed on this step.
        ``None`` if no demonstrator was present or their action was not visible.
    social_reward
        The reward the demonstrator received, if it was visible to the
        participant. This is only populated when the task schema explicitly
        marks the demonstrator's reward as observable. ``None`` otherwise.
    metadata
        Any additional bookkeeping information attached by the extractor.
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
    """Walk through a trial event-by-event and emit signals the model should respond to.

    This function is the bridge between raw data and model fitting. It reads a
    trial's events in order and, at the right moments, yields a signal telling
    the caller what the model should do next. There are two kinds of signals:

    - **DECISION** (``EventPhase.DECISION``): the participant just made a
      choice. The caller should ask the model "how probable was this action?"
      and record that probability (e.g. for computing log-likelihood).
      Only emitted for the participant's own choices, not a demonstrator's.

    - **UPDATE** (``EventPhase.UPDATE``): a learning step should happen now.
      The caller should advance the model's internal state (e.g. update
      action values) using the choice and reward provided in the view. The
      UPDATE payload already contains both the choice that was made and the
      reward that resulted, so there is no need to remember information from
      earlier events.

    Design note — why carry choice and reward in the UPDATE payload?
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
                        choice=int(event.payload["action"]),
                        reward=None,
                        observation=input_event.payload.get("observation", {}),
                        social_action=None,
                        social_reward=None,
                    ),
                )

        elif step.phase == EventPhase.OUTCOME:
            # Reward flows directly into the UPDATE event payload below.
            pass

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
