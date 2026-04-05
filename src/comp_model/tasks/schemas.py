"""Trial schemas — scripts that define the event order within a single trial.

A schema is a precise description of what happens in a trial: which events
occur, in what order, who is the actor for each event (subject or
demonstrator), and what information is available to the learner at each step.

Schemas serve three purposes, all using the same definition:

1. Driving simulation: the environment reads the schema step by step to know
   what event type to generate next.
2. Validating recorded data: when loading real experimental data, each trial
   is checked against the schema to confirm that events appear in the correct
   order with the correct actors.
3. Guiding data extraction: the schema tells the data extractor how to
   interpret each event — for example, which events carry the subject's own
   choice versus the demonstrator's observable outcome.

Pre-built schemas for common experimental designs are provided at the bottom of
this module. Choose the schema that matches your experimental protocol and pass
it to the environment or fitting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from comp_model.data.schema import EventPhase, Trial
from comp_model.data.validation import validate_event_payload


@dataclass(frozen=True, slots=True)
class TrialSchemaStep:
    """One event slot in a trial script.

    Each step describes a single event that should occur at that position in
    the trial. Steps are matched positionally — the first step describes the
    first event, the second step describes the second event, and so on.

    Attributes
    ----------
    phase
        The type of event expected at this position. Common phases are INPUT
        (options become available), DECISION (a choice is recorded), OUTCOME
        (a reward is delivered), and UPDATE (beliefs are updated).
    node_id
        The name of the decision point this event belongs to. Used to match
        recorded events to the correct position in the schema.
    actor_id
        Who performs the action at this step — either ``"subject"`` (the
        participant) or ``"demonstrator"``. Defaults to ``"subject"``.
    learner_id
        Whose internal model is updated at this step. Relevant only for UPDATE
        steps. Defaults to ``"subject"``. For social learning steps the
        ``learner_id`` and ``actor_id`` differ: the actor is the demonstrator
        but the learner is the subject.
    observable_fields
        For UPDATE steps where the learner is watching the actor (i.e.
        ``actor_id`` and ``learner_id`` differ): which pieces of the actor's
        outcome are visible to the learner. Must be specified for every such
        social update step. Allowed values are ``"action"`` (the demonstrator's
        choice is visible) and ``"reward"`` (the demonstrator's reward is
        visible). Omitting ``"reward"`` models action-only social observation.
    """

    phase: EventPhase
    node_id: str
    actor_id: str = "subject"
    learner_id: str = "subject"
    observable_fields: frozenset[str] = field(default_factory=lambda: frozenset[str]())


@dataclass(frozen=True, slots=True)
class TrialSchema:
    """A complete script for one type of trial.

    A ``TrialSchema`` is an ordered list of steps (``TrialSchemaStep``) that
    describes everything that should happen in a trial of this type: which
    events occur, in what order, and who acts or learns at each step.

    The same schema object drives simulation, validates recorded data, and
    guides feature extraction — there is a single source of truth for the
    trial structure.

    Attributes
    ----------
    schema_id
        A short, stable name for this schema (e.g. ``"asocial_bandit"``). Used
        for logging and error messages.
    steps
        The ordered list of event slots that every trial of this type must
        match, position by position.
    """

    schema_id: str
    steps: tuple[TrialSchemaStep, ...]

    def __post_init__(self) -> None:
        for i, step in enumerate(self.steps):
            if (
                step.phase == EventPhase.UPDATE
                and step.actor_id != step.learner_id
                and not step.observable_fields
            ):
                raise ValueError(
                    f"Schema {self.schema_id!r}, step {i}: social UPDATE step "
                    f"(actor_id={step.actor_id!r}, learner_id={step.learner_id!r}) "
                    f"must declare observable_fields"
                )

    def validate_trial(self, trial: Trial) -> None:
        """Check that a recorded trial matches this schema exactly.

        Compares the trial's events against the schema step by step. If
        anything does not match — wrong number of events, wrong event type,
        wrong actor, wrong position — a descriptive error is raised so you can
        identify the problematic trial and step.

        Parameters
        ----------
        trial
            A single recorded trial to check.

        Returns
        -------
        None
            Returns silently if the trial is valid.

        Raises
        ------
        ValueError
            Raised with a descriptive message when the trial does not conform
            to the schema (e.g. wrong number of events, unexpected actor,
            mismatched event type, or missing required data fields).
        """

        if len(trial.events) != len(self.steps):
            raise ValueError(
                f"Trial {trial.trial_index}: expected {len(self.steps)} events from "
                f"schema {self.schema_id!r}, got {len(trial.events)}"
            )

        for step_index, (event, step) in enumerate(zip(trial.events, self.steps, strict=True)):
            if event.phase != step.phase:
                raise ValueError(
                    f"Trial {trial.trial_index}, event {step_index}: expected phase "
                    f"{step.phase!r}, got {event.phase!r}"
                )
            if event.node_id != step.node_id:
                raise ValueError(
                    f"Trial {trial.trial_index}, event {step_index}: expected node_id "
                    f"{step.node_id!r}, got {event.node_id!r}"
                )
            if event.actor_id != step.actor_id:
                raise ValueError(
                    f"Trial {trial.trial_index}, event {step_index}: expected actor_id "
                    f"{step.actor_id!r}, got {event.actor_id!r}"
                )
            if event.event_index != step_index:
                raise ValueError(
                    f"Trial {trial.trial_index}, event {step_index}: expected event_index "
                    f"{step_index}, got {event.event_index}"
                )
            validate_event_payload(event, trial.trial_index, step_index)

    @property
    def social_observable_fields(self) -> frozenset[str]:
        """Union of observable fields across all social UPDATE steps for the subject.

        A social UPDATE step is one where ``actor_id != learner_id`` and
        ``learner_id == "subject"``.  This property collects all field names
        that the subject can observe from such steps.

        Returns
        -------
        frozenset[str]
            The union of ``observable_fields`` for every social update step
            directed at the subject.  Empty for purely asocial schemas.
        """

        fields: set[str] = set()
        for step in self.steps:
            if (
                step.phase == EventPhase.UPDATE
                and step.actor_id != step.learner_id
                and step.learner_id == "subject"
            ):
                fields.update(step.observable_fields)
        return frozenset(fields)

    @property
    def decision_step_indices(self) -> tuple[int, ...]:
        """The positions (0-based) in the trial where a choice is made.

        A trial may contain more than one decision step — for example, in
        social designs both the demonstrator and the subject make a choice.
        This property returns the indices of all such steps so that downstream
        code can locate choices without needing to know the schema structure.

        Returns
        -------
        tuple[int, ...]
            Zero-based indices of all DECISION steps in the schema, in order.
        """

        return tuple(
            index for index, step in enumerate(self.steps) if step.phase == EventPhase.DECISION
        )

    @property
    def has_subject_reward(self) -> bool:
        """Whether the schema contains a subject-owned reward-bearing step.

        Returns
        -------
        bool
            ``True`` when the schema includes a subject ``OUTCOME`` or subject
            ``UPDATE`` step that requires a concrete reward value during trial
            reconstruction.
        """

        return any(
            step.actor_id == "subject" and step.phase in {EventPhase.OUTCOME, EventPhase.UPDATE}
            for step in self.steps
        )


# Standard solo learning trial.
# Trial order: options appear → subject chooses → subject receives reward → beliefs updated.
# No demonstrator is present. Use this for basic multi-armed bandit experiments.
ASOCIAL_BANDIT_SCHEMA = TrialSchema(
    schema_id="asocial_bandit",
    steps=(
        TrialSchemaStep(EventPhase.INPUT, "main"),
        TrialSchemaStep(EventPhase.DECISION, "main"),
        TrialSchemaStep(EventPhase.OUTCOME, "main"),
        TrialSchemaStep(EventPhase.UPDATE, "main"),
    ),
)

# Demonstrator-first social learning trial (subject sees choice AND reward before acting).
# Trial order: demonstrator sees options → demonstrator chooses → demonstrator gets reward →
#   demonstrator's beliefs updated → subject observes demo's choice+reward and updates →
#   subject sees options → subject chooses → subject gets reward → subject's beliefs updated.
# Use this when participants watch the demonstrator's full outcome before making their own choice.
# Social learning can therefore influence the current trial's choice.
SOCIAL_PRE_CHOICE_SCHEMA = TrialSchema(
    schema_id="social_pre_choice",
    steps=(
        TrialSchemaStep(EventPhase.INPUT, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.DECISION, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.OUTCOME, "main", actor_id="demonstrator"),
        TrialSchemaStep(
            EventPhase.UPDATE, "main", actor_id="demonstrator", learner_id="demonstrator"
        ),
        TrialSchemaStep(
            EventPhase.UPDATE,
            "main",
            actor_id="demonstrator",
            learner_id="subject",
            observable_fields=frozenset({"action", "reward"}),
        ),
        TrialSchemaStep(EventPhase.INPUT, "main"),
        TrialSchemaStep(EventPhase.DECISION, "main"),
        TrialSchemaStep(EventPhase.OUTCOME, "main"),
        TrialSchemaStep(EventPhase.UPDATE, "main"),
    ),
)

# Demonstrator-first social learning trial (subject sees choice ONLY, not reward, before acting).
# Identical structure to SOCIAL_PRE_CHOICE_SCHEMA except that when the subject observes the
# demonstrator, only the demonstrator's choice is visible — the reward is hidden.
# Use this to model action imitation without outcome knowledge.
SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA = TrialSchema(
    schema_id="social_pre_choice_action_only",
    steps=(
        TrialSchemaStep(EventPhase.INPUT, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.DECISION, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.OUTCOME, "main", actor_id="demonstrator"),
        TrialSchemaStep(
            EventPhase.UPDATE, "main", actor_id="demonstrator", learner_id="demonstrator"
        ),
        TrialSchemaStep(
            EventPhase.UPDATE,
            "main",
            actor_id="demonstrator",
            learner_id="subject",
            observable_fields=frozenset({"action"}),
        ),
        TrialSchemaStep(EventPhase.INPUT, "main"),
        TrialSchemaStep(EventPhase.DECISION, "main"),
        TrialSchemaStep(EventPhase.OUTCOME, "main"),
        TrialSchemaStep(EventPhase.UPDATE, "main"),
    ),
)

# Subject-first social learning trial (subject acts, then watches demonstrator's choice+reward).
# Trial order: subject sees options → subject chooses → subject gets reward →
#   subject's beliefs updated → demonstrator sees options → demonstrator chooses →
#   demonstrator gets reward → demonstrator's beliefs updated →
#   subject observes demo's choice+reward and updates.
# Because the demonstrator is observed after the subject has already chosen, the social
# information can only influence future trials, not the current one.
SOCIAL_POST_OUTCOME_SCHEMA = TrialSchema(
    schema_id="social_post_outcome",
    steps=(
        TrialSchemaStep(EventPhase.INPUT, "main"),
        TrialSchemaStep(EventPhase.DECISION, "main"),
        TrialSchemaStep(EventPhase.OUTCOME, "main"),
        TrialSchemaStep(EventPhase.UPDATE, "main"),
        TrialSchemaStep(EventPhase.INPUT, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.DECISION, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.OUTCOME, "main", actor_id="demonstrator"),
        TrialSchemaStep(
            EventPhase.UPDATE, "main", actor_id="demonstrator", learner_id="demonstrator"
        ),
        TrialSchemaStep(
            EventPhase.UPDATE,
            "main",
            actor_id="demonstrator",
            learner_id="subject",
            observable_fields=frozenset({"action", "reward"}),
        ),
    ),
)

# Subject-first social learning trial (subject acts, then watches demonstrator's choice ONLY).
# Identical structure to SOCIAL_POST_OUTCOME_SCHEMA except that the subject only sees the
# demonstrator's choice, not the reward. Social learning therefore carries no outcome signal.
SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA = TrialSchema(
    schema_id="social_post_outcome_action_only",
    steps=(
        TrialSchemaStep(EventPhase.INPUT, "main"),
        TrialSchemaStep(EventPhase.DECISION, "main"),
        TrialSchemaStep(EventPhase.OUTCOME, "main"),
        TrialSchemaStep(EventPhase.UPDATE, "main"),
        TrialSchemaStep(EventPhase.INPUT, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.DECISION, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.OUTCOME, "main", actor_id="demonstrator"),
        TrialSchemaStep(
            EventPhase.UPDATE, "main", actor_id="demonstrator", learner_id="demonstrator"
        ),
        TrialSchemaStep(
            EventPhase.UPDATE,
            "main",
            actor_id="demonstrator",
            learner_id="subject",
            observable_fields=frozenset({"action"}),
        ),
    ),
)

# Schemas where the subject acts but receives no feedback on their own outcome.
# The subject makes a choice but is never told whether it was rewarded.
# Only the demonstrator's outcome can drive learning in these designs.

# Demonstrator-first, no self-feedback trial.
# Trial order: demonstrator acts and gets reward → subject observes demo's choice+reward →
#   subject chooses (but receives no reward and gets no personal update).
# Use this to isolate pure social learning: only the demonstrator's outcome can shape beliefs.
SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA = TrialSchema(
    schema_id="social_pre_choice_no_self_outcome",
    steps=(
        TrialSchemaStep(EventPhase.INPUT, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.DECISION, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.OUTCOME, "main", actor_id="demonstrator"),
        TrialSchemaStep(
            EventPhase.UPDATE, "main", actor_id="demonstrator", learner_id="demonstrator"
        ),
        TrialSchemaStep(
            EventPhase.UPDATE,
            "main",
            actor_id="demonstrator",
            learner_id="subject",
            observable_fields=frozenset({"action", "reward"}),
        ),
        TrialSchemaStep(EventPhase.INPUT, "main"),
        TrialSchemaStep(EventPhase.DECISION, "main"),
    ),
)

# Subject-first, no self-feedback trial.
# Trial order: subject chooses (no reward given) → demonstrator acts and gets reward →
#   subject observes demo's choice+reward and updates.
# Use this when you want the subject to commit to a choice before seeing the demo,
# but still learn only from the demonstrator's outcome (no personal feedback).
SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA = TrialSchema(
    schema_id="social_post_outcome_no_self_outcome",
    steps=(
        TrialSchemaStep(EventPhase.INPUT, "main"),
        TrialSchemaStep(EventPhase.DECISION, "main"),
        TrialSchemaStep(EventPhase.INPUT, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.DECISION, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.OUTCOME, "main", actor_id="demonstrator"),
        TrialSchemaStep(
            EventPhase.UPDATE, "main", actor_id="demonstrator", learner_id="demonstrator"
        ),
        TrialSchemaStep(
            EventPhase.UPDATE,
            "main",
            actor_id="demonstrator",
            learner_id="subject",
            observable_fields=frozenset({"action", "reward"}),
        ),
    ),
)

# Bidirectional social learning schemas — both the subject and the demonstrator
# learn from each other's outcomes.
# In these designs the demonstrator also observes the subject's choice and reward
# and updates their own beliefs accordingly. This models dyadic or reciprocal
# social learning where influence flows in both directions.

# Demonstrator-first, bidirectional learning trial.
# Trial order: demonstrator acts+gets reward → subject observes demo (choice+reward) →
#   subject acts+gets reward → demonstrator observes subject (choice+reward).
# Both agents learn from each other. The subject still observes the demonstrator before
# choosing, so social information can influence the current trial's choice.
SOCIAL_PRE_CHOICE_DEMO_LEARNS_SCHEMA = TrialSchema(
    schema_id="social_pre_choice_demo_learns",
    steps=(
        TrialSchemaStep(EventPhase.INPUT, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.DECISION, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.OUTCOME, "main", actor_id="demonstrator"),
        TrialSchemaStep(
            EventPhase.UPDATE, "main", actor_id="demonstrator", learner_id="demonstrator"
        ),
        TrialSchemaStep(
            EventPhase.UPDATE,
            "main",
            actor_id="demonstrator",
            learner_id="subject",
            observable_fields=frozenset({"action", "reward"}),
        ),
        TrialSchemaStep(EventPhase.INPUT, "main"),
        TrialSchemaStep(EventPhase.DECISION, "main"),
        TrialSchemaStep(EventPhase.OUTCOME, "main"),
        TrialSchemaStep(EventPhase.UPDATE, "main"),
        TrialSchemaStep(
            EventPhase.UPDATE,
            "main",
            actor_id="subject",
            learner_id="demonstrator",
            observable_fields=frozenset({"action", "reward"}),
        ),
    ),
)

# Subject-first, bidirectional learning trial.
# Trial order: subject acts+gets reward → demonstrator observes subject (choice+reward) →
#   demonstrator acts+gets reward → subject observes demo (choice+reward).
# Both agents learn from each other. Because the subject acts before watching the demo,
# social information from this trial can only shape future choices.
SOCIAL_POST_OUTCOME_DEMO_LEARNS_SCHEMA = TrialSchema(
    schema_id="social_post_outcome_demo_learns",
    steps=(
        TrialSchemaStep(EventPhase.INPUT, "main"),
        TrialSchemaStep(EventPhase.DECISION, "main"),
        TrialSchemaStep(EventPhase.OUTCOME, "main"),
        TrialSchemaStep(EventPhase.UPDATE, "main"),
        TrialSchemaStep(
            EventPhase.UPDATE,
            "main",
            actor_id="subject",
            learner_id="demonstrator",
            observable_fields=frozenset({"action", "reward"}),
        ),
        TrialSchemaStep(EventPhase.INPUT, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.DECISION, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.OUTCOME, "main", actor_id="demonstrator"),
        TrialSchemaStep(
            EventPhase.UPDATE, "main", actor_id="demonstrator", learner_id="demonstrator"
        ),
        TrialSchemaStep(
            EventPhase.UPDATE,
            "main",
            actor_id="demonstrator",
            learner_id="subject",
            observable_fields=frozenset({"action", "reward"}),
        ),
    ),
)
