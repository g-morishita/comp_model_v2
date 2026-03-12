"""Declarative event-order schemas for trial structure."""

from __future__ import annotations

from dataclasses import dataclass

from comp_model.data.schema import EventPhase, Trial
from comp_model.data.validation import validate_event_payload


@dataclass(frozen=True, slots=True)
class TrialSchemaStep:
    """Single positional step inside a declarative trial schema.

    Attributes
    ----------
    phase
        Phase expected at this step.
    node_id
        Decision-point identifier expected at this step.
    actor_id
        Actor expected at this step.
    action_required
        Whether the simulation engine must supply an action here.
    """

    phase: EventPhase
    node_id: str
    actor_id: str = "subject"
    action_required: bool = False


@dataclass(frozen=True, slots=True)
class TrialSchema:
    """Valid event ordering for one trial type.

    Attributes
    ----------
    schema_id
        Stable identifier for the schema.
    steps
        Ordered schema steps that each trial must match positionally.
    """

    schema_id: str
    steps: tuple[TrialSchemaStep, ...]

    def validate_trial(self, trial: Trial) -> None:
        """Validate a trial against the schema definition.

        Parameters
        ----------
        trial
            Trial to validate.

        Returns
        -------
        None
            This function raises on any mismatch.
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
    def decision_step_indices(self) -> tuple[int, ...]:
        """Indices of all decision steps in the schema.

        Returns
        -------
        tuple[int, ...]
            Positions of decision steps.
        """

        return tuple(
            index for index, step in enumerate(self.steps) if step.phase == EventPhase.DECISION
        )

    @property
    def action_required_indices(self) -> tuple[int, ...]:
        """Indices of steps that require an externally supplied action.

        Returns
        -------
        tuple[int, ...]
            Positions whose `action_required` flag is true.
        """

        return tuple(index for index, step in enumerate(self.steps) if step.action_required)


ASOCIAL_BANDIT_SCHEMA = TrialSchema(
    schema_id="asocial_bandit",
    steps=(
        TrialSchemaStep(EventPhase.INPUT, "main"),
        TrialSchemaStep(EventPhase.DECISION, "main", action_required=True),
        TrialSchemaStep(EventPhase.OUTCOME, "main"),
        TrialSchemaStep(EventPhase.UPDATE, "main"),
    ),
)

SOCIAL_PRE_CHOICE_SCHEMA = TrialSchema(
    schema_id="social_pre_choice",
    steps=(
        TrialSchemaStep(EventPhase.INPUT, "main"),
        TrialSchemaStep(EventPhase.INPUT, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.DECISION, "main", action_required=True),
        TrialSchemaStep(EventPhase.OUTCOME, "main"),
        TrialSchemaStep(EventPhase.UPDATE, "main"),
    ),
)

SOCIAL_POST_OUTCOME_SCHEMA = TrialSchema(
    schema_id="social_post_outcome",
    steps=(
        TrialSchemaStep(EventPhase.INPUT, "main"),
        TrialSchemaStep(EventPhase.DECISION, "main", action_required=True),
        TrialSchemaStep(EventPhase.OUTCOME, "main"),
        TrialSchemaStep(EventPhase.INPUT, "main", actor_id="demonstrator"),
        TrialSchemaStep(EventPhase.UPDATE, "main"),
    ),
)
