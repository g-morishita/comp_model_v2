"""Validation helpers for canonical event-based data structures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from comp_model.data.schema import Block, Dataset, Event, EventPhase, SubjectData, Trial


class _TrialSchemaProtocol(Protocol):
    """Structural protocol for schema-based trial validation.

    Methods
    -------
    validate_trial(trial)
        Validate a trial against an external schema.
    """

    def validate_trial(self, trial: Trial) -> None:
        """Validate a trial against the schema.

        Parameters
        ----------
        trial
            Trial to validate.

        Returns
        -------
        None
            This function raises on invalid trials.
        """


def validate_event_payload(event: Event, trial_index: int, step_index: int) -> None:
    """Validate required payload keys for one event.

    Parameters
    ----------
    event
        Event whose payload is being validated.
    trial_index
        Index of the containing trial.
    step_index
        Expected position of the event within the trial.

    Returns
    -------
    None
        This function raises on invalid payloads.
    """

    prefix = f"Trial {trial_index}, event {step_index}"
    payload = event.payload
    if event.phase == EventPhase.INPUT:
        if "available_actions" not in payload:
            raise ValueError(f"{prefix}: INPUT missing 'available_actions'")
        available_actions = payload["available_actions"]
        if not isinstance(available_actions, Sequence) or isinstance(available_actions, str):
            raise ValueError(f"{prefix}: 'available_actions' must be a non-empty sequence")
        if len(available_actions) == 0:
            raise ValueError(f"{prefix}: 'available_actions' must be non-empty")
    elif event.phase == EventPhase.DECISION:
        if "action" not in payload:
            raise ValueError(f"{prefix}: DECISION missing 'action'")
    elif event.phase == EventPhase.OUTCOME and "reward" not in payload:
        raise ValueError(f"{prefix}: OUTCOME missing 'reward'")


def validate_event(event: Event, trial_index: int, step_index: int) -> None:
    """Validate one event's structural integrity.

    Parameters
    ----------
    event
        Event to validate.
    trial_index
        Trial index used for descriptive errors.
    step_index
        Expected index of the event within the trial.

    Returns
    -------
    None
        This function raises on invalid events.
    """

    if not event.node_id or not event.node_id.strip():
        raise ValueError(f"Trial {trial_index}, event {step_index}: node_id must be non-empty")
    if not event.actor_id or not event.actor_id.strip():
        raise ValueError(f"Trial {trial_index}, event {step_index}: actor_id must be non-empty")
    if event.event_index != step_index:
        raise ValueError(
            f"Trial {trial_index}, event {step_index}: event_index mismatch "
            f"({event.event_index} != {step_index})"
        )
    validate_event_payload(event, trial_index, step_index)


def validate_trial(trial: Trial, schema: _TrialSchemaProtocol | None = None) -> None:
    """Validate a trial, optionally against a schema.

    Parameters
    ----------
    trial
        Trial to validate.
    schema
        Optional trial schema used for positional validation.

    Returns
    -------
    None
        This function raises on invalid trials.
    """

    if schema is not None:
        schema.validate_trial(trial)
        return

    expected_indices = list(range(len(trial.events)))
    actual_indices = [event.event_index for event in trial.events]
    if actual_indices != expected_indices:
        raise ValueError(
            f"Trial {trial.trial_index}: event indices must be contiguous starting at 0, "
            f"got {actual_indices}"
        )

    for step_index, event in enumerate(trial.events):
        validate_event(event, trial.trial_index, step_index)


def validate_block(block: Block, schema: _TrialSchemaProtocol | None = None) -> None:
    """Validate a block and each of its trials.

    Parameters
    ----------
    block
        Block to validate.
    schema
        Optional schema applied to each trial.

    Returns
    -------
    None
        This function raises on invalid blocks.
    """

    if not block.condition or not block.condition.strip():
        raise ValueError(f"Block {block.block_index}: condition must be non-empty")
    expected_indices = list(range(len(block.trials)))
    actual_indices = [trial.trial_index for trial in block.trials]
    if actual_indices != expected_indices:
        raise ValueError(
            f"Block {block.block_index}: trial indices must be contiguous starting at 0, "
            f"got {actual_indices}"
        )
    for trial in block.trials:
        validate_trial(trial, schema)


def validate_subject(subject: SubjectData, schema: _TrialSchemaProtocol | None = None) -> None:
    """Validate a subject's hierarchical data.

    Parameters
    ----------
    subject
        Subject data to validate.
    schema
        Optional schema applied to each trial.

    Returns
    -------
    None
        This function raises on invalid subject data.
    """

    if not subject.subject_id or not subject.subject_id.strip():
        raise ValueError("subject_id must be non-empty")
    expected_block_indices = list(range(len(subject.blocks)))
    actual_block_indices = [block.block_index for block in subject.blocks]
    if actual_block_indices != expected_block_indices:
        raise ValueError(
            f"Subject {subject.subject_id}: block indices must be contiguous starting at 0, "
            f"got {actual_block_indices}"
        )
    for block in subject.blocks:
        validate_block(block, schema)


def validate_dataset(dataset: Dataset, schema: _TrialSchemaProtocol | None = None) -> None:
    """Validate a full dataset.

    Parameters
    ----------
    dataset
        Dataset to validate.
    schema
        Optional schema applied to each subject's trials.

    Returns
    -------
    None
        This function raises on invalid datasets.
    """

    subject_ids = [subject.subject_id for subject in dataset.subjects]
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("Subject IDs must be unique within dataset")
    for subject in dataset.subjects:
        validate_subject(subject, schema)


__all__ = [
    "validate_block",
    "validate_dataset",
    "validate_event",
    "validate_event_payload",
    "validate_subject",
    "validate_trial",
]
