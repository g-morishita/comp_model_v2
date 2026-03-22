"""Tests for canonical data validation."""

import pytest

from comp_model.data.schema import Block, Dataset, Event, EventPhase, SubjectData, Trial
from comp_model.data.validation import (
    validate_block,
    validate_dataset,
    validate_event,
    validate_subject,
    validate_trial,
)


def _valid_trial() -> Trial:
    """Build a valid asocial trial for validation tests.

    Returns
    -------
    Trial
        A minimal structurally valid trial.
    """

    return Trial(
        trial_index=0,
        events=(
            Event(
                phase=EventPhase.INPUT,
                event_index=0,
                payload={"available_actions": (0, 1)},
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=1,
                payload={"action": 1},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=2,
                payload={"reward": 1.0},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=3,
                payload={"choice": 1, "reward": 1.0},
            ),
        ),
    )


def test_validate_event_rejects_missing_available_actions() -> None:
    """Ensure INPUT validation enforces available actions.

    Returns
    -------
    None
        This test raises on invalid payloads.
    """

    event = Event(phase=EventPhase.INPUT, event_index=0, payload={})

    with pytest.raises(ValueError, match="available_actions"):
        validate_event(event, trial_index=0, step_index=0)


def test_validate_trial_accepts_valid_trial() -> None:
    """Ensure trial validation passes for a valid trial.

    Returns
    -------
    None
        This test only checks successful validation.
    """

    validate_trial(_valid_trial())


def test_validate_block_rejects_non_contiguous_trial_indices() -> None:
    """Ensure block validation enforces contiguous trial indices.

    Returns
    -------
    None
        This test raises on malformed block structure.
    """

    bad_trial = Trial(
        trial_index=2,
        events=_valid_trial().events,
    )
    block = Block(block_index=0, condition="test", trials=(bad_trial,))

    with pytest.raises(ValueError, match="contiguous"):
        validate_block(block)


def test_validate_subject_rejects_empty_subject_id() -> None:
    """Ensure subject identifiers are required.

    Returns
    -------
    None
        This test raises on invalid subject metadata.
    """

    subject = SubjectData(subject_id="", blocks=())

    with pytest.raises(ValueError, match="subject_id"):
        validate_subject(subject)


def test_validate_dataset_rejects_duplicate_subject_ids() -> None:
    """Ensure datasets cannot contain duplicate subject identifiers.

    Returns
    -------
    None
        This test raises on duplicate IDs.
    """

    block = Block(block_index=0, condition="a", trials=(_valid_trial(),))
    subject = SubjectData(subject_id="dup", blocks=(block,))
    dataset = Dataset(subjects=(subject, subject))

    with pytest.raises(ValueError, match="unique"):
        validate_dataset(dataset)
