"""Canonical event-based data structures and validation helpers."""

from comp_model.data.compatibility import (
    check_kernel_schema_compatibility,
    check_spec_schema_compatibility,
)
from comp_model.data.extractors import DecisionTrialView, replay_trial_steps
from comp_model.data.schema import Block, Dataset, Event, EventPhase, SubjectData, Trial
from comp_model.data.validation import (
    validate_block,
    validate_dataset,
    validate_event,
    validate_event_payload,
    validate_subject,
    validate_trial,
)

__all__ = [
    "Block",
    "Dataset",
    "DecisionTrialView",
    "Event",
    "EventPhase",
    "SubjectData",
    "Trial",
    "check_kernel_schema_compatibility",
    "check_spec_schema_compatibility",
    "replay_trial_steps",
    "validate_block",
    "validate_dataset",
    "validate_event",
    "validate_event_payload",
    "validate_subject",
    "validate_trial",
]
