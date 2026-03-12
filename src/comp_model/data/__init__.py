"""Canonical event-based data structures and validation helpers."""

from comp_model.data.extractors import DecisionTrialView, extract_decision_views
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
    "extract_decision_views",
    "validate_block",
    "validate_dataset",
    "validate_event",
    "validate_event_payload",
    "validate_subject",
    "validate_trial",
]
