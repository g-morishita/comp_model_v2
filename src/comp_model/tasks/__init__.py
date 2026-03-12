"""Task specifications and declarative trial schemas."""

from comp_model.tasks.schemas import (
    ASOCIAL_BANDIT_SCHEMA,
    SOCIAL_POST_OUTCOME_SCHEMA,
    SOCIAL_PRE_CHOICE_SCHEMA,
    TrialSchema,
    TrialSchemaStep,
)
from comp_model.tasks.spec import BlockSpec, TaskSpec

__all__ = [
    "ASOCIAL_BANDIT_SCHEMA",
    "SOCIAL_POST_OUTCOME_SCHEMA",
    "SOCIAL_PRE_CHOICE_SCHEMA",
    "BlockSpec",
    "TaskSpec",
    "TrialSchema",
    "TrialSchemaStep",
]
