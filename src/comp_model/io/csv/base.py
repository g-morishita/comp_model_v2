"""Shared protocol and row shapes for schema-specific trial CSV conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from comp_model.data import Trial


COMMON_FIELDNAMES = (
    "subject_id",
    "block_index",
    "condition",
    "schema_id",
    "trial_index",
    "available_actions",
    "choice",
    "reward",
)
SOCIAL_FIELDNAMES = (*COMMON_FIELDNAMES, "demonstrator_choice", "demonstrator_reward")


class TrialCsvConverter(Protocol):
    """Protocol for schema-specific trial-row CSV converters."""

    @property
    def schema_id(self) -> str:
        """Return the schema identifier handled by this converter."""

        ...

    @property
    def fieldnames(self) -> tuple[str, ...]:
        """Return the exact CSV header expected by this converter."""

        ...

    def trial_to_row(
        self,
        *,
        subject_id: str,
        block_index: int,
        condition: str,
        schema_id: str,
        trial: Trial,
    ) -> dict[str, str]:
        """Flatten one canonical trial into one CSV row."""

        ...

    def row_to_trial(self, row: Mapping[str, str], *, trial_index: int) -> Trial:
        """Rebuild one canonical trial from one CSV row."""

        ...
