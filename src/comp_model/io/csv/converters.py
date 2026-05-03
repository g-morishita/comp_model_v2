"""Built-in schema-specific trial CSV converters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from comp_model.io.csv.base import COMMON_FIELDNAMES, SOCIAL_FIELDNAMES, TrialCsvConverter
from comp_model.io.csv.parsing import (
    parse_available_actions,
    parse_float_field,
    parse_int_field,
    parse_optional_float_field,
    parse_optional_int_field,
    require_social_action,
    require_social_reward,
    subject_reward_for_csv_export,
    validate_action_in_available_set,
)
from comp_model.io.csv.views import (
    build_common_row,
    build_trial_from_schema,
    extract_single_view,
)
from comp_model.tasks import (
    ASOCIAL_BANDIT_SCHEMA,
    SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA,
    SOCIAL_POST_OUTCOME_DEMO_LEARNS_SCHEMA,
    SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA,
    SOCIAL_POST_OUTCOME_SCHEMA,
    SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
    SOCIAL_PRE_CHOICE_DEMO_LEARNS_SCHEMA,
    SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA,
    SOCIAL_PRE_CHOICE_SCHEMA,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from comp_model.data import Trial
    from comp_model.tasks import TrialSchema


@dataclass(frozen=True, slots=True)
class AsocialBanditTrialCsvConverter:
    """Trial-row CSV converter for the asocial bandit schema."""

    schema: TrialSchema = field(default=ASOCIAL_BANDIT_SCHEMA, init=False, repr=False)
    fieldnames: tuple[str, ...] = field(default=COMMON_FIELDNAMES, init=False)

    @property
    def schema_id(self) -> str:
        """Return the schema identifier for the asocial bandit schema.

        Returns
        -------
        str
            Stable schema identifier derived from the bound schema.
        """

        return self.schema.schema_id

    def trial_to_row(
        self,
        *,
        subject_id: str,
        block_index: int,
        condition: str,
        schema_id: str,
        trial: Trial,
    ) -> dict[str, str]:
        """Flatten one asocial trial into one CSV row.

        Parameters
        ----------
        subject_id
            Subject identifier for the containing subject.
        block_index
            Block index for the containing block.
        condition
            Condition label for the containing block.
        schema_id
            Schema identifier for the containing block.
        trial
            Canonical trial to flatten.

        Returns
        -------
        dict[str, str]
            String-valued CSV row for the asocial schema.
        """

        view = extract_single_view(trial, self.schema)
        reward = subject_reward_for_csv_export(
            choice=view.choice,
            reward=view.reward,
            schema=self.schema,
            trial_index=trial.trial_index,
        )
        return build_common_row(
            subject_id=subject_id,
            block_index=block_index,
            condition=condition,
            schema_id=schema_id,
            trial_index=trial.trial_index,
            available_actions=view.available_actions,
            choice=view.choice,
            reward=reward,
        )

    def row_to_trial(self, row: Mapping[str, str], *, trial_index: int) -> Trial:
        """Rebuild one asocial trial from one CSV row.

        Parameters
        ----------
        row
            Parsed CSV row keyed by header name.
        trial_index
            Trial index assigned by the caller.

        Returns
        -------
        Trial
            Canonical asocial trial in schema order. Blank/NA ``choice`` and
            ``reward`` cells are reconstructed as a timeout-style self event
            with ``None`` payload values.
        """

        available_actions = parse_available_actions(row["available_actions"])
        choice = parse_optional_int_field(row, "choice")
        reward = parse_optional_float_field(row, "reward")
        if choice is None:
            if reward is not None:
                raise ValueError("Field 'reward' must be empty when 'choice' is missing")
        else:
            validate_action_in_available_set(
                action=choice,
                available_actions=available_actions,
                field_name="choice",
            )
            if reward is None:
                raise ValueError("Field 'reward' must be a float when 'choice' is present")
        return build_trial_from_schema(
            schema=self.schema,
            trial_index=trial_index,
            available_actions=available_actions,
            choice=choice,
            reward=reward,
        )


@dataclass(frozen=True, slots=True)
class SocialTrialCsvConverter:
    """Trial-row CSV converter shared by one-choice social schemas.

    Notes
    -----
    Pre-choice, post-outcome, action-only, no-self-outcome, and bidirectional
    schemas all flatten to the same social row shape. The bound schema still
    matters because it decides which canonical events exist and whether the
    subject's own ``reward`` cell should contain a float or remain empty.
    """

    schema: TrialSchema
    fieldnames: tuple[str, ...] = field(default=SOCIAL_FIELDNAMES, init=False)

    @property
    def schema_id(self) -> str:
        """Return the schema identifier for the bound social schema.

        Returns
        -------
        str
            Stable schema identifier derived from the bound schema.
        """

        return self.schema.schema_id

    def trial_to_row(
        self,
        *,
        subject_id: str,
        block_index: int,
        condition: str,
        schema_id: str,
        trial: Trial,
    ) -> dict[str, str]:
        """Flatten one social trial into one CSV row.

        Parameters
        ----------
        subject_id
            Subject identifier for the containing subject.
        block_index
            Block index for the containing block.
        condition
            Condition label for the containing block.
        schema_id
            Schema identifier for the containing block.
        trial
            Canonical trial to flatten.

        Returns
        -------
        dict[str, str]
            String-valued CSV row for the social schema. Schemas without a
            subject outcome serialize ``reward`` as ``""``.
        """

        view = extract_single_view(trial, self.schema)
        subject_reward = subject_reward_for_csv_export(
            choice=view.choice,
            reward=view.reward,
            schema=self.schema,
            trial_index=trial.trial_index,
        )
        return {
            **build_common_row(
                subject_id=subject_id,
                block_index=block_index,
                condition=condition,
                schema_id=schema_id,
                trial_index=trial.trial_index,
                available_actions=view.available_actions,
                choice=view.choice,
                reward=subject_reward,
            ),
            "demonstrator_choice": str(
                require_social_action(view.social_action, self.schema_id, trial.trial_index)
            ),
            "demonstrator_reward": str(
                require_social_reward(view.social_reward, self.schema_id, trial.trial_index)
            ),
        }

    def row_to_trial(self, row: Mapping[str, str], *, trial_index: int) -> Trial:
        """Rebuild one social trial from one CSV row.

        Parameters
        ----------
        row
            Parsed CSV row keyed by header name.
        trial_index
            Trial index assigned by the caller.

        Returns
        -------
        Trial
            Canonical social trial rebuilt in schema order. Schemas without a
            subject outcome accept ``reward=""`` and reconstruct no subject
            OUTCOME or self-UPDATE event. Blank/NA subject ``choice`` and
            ``reward`` cells are reconstructed as a timeout-style subject row
            while preserving demonstrator information.
        """

        available_actions = parse_available_actions(row["available_actions"])
        choice = parse_optional_int_field(row, "choice")
        reward = parse_optional_float_field(row, "reward")
        demonstrator_choice = parse_int_field(row, "demonstrator_choice")
        demonstrator_reward = parse_float_field(row, "demonstrator_reward")
        if choice is None and reward is not None:
            raise ValueError("Field 'reward' must be empty when 'choice' is missing")
        if self.schema.has_subject_reward:
            if choice is not None and reward is None:
                raise ValueError(f"Field 'reward' must be a float for schema {self.schema_id!r}")
        elif reward is not None:
            raise ValueError(f"Field 'reward' must be empty for schema {self.schema_id!r}")
        if choice is not None:
            validate_action_in_available_set(
                action=choice,
                available_actions=available_actions,
                field_name="choice",
            )
        validate_action_in_available_set(
            action=demonstrator_choice,
            available_actions=available_actions,
            field_name="demonstrator_choice",
        )
        return build_trial_from_schema(
            schema=self.schema,
            trial_index=trial_index,
            available_actions=available_actions,
            choice=choice,
            reward=reward,
            demonstrator_observation={
                "social_action": demonstrator_choice,
                "social_reward": demonstrator_reward,
            },
        )


def builtin_trial_csv_converters() -> tuple[TrialCsvConverter, ...]:
    """Return built-in schema-specific trial CSV converters."""

    return (
        AsocialBanditTrialCsvConverter(),
        SocialTrialCsvConverter(SOCIAL_PRE_CHOICE_SCHEMA),
        SocialTrialCsvConverter(SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA),
        SocialTrialCsvConverter(SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA),
        SocialTrialCsvConverter(SOCIAL_PRE_CHOICE_DEMO_LEARNS_SCHEMA),
        SocialTrialCsvConverter(SOCIAL_POST_OUTCOME_SCHEMA),
        SocialTrialCsvConverter(SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA),
        SocialTrialCsvConverter(SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA),
        SocialTrialCsvConverter(SOCIAL_POST_OUTCOME_DEMO_LEARNS_SCHEMA),
    )
