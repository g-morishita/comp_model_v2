"""Schema-specific CSV converters for fitting-oriented trial tables.

Each converter owns one flat CSV row contract for a specific
``TrialSchema.schema_id``. Export collapses the canonical event sequence for
one trial into that row shape; import performs the inverse reconstruction using
the schema's declared event order. Social schemas intentionally share one row
shape even when they differ in timing or whether the subject receives an
outcome, so some cells may be intentionally blank for certain schemas.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from comp_model.data import (
    Block,
    Dataset,
    Event,
    EventPhase,
    SubjectData,
    Trial,
    replay_trial_steps,
    validate_dataset,
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
    TrialSchema,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_COMMON_FIELDNAMES = (
    "subject_id",
    "block_index",
    "condition",
    "schema_id",
    "trial_index",
    "available_actions",
    "choice",
    "reward",
)
# Social schemas keep one stable header across variants. Schemas without a
# subject outcome still include the ``reward`` column, but export it as "".
_SOCIAL_FIELDNAMES = (*_COMMON_FIELDNAMES, "demonstrator_choice", "demonstrator_reward")


class TrialCsvConverter(Protocol):
    """Protocol for schema-specific trial-row CSV converters.

    Attributes
    ----------
    schema_id
        Schema identifier handled by this converter.
    fieldnames
        Exact CSV header expected by the converter.
    """

    @property
    def schema_id(self) -> str:
        """Return the schema identifier handled by the converter.

        Returns
        -------
        str
            Stable schema identifier.
        """

        ...

    @property
    def fieldnames(self) -> tuple[str, ...]:
        """Return the exact CSV header expected by the converter.

        Returns
        -------
        tuple[str, ...]
            Ordered CSV header columns.
        """

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
        """Flatten one canonical trial into one CSV row.

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
            String-valued CSV row matching ``fieldnames`` exactly.
        """

        ...

    def row_to_trial(self, row: Mapping[str, str], *, trial_index: int) -> Trial:
        """Rebuild one canonical trial from one CSV row.

        Parameters
        ----------
        row
            Parsed CSV row keyed by header name.
        trial_index
            Trial index assigned by the caller.

        Returns
        -------
        Trial
            Canonical trial rebuilt in the converter's schema order.
        """

        ...


def _empty_trial_mapping() -> dict[int, Trial]:
    """Create an empty trial-index mapping for CSV loading.

    Returns
    -------
    dict[int, Trial]
        Empty mutable mapping keyed by trial index.
    """

    return {}


@dataclass(slots=True)
class _BlockAccumulator:
    """Mutable block accumulator used while loading CSV rows.

    Attributes
    ----------
    condition
        Condition label associated with the block.
    trials
        Trials keyed by trial index until final reconstruction.
    """

    condition: str
    trials: dict[int, Trial] = field(default_factory=_empty_trial_mapping)


@dataclass(frozen=True, slots=True)
class _AsocialBanditTrialCsvConverter:
    """Trial-row CSV converter for the asocial bandit schema."""

    schema: TrialSchema = field(default=ASOCIAL_BANDIT_SCHEMA, init=False, repr=False)
    fieldnames: tuple[str, ...] = field(default=_COMMON_FIELDNAMES, init=False)

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

        view = _extract_single_view(trial, self.schema)
        return _build_common_row(
            subject_id=subject_id,
            block_index=block_index,
            condition=condition,
            schema_id=schema_id,
            trial_index=trial.trial_index,
            available_actions=view.available_actions,
            choice=_require_choice(view.choice, self.schema_id, trial.trial_index),
            reward=_require_reward(view.reward, self.schema_id, trial.trial_index),
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

        available_actions = _parse_available_actions(row["available_actions"])
        choice = _parse_optional_int_field(row, "choice")
        reward = _parse_optional_float_field(row, "reward")
        if choice is None:
            if reward is not None:
                raise ValueError("Field 'reward' must be empty when 'choice' is missing")
        else:
            _validate_action_in_available_set(
                action=choice,
                available_actions=available_actions,
                field_name="choice",
            )
            if reward is None:
                raise ValueError("Field 'reward' must be a float when 'choice' is present")
        return _build_trial_from_schema(
            schema=self.schema,
            trial_index=trial_index,
            available_actions=available_actions,
            choice=choice,
            reward=reward,
        )


@dataclass(frozen=True, slots=True)
class _SocialTrialCsvConverter:
    """Trial-row CSV converter shared by one-choice social schemas.

    Notes
    -----
    Pre-choice, post-outcome, action-only, no-self-outcome, and bidirectional
    schemas all flatten to the same social row shape. The bound schema still
    matters because it decides which canonical events exist and whether the
    subject's own ``reward`` cell should contain a float or remain empty.
    """

    schema: TrialSchema
    fieldnames: tuple[str, ...] = field(default=_SOCIAL_FIELDNAMES, init=False)

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

        view = _extract_single_view(trial, self.schema)
        subject_reward = (
            _require_reward(view.reward, self.schema_id, trial.trial_index)
            if self.schema.has_subject_reward
            else None
        )
        return {
            **_build_common_row(
                subject_id=subject_id,
                block_index=block_index,
                condition=condition,
                schema_id=schema_id,
                trial_index=trial.trial_index,
                available_actions=view.available_actions,
                choice=_require_choice(view.choice, self.schema_id, trial.trial_index),
                reward=subject_reward,
            ),
            "demonstrator_choice": str(
                _require_social_action(view.social_action, self.schema_id, trial.trial_index)
            ),
            "demonstrator_reward": str(
                _require_social_reward(view.social_reward, self.schema_id, trial.trial_index)
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

        available_actions = _parse_available_actions(row["available_actions"])
        choice = _parse_optional_int_field(row, "choice")
        reward = _parse_optional_float_field(row, "reward")
        demonstrator_choice = _parse_int_field(row, "demonstrator_choice")
        demonstrator_reward = _parse_float_field(row, "demonstrator_reward")
        if choice is None and reward is not None:
            raise ValueError("Field 'reward' must be empty when 'choice' is missing")
        if self.schema.has_subject_reward:
            if choice is not None and reward is None:
                raise ValueError(f"Field 'reward' must be a float for schema {self.schema_id!r}")
        elif reward is not None:
            raise ValueError(f"Field 'reward' must be empty for schema {self.schema_id!r}")
        if choice is not None:
            _validate_action_in_available_set(
                action=choice,
                available_actions=available_actions,
                field_name="choice",
            )
        _validate_action_in_available_set(
            action=demonstrator_choice,
            available_actions=available_actions,
            field_name="demonstrator_choice",
        )
        return _build_trial_from_schema(
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


_TRIAL_CSV_CONVERTERS: dict[str, TrialCsvConverter] = {}


def register_trial_csv_converter(converter: TrialCsvConverter) -> None:
    """Register a schema-specific trial CSV converter.

    Parameters
    ----------
    converter
        Converter instance keyed by its ``schema_id``.

    Returns
    -------
    None
        This function mutates the module-level converter registry.

    Raises
    ------
    ValueError
        Raised when a converter is already registered for the same schema id.
    """

    existing_converter = _TRIAL_CSV_CONVERTERS.get(converter.schema_id)
    if existing_converter is not None:
        raise ValueError(f"CSV converter already registered for schema_id {converter.schema_id!r}")
    _TRIAL_CSV_CONVERTERS[converter.schema_id] = converter


def get_trial_csv_converter(schema: TrialSchema | str) -> TrialCsvConverter:
    """Return the registered converter for a schema.

    Parameters
    ----------
    schema
        Trial schema object or schema identifier.

    Returns
    -------
    TrialCsvConverter
        Registered converter for the requested schema.

    Raises
    ------
    ValueError
        Raised when no converter is registered for the requested schema id.
    """

    schema_id = schema if isinstance(schema, str) else schema.schema_id
    converter = _TRIAL_CSV_CONVERTERS.get(schema_id)
    if converter is None:
        raise ValueError(f"No CSV converter registered for schema_id {schema_id!r}")
    return converter


def save_dataset_to_csv(dataset: Dataset, *, schema: TrialSchema, path: str | Path) -> None:
    """Save a dataset to a schema-specific trial CSV file.

    Parameters
    ----------
    dataset
        Dataset to export.
    schema
        Trial schema shared by every exported row.
    path
        Destination CSV path.

    Returns
    -------
    None
        This function writes the CSV file to disk.

    Raises
    ------
    ValueError
        Raised when a trial does not match the schema or cannot be flattened by
        the selected converter. For schemas without a subject outcome, the
        exported ``reward`` cell is left blank instead of fabricating one.
    """

    converter = get_trial_csv_converter(schema)
    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(converter.fieldnames))
        writer.writeheader()
        for subject in dataset.subjects:
            for block in subject.blocks:
                if block.schema_id != schema.schema_id:
                    raise ValueError(
                        f"Subject {subject.subject_id!r}, block {block.block_index}: "
                        f"schema_id mismatch — block has {block.schema_id!r} but "
                        f"export schema is {schema.schema_id!r}"
                    )
                for trial in block.trials:
                    schema.validate_trial(trial)
                    writer.writerow(
                        _normalize_output_row(
                            converter.trial_to_row(
                                subject_id=subject.subject_id,
                                block_index=block.block_index,
                                condition=block.condition,
                                schema_id=block.schema_id,
                                trial=trial,
                            ),
                            expected_fields=converter.fieldnames,
                        )
                    )


def _infer_available_actions(
    rows: Sequence[Mapping[str | None, object]], *, is_social: bool
) -> str:
    """Infer ``available_actions`` from buffered CSV rows.

    Collects every unique integer that appears in the ``choice`` column (and
    ``demonstrator_choice`` when ``is_social`` is true), then returns the
    sorted union formatted as a pipe-delimited string (e.g. ``"0|1|2"``).

    Parameters
    ----------
    rows
        Buffered CSV rows (list of dicts from :class:`csv.DictReader`).
    is_social
        When true, also include values from the ``demonstrator_choice`` column.

    Returns
    -------
    str
        Pipe-delimited available actions inferred from the data.

    Raises
    ------
    ValueError
        Raised when the rows are empty or action values are not integers.
    """

    action_columns = ["choice"]
    if is_social:
        action_columns.append("demonstrator_choice")
    actions: set[int] = set()
    for row_number, row in enumerate(rows, start=2):
        for col_name in action_columns:
            raw_val = row.get(col_name)
            if raw_val is None:
                raise ValueError(f"Row {row_number}: missing '{col_name}' column")
            if isinstance(raw_val, str) and _is_missing_csv_value(raw_val):
                continue
            try:
                actions.add(int(raw_val))  # type: ignore[arg-type]
            except (ValueError, TypeError) as error:
                raise ValueError(f"Row {row_number}: '{col_name}' must be an integer") from error
    if not actions:
        raise ValueError("Cannot infer available_actions from an empty CSV file")
    return _format_available_actions(tuple(sorted(actions)))


def load_dataset_from_csv(path: str | Path, *, schema: TrialSchema) -> Dataset:
    """Load a dataset from a schema-specific trial CSV file.

    Parameters
    ----------
    path
        Source CSV path.
    schema
        Trial schema shared by every row in the file.

    Returns
    -------
    Dataset
        Reconstructed canonical dataset.

    Raises
    ------
    ValueError
        Raised when headers are invalid, block conditions conflict, duplicate
        trial keys appear, or rows cannot be reconstructed for ``schema``.
        Schemas without a subject outcome require ``reward=""`` in each row.
        Blank/NA subject ``choice`` cells are reconstructed as timeout-style
        self events instead of being dropped.
    """

    converter = get_trial_csv_converter(schema)
    source = Path(path)
    subjects_by_id: dict[str, dict[int, _BlockAccumulator]] = {}
    seen_trial_keys: set[tuple[str, int, int]] = set()

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        absent_optional = _validate_header_row(
            reader.fieldnames,
            expected_fields=converter.fieldnames,
            optional_fields=frozenset({"available_actions", "schema_id"}),
        )
        infer_actions = "available_actions" in absent_optional
        infer_schema_id = "schema_id" in absent_optional
        # Optional columns can be inferred only after seeing the whole file, so
        # we buffer rows once before reconstructing trials.
        buffered_rows = list(reader)
        if infer_actions:
            is_social = "demonstrator_choice" in converter.fieldnames
            inferred_available_actions = _infer_available_actions(
                buffered_rows, is_social=is_social
            )
        else:
            inferred_available_actions = None
        effective_fields = tuple(f for f in converter.fieldnames if f not in absent_optional)

        for row_number, raw_row in enumerate(buffered_rows, start=2):
            row = _normalize_input_row(
                raw_row,
                expected_fields=effective_fields,
                row_number=row_number,
            )
            if infer_actions:
                row["available_actions"] = inferred_available_actions  # type: ignore[assignment]
            if infer_schema_id:
                row["schema_id"] = schema.schema_id
            subject_id = row["subject_id"]
            block_index = _parse_non_negative_int(row["block_index"], field_name="block_index")
            condition = row["condition"]
            row_schema_id = row["schema_id"]
            if row_schema_id != schema.schema_id:
                raise ValueError(
                    f"Row {row_number}: schema_id mismatch — row has "
                    f"{row_schema_id!r} but expected {schema.schema_id!r}"
                )
            trial_index = _parse_non_negative_int(row["trial_index"], field_name="trial_index")
            trial_key = (subject_id, block_index, trial_index)
            if trial_key in seen_trial_keys:
                raise ValueError(
                    "Duplicate trial key encountered for "
                    f"subject_id={subject_id!r}, block_index={block_index}, "
                    f"trial_index={trial_index}"
                )
            seen_trial_keys.add(trial_key)

            try:
                trial = converter.row_to_trial(row, trial_index=trial_index)
            except ValueError as error:
                raise ValueError(f"Row {row_number}: {error}") from error
            schema.validate_trial(trial)

            subject_blocks = subjects_by_id.setdefault(subject_id, {})
            block = subject_blocks.get(block_index)
            if block is None:
                block = _BlockAccumulator(condition=condition)
                subject_blocks[block_index] = block
            elif block.condition != condition:
                raise ValueError(
                    "Inconsistent condition for "
                    f"subject_id={subject_id!r}, block_index={block_index}: "
                    f"{block.condition!r} != {condition!r}"
                )
            block.trials[trial_index] = trial

    dataset = Dataset(
        subjects=tuple(
            SubjectData(
                subject_id=subject_id,
                blocks=tuple(
                    Block(
                        block_index=block_index,
                        condition=block.condition,
                        schema_id=schema.schema_id,
                        trials=tuple(
                            trial
                            for _, trial in sorted(block.trials.items(), key=lambda item: item[0])
                        ),
                    )
                    for block_index, block in sorted(
                        subject_blocks.items(),
                        key=lambda item: item[0],
                    )
                ),
            )
            for subject_id, subject_blocks in sorted(
                subjects_by_id.items(), key=lambda item: item[0]
            )
        )
    )
    validate_dataset(dataset, schema)
    return dataset


@dataclass(frozen=True, slots=True)
class _CombinedTrialView:
    """Merged per-trial data for CSV row converters.

    Aggregates the subject's choice, optional self reward, and observed social
    information from every replay step in one trial. This is an internal helper
    for flat CSV export, not a general decision view, because it intentionally
    collapses multiple replay events into one row-shaped record.
    """

    trial_index: int
    available_actions: tuple[int, ...]
    choice: int
    reward: float | None
    social_action: int | None
    social_reward: float | None
    observation: dict[str, Any]


def _extract_single_view(trial: Trial, schema: TrialSchema) -> _CombinedTrialView:
    """Collapse one trial into the row-shaped view used by built-in converters.

    Parameters
    ----------
    trial
        Canonical trial to flatten.
    schema
        Schema used to validate and extract the trial.

    Returns
    -------
    _CombinedTrialView
        Merged per-trial record combining the subject's choice, optional self
        reward, observation, and any observed social information.

    Raises
    ------
    ValueError
        Raised when the schema does not yield exactly one subject action step.
    """

    choice: int | None = None
    available_actions: tuple[int, ...] = ()
    reward: float | None = None
    social_action: int | None = None
    social_reward: float | None = None
    observation: dict[str, Any] = {}

    for event_type, learner_id, view in replay_trial_steps(trial, schema):
        if event_type == EventPhase.DECISION and learner_id == "subject":
            # The flat CSV row stores only the subject-facing decision state.
            choice = view.action
            available_actions = view.available_actions
            observation = dict(view.observation)
        elif event_type == EventPhase.UPDATE:
            # Subject-owned reward can appear in self-updates and in
            # demonstrator-facing updates for bidirectional schemas.
            if view.actor_id == "subject" and view.reward is not None:
                reward = view.reward
            if view.actor_id == "demonstrator":
                # Demonstrator reward must come from the actor's own update
                # when the subject-facing social view intentionally hides it.
                if view.action is not None:
                    social_action = view.action
                if view.reward is not None:
                    social_reward = view.reward

    if choice is None:
        raise ValueError(
            f"Schema {schema.schema_id!r} expected at least one subject action step, got none"
        )
    return _CombinedTrialView(
        trial_index=trial.trial_index,
        available_actions=available_actions,
        choice=choice,
        reward=reward,
        observation=observation,
        social_action=social_action,
        social_reward=social_reward,
    )


def _build_common_row(
    *,
    subject_id: str,
    block_index: int,
    condition: str,
    schema_id: str,
    trial_index: int,
    available_actions: tuple[int, ...],
    choice: int,
    reward: float | None,
) -> dict[str, str]:
    """Build the shared CSV columns for one trial row.

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
    trial_index
        Trial index within the block.
    available_actions
        Legal action values for the trial.
    choice
        Chosen action value.
    reward
        Observed reward, or ``None`` for schemas with no subject outcome.

    Returns
    -------
    dict[str, str]
        Shared CSV row columns as strings.
    """

    return {
        "subject_id": subject_id,
        "block_index": str(block_index),
        "condition": condition,
        "schema_id": schema_id,
        "trial_index": str(trial_index),
        "available_actions": _format_available_actions(available_actions),
        "choice": str(choice),
        "reward": "" if reward is None else str(reward),
    }


def _build_trial_from_schema(
    *,
    schema: TrialSchema,
    trial_index: int,
    available_actions: tuple[int, ...],
    choice: int | None,
    reward: float | None,
    demonstrator_observation: Mapping[str, Any] | None = None,
) -> Trial:
    """Build one canonical trial using the declared schema order.

    Parameters
    ----------
    schema
        Schema whose positional steps define event order.
    trial_index
        Trial index assigned to the rebuilt trial.
    available_actions
        Legal actions for subject and demonstrator input events.
    choice
        Chosen action value, or ``None`` for timeout-style subject rows.
    reward
        Observed reward value, or ``None`` when the schema omits the subject's
        own outcome entirely or the subject timed out.
    demonstrator_observation
        Demonstrator observation payload for non-subject input events, if any.

    Returns
    -------
    Trial
        Canonical trial rebuilt in schema order.

    Raises
    ------
    ValueError
        Raised when the schema expects a demonstrator input but no observation
        payload was supplied.
    """

    demonstrator_choice: int | None = None
    demonstrator_reward: float | None = None
    if demonstrator_observation is not None:
        demonstrator_choice = demonstrator_observation.get("social_action")
        demonstrator_reward = demonstrator_observation.get("social_reward")

    events: list[Event] = []
    for step_index, step in enumerate(schema.steps):
        # Reconstruction follows the schema verbatim: the row supplies values,
        # while the schema decides which event types actually appear.
        if step.phase == EventPhase.INPUT:
            payload: dict[str, Any] = {"available_actions": available_actions}
        elif step.phase == EventPhase.DECISION:
            if step.actor_id == "subject":
                payload = {"action": choice}
            else:
                if demonstrator_choice is None:
                    raise ValueError(f"Schema {schema.schema_id!r} requires demonstrator choice")
                payload = {"action": demonstrator_choice}
        elif step.phase == EventPhase.OUTCOME:
            if step.actor_id == "subject":
                payload = {"reward": reward}
            else:
                if demonstrator_reward is None:
                    raise ValueError(f"Schema {schema.schema_id!r} requires demonstrator reward")
                payload = {"reward": demonstrator_reward}
        elif step.phase == EventPhase.UPDATE:
            # Payload carries the actor's choice and reward for replay.
            if step.actor_id == "subject":
                payload = {"choice": choice, "reward": reward}
            else:
                if demonstrator_choice is None or demonstrator_reward is None:
                    raise ValueError(f"Schema {schema.schema_id!r} requires demonstrator data")
                payload = {"choice": demonstrator_choice, "reward": demonstrator_reward}
        else:
            raise ValueError(f"Unsupported event phase {step.phase!r}")

        events.append(
            Event(
                phase=step.phase,
                event_index=step_index,
                node_id=step.node_id,
                actor_id=step.actor_id,
                payload=payload,
            )
        )
    return Trial(trial_index=trial_index, events=tuple(events))


def _normalize_output_row(
    row: Mapping[str, str], *, expected_fields: tuple[str, ...]
) -> dict[str, str]:
    """Validate and order a converter-produced CSV row.

    Parameters
    ----------
    row
        Converter-produced row mapping.
    expected_fields
        Exact field order required by the converter.

    Returns
    -------
    dict[str, str]
        Ordered row containing exactly the required fields.

    Raises
    ------
    ValueError
        Raised when the converter omits a required field or emits an unknown
        field.
    """

    missing_fields = [field for field in expected_fields if field not in row]
    unknown_fields = sorted(field for field in row if field not in expected_fields)
    if missing_fields:
        raise ValueError(f"Converter row is missing required fields: {missing_fields}")
    if unknown_fields:
        raise ValueError(f"Converter row includes unknown fields: {unknown_fields}")
    return {field: row[field] for field in expected_fields}


def _validate_header_row(
    fieldnames: Sequence[str] | None,
    *,
    expected_fields: tuple[str, ...],
    optional_fields: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Validate a CSV header against a converter's declared columns.

    Parameters
    ----------
    fieldnames
        Header row returned by :class:`csv.DictReader`.
    expected_fields
        Exact field set required by the converter.
    optional_fields
        Field names that may be absent from the header without raising.

    Returns
    -------
    frozenset[str]
        Subset of ``optional_fields`` that were actually absent from the
        header.

    Raises
    ------
    ValueError
        Raised when the header is missing required columns or contains unknown
        columns.
    """

    if fieldnames is None:
        raise ValueError("CSV file is missing a header row")
    actual_fields = set(fieldnames)
    expected_field_set = set(expected_fields)
    all_missing = expected_field_set - actual_fields
    absent_optional = frozenset(all_missing & optional_fields)
    required_missing = sorted(all_missing - optional_fields)
    unknown_fields = sorted(actual_fields - expected_field_set)
    if required_missing:
        raise ValueError(f"Missing required columns: {required_missing}")
    if unknown_fields:
        raise ValueError(f"Unknown columns: {unknown_fields}")
    return absent_optional


def _normalize_input_row(
    raw_row: Mapping[str | None, object],
    *,
    expected_fields: tuple[str, ...],
    row_number: int,
) -> dict[str, str]:
    """Normalize one parsed CSV row to a strict string mapping.

    Parameters
    ----------
    raw_row
        Raw row mapping returned by :class:`csv.DictReader`.
    expected_fields
        Exact field set required by the converter.
    row_number
        One-based row number in the CSV file, including the header row.

    Returns
    -------
    dict[str, str]
        Row keyed only by expected fields with string values.

    Raises
    ------
    ValueError
        Raised when the row contains too many columns or missing cells.
    """

    if None in raw_row:
        raise ValueError(f"Row {row_number}: row has more columns than the header")
    normalized_row: dict[str, str] = {}
    for column_name in expected_fields:
        value = raw_row.get(column_name)
        if value is None:
            raise ValueError(f"Row {row_number}: field {column_name!r} is missing")
        if not isinstance(value, str):
            raise ValueError(f"Row {row_number}: field {column_name!r} must be a string")
        normalized_row[column_name] = value
    return normalized_row


def _format_available_actions(available_actions: tuple[int, ...]) -> str:
    """Encode legal actions into the stable CSV string form.

    Parameters
    ----------
    available_actions
        Legal actions for one trial.

    Returns
    -------
    str
        ``|``-delimited integer encoding such as ``"0|1|2"``.
    """

    return "|".join(str(action) for action in available_actions)


def _parse_available_actions(value: str) -> tuple[int, ...]:
    """Parse the stable CSV encoding for legal actions.

    Parameters
    ----------
    value
        ``|``-delimited integer action list.

    Returns
    -------
    tuple[int, ...]
        Parsed legal actions in file order.

    Raises
    ------
    ValueError
        Raised when the field is empty or contains invalid integers.
    """

    if value == "":
        raise ValueError("Field 'available_actions' must not be empty")
    tokens = value.split("|")
    if any(token == "" for token in tokens):
        raise ValueError("Field 'available_actions' contains an empty token")
    try:
        available_actions = tuple(int(token) for token in tokens)
    except ValueError as error:
        raise ValueError("Field 'available_actions' must contain only integers") from error
    if len(available_actions) == 0:
        raise ValueError("Field 'available_actions' must contain at least one action")
    if len(set(available_actions)) != len(available_actions):
        raise ValueError("Field 'available_actions' must not contain duplicate values")
    return available_actions


def _parse_non_negative_int(value: str, *, field_name: str) -> int:
    """Parse a non-negative integer CSV field.

    Parameters
    ----------
    value
        Raw CSV cell value.
    field_name
        Field name used in error messages.

    Returns
    -------
    int
        Parsed non-negative integer.

    Raises
    ------
    ValueError
        Raised when the field is not a non-negative integer.
    """

    parsed_value = _parse_int_value(value, field_name=field_name)
    if parsed_value < 0:
        raise ValueError(f"Field {field_name!r} must be non-negative")
    return parsed_value


def _parse_int_field(row: Mapping[str, str], field_name: str) -> int:
    """Parse an integer field from one normalized row.

    Parameters
    ----------
    row
        Normalized CSV row.
    field_name
        Field name to parse.

    Returns
    -------
    int
        Parsed integer value.
    """

    return _parse_int_value(row[field_name], field_name=field_name)


def _parse_optional_int_field(row: Mapping[str, str], field_name: str) -> int | None:
    """Parse an integer field that may use a blank/NA marker.

    Parameters
    ----------
    row
        Normalized CSV row.
    field_name
        Field name to parse.

    Returns
    -------
    int | None
        Parsed integer value, or ``None`` when the cell is blank or marked as
        missing.

    Raises
    ------
    ValueError
        Raised when the field is non-empty but not an integer.
    """

    if _is_missing_csv_value(row[field_name]):
        return None
    return _parse_int_field(row, field_name)


def _parse_int_value(value: str, *, field_name: str) -> int:
    """Parse a raw string as an integer field value.

    Parameters
    ----------
    value
        Raw CSV cell value.
    field_name
        Field name used in error messages.

    Returns
    -------
    int
        Parsed integer value.

    Raises
    ------
    ValueError
        Raised when the value is not an integer.
    """

    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"Field {field_name!r} must be an integer") from error


def _parse_float_field(row: Mapping[str, str], field_name: str) -> float:
    """Parse a floating-point field from one normalized row.

    Parameters
    ----------
    row
        Normalized CSV row.
    field_name
        Field name to parse.

    Returns
    -------
    float
        Parsed floating-point value.

    Raises
    ------
    ValueError
        Raised when the field is not a floating-point value.
    """

    try:
        return float(row[field_name])
    except ValueError as error:
        raise ValueError(f"Field {field_name!r} must be a float") from error


def _parse_optional_float_field(row: Mapping[str, str], field_name: str) -> float | None:
    """Parse a floating-point field that may use a blank/NA marker.

    Parameters
    ----------
    row
        Normalized CSV row.
    field_name
        Field name to parse.

    Returns
    -------
    float | None
        Parsed float value, or ``None`` when the cell is blank or marked as
        missing.

    Raises
    ------
    ValueError
        Raised when the field is non-empty but not a floating-point value.
    """

    if _is_missing_csv_value(row[field_name]):
        return None
    return _parse_float_field(row, field_name)


def _is_missing_csv_value(value: str) -> bool:
    """Return whether a CSV cell is using a common missing-value marker.

    Parameters
    ----------
    value
        Raw CSV cell value.

    Returns
    -------
    bool
        ``True`` when the cell is blank or contains a common NA token such as
        ``"NA"`` or ``"NaN"``.
    """

    return value.strip().lower() in {"", "n/a", "na", "nan", "none", "null"}


def _validate_action_in_available_set(
    *, action: int, available_actions: tuple[int, ...], field_name: str
) -> None:
    """Validate that an action appears in the legal action set.

    Parameters
    ----------
    action
        Parsed action value to validate.
    available_actions
        Legal actions for the row.
    field_name
        Field name used in error messages.

    Returns
    -------
    None
        This function raises when the action is illegal.

    Raises
    ------
    ValueError
        Raised when ``action`` is not present in ``available_actions``.
    """

    if action not in available_actions:
        raise ValueError(
            f"Field {field_name!r} must be one of available_actions {available_actions!r}"
        )


def _require_choice(choice: int | None, schema_id: str, trial_index: int) -> int:
    """Require a subject choice during CSV export.

    Parameters
    ----------
    choice
        Extracted subject choice.
    schema_id
        Schema identifier used in the error message.
    trial_index
        Trial index used in the error message.

    Returns
    -------
    int
        Concrete choice value.

    Raises
    ------
    ValueError
        Raised when the extracted trial has no subject choice.
    """

    if choice is None:
        raise ValueError(
            f"Schema {schema_id!r}, trial {trial_index}: subject choice is required for CSV export"
        )
    return choice


def _require_reward(reward: float | None, schema_id: str, trial_index: int) -> float:
    """Require a reward value during CSV export.

    Parameters
    ----------
    reward
        Extracted reward value.
    schema_id
        Schema identifier used in the error message.
    trial_index
        Trial index used in the error message.

    Returns
    -------
    float
        Concrete reward value.

    Raises
    ------
    ValueError
        Raised when the extracted trial has no reward.
    """

    if reward is None:
        raise ValueError(
            f"Schema {schema_id!r}, trial {trial_index}: reward is required for CSV export"
        )
    return reward


def _require_social_action(action: int | None, schema_id: str, trial_index: int) -> int:
    """Require a demonstrator action during social CSV export.

    Parameters
    ----------
    action
        Extracted demonstrator action.
    schema_id
        Schema identifier used in the error message.
    trial_index
        Trial index used in the error message.

    Returns
    -------
    int
        Concrete demonstrator action value.

    Raises
    ------
    ValueError
        Raised when the extracted trial has no demonstrator action.
    """

    if action is None:
        raise ValueError(
            f"Schema {schema_id!r}, trial {trial_index}: demonstrator action is required "
            "for CSV export"
        )
    return action


def _require_social_reward(reward: float | None, schema_id: str, trial_index: int) -> float:
    """Require a demonstrator reward during social CSV export.

    Parameters
    ----------
    reward
        Extracted demonstrator reward.
    schema_id
        Schema identifier used in the error message.
    trial_index
        Trial index used in the error message.

    Returns
    -------
    float
        Concrete demonstrator reward value.

    Raises
    ------
    ValueError
        Raised when the extracted trial has no demonstrator reward.
    """

    if reward is None:
        raise ValueError(
            f"Schema {schema_id!r}, trial {trial_index}: demonstrator reward is required "
            "for CSV export"
        )
    return reward


def _register_builtin_converters() -> None:
    """Populate the module registry with built-in schema converters.

    This function is idempotent: it skips any schema already present in the
    registry so that module reloads in test environments do not raise.

    Returns
    -------
    None
        This function mutates the module-level registry during import.
    """

    builtin_converters: tuple[TrialCsvConverter, ...] = (
        _AsocialBanditTrialCsvConverter(),
        _SocialTrialCsvConverter(SOCIAL_PRE_CHOICE_SCHEMA),
        _SocialTrialCsvConverter(SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA),
        _SocialTrialCsvConverter(SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA),
        _SocialTrialCsvConverter(SOCIAL_PRE_CHOICE_DEMO_LEARNS_SCHEMA),
        _SocialTrialCsvConverter(SOCIAL_POST_OUTCOME_SCHEMA),
        _SocialTrialCsvConverter(SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA),
        _SocialTrialCsvConverter(SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA),
        _SocialTrialCsvConverter(SOCIAL_POST_OUTCOME_DEMO_LEARNS_SCHEMA),
    )
    for converter in builtin_converters:
        if converter.schema_id not in _TRIAL_CSV_CONVERTERS:
            register_trial_csv_converter(converter)


_register_builtin_converters()
