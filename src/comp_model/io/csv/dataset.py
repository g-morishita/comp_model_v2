"""File-level import and export for schema-specific trial CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from comp_model.data import Block, Dataset, SubjectData, Trial, validate_dataset
from comp_model.io.csv.parsing import (
    format_available_actions,
    is_missing_csv_value,
    normalize_input_row,
    normalize_output_row,
    parse_non_negative_int,
    validate_header_row,
)
from comp_model.io.csv.registry import get_trial_csv_converter

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from comp_model.tasks import TrialSchema


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
                        normalize_output_row(
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
            if is_missing_csv_value(cast("str", raw_val)):
                continue
            try:
                actions.add(int(raw_val))  # type: ignore[arg-type]
            except (ValueError, TypeError) as error:
                raise ValueError(f"Row {row_number}: '{col_name}' must be an integer") from error
    if not actions:
        raise ValueError("Cannot infer available_actions from an empty CSV file")
    return format_available_actions(tuple(sorted(actions)))


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
        absent_optional = validate_header_row(
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
            row = normalize_input_row(
                raw_row,
                expected_fields=effective_fields,
                row_number=row_number,
            )
            if infer_actions:
                row["available_actions"] = inferred_available_actions  # type: ignore[assignment]
            if infer_schema_id:
                row["schema_id"] = schema.schema_id
            subject_id = row["subject_id"]
            block_index = parse_non_negative_int(row["block_index"], field_name="block_index")
            condition = row["condition"]
            row_schema_id = row["schema_id"]
            if row_schema_id != schema.schema_id:
                raise ValueError(
                    f"Row {row_number}: schema_id mismatch — row has "
                    f"{row_schema_id!r} but expected {schema.schema_id!r}"
                )
            trial_index = parse_non_negative_int(row["trial_index"], field_name="trial_index")
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
