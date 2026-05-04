"""Parsing and validation helpers for trial CSV rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from comp_model.tasks import TrialSchema

_NA_MARKERS = frozenset({"", "n/a", "na", "nan", "none", "null"})


def normalize_output_row(
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


def validate_header_row(
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


def normalize_input_row(
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


def format_available_actions(available_actions: tuple[int, ...]) -> str:
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


def parse_available_actions(value: str) -> tuple[int, ...]:
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


def parse_non_negative_int(value: str, *, field_name: str) -> int:
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


def parse_int_field(row: Mapping[str, str], field_name: str) -> int:
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


def parse_optional_int_field(row: Mapping[str, str], field_name: str) -> int | None:
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

    if is_missing_csv_value(row[field_name]):
        return None
    return parse_int_field(row, field_name)


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


def parse_float_field(row: Mapping[str, str], field_name: str) -> float:
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


def parse_optional_float_field(row: Mapping[str, str], field_name: str) -> float | None:
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

    if is_missing_csv_value(row[field_name]):
        return None
    return parse_float_field(row, field_name)


def is_missing_csv_value(value: str) -> bool:
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

    return value.strip().lower() in _NA_MARKERS


def validate_action_in_available_set(
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


def subject_reward_for_csv_export(
    *,
    choice: int | None,
    reward: float | None,
    schema: TrialSchema,
    trial_index: int,
) -> float | None:
    """Normalize the subject reward field when exporting one CSV row.

    Parameters
    ----------
    choice
        Subject choice extracted from the canonical trial, or ``None`` for a
        timeout-style row.
    reward
        Subject reward extracted from the canonical trial.
    schema
        Trial schema driving export.
    trial_index
        Trial index used in error messages.

    Returns
    -------
    float | None
        Reward value to serialize, or ``None`` when the CSV row should leave
        the reward cell blank.

    Raises
    ------
    ValueError
        Raised when a timeout row still carries a reward, or when a
        reward-bearing schema has a concrete choice but no reward.
    """

    if choice is None:
        if reward is not None:
            raise ValueError(
                f"Schema {schema.schema_id!r}, trial {trial_index}: timeout rows must not "
                "carry a subject reward"
            )
        return None
    if not schema.has_subject_reward:
        return None
    return _require_reward(reward, schema.schema_id, trial_index)


def require_social_action(action: int | None, schema_id: str, trial_index: int) -> int:
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


def require_social_reward(reward: float | None, schema_id: str, trial_index: int) -> float:
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
