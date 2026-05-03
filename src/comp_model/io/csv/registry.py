"""Registry for schema-specific trial CSV converters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from comp_model.io.csv.converters import builtin_trial_csv_converters

if TYPE_CHECKING:
    from comp_model.io.csv.base import TrialCsvConverter
    from comp_model.tasks import TrialSchema

_TRIAL_CSV_CONVERTERS: dict[str, TrialCsvConverter] = {}


def register_trial_csv_converter(converter: TrialCsvConverter) -> None:
    """Register a schema-specific trial CSV converter."""

    existing_converter = _TRIAL_CSV_CONVERTERS.get(converter.schema_id)
    if existing_converter is not None:
        raise ValueError(f"CSV converter already registered for schema_id {converter.schema_id!r}")
    _TRIAL_CSV_CONVERTERS[converter.schema_id] = converter


def get_trial_csv_converter(schema: TrialSchema | str) -> TrialCsvConverter:
    """Return the registered converter for a schema."""

    schema_id = schema if isinstance(schema, str) else schema.schema_id
    converter = _TRIAL_CSV_CONVERTERS.get(schema_id)
    if converter is None:
        raise ValueError(f"No CSV converter registered for schema_id {schema_id!r}")
    return converter


def _register_builtin_converters() -> None:
    """Populate the module registry with built-in schema converters."""

    for converter in builtin_trial_csv_converters():
        if converter.schema_id not in _TRIAL_CSV_CONVERTERS:
            register_trial_csv_converter(converter)


_register_builtin_converters()
