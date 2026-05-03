# Trial CSV Refactor Design

## Context

`src/comp_model/io/trial_csv.py` currently owns the whole schema-specific CSV
surface in one file. It contains the converter protocol, field constants,
asocial and social converters, converter registration, dataset load/save,
trial-view extraction, trial reconstruction, row normalization, scalar parsing,
and validation helpers. The file is about 1,470 lines, while
`tests/test_io/test_trial_csv.py` provides broad behavior coverage for
round-trips, timeout rows, header validation, schema mismatch checks, inferred
columns, and converter registration.

This repository is not public yet, so the refactor does not need to preserve
direct imports from `comp_model.io.trial_csv`.

## Goal

Split the CSV implementation into a focused `comp_model.io.csv` package while
preserving existing CSV behavior, row contracts, and the high-level
`comp_model.io` import surface.

## Non-Goals

- Do not change CSV column names or serialized values.
- Do not add support for new schemas.
- Do not change the public functions exported from `comp_model.io`.
- Do not preserve `comp_model.io.trial_csv` as a compatibility module.
- Do not refactor model kernels, Stan adapters, or runtime simulation in this
  change.

## Proposed Package Layout

```text
src/comp_model/io/csv/
  __init__.py
  base.py
  registry.py
  dataset.py
  views.py
  parsing.py
  converters.py
```

`src/comp_model/io/trial_csv.py` will be deleted after its contents are moved.

## Module Responsibilities

### `base.py`

Owns shared protocol and row shape constants:

- `TrialCsvConverter`
- `_COMMON_FIELDNAMES`
- `_SOCIAL_FIELDNAMES`

The protocol remains intentionally small: `schema_id`, `fieldnames`,
`trial_to_row()`, and `row_to_trial()`.

### `parsing.py`

Owns CSV-cell and row-level parsing helpers:

- missing-value marker handling
- `available_actions` formatting and parsing
- integer and float parsing
- action-in-available-set validation
- input/output row normalization
- header validation
- subject/social reward requirement helpers

This module has no dependency on concrete converters.

### `views.py`

Owns conversion between canonical event traces and row-shaped trial data:

- `_CombinedTrialView`
- `_extract_single_view()`
- `_build_common_row()`
- `_build_trial_from_schema()`

This module depends on canonical data objects, schemas, and
`replay_trial_steps()`, but not on CSV registry state.

### `converters.py`

Owns built-in schema converters:

- `AsocialBanditTrialCsvConverter`
- `SocialTrialCsvConverter`
- `builtin_trial_csv_converters()`

The converter classes can become public within `comp_model.io.csv` because the
old private names are being removed with `trial_csv.py`.

### `registry.py`

Owns converter registration and lookup:

- `register_trial_csv_converter()`
- `get_trial_csv_converter()`

At import time, it registers `builtin_trial_csv_converters()`. Registration
keeps the existing duplicate-schema error behavior.

### `dataset.py`

Owns file-level dataset import/export:

- `_BlockAccumulator`
- `_infer_available_actions()`
- `save_dataset_to_csv()`
- `load_dataset_from_csv()`

This module uses `get_trial_csv_converter()` to select the converter, then
performs the same header validation, row buffering, duplicate-key checks,
schema validation, and canonical `Dataset` reconstruction as the current file.

### `__init__.py`

Re-exports the intended direct CSV API:

- `TrialCsvConverter`
- `get_trial_csv_converter`
- `load_dataset_from_csv`
- `register_trial_csv_converter`
- `save_dataset_to_csv`

`src/comp_model/io/__init__.py` will import these from `comp_model.io.csv`
instead of `comp_model.io.trial_csv`.

## Import Updates

All repository imports of `comp_model.io.trial_csv` will move to either:

- `from comp_model.io import ...` for normal usage, or
- `from comp_model.io.csv import ...` for direct CSV-package usage.

Examples and tests that already import from `comp_model.io` should require no
semantic change. Tests that deliberately import `comp_model.io.trial_csv` should
be updated because the old module is intentionally removed.

## Behavior Preservation

The refactor must preserve:

- built-in converter lookup for every currently registered schema
- duplicate converter registration errors
- exact asocial and social CSV headers
- round-trip preservation of fitting-relevant replay views
- action-only social schemas serializing demonstrator reward while hiding it
  from subject-facing social updates
- no-self-outcome schemas using blank subject reward cells
- timeout-style rows with blank/NA choices and rewards
- optional `available_actions` and `schema_id` inference on load
- duplicate trial-key rejection
- inconsistent block-condition rejection
- unknown/missing header rejection
- schema mismatch rejection

## Error Handling

The refactor should keep existing error messages close to their current wording
where tests or downstream diagnostics rely on them. When wrapping converter
errors during load, row-number context remains attached as `Row N: ...`.

## Testing

Run the IO-focused tests first:

```bash
uv run python -m pytest tests/test_io/test_trial_csv.py -q
```

Then run schema safeguard tests that import CSV helpers directly:

```bash
uv run python -m pytest tests/test_data/test_schema_safeguard.py -q
```

Finally run a broader non-Stan check if the focused tests pass:

```bash
uv run python -m pytest -m "not stan" --tb=short -q
```

Use `uv run python -m pytest` rather than `uv run pytest` in this workspace,
because the local `.venv/bin/pytest` entry point has previously had a stale
shebang.

## Rollout

1. Create `src/comp_model/io/csv/`.
2. Move helpers into the new modules with imports adjusted.
3. Update `src/comp_model/io/__init__.py`.
4. Update direct `comp_model.io.trial_csv` imports in tests and docs.
5. Delete `src/comp_model/io/trial_csv.py`.
6. Run focused tests, then the non-Stan suite.

## Risks

The main risk is circular imports between `registry.py`, `dataset.py`, and
`converters.py`. Keep dependencies one-directional:

```text
base/parsing/views -> converters -> registry -> dataset -> io.__init__
```

Another risk is accidentally making formerly private helper names part of the
stable API. Limit `__all__` in `comp_model.io.csv` to the intended direct API.
