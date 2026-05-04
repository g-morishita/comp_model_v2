# Trial CSV Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `src/comp_model/io/trial_csv.py` into a focused `comp_model.io.csv` package while preserving existing CSV behavior and deleting the old direct module.

**Architecture:** Move the current monolithic implementation into `base`, `parsing`, `views`, `converters`, `registry`, and `dataset` modules under `src/comp_model/io/csv/`. Keep `comp_model.io` as the high-level import surface, expose direct CSV imports from `comp_model.io.csv`, and remove `comp_model.io.trial_csv` without a compatibility shim.

**Tech Stack:** Python 3.11, dataclasses, Protocol typing, stdlib `csv`, pytest, ruff, pyright.

---

## File Structure

Create:

- `src/comp_model/io/csv/__init__.py`: direct CSV package exports.
- `src/comp_model/io/csv/base.py`: protocol and fieldname constants.
- `src/comp_model/io/csv/parsing.py`: row normalization, scalar parsing, missing-value handling, and validation helpers.
- `src/comp_model/io/csv/views.py`: row-shaped trial view extraction and canonical trial reconstruction.
- `src/comp_model/io/csv/converters.py`: built-in asocial and social converters.
- `src/comp_model/io/csv/registry.py`: converter registry and built-in registration.
- `src/comp_model/io/csv/dataset.py`: file-level `save_dataset_to_csv()` and `load_dataset_from_csv()`.

Modify:

- `src/comp_model/io/__init__.py`: import from `comp_model.io.csv`.
- `tests/test_io/test_trial_csv.py`: add direct `comp_model.io.csv` export coverage.
- `tests/test_data/test_schema_safeguard.py`: replace direct imports from `comp_model.io.trial_csv`.
- `docs/mental_model.md`: update the CSV module reference.
- `docs/refactoring_points.md`: if present in the worktree, mark the CSV split as in progress or update the module path reference.

Delete:

- `src/comp_model/io/trial_csv.py`.

Do not modify unrelated untracked files or generated Stan executable artifacts.

### Task 1: Add Direct CSV Package Import Test

**Files:**

- Modify: `tests/test_io/test_trial_csv.py`

- [ ] **Step 1: Add a failing import-surface test**

Add this test after `test_get_trial_csv_converter_returns_registered_builtin()`:

```python
def test_csv_package_exports_public_api() -> None:
    """Ensure the direct CSV package exposes the intended public functions.

    Returns
    -------
    None
        This test asserts direct ``comp_model.io.csv`` imports only.
    """

    from comp_model.io.csv import (
        TrialCsvConverter,
        get_trial_csv_converter as direct_get_trial_csv_converter,
        load_dataset_from_csv as direct_load_dataset_from_csv,
        register_trial_csv_converter as direct_register_trial_csv_converter,
        save_dataset_to_csv as direct_save_dataset_to_csv,
    )

    assert TrialCsvConverter is not None
    assert direct_get_trial_csv_converter is get_trial_csv_converter
    assert direct_load_dataset_from_csv is load_dataset_from_csv
    assert direct_register_trial_csv_converter is register_trial_csv_converter
    assert direct_save_dataset_to_csv is save_dataset_to_csv
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
uv run python -m pytest tests/test_io/test_trial_csv.py::test_csv_package_exports_public_api -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'comp_model.io.csv'`.

- [ ] **Step 3: Commit the failing test**

Run:

```bash
git add tests/test_io/test_trial_csv.py
git commit --no-verify -m "test: cover direct csv io package exports"
```

Use `--no-verify` because this workspace's generated pre-commit hook points at
`comp_model_v3`. Before committing, run `git diff --check -- tests/test_io/test_trial_csv.py`.

### Task 2: Create CSV Package Base And Parsing Modules

**Files:**

- Create: `src/comp_model/io/csv/base.py`
- Create: `src/comp_model/io/csv/parsing.py`

- [ ] **Step 1: Create `base.py`**

Create `src/comp_model/io/csv/base.py` with:

```python
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
```

- [ ] **Step 2: Create `parsing.py`**

Move these definitions verbatim from `src/comp_model/io/trial_csv.py` into
`src/comp_model/io/csv/parsing.py`, preserving behavior and docstrings:

- `_NA_MARKERS`
- `_normalize_output_row()`
- `_validate_header_row()`
- `_normalize_input_row()`
- `_format_available_actions()`
- `_parse_available_actions()`
- `_parse_non_negative_int()`
- `_parse_int_field()`
- `_parse_optional_int_field()`
- `_parse_int_value()`
- `_parse_float_field()`
- `_parse_optional_float_field()`
- `_is_missing_csv_value()`
- `_validate_action_in_available_set()`
- `_require_reward()`
- `_subject_reward_for_csv_export()`
- `_require_social_action()`
- `_require_social_reward()`

Use this import header:

```python
"""Parsing and validation helpers for trial CSV rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from comp_model.tasks import TrialSchema

_NA_MARKERS = frozenset({"", "n/a", "na", "nan", "none", "null"})
```

- [ ] **Step 3: Run syntax import check**

Run:

```bash
uv run python -m py_compile src/comp_model/io/csv/base.py src/comp_model/io/csv/parsing.py
```

Expected: exit 0.

- [ ] **Step 4: Commit base and parsing modules**

Run:

```bash
git add src/comp_model/io/csv/base.py src/comp_model/io/csv/parsing.py
git commit --no-verify -m "refactor: add csv base and parsing modules"
```

Before committing, run:

```bash
git diff --check -- src/comp_model/io/csv/base.py src/comp_model/io/csv/parsing.py
```

### Task 3: Create Trial View And Converter Modules

**Files:**

- Create: `src/comp_model/io/csv/views.py`
- Create: `src/comp_model/io/csv/converters.py`

- [ ] **Step 1: Create `views.py`**

Move these definitions verbatim from `src/comp_model/io/trial_csv.py` into
`src/comp_model/io/csv/views.py`, preserving behavior and docstrings:

- `_CombinedTrialView`
- `_extract_single_view()`
- `_build_common_row()`
- `_build_trial_from_schema()`

Use this import header:

```python
"""Conversion between canonical trial events and row-shaped CSV views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from comp_model.data import Event, EventPhase, Trial, replay_trial_steps
from comp_model.io.csv.parsing import _format_available_actions

if TYPE_CHECKING:
    from collections.abc import Mapping

    from comp_model.tasks import TrialSchema
```

- [ ] **Step 2: Create `converters.py`**

Move the current converter classes from `src/comp_model/io/trial_csv.py` into
`src/comp_model/io/csv/converters.py`, renaming:

- `_AsocialBanditTrialCsvConverter` to `AsocialBanditTrialCsvConverter`
- `_SocialTrialCsvConverter` to `SocialTrialCsvConverter`

Use this import header:

```python
"""Built-in schema-specific trial CSV converters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from comp_model.io.csv.base import COMMON_FIELDNAMES, SOCIAL_FIELDNAMES, TrialCsvConverter
from comp_model.io.csv.parsing import (
    _parse_available_actions,
    _parse_float_field,
    _parse_int_field,
    _parse_optional_float_field,
    _parse_optional_int_field,
    _require_social_action,
    _require_social_reward,
    _subject_reward_for_csv_export,
    _validate_action_in_available_set,
)
from comp_model.io.csv.views import _build_common_row, _build_trial_from_schema, _extract_single_view
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
```

Define the built-in converter factory at the bottom of `converters.py`:

```python
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
```

- [ ] **Step 3: Run syntax import check**

Run:

```bash
uv run python -m py_compile src/comp_model/io/csv/views.py src/comp_model/io/csv/converters.py
```

Expected: exit 0.

- [ ] **Step 4: Commit view and converter modules**

Run:

```bash
git add src/comp_model/io/csv/views.py src/comp_model/io/csv/converters.py
git commit --no-verify -m "refactor: move csv trial views and converters"
```

Before committing, run:

```bash
git diff --check -- src/comp_model/io/csv/views.py src/comp_model/io/csv/converters.py
```

### Task 4: Create Registry And Dataset Modules

**Files:**

- Create: `src/comp_model/io/csv/registry.py`
- Create: `src/comp_model/io/csv/dataset.py`

- [ ] **Step 1: Create `registry.py`**

Create `src/comp_model/io/csv/registry.py` with:

```python
"""Registry for schema-specific trial CSV converters."""

from __future__ import annotations

from comp_model.io.csv.base import TrialCsvConverter
from comp_model.io.csv.converters import builtin_trial_csv_converters
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
```

- [ ] **Step 2: Create `dataset.py`**

Move these definitions verbatim from `src/comp_model/io/trial_csv.py` into
`src/comp_model/io/csv/dataset.py`, preserving behavior and docstrings:

- `_empty_trial_mapping()`
- `_BlockAccumulator`
- `save_dataset_to_csv()`
- `_infer_available_actions()`
- `load_dataset_from_csv()`

Use this import header:

```python
"""File-level import and export for schema-specific trial CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from comp_model.data import Block, Dataset, SubjectData, Trial, validate_dataset
from comp_model.io.csv.parsing import (
    _format_available_actions,
    _is_missing_csv_value,
    _normalize_input_row,
    _normalize_output_row,
    _parse_non_negative_int,
    _validate_header_row,
)
from comp_model.io.csv.registry import get_trial_csv_converter

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from comp_model.tasks import TrialSchema
```

- [ ] **Step 3: Run syntax import check**

Run:

```bash
uv run python -m py_compile src/comp_model/io/csv/registry.py src/comp_model/io/csv/dataset.py
```

Expected: exit 0.

- [ ] **Step 4: Commit registry and dataset modules**

Run:

```bash
git add src/comp_model/io/csv/registry.py src/comp_model/io/csv/dataset.py
git commit --no-verify -m "refactor: move csv registry and dataset io"
```

Before committing, run:

```bash
git diff --check -- src/comp_model/io/csv/registry.py src/comp_model/io/csv/dataset.py
```

### Task 5: Wire Public Imports And Remove Old Module

**Files:**

- Create: `src/comp_model/io/csv/__init__.py`
- Modify: `src/comp_model/io/__init__.py`
- Delete: `src/comp_model/io/trial_csv.py`

- [ ] **Step 1: Create `src/comp_model/io/csv/__init__.py`**

Create:

```python
"""Schema-specific CSV import and export helpers."""

from comp_model.io.csv.base import TrialCsvConverter
from comp_model.io.csv.dataset import load_dataset_from_csv, save_dataset_to_csv
from comp_model.io.csv.registry import get_trial_csv_converter, register_trial_csv_converter

__all__ = [
    "TrialCsvConverter",
    "get_trial_csv_converter",
    "load_dataset_from_csv",
    "register_trial_csv_converter",
    "save_dataset_to_csv",
]
```

- [ ] **Step 2: Update `src/comp_model/io/__init__.py`**

Replace the import block with:

```python
"""File import and export helpers."""

from comp_model.io.csv import (
    TrialCsvConverter,
    get_trial_csv_converter,
    load_dataset_from_csv,
    register_trial_csv_converter,
    save_dataset_to_csv,
)
```

Keep the existing `__all__` list unchanged.

- [ ] **Step 3: Delete the old module**

Run:

```bash
git rm src/comp_model/io/trial_csv.py
```

Expected: `rm 'src/comp_model/io/trial_csv.py'`.

- [ ] **Step 4: Run focused import test**

Run:

```bash
uv run python -m pytest tests/test_io/test_trial_csv.py::test_csv_package_exports_public_api -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit public import wiring**

Run:

```bash
git add src/comp_model/io/__init__.py src/comp_model/io/csv/__init__.py
git commit --no-verify -m "refactor: expose csv io package"
```

Before committing, run:

```bash
git diff --check -- src/comp_model/io/__init__.py src/comp_model/io/csv/__init__.py src/comp_model/io/trial_csv.py
```

### Task 6: Update Direct Imports And Documentation References

**Files:**

- Modify: `tests/test_data/test_schema_safeguard.py`
- Modify: `docs/mental_model.md`
- Modify: `docs/refactoring_points.md` if present in the worktree

- [ ] **Step 1: Replace direct imports in schema safeguard tests**

In `tests/test_data/test_schema_safeguard.py`, replace every import of
`comp_model.io.trial_csv` with `comp_model.io`:

```python
from comp_model.io import load_dataset_from_csv
```

and:

```python
from comp_model.io import load_dataset_from_csv, save_dataset_to_csv
```

and:

```python
from comp_model.io import save_dataset_to_csv
```

- [ ] **Step 2: Update tracked docs reference**

In `docs/mental_model.md`, replace:

```markdown
Real data comes in as CSV → `io.trial_csv` reconstructs this hierarchy.
```

with:

```markdown
Real data comes in as CSV → `io.csv` reconstructs this hierarchy.
```

- [ ] **Step 3: Update untracked refactoring doc if it exists**

If `docs/refactoring_points.md` exists in this worktree, replace the first
section title and opening sentence with:

```markdown
## 1. Keep CSV Conversion Split By Schema Family

The CSV implementation now lives under `src/comp_model/io/csv/`, split into
base protocol, parsing helpers, row views, converters, registry, and dataset
load/save modules.
```

Do not add this untracked file to the commit unless the user explicitly asks to
include the earlier documentation pass.

- [ ] **Step 4: Verify no old direct imports remain in tracked code**

Run:

```bash
rg -n "comp_model\\.io\\.trial_csv|io\\.trial_csv" src tests docs/mental_model.md
```

Expected: no output.

- [ ] **Step 5: Commit tracked import/doc updates**

Run:

```bash
git add tests/test_data/test_schema_safeguard.py docs/mental_model.md
git commit --no-verify -m "refactor: remove trial_csv import references"
```

Before committing, run:

```bash
git diff --check -- tests/test_data/test_schema_safeguard.py docs/mental_model.md
```

### Task 7: Run Focused Behavior Tests And Fix Mechanical Issues

**Files:**

- Modify only files created or changed in Tasks 1-6.

- [ ] **Step 1: Run IO tests**

Run:

```bash
uv run python -m pytest tests/test_io/test_trial_csv.py -q
```

Expected: all tests pass.

- [ ] **Step 2: If imports fail, inspect the exact cycle**

Run:

```bash
uv run python -c "import comp_model.io; import comp_model.io.csv; print('ok')"
```

Expected: `ok`.

If this fails with a circular import, fix only the dependency direction so it
matches:

```text
base/parsing/views -> converters -> registry -> dataset -> io.__init__
```

- [ ] **Step 3: Run schema safeguard tests**

Run:

```bash
uv run python -m pytest tests/test_data/test_schema_safeguard.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit mechanical fixes, if any**

If Steps 1-3 required code edits, run:

```bash
git add src/comp_model/io tests/test_io/test_trial_csv.py tests/test_data/test_schema_safeguard.py docs/mental_model.md
git commit --no-verify -m "fix: preserve csv io behavior after split"
```

If no edits were required after the previous commit, skip this commit.

### Task 8: Run Type, Lint, And Non-Stan Verification

**Files:**

- Modify only files needed to satisfy verification for the CSV refactor.

- [ ] **Step 1: Run ruff on touched source and tests**

Run:

```bash
uv run ruff check src/comp_model/io tests/test_io/test_trial_csv.py tests/test_data/test_schema_safeguard.py
```

Expected: no ruff errors.

- [ ] **Step 2: Run pyright on source**

Run:

```bash
uv run pyright src/
```

Expected: no pyright errors.

- [ ] **Step 3: Run non-Stan tests**

Run:

```bash
uv run python -m pytest -m "not stan" --tb=short -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Check diff hygiene**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 5: Commit verification fixes, if any**

If verification required edits, run:

```bash
git add src/comp_model/io tests/test_io/test_trial_csv.py tests/test_data/test_schema_safeguard.py docs/mental_model.md
git commit --no-verify -m "fix: satisfy csv refactor verification"
```

If no edits were required after Task 7, skip this commit.

## Completion Checklist

- [ ] `src/comp_model/io/trial_csv.py` is deleted.
- [ ] `src/comp_model/io/csv/__init__.py` exports the intended direct CSV API.
- [ ] `src/comp_model/io/__init__.py` still exports the same five high-level names.
- [ ] No tracked source, test, or `docs/mental_model.md` reference points at `comp_model.io.trial_csv`.
- [ ] `tests/test_io/test_trial_csv.py` passes.
- [ ] `tests/test_data/test_schema_safeguard.py` passes.
- [ ] `uv run ruff check src/comp_model/io tests/test_io/test_trial_csv.py tests/test_data/test_schema_safeguard.py` passes.
- [ ] `uv run pyright src/` passes.
- [ ] `uv run python -m pytest -m "not stan" --tb=short -q` passes.
