# Plan 02: Complete the Public Schema Workflow Surface

## Priority: 2

## Problem

The `tasks` package exports 9 schemas, but only 5 have CSV converter support.  The 4
unsupported schemas are:

| Schema | Missing support |
|---|---|
| `SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA` | No CSV converter |
| `SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA` | No CSV converter |
| `SOCIAL_PRE_CHOICE_DEMO_LEARNS_SCHEMA` | No CSV converter |
| `SOCIAL_POST_OUTCOME_DEMO_LEARNS_SCHEMA` | No CSV converter |

A user who picks one of these exported schemas can simulate data and fit models, but
cannot save/load CSV or use CSV-dependent recovery reporting.  This is a broken contract.

Additionally, the action-only schemas (`SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA`,
`SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA`) share the same `_SocialTrialCsvConverter` as
full-observation schemas.  The CSV row includes a `demonstrator_reward` column even when
the schema says reward is unobservable.  The semantics of this column for action-only
schemas need to be defined explicitly.

## Goal

Every schema exported from `tasks/__init__.py` must be usable through the full workflow:
simulate → validate → CSV export → CSV import → fit → recovery.  Schemas that cannot be
supported should not be exported.

## Design Decisions

### Fate of the 4 unsupported schemas

**No kernel currently implements `no_self_outcome` or `demo_learns` patterns.**  There are
no example scripts, no Stan programs, and no adapters for these schemas.  They are purely
aspirational.

**Recommended approach: stop exporting them.**

- Remove from `tasks/__init__.py` `__all__`
- Keep the definitions in `schemas.py` (they are valid schema definitions, just not
  part of the supported public API yet)
- Add a comment in `schemas.py` marking them as internal/experimental
- When a kernel that needs these schemas is added, re-export them and add CSV support
  at that time

**Alternative (if you want to keep them exported):** Add CSV converters for all 4.  This
is more work but straightforward:
- `no_self_outcome` schemas lack a subject OUTCOME/UPDATE, so the CSV row would have
  `choice` but no `reward`.  Needs a new converter class or a flag on
  `_SocialTrialCsvConverter`.
- `demo_learns` schemas have 11 events with bidirectional updates.  The CSV row needs
  additional columns or a second row per trial for the demonstrator's perspective.

### Action-only CSV semantics

The current `_SocialTrialCsvConverter` writes `demonstrator_reward` for all social
schemas.  For action-only schemas, the simulation engine stores the demonstrator's reward
in the OUTCOME event (the demonstrator still *gets* a reward), but the schema says the
subject cannot *observe* it.

**Recommended approach: write empty string for unobservable fields.**

- On export: if the schema's social UPDATE step does not include `"reward"` in
  `observable_fields`, write `demonstrator_reward` as `""` (empty string).
- On import: if the schema is action-only and `demonstrator_reward` is non-empty, raise
  `ValueError` — the CSV contains data the subject shouldn't see.
- This preserves the shared CSV column shape (no schema-specific headers) while making
  the semantics explicit.

**Alternative: always write the actual value.** Simpler, but allows loading action-only
CSV data under a full-observation schema without error, which defeats the schema
provenance safeguard.

## Implementation Steps

### Commit 1: Remove unsupported schemas from public exports

**Files:**
- `src/comp_model/tasks/__init__.py` — remove the 4 schemas from `__all__`
- `src/comp_model/tasks/schemas.py` — add comment marking them as internal

**Tests:**
- Verify the 4 schemas are no longer importable from `comp_model.tasks`
- Verify the 5 supported schemas are still importable

### Commit 2: Fix action-only CSV export semantics

**Files:**
- `src/comp_model/io/trial_csv.py`:
  - `_SocialTrialCsvConverter.__init__` already stores the schema — use its
    `observable_fields` to decide whether to write `demonstrator_reward`
  - In `trial_to_row`: if `"reward"` not in the schema's social UPDATE
    `observable_fields`, write `""` for `demonstrator_reward`
  - In `row_to_trial`: if `"reward"` not in `observable_fields` and
    `demonstrator_reward` is non-empty, raise `ValueError`

**Tests:**
- Round-trip test for `SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA`: export writes empty
  `demonstrator_reward`, import succeeds
- Round-trip test for `SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA`: same
- Negative test: CSV with non-empty `demonstrator_reward` loaded under action-only
  schema raises
- Existing full-observation tests still pass

### Commit 3: Audit recovery/reporting layers for schema dead-ends

**Files:**
- `src/comp_model/recovery/parameter/` — verify CSV export works for all 5 supported
  schemas (parameter recovery saves per-subject results, not trial data, so likely fine)
- `src/comp_model/recovery/model/` — same audit
- Example scripts — verify none reference the removed schemas

**Tests:**
- No new tests needed if recovery layers don't touch trial CSV directly (they don't)

## Verification

```bash
uv run ruff check && uv run ruff format --check
uv run pyright
uv run pytest tests/ -q
```

## Dependencies

- Plan 01 (kernel+schema validation) should land first.  If the 4 schemas are removed
  from exports, Plan 01's test matrix is simpler.
- Plan 03 (acceptance tests) will exercise the CSV round-trip for supported schemas.
