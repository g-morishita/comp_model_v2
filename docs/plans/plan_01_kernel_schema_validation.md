# Plan 01: Kernel + Schema Compatibility Validation

## Priority: 1 (highest — correctness bug)

## Problem

There is **zero validation** that a kernel's requirements match the schema it is being
used with.  The `fit()` dispatch, `simulate_dataset()`, Stan `build_stan_data()`, and
both recovery runners all silently accept any kernel+schema combination.

Concrete failure modes today:

| Combination | What happens | Severity |
|---|---|---|
| Social kernel + asocial schema | Social fields are all-zero arrays; kernel learns from ghost data | **Silent wrong results** |
| Social kernel + action-only schema when it expects reward | `social_reward` is zero; kernel treats missing reward as zero reward | **Silent wrong results** |
| Asocial kernel + social schema | Kernel ignores social UPDATE steps via `actor_id != learner_id` guard | Harmless but confusing |

The first two can produce publishable-but-wrong parameter estimates with no error or
warning.

## Goal

Add a single validation layer that rejects unsupported `kernel + schema` combinations at
every public entrypoint, before any computation begins.

## Design Decisions

### What metadata exists today

- `ModelKernelSpec.requires_social: bool` — whether the kernel needs demonstrator data.
- `TrialSchemaStep.observable_fields: frozenset[str]` — what the learner can see on a
  social UPDATE step (`"action"`, `"reward"`, or both).
- `TrialSchema.steps` — the full event sequence including all UPDATE steps.

### What the check needs to answer

1. **Does the kernel need social data?**
   `kernel.spec().requires_social`

2. **Does the schema provide social data?**
   A schema is "social" if it contains at least one UPDATE step where
   `actor_id != learner_id`.

3. **Does the schema provide the specific fields the kernel needs?**
   This requires knowing *which* observable fields the kernel consumes.  Today
   `requires_social` is a bool — it does not distinguish "needs action only" from "needs
   action + reward".

### Proposed approach: extend `ModelKernelSpec`

Add a new field to `ModelKernelSpec`:

```python
@dataclass(frozen=True, slots=True)
class ModelKernelSpec:
    ...
    requires_social: bool = False
    required_social_fields: frozenset[str] = field(default_factory=frozenset)
    ...
```

- Asocial kernels: `requires_social=False`, `required_social_fields=frozenset()`
- `SocialRlSelfRewardDemoRewardKernel`: `requires_social=True`,
  `required_social_fields=frozenset({"action", "reward"})`
- `SocialRlSelfRewardDemoMixtureKernel`: `requires_social=True`,
  `required_social_fields=frozenset({"action", "reward"})`

Validation rule: for every social UPDATE step in the schema where
`learner_id == "subject"`, the step's `observable_fields` must be a superset of
`required_social_fields`.

### Where to put the validation function

Create `src/comp_model/data/compatibility.py` with:

```python
def check_kernel_schema_compatibility(
    kernel: ModelKernel,
    schema: TrialSchema,
) -> None:
    """Raise ValueError if kernel requirements are incompatible with schema."""
```

This is a pure data-layer check with no inference dependencies — it only needs
`ModelKernelSpec` and `TrialSchema`.

### Where to enforce

| Entrypoint | File | Call site |
|---|---|---|
| `fit()` | `inference/dispatch.py` | Before routing to MLE or Stan |
| `simulate_subject()` | `runtime/engine.py` | Before simulation loop |
| `simulate_dataset()` | `runtime/engine.py` | Before simulation loop |
| `run_parameter_recovery()` | `recovery/parameter/runner.py` | After `_check_schema_consistency`, before simulation |
| `run_model_recovery()` | `recovery/model/runner.py` | After `_check_schema_consistency`, before simulation |

Recovery runners also need to check each *candidate* kernel against the schema, not just
the generating kernel.

### What about asocial kernel on social schema?

This is intentionally allowed for model comparison: you fit an asocial model to social
data to see if social information improves fit.  The kernel safely ignores social UPDATE
steps.  The validation should **not** reject this combination.

Rule: reject only when `kernel.spec().requires_social is True` and the schema cannot
satisfy `required_social_fields`.

### Helper on TrialSchema

Add a property to `TrialSchema`:

```python
@property
def social_observable_fields(self) -> frozenset[str]:
    """Union of observable_fields across all social UPDATE steps for the subject."""
```

This collects all fields visible to the subject from demonstrator updates.

## Implementation Steps

### Commit 1: Extend `ModelKernelSpec` with `required_social_fields`

**Files:**
- `src/comp_model/models/kernels/base.py` — add `required_social_fields` field
- `src/comp_model/models/kernels/social_rl_self_reward_demo_reward.py` — set field
- `src/comp_model/models/kernels/social_rl_self_reward_demo_mixture.py` — set field
- Asocial kernels — no change needed (default `frozenset()` is correct)

**Tests:**
- Unit test that each kernel's `spec()` returns the expected `required_social_fields`

### Commit 2: Add `social_observable_fields` property to `TrialSchema`

**Files:**
- `src/comp_model/tasks/schemas.py` — add property

**Tests:**
- `tests/test_tasks/` — verify property returns expected values for each of the 9 schemas

### Commit 3: Add `check_kernel_schema_compatibility()` validation

**Files:**
- Create `src/comp_model/data/compatibility.py`

**Tests:**
- Social kernel + asocial schema → raises
- Social kernel + action-only schema (when kernel needs reward) → raises
- Social kernel + full-observation schema → passes
- Asocial kernel + social schema → passes (intentionally allowed)
- Asocial kernel + asocial schema → passes

### Commit 4: Enforce at all public entrypoints

**Files:**
- `src/comp_model/inference/dispatch.py` — call in `fit()`
- `src/comp_model/runtime/engine.py` — call in `simulate_subject()` and `simulate_dataset()`
- `src/comp_model/recovery/parameter/runner.py` — call in `run_parameter_recovery()`
- `src/comp_model/recovery/model/runner.py` — call in `run_model_recovery()` (check each
  generating model AND each candidate model)

**Tests:**
- Integration tests that confirm each entrypoint rejects mismatched combinations
- Add to existing `test_schema_safeguard.py` or create new test file

## Verification

```bash
uv run ruff check && uv run ruff format --check
uv run pyright
uv run pytest tests/ -q
```

## Open Questions

1. Should `check_kernel_schema_compatibility` also verify that asocial kernels are NOT
   given `requires_social=True`?  (Defensive consistency check on kernel metadata.)
2. Should the `adapter` also be validated against the kernel at the `fit()` entrypoint?
   Today `fit()` trusts that the caller passes a matching adapter.  This could be a
   follow-up.
