# Plan 03: End-to-End Acceptance Tests

## Priority: 3

## Problem

The codebase has 167 unit tests but no true end-to-end acceptance tests that exercise the
full public workflow: simulate → validate → CSV export → CSV import → fit → summarize.

Unit tests verify individual functions in isolation.  They do not catch integration
failures like:

- Simulation produces data that the CSV exporter cannot serialize
- CSV importer produces data that the fitter rejects
- Schema metadata flows correctly through all layers
- Recovery pipelines produce sensible results on synthetic data

## Goal

Add acceptance tests that cover the public workflow for every supported schema × backend
combination.  These tests are the contract: if they pass, the documented workflow works.

## Design Decisions

### Test matrix

After Plan 02 lands, the supported schemas are:

| Schema | Kernel(s) | MLE | Stan |
|---|---|---|---|
| `ASOCIAL_BANDIT_SCHEMA` | `AsocialQLearningKernel`, `AsocialRlAsymmetricKernel` | Yes | Yes |
| `SOCIAL_PRE_CHOICE_SCHEMA` | `SocialRlSelfRewardDemoRewardKernel`, `SocialRlSelfRewardDemoMixtureKernel` | Yes | Yes |
| `SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA` | `SocialRlSelfRewardDemoMixtureKernel` | Yes | Yes |
| `SOCIAL_POST_OUTCOME_SCHEMA` | `SocialRlSelfRewardDemoRewardKernel`, `SocialRlSelfRewardDemoMixtureKernel` | Yes | Yes |
| `SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA` | `SocialRlSelfRewardDemoMixtureKernel` | Yes | Yes |

**Note:** `SocialRlSelfRewardDemoRewardKernel` requires `"reward"` in observable fields,
so it is incompatible with action-only schemas.  Plan 01's validation will enforce this.

### Stan tests are slow

Stan tests require CmdStan compilation and MCMC sampling.  They should be marked with
`@pytest.mark.stan` so CI can skip them by default.

**Strategy:**
- MLE acceptance tests: run in default CI (fast, ~seconds)
- Stan acceptance tests: marked `@pytest.mark.stan`, run on demand

### Test structure

One test file: `tests/test_acceptance/test_end_to_end.py`

Each test follows the same pattern:
1. Define a minimal `TaskSpec` (1 block, 10 trials)
2. Simulate a small dataset (2 subjects)
3. Validate the dataset
4. Export to CSV (tmp_path)
5. Load from CSV
6. Assert loaded dataset matches original
7. Fit with MLE (or Stan)
8. Assert fit result has expected structure

### Negative tests (validation matrix)

Separate file: `tests/test_acceptance/test_unsupported_combinations.py`

Test that unsupported combinations raise at the entrypoint:

- Social kernel + asocial schema → raises at `fit()` / `simulate_subject()`
- Social kernel (needs reward) + action-only schema → raises
- Stan backend without adapter → raises (already tested, but include for completeness)
- Condition hierarchy without layout → raises (already tested)

## Implementation Steps

### Commit 1: Add MLE end-to-end acceptance tests

**File:** `tests/test_acceptance/test_end_to_end.py`

**Tests (parametrized):**

```python
@pytest.mark.parametrize("schema, kernel, env_factory", [
    (ASOCIAL_BANDIT_SCHEMA, AsocialQLearningKernel(), make_asocial_env),
    (ASOCIAL_BANDIT_SCHEMA, AsocialRlAsymmetricKernel(), make_asocial_env),
    (SOCIAL_PRE_CHOICE_SCHEMA, SocialRlSelfRewardDemoRewardKernel(), make_social_env),
    (SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA, SocialRlSelfRewardDemoMixtureKernel(), make_social_env),
    (SOCIAL_POST_OUTCOME_SCHEMA, SocialRlSelfRewardDemoRewardKernel(), make_social_env),
    (SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA, SocialRlSelfRewardDemoMixtureKernel(), make_social_env),
])
def test_simulate_csv_roundtrip_mle_fit(schema, kernel, env_factory, tmp_path):
    ...
```

Each test:
1. Build `TaskSpec` with schema
2. `simulate_dataset(...)` with known params
3. `validate_dataset(dataset, schema)`
4. `save_dataset_to_csv(dataset, schema=schema, path=csv_path)`
5. `loaded = load_dataset_from_csv(csv_path, schema=schema)`
6. Assert `len(loaded.subjects) == len(dataset.subjects)`
7. Assert block schema_ids match
8. `fit(mle_config, kernel, subject, schema)` for each subject
9. Assert result has parameter values within bounds

### Commit 2: Add Stan end-to-end acceptance tests

**File:** Same file, additional tests marked `@pytest.mark.stan`

```python
@pytest.mark.stan
@pytest.mark.parametrize("schema, kernel, adapter, env_factory", [
    (ASOCIAL_BANDIT_SCHEMA, AsocialQLearningKernel(), AsocialQLearningStanAdapter(), make_asocial_env),
    (SOCIAL_PRE_CHOICE_SCHEMA, SocialRlSelfRewardDemoRewardKernel(), SocialRlSelfRewardDemoRewardStanAdapter(), make_social_env),
])
def test_simulate_csv_roundtrip_stan_fit(schema, kernel, adapter, env_factory, tmp_path):
    ...
```

Minimal Stan tests (2-3 combinations, short chains) to verify the full path without
excessive CI cost.

### Commit 3: Add unsupported-combination negative tests

**File:** `tests/test_acceptance/test_unsupported_combinations.py`

```python
def test_social_kernel_on_asocial_schema_raises():
    """Social kernel cannot simulate on asocial schema."""
    ...

def test_reward_kernel_on_action_only_schema_raises():
    """Kernel requiring reward fails on action-only schema."""
    ...
```

These depend on Plan 01 landing first.

## Test Helpers

Create `tests/test_acceptance/conftest.py` with shared fixtures:

```python
def make_asocial_env() -> Environment:
    """Factory for a 2-armed bandit."""

def make_social_env() -> Environment:
    """Factory for a 2-armed social bandit with demonstrator."""

def make_task(schema: TrialSchema, n_trials: int = 10) -> TaskSpec:
    """Build a single-block TaskSpec."""

def make_params(kernel: ModelKernel) -> dict[str, float]:
    """Return reasonable default parameters for any kernel."""
```

## Verification

```bash
uv run ruff check && uv run ruff format --check
uv run pyright
uv run pytest tests/test_acceptance/ -q          # MLE only
uv run pytest tests/test_acceptance/ -q -m stan  # Stan (slow)
```

## Dependencies

- **Plan 01** must land first (validation layer exists for negative tests)
- **Plan 02** should land first (schema surface is settled, action-only CSV semantics
  defined)
