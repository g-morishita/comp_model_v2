# Plan 04: Fix Examples and Optional-Feature Contract

## Priority: 4

## Problem

There are 43 example scripts across 5 directories.  Several issues:

1. **Plotting guards:** PR #55 added `try/except ImportError` guards for matplotlib in
   the model recovery examples.  The remaining ~40 scripts that use plotting are
   unguarded and will crash without the `plot` extra installed.

2. **No smoke tests:** No automated test verifies that example scripts run without error.
   Examples can silently break when APIs change.

3. **No documentation header:** Example scripts do not document which extras are required
   (`pip install .[plot]`, `pip install .[stan]`).

## Goal

- Every example script runs under `uv run python example/...` with only core dependencies
  (matplotlib failures print a skip message, Stan scripts are clearly marked).
- A parametrized smoke test catches broken examples in CI.

## Design Decisions

### Guard strategy for matplotlib

**Recommended: guard at the call site, not the import.**

Pattern used in PR #55 model recovery examples:

```python
try:
    from comp_model.recovery.model.plotting import plot_confusion_matrix
    plot_confusion_matrix(result, save_path=output_dir / "confusion.png")
except ImportError:
    print("Install the 'plot' extra for visualisations: pip install .[plot]")
```

Apply this pattern to all example scripts that call plotting functions.

### Stan examples

Stan examples require `cmdstanpy` and compiled Stan programs.  These cannot be guarded
the same way — they fail fundamentally without Stan.

**Recommended: add a header comment and skip in smoke tests.**

```python
"""Hierarchical Stan fit for asocial Q-learning.

Requires: pip install .[stan]
"""
```

Smoke tests for Stan examples should be marked `@pytest.mark.stan`.

### Smoke test design

**One parametrized test per example family, not 43 individual tests.**

Each example script should expose a `main()` function (or already does).  The smoke test:
1. Imports the example module
2. Calls `main()` with minimal parameters (small N, few trials)
3. Asserts no exception

For scripts that don't have a `main()`, refactor them to add one.  This is a small
change per file and makes testing possible.

**Alternative (simpler, less robust):** Run each script as a subprocess with a timeout.
This avoids refactoring but gives worse error messages and is slower.

**Recommended: subprocess approach for now.** Refactoring 43 scripts to add `main()` is
a large change that should be a separate effort.

```python
@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
def test_example_smoke(script):
    result = subprocess.run(
        ["uv", "run", "python", str(script)],
        capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr.decode()
```

Filter Stan scripts with `@pytest.mark.stan` based on filename pattern.

## Implementation Steps

### Commit 1: Add matplotlib guards to all example scripts

**Files:** All example scripts that import or call matplotlib/plotting functions.

**Inventory (by directory):**

- `example/asocial_rl/` — 10 scripts.  Check which call plotting.
- `example/asocial_rl_asymmetric/` — 10 scripts.  Same audit.
- `example/social_rl_self_reward_demo_reward/` — 7 scripts.
- `example/social_rl_self_reward_demo_mixture/` — 13 scripts.
- `example/model_recovery/` — 3 scripts (already guarded from PR #55).

For each script:
1. Find all plotting imports and calls
2. Wrap in `try/except ImportError` with skip message
3. Ensure the rest of the script (simulation, fitting, CSV export) still runs

### Commit 2: Add header comments documenting required extras

**Files:** All example scripts.

Add a one-line `Requires:` in the module docstring:

- Scripts using Stan: `Requires: pip install .[stan]`
- Scripts using both: `Requires: pip install .[stan,plot]`
- Scripts using only core + plot: `Requires: pip install .[plot]` (optional)

### Commit 3: Add smoke tests for example scripts

**File:** `tests/test_examples/test_smoke.py`

```python
import subprocess
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "example"

MLE_SCRIPTS = sorted(EXAMPLE_DIR.rglob("mle*.py"))
STAN_SCRIPTS = sorted(EXAMPLE_DIR.rglob("stan*.py"))
OTHER_SCRIPTS = [
    s for s in sorted(EXAMPLE_DIR.rglob("*.py"))
    if s not in MLE_SCRIPTS and s not in STAN_SCRIPTS
]

@pytest.mark.parametrize("script", MLE_SCRIPTS, ids=lambda p: str(p.relative_to(EXAMPLE_DIR)))
def test_mle_example(script):
    """MLE example scripts run without error."""
    result = subprocess.run(
        ["uv", "run", "python", str(script)],
        capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr.decode()

@pytest.mark.stan
@pytest.mark.parametrize("script", STAN_SCRIPTS, ids=lambda p: str(p.relative_to(EXAMPLE_DIR)))
def test_stan_example(script):
    """Stan example scripts run without error (requires Stan)."""
    result = subprocess.run(
        ["uv", "run", "python", str(script)],
        capture_output=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr.decode()
```

**Practical note:** Some example scripts may take too long with default parameters.
Consider adding a `--smoke` or `--quick` CLI flag to examples that reduces N_SUBJECTS
and N_TRIALS for testing purposes.  This is optional and can be deferred.

## Verification

```bash
uv run ruff check && uv run ruff format --check
uv run pyright
uv run pytest tests/test_examples/ -q              # MLE examples only
uv run pytest tests/test_examples/ -q -m stan      # Stan examples (slow)
```

## Dependencies

- **Plan 02** should land first (if schemas are removed from exports, example scripts
  that reference them need updating — though currently no example uses the 4 unsupported
  schemas).
- **Plan 01** should land first (if examples use mismatched kernel+schema, they will
  start failing after validation is added).

## Scope Risk

43 scripts is a lot.  The matplotlib guard (Commit 1) is mechanical but tedious.
Consider doing it in batches:

1. First batch: scripts that currently fail without matplotlib (highest priority)
2. Second batch: scripts where plotting is a minor final step
