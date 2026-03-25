# Plan 05: Library-Level Documentation (README)

## Priority: 5 (last — document the settled contract)

## Problem

The README contains only `# comp_model`.  A new user has no way to understand:

- What the library does
- What schemas/kernels are supported
- What the canonical workflow looks like
- What optional extras exist
- What combinations are guaranteed to work

## Goal

Replace the empty README with a short, contract-level guide that makes the library
self-documenting for first use.  Not exhaustive API docs — just enough to orient a user
and point them to the right entry points.

## Design Decisions

### Scope

The README is a contract document, not a tutorial.  It should answer:

1. What is this library?
2. How do I install it?
3. What is the canonical workflow?
4. What schemas and kernels exist?
5. What combinations are supported?
6. What optional extras are available?
7. Where are the examples?

Detailed internals stay in `docs/`.

### Support matrix

The core of the README is the support matrix — a table showing which
`schema × kernel × backend × hierarchy` combinations are guaranteed to work end-to-end.
This table is authoritative: if a combination is in the table, the acceptance tests
(Plan 03) cover it.

### Format

Plain markdown.  No generated docs, no Sphinx, no MkDocs.  The README is the single
source of truth for the public contract.

## Implementation Steps

### Commit 1: Write README.md

**File:** `README.md`

**Sections:**

```
# comp_model

Computational modeling library for reinforcement learning experiments with
social and asocial learning paradigms.

## Installation

    pip install .                    # core (simulation, MLE fitting, CSV I/O)
    pip install .[stan]              # + Bayesian inference via CmdStan
    pip install .[plot]              # + matplotlib visualisations
    pip install .[stan,plot]         # everything

## Canonical Workflow

1. Define a task (`TaskSpec`) with a trial schema
2. Simulate data (`simulate_dataset`)
3. Export / import CSV (`save_dataset_to_csv` / `load_dataset_from_csv`)
4. Fit models (`fit` with MLE or Stan backend)
5. Run recovery studies (`run_parameter_recovery`, `run_model_recovery`)

## Supported Schemas

| Schema | Description |
|---|---|
| `ASOCIAL_BANDIT_SCHEMA` | Solo multi-armed bandit |
| `SOCIAL_PRE_CHOICE_SCHEMA` | Observe demonstrator (action + reward) before choosing |
| `SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA` | Observe demonstrator (action only) before choosing |
| `SOCIAL_POST_OUTCOME_SCHEMA` | Choose first, then observe demonstrator (action + reward) |
| `SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA` | Choose first, then observe demonstrator (action only) |

## Supported Kernels

| Kernel | Social | Parameters |
|---|---|---|
| `AsocialQLearningKernel` | No | alpha, beta |
| `AsocialRlAsymmetricKernel` | No | alpha_pos, alpha_neg, beta |
| `SocialRlSelfRewardDemoRewardKernel` | Yes (action + reward) | alpha_self, alpha_other, beta |
| `SocialRlSelfRewardDemoMixtureKernel` | Yes (action + reward) | alpha_self, alpha_other_outcome, alpha_other_action, w_imitation, beta |

## Support Matrix

[Table mapping schema × kernel × backend to supported/unsupported]

## Hierarchy Structures

| Structure | Description |
|---|---|
| `SUBJECT_SHARED` | Single subject, shared parameters |
| `SUBJECT_BLOCK_CONDITION` | Single subject, condition-specific parameters (requires layout) |
| `STUDY_SUBJECT` | Multi-subject hierarchical |
| `STUDY_SUBJECT_BLOCK_CONDITION` | Multi-subject hierarchical with condition parameters (requires layout) |

## Examples

See `example/` for runnable scripts covering each kernel and backend.

## Intentionally Unsupported

- Asocial kernels reject social schemas that require social observation
- Social kernels reject asocial schemas
- Schemas without CSV converters are not part of the public API
- No backward-compatibility shims
```

### Commit 2: Add support matrix to `docs/`

**File:** `docs/support_matrix.md`

Detailed matrix with every tested combination, linking to the acceptance test that
covers it.  The README links to this file for the full picture.

## Verification

- Review the README for accuracy against actual code
- Verify all schemas, kernels, and hierarchies listed actually exist
- Verify the support matrix matches acceptance test coverage (Plan 03)

## Dependencies

- **All other plans should land first.**  The README documents the settled contract.
  Writing it before Plans 01-04 means it will need updates.
- In particular:
  - Plan 01 settles which kernel+schema combinations are valid
  - Plan 02 settles which schemas are exported
  - Plan 03 settles which combinations are acceptance-tested
  - Plan 04 settles the example contract
