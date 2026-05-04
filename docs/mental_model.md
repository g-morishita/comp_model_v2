# Mental Model: comp_model Library

## The Three Pillars

Everything in the library orbits three independent objects that never directly depend on each other:

```
TaskSpec          ModelKernel         Environment
(what to do)     (how to learn)      (what happens)
```

- **`TaskSpec`** is a script: ordered blocks, each with a schema that says which events fire in what order (`INPUT → DECISION → OUTCOME → UPDATE`).
- **`ModelKernel`** is an artificial mind: it holds beliefs (state), picks actions, and updates after outcomes. It never sees tasks or events directly.
- **`Environment`** is a pure reward oracle: give it an action, it returns a float. Nothing more.

---

## The Central Abstraction: `DecisionTrialView`

Kernels never touch raw events. Everything passes through `DecisionTrialView` — a flat snapshot of one decision moment:

```
choice, reward, available_actions, social_action, social_reward, ...
```

`replay_trial_steps()` in `extractors.py` is the translator: it walks raw events according to a schema and emits these views. Both simulation and fitting use this same translator, so the kernel code is identical for real data and synthetic data.

---

## Two Main Workflows

### Simulation (generating synthetic data)

```
TaskSpec + ModelKernel + Environment
        ↓  engine.simulate_subject()
     SubjectData   ← identical format to real data
```

The engine walks the schema step by step, calls `kernel.action_probabilities()` to sample a choice, calls `env.step()` to get a reward, then calls `kernel.update()` to advance state.

### Fitting (recovering parameters from data)

```
SubjectData + ModelKernel + TrialSchema
        ↓  mle.objective.log_likelihood()
    log P(data | params)
        ↓  scipy.optimize / Stan
     MleFitResult / BayesFitResult
```

The objective replays each trial via `replay_trial_steps()`, asks the kernel for `action_probabilities()`, accumulates `log p(observed choice)`, and calls `update()` to advance state — exactly mirroring simulation but in reverse (inferring params rather than generating choices).

---

## The Canonical Data Hierarchy

```
Dataset
 └─ SubjectData  (one per participant)
     └─ Block    (one per condition)
         └─ Trial    (one decision round)
             └─ Event    (INPUT / DECISION / OUTCOME / UPDATE)
```

Real data comes in as CSV → `io.csv` reconstructs this hierarchy. Simulated data comes out of `engine.py` in the same hierarchy. Every downstream piece of code (fitting, validation, recovery) works on this hierarchy regardless of source.

---

## Parameter Flow

Parameters live on two scales, always:

```
unconstrained (ℝ)  ←→  constrained (natural scale)
     via transforms.py (sigmoid, softplus, ...)
```

The optimiser and Stan work on the unconstrained scale. The kernel always receives constrained values via `parse_params()`. The same `transform_id` string in `ParameterSpec` drives both the Python transform and the generated Stan code.

---

## Module Responsibilities

| Module | Role |
|---|---|
| `data/schema.py` | Canonical event hierarchy |
| `data/extractors.py` | Raw events → `DecisionTrialView` |
| `data/validation.py` | Structural integrity checks |
| `tasks/schemas.py` | Trial event-order contracts |
| `tasks/spec.py` | Experiment design (blocks, conditions) |
| `models/kernels/` | Learning rule + choice rule implementations |
| `environments/` | Reward oracle |
| `runtime/engine.py` | Simulation loop |
| `inference/mle/` | Maximum likelihood fitting |
| `inference/bayes/stan/` | Bayesian fitting via Stan |
| `recovery/` | Simulate → fit → validate parameter recovery |
