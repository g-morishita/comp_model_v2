# social_rl_demo_action_bias_sticky

This directory contains fixed example scripts for the `social_rl_demo_action_bias_sticky` model.

## Parameters

- `demo_bias` (identity)
- `stickiness` (identity)

## Scripts

- `manual_subject_mle.py`: Simulate one subject with fixed parameters and fit with MLE.
- `manual_subject_stan.py`: Simulate one subject with fixed parameters, set priors, and fit with Stan.
- `flat_population_mle.py`: Sample one parameter set per subject from flat distributions and fit with MLE.
- `hierarchical_population_stan.py`: Sample a population hierarchically, set priors, and fit with hierarchical Stan.

## Run

```bash
uv run python example/models/social_rl_demo_action_bias_sticky/manual_subject_mle.py
```
