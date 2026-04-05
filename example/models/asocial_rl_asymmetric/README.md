# asocial_rl_asymmetric

This directory contains fixed example scripts for the `asocial_rl_asymmetric` model.

## Parameters

- `alpha_pos` (sigmoid)
- `alpha_neg` (sigmoid)
- `beta` (softplus)

## Scripts

- `manual_subject_mle.py`: Simulate one subject with fixed parameters and fit with MLE.
- `manual_subject_stan.py`: Simulate one subject with fixed parameters, set priors, and fit with Stan.
- `flat_population_mle.py`: Sample one parameter set per subject from flat distributions and fit with MLE.
- `hierarchical_population_stan.py`: Sample a population hierarchically, set priors, and fit with hierarchical Stan.

## Run

```bash
uv run python example/models/asocial_rl_asymmetric/manual_subject_mle.py
```
