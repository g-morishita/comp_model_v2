# Model Workflows

Every model directory contains the same fixed workflow scripts:

- `manual_subject_mle.py`
- `manual_subject_stan.py`
- `flat_population_mle.py`
- `hierarchical_population_stan.py`
- `condition_hierarchical_stan.py`

These scripts are intentionally explicit. They do not hide the workflow behind helper modules or model-selection flags.

## Example commands

```bash
uv run python example/models/asocial_q_learning/manual_subject_mle.py
uv run python example/models/asocial_q_learning/hierarchical_population_stan.py
uv run python example/models/asocial_q_learning/condition_hierarchical_stan.py
uv run python example/models/social_rl_demo_reward/flat_population_mle.py
uv run python example/models/social_rl_demo_action_bias_sticky/manual_subject_stan.py
```

## Models

- [`asocial_q_learning`](asocial_q_learning/README.md)
- [`asocial_rl_asymmetric`](asocial_rl_asymmetric/README.md)
- [`asocial_rl_sticky`](asocial_rl_sticky/README.md)
- [`social_rl_demo_action_bias_sticky`](social_rl_demo_action_bias_sticky/README.md)
- [`social_rl_demo_reward`](social_rl_demo_reward/README.md)
- [`social_rl_demo_reward_sticky`](social_rl_demo_reward_sticky/README.md)
- [`social_rl_demo_mixture`](social_rl_demo_mixture/README.md)
- [`social_rl_demo_mixture_sticky`](social_rl_demo_mixture_sticky/README.md)
- [`social_rl_self_reward_demo_action_mixture`](social_rl_self_reward_demo_action_mixture/README.md)
- [`social_rl_self_reward_demo_action_mixture_sticky`](social_rl_self_reward_demo_action_mixture_sticky/README.md)
- [`social_rl_self_reward_demo_mixture`](social_rl_self_reward_demo_mixture/README.md)
- [`social_rl_self_reward_demo_mixture_sticky`](social_rl_self_reward_demo_mixture_sticky/README.md)
- [`social_rl_self_reward_demo_reward`](social_rl_self_reward_demo_reward/README.md)
- [`social_rl_self_reward_demo_reward_sticky`](social_rl_self_reward_demo_reward_sticky/README.md)
