# Complete Workflow Examples

This directory is a single, structured learning path for `comp_model`.

Each script is standalone. You can run them in order, or jump directly to the
topic you need. Every script also accepts `--model`, so the same workflow can
be run for each public model.

## Workflow

1. `01_model_and_task.py`
   Understand the core pieces: task, environment, kernel, parameters, and a
   simulated subject.
2. `02_priors_and_parameter_sampling.py`
   Define Bayesian priors and sample synthetic ground-truth parameters.
3. `03_fit_with_mle.py`
   Simulate a dataset and fit each subject with maximum likelihood.
4. `04_fit_with_stan.py`
   Simulate a hierarchical dataset, set `PriorSpec`s, and fit with Stan.
   Requires `pip install .[stan]` and a working CmdStan installation.
5. `05_model_comparison.py`
   Run a model recovery study and inspect the confusion matrix / recovery rates.
6. `06_parameter_comparison.py`
   Run a parameter recovery study and compare true versus recovered parameters.

## Recommended Commands

```bash
uv run python example/complete_workflow/01_model_and_task.py --model asocial_q_learning
uv run python example/complete_workflow/02_priors_and_parameter_sampling.py --model asocial_q_learning
uv run python example/complete_workflow/03_fit_with_mle.py --model asocial_q_learning
uv run python example/complete_workflow/04_fit_with_stan.py --model asocial_q_learning
uv run python example/complete_workflow/05_model_comparison.py --model asocial_q_learning
uv run python example/complete_workflow/06_parameter_comparison.py --model asocial_q_learning
```

To test a specific social model:

```bash
uv run python example/complete_workflow/03_fit_with_mle.py --model social_rl_demo_reward
uv run python example/complete_workflow/03_fit_with_mle.py --model social_rl_self_reward_demo_mixture_sticky
```

For faster smoke-sized runs:

```bash
uv run python example/complete_workflow/03_fit_with_mle.py --model social_rl_demo_reward --quick
uv run python example/complete_workflow/05_model_comparison.py --model social_rl_demo_reward --quick
uv run python example/complete_workflow/06_parameter_comparison.py --model social_rl_demo_reward --quick
```

## Supported Models

- `asocial_q_learning`
- `asocial_rl_asymmetric`
- `asocial_rl_sticky`
- `social_rl_demo_reward`
- `social_rl_demo_reward_sticky`
- `social_rl_demo_mixture`
- `social_rl_demo_mixture_sticky`
- `social_rl_self_reward_demo_reward`
- `social_rl_self_reward_demo_reward_sticky`
- `social_rl_self_reward_demo_action_mixture`
- `social_rl_self_reward_demo_action_mixture_sticky`
- `social_rl_self_reward_demo_mixture`
- `social_rl_self_reward_demo_mixture_sticky`

## Design Notes

- The examples all use the same two-armed bandit environment. The schema
  switches automatically between asocial, full-social, and action-only social
  variants depending on the selected model.
- `03_fit_with_mle.py` is the shortest end-to-end path.
- `04_fit_with_stan.py` focuses on priors and hierarchical fitting.
- `05_model_comparison.py` and `06_parameter_comparison.py` use the recovery
  utilities because they are the package's structured comparison surfaces.

Pass `--output-dir /path/to/output` to any script that offers file output if
you want CSV artifacts written alongside the printed summaries.
