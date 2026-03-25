# Kernel Naming Convention

## Structure

```
{asocial|social}_rl_{signals}_{sticky?}
```

- **`asocial` / `social`**: whether the model uses any demonstrator information
- **`rl`**: reinforcement learning (replaces the old `q` suffix)
- **signals**: ordered combination of information sources and their mechanisms (see below)
- **`sticky`** (optional suffix): includes a stickiness/perseveration term biasing the agent to repeat its own previous choice

---

## Information Sources

Three binary signals, always written in this fixed order when present:

| Token | Meaning |
|-------|---------|
| `self_reward` | Agent learns from its own experienced reward |
| `demo_action` | Agent observes the demonstrator's action |
| `demo_reward` | Agent observes the demonstrator's reward/outcome |

### Mechanism for `demo_action`

When `demo_action` is present, the mechanism must be specified as a suffix on `demo_action`:

| Suffix | Meaning | Reference |
|--------|---------|-----------|
| `_bias` | **Decision bias**: demo action transiently biases action selection; Q-values unchanged | Najar et al. (2020) |
| `_shaping` | **Value shaping**: demo action treated as pseudo-reward, directly updates Q-values | Najar et al. (2020) |

When `demo_action` is present without `demo_reward` but with `self_reward`, a mixture mechanism is also available:

| Token | Meaning | Reference |
|-------|---------|-----------|
| `demo_action_mixture` | **Mixture**: two parallel systems — (1) outcome-based tracking via `self_reward`, (2) action likelihood tracking via `demo_action` | Adapted from Kang et al. (2021) |

When `demo_action` and `demo_reward` are both present, an additional mechanism is available:

| Token | Meaning | Reference |
|-------|---------|-----------|
| `demo_mixture` | **Mixture**: two parallel systems — (1) outcome-based tracking via `demo_reward`, (2) action likelihood tracking via `demo_action` | Kang et al. (2021) |

Note: when `demo_mixture` is used, `demo_action` and `demo_reward` are not written separately.

---

## Full Model List

| # | self_reward | demo_action | demo_reward | File | Class |
|---|:-----------:|:-----------:|:-----------:|------|-------|
| 1 | ✓ | | | `asocial_rl` | `AsocialRlKernel` |
| 2a | | ✓ (bias) | | `social_rl_demo_action_bias` | `SocialRlDemoActionBiasKernel` |
| 2b | | ✓ (shaping) | | `social_rl_demo_action_shaping` | `SocialRlDemoActionShapingKernel` |
| 3 | ✓ | | ✓ | `social_rl_self_reward_demo_reward` | `SocialRlSelfRewardDemoRewardKernel` |
| 4a | ✓ | ✓ (bias) | | `social_rl_self_reward_demo_action_bias` | `SocialRlSelfRewardDemoActionBiasKernel` |
| 4b | ✓ | ✓ (shaping) | | `social_rl_self_reward_demo_action_shaping` | `SocialRlSelfRewardDemoActionShapingKernel` |
| 4c | ✓ | ✓ (mixture) | | `social_rl_self_reward_demo_action_mixture` | `SocialRlSelfRewardDemoActionMixtureKernel` |
| 5 | | | ✓ | `social_rl_demo_reward` | `SocialRlDemoRewardKernel` |
| 6a | | ✓ (bias) | ✓ | `social_rl_demo_action_bias_demo_reward` | `SocialRlDemoActionBiasDemoRewardKernel` |
| 6b | | ✓ (shaping) | ✓ | `social_rl_demo_action_shaping_demo_reward` | `SocialRlDemoActionShapingDemoRewardKernel` |
| 6c | | ✓ + ✓ | ✓ | `social_rl_demo_mixture` | `SocialRlDemoMixtureKernel` |
| 7a | ✓ | ✓ (bias) | ✓ | `social_rl_self_reward_demo_action_bias_demo_reward` | `SocialRlSelfRewardDemoActionBiasDemoRewardKernel` |
| 7b | ✓ | ✓ (shaping) | ✓ | `social_rl_self_reward_demo_action_shaping_demo_reward` | `SocialRlSelfRewardDemoActionShapingDemoRewardKernel` |
| 7c | ✓ | ✓ + ✓ | ✓ | `social_rl_self_reward_demo_mixture` | `SocialRlSelfRewardDemoMixtureKernel` |

Each model also has a `_sticky` variant (e.g. `asocial_rl_sticky`, `social_rl_self_reward_demo_reward_sticky`), giving **28 models** in total.

### Asocial variants

The asocial model also has a mechanistic variant for the self-reward update:

| File | Class | Description |
|------|-------|-------------|
| `asocial_rl_asymmetric` | `AsocialRlAsymmetricKernel` | Separate learning rates for positive (`α+`) and negative (`α−`) prediction errors |

---

## Existing Kernels to Rename

| Old name | New name |
|----------|----------|
| `AsocialQLearningKernel` (`asocial_q_learning.py`) | `AsocialRlKernel` (`asocial_rl.py`) |
| `AsymmetricQLearningKernel` (`asymmetric_q_learning.py`) | `AsocialRlAsymmetricKernel` (`asocial_rl_asymmetric.py`) |
| `SocialObservedOutcomeQKernel` (`social_observed_outcome_q.py`) | `SocialRlSelfRewardDemoRewardKernel` (`social_rl_self_reward_demo_reward.py`) |

---

## References

- Najar, A., Bonnet, E., Bahrami, B., & Palminteri, S. (2020). The actions of others act as a pseudo-reward to drive imitation in the context of social reinforcement learning. *PLOS Biology*, 18(12), e3001028. https://doi.org/10.1371/journal.pbio.3001028
- Kang, Y., Burke, C. J., & Tobler, P. N. (2021). Why We Learn Less from Observing Outgroups. *Journal of Neuroscience*, 41(1), 144–152. https://doi.org/10.1523/JNEUROSCI.0926-20.2020
