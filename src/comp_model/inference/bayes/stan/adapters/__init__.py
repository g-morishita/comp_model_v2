"""Stan adapter implementations."""

from comp_model.inference.bayes.stan.adapters.asocial_q_learning import AsocialQLearningStanAdapter
from comp_model.inference.bayes.stan.adapters.asocial_rl_asymmetric import (
    AsocialRlAsymmetricStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.asocial_rl_sticky import (
    AsocialRlStickyStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.base import StanAdapter
from comp_model.inference.bayes.stan.adapters.social_rl_demo_mixture import (
    SocialRlDemoMixtureStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_demo_reward import (
    SocialRlDemoRewardStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_demo_reward_sticky import (
    SocialRlDemoRewardStickyStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_self_reward_demo_action_mixture import (
    SocialRlSelfRewardDemoActionMixtureStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_self_reward_demo_mixture import (
    SocialRlSelfRewardDemoMixtureStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_self_reward_demo_mixture_sticky import (
    SocialRlSelfRewardDemoMixtureStickyStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_self_reward_demo_reward import (
    SocialRlSelfRewardDemoRewardStanAdapter,
)

__all__ = (
    "AsocialQLearningStanAdapter",
    "AsocialRlAsymmetricStanAdapter",
    "AsocialRlStickyStanAdapter",
    "SocialRlDemoMixtureStanAdapter",
    "SocialRlDemoRewardStanAdapter",
    "SocialRlDemoRewardStickyStanAdapter",
    "SocialRlSelfRewardDemoActionMixtureStanAdapter",
    "SocialRlSelfRewardDemoMixtureStanAdapter",
    "SocialRlSelfRewardDemoMixtureStickyStanAdapter",
    "SocialRlSelfRewardDemoRewardStanAdapter",
    "StanAdapter",
)
