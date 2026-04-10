"""Stan adapter implementations."""

from comp_model.inference.bayes.stan.adapters.asocial_q_learning import AsocialQLearningStanAdapter
from comp_model.inference.bayes.stan.adapters.asocial_rl_asymmetric import (
    AsocialRlAsymmetricStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.asocial_rl_sticky import (
    AsocialRlStickyStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.base import StanAdapter
from comp_model.inference.bayes.stan.adapters.social_rl_demo_action import (
    SocialRlDemoActionStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_demo_action_bias import (
    SocialRlDemoActionBiasStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_demo_action_bias_sticky import (
    SocialRlDemoActionBiasStickyStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_demo_action_sticky import (
    SocialRlDemoActionStickyStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_demo_mixture import (
    SocialRlDemoMixtureStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_demo_mixture_sticky import (
    SocialRlDemoMixtureStickyStanAdapter,
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

from .social_rl_self_reward_demo_action_mixture_sticky import (
    SocialRlSelfRewardDemoActionMixtureStickyStanAdapter,
)
from .social_rl_self_reward_demo_reward_sticky import (
    SocialRlSelfRewardDemoRewardStickyStanAdapter,
)

__all__ = (
    "AsocialQLearningStanAdapter",
    "AsocialRlAsymmetricStanAdapter",
    "AsocialRlStickyStanAdapter",
    "SocialRlDemoActionBiasStanAdapter",
    "SocialRlDemoActionBiasStickyStanAdapter",
    "SocialRlDemoActionStanAdapter",
    "SocialRlDemoActionStickyStanAdapter",
    "SocialRlDemoMixtureStanAdapter",
    "SocialRlDemoMixtureStickyStanAdapter",
    "SocialRlDemoRewardStanAdapter",
    "SocialRlDemoRewardStickyStanAdapter",
    "SocialRlSelfRewardDemoActionMixtureStanAdapter",
    "SocialRlSelfRewardDemoActionMixtureStickyStanAdapter",
    "SocialRlSelfRewardDemoMixtureStanAdapter",
    "SocialRlSelfRewardDemoMixtureStickyStanAdapter",
    "SocialRlSelfRewardDemoRewardStanAdapter",
    "SocialRlSelfRewardDemoRewardStickyStanAdapter",
    "StanAdapter",
)
