"""Stan adapter implementations."""

from comp_model.inference.bayes.stan.adapters.asocial_q_learning import AsocialQLearningStanAdapter
from comp_model.inference.bayes.stan.adapters.base import StanAdapter

__all__ = ("AsocialQLearningStanAdapter", "StanAdapter")
