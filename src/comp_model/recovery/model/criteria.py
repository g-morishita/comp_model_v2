"""Model selection criteria for model recovery analysis.

All scoring functions return values on a **higher-is-better** scale so that
``select_winner`` can uniformly take the argmax regardless of criterion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from comp_model.inference.bayes.result import BayesFitResult
    from comp_model.inference.mle.optimize import MleFitResult

MleCriterion = Literal["aic", "bic", "log_likelihood"]
BayesCriterion = Literal["waic"]
AnyMleCriterion = MleCriterion
AnyBayesCriterion = BayesCriterion


def score_candidate_mle(
    results: list[MleFitResult],
    criterion: MleCriterion,
) -> float:
    """Aggregate MLE fit results into a single score (higher = better).

    Parameters
    ----------
    results
        Per-subject MLE fit results.
    criterion
        Scoring criterion.  ``"aic"`` and ``"bic"`` are negated (lower raw
        value is better); ``"log_likelihood"`` is summed directly.

    Returns
    -------
    float
        Aggregated score on a higher-is-better scale.
    """

    if criterion == "log_likelihood":
        return sum(r.log_likelihood for r in results)
    if criterion == "aic":
        return -sum(r.aic for r in results)
    if criterion == "bic":
        return -sum(r.bic for r in results)
    raise ValueError(f"Unknown MLE criterion: {criterion!r}")


def score_candidate_bayes(
    result: BayesFitResult,
    criterion: BayesCriterion,
) -> float:
    """Score a Bayesian fit result (higher = better).

    Parameters
    ----------
    result
        Bayesian fit result containing posterior log-likelihood draws.
    criterion
        ``"waic"`` uses the Watanabe-Akaike Information Criterion computed
        from ``result.log_lik``.

    Returns
    -------
    float
        Score on a higher-is-better scale.

    Raises
    ------
    ValueError
        If an unknown criterion is supplied.
    """

    log_lik = result.log_lik  # shape: (draws, observations)
    if log_lik.ndim != 2:
        raise ValueError(
            f"BayesFitResult.log_lik must be 2-D (draws x obs), got shape {log_lik.shape}"
        )

    if criterion == "waic":
        return _waic_score(log_lik)
    raise ValueError(f"Unknown Bayesian criterion: {criterion!r}")


def select_winner(scores: dict[str, float]) -> tuple[str, float, str | None, float | None]:
    """Select the winning model from a mapping of candidate scores.

    Parameters
    ----------
    scores
        Mapping from candidate model name to score (higher = better).

    Returns
    -------
    winner
        Name of the best candidate.
    winner_score
        Score of the best candidate.
    second_best
        Name of the second-best candidate, or ``None`` if only one candidate.
    delta_to_second
        Difference ``winner_score - second_score``, or ``None`` if only one.
    """

    if not scores:
        raise ValueError("scores must be non-empty")

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    winner, winner_score = ranked[0]

    if len(ranked) >= 2:
        second_best, second_score = ranked[1]
        return winner, winner_score, second_best, winner_score - second_score

    return winner, winner_score, None, None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _waic_score(log_lik: np.ndarray) -> float:
    """Compute WAIC score (higher = better) from log-likelihood draws.

    Uses the log-sum-exp trick for numerical stability.

    Parameters
    ----------
    log_lik
        Array of shape ``(draws, observations)``.

    Returns
    -------
    float
        ``lppd - p_waic`` where lppd is the log pointwise predictive density
        and p_waic is the effective number of parameters penalty.
        (Equivalent to ``-waic / 2``; higher is better.)
    """

    n_draws = log_lik.shape[0]

    # lppd: log pointwise predictive density
    # log mean_s exp(log_lik_ns) = logsumexp(log_lik_ns) - log(S)
    log_sum = np.logaddexp.reduce(log_lik, axis=0)  # (obs,)
    lppd = float(np.sum(log_sum - np.log(n_draws)))

    # p_waic: variance-based penalty
    p_waic = float(np.sum(np.var(log_lik, axis=0, ddof=1)))

    return lppd - p_waic
