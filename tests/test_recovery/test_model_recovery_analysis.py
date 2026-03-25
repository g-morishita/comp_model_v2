"""Tests for model recovery analysis utilities (analysis.py, display.py, criteria.py)."""

from __future__ import annotations

import numpy as np
import pytest

from comp_model.inference.bayes.result import BayesFitResult
from comp_model.inference.config import HierarchyStructure
from comp_model.inference.mle.optimize import MleFitResult
from comp_model.recovery.model.analysis import compute_confusion_matrix, compute_recovery_rates
from comp_model.recovery.model.criteria import (
    _waic_score,
    score_candidate_bayes,
    score_candidate_mle,
    select_winner,
)
from comp_model.recovery.model.display import (
    model_recovery_confusion_table,
    model_recovery_rate_table,
)
from comp_model.recovery.model.runner import ModelRecoveryResult, ReplicationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mle_result(
    log_likelihood: float = -10.0,
    aic: float = 24.0,
    bic: float = 26.0,
    subject_id: str = "sub_00",
) -> MleFitResult:
    return MleFitResult(
        subject_id=subject_id,
        model_id="test",
        log_likelihood=log_likelihood,
        n_params=2,
        raw_params={"alpha": 0.0, "beta": 0.0},
        constrained_params={"alpha": 0.5, "beta": 3.0},
        aic=aic,
        bic=bic,
        n_trials=50,
        converged=True,
        n_restarts=1,
        all_candidates=({"alpha": 0.5, "beta": 3.0},),
        all_log_likelihoods=(-10.0,),
        params_by_condition=None,
    )


def _make_bayes_result(log_lik: np.ndarray) -> BayesFitResult:
    return BayesFitResult(
        model_id="test",
        hierarchy=HierarchyStructure.STUDY_SUBJECT,
        posterior_samples={},
        log_lik=log_lik,
        subject_params=None,
        diagnostics={},
    )


def _make_replication(
    rep_idx: int,
    generating_model: str,
    selected_model: str,
) -> ReplicationResult:
    return ReplicationResult(
        replication_index=rep_idx,
        generating_model=generating_model,
        candidate_scores={"ModelA": 0.5, "ModelB": 0.3},
        selected_model=selected_model,
        winner_score=0.5,
        second_best_model="ModelB",
        delta_to_second=0.2,
    )


# ---------------------------------------------------------------------------
# criteria: score_candidate_mle
# ---------------------------------------------------------------------------


class TestScoreCandidateMle:
    def test_log_likelihood_sums(self) -> None:
        results = [_make_mle_result(log_likelihood=-5.0), _make_mle_result(log_likelihood=-8.0)]
        assert score_candidate_mle(results, "log_likelihood") == pytest.approx(-13.0)

    def test_aic_negated(self) -> None:
        results = [_make_mle_result(aic=20.0), _make_mle_result(aic=30.0)]
        assert score_candidate_mle(results, "aic") == pytest.approx(-50.0)

    def test_bic_negated(self) -> None:
        results = [_make_mle_result(bic=15.0), _make_mle_result(bic=25.0)]
        assert score_candidate_mle(results, "bic") == pytest.approx(-40.0)

    def test_unknown_criterion_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown MLE criterion"):
            score_candidate_mle([_make_mle_result()], "waic")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# criteria: _waic_score
# ---------------------------------------------------------------------------


class TestWaicScore:
    def test_returns_float(self) -> None:
        rng = np.random.default_rng(0)
        log_lik = rng.normal(-1.5, 0.3, size=(500, 100))
        score = _waic_score(log_lik)
        assert isinstance(score, float)

    def test_better_model_higher_score(self) -> None:
        rng = np.random.default_rng(42)
        # Good model: concentrated log-likelihoods close to 0
        good_ll = rng.normal(-0.5, 0.1, size=(400, 80))
        # Bad model: low, noisy log-likelihoods
        bad_ll = rng.normal(-5.0, 1.0, size=(400, 80))
        assert _waic_score(good_ll) > _waic_score(bad_ll)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            score_candidate_bayes(_make_bayes_result(np.zeros((100,))), "waic")


# ---------------------------------------------------------------------------
# criteria: select_winner
# ---------------------------------------------------------------------------


class TestSelectWinner:
    def test_winner_is_highest_score(self) -> None:
        scores = {"A": -5.0, "B": -3.0, "C": -7.0}
        winner, winner_score, second, delta = select_winner(scores)
        assert winner == "B"
        assert winner_score == pytest.approx(-3.0)
        assert second == "A"
        assert delta == pytest.approx(2.0)

    def test_single_candidate(self) -> None:
        winner, _winner_score, second, delta = select_winner({"OnlyModel": 0.0})
        assert winner == "OnlyModel"
        assert second is None
        assert delta is None

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            select_winner({})


# ---------------------------------------------------------------------------
# analysis: confusion_matrix
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    def _make_result(self, replications: list[ReplicationResult]) -> ModelRecoveryResult:
        """Build a minimal ModelRecoveryResult with two generating models."""
        from unittest.mock import MagicMock

        from comp_model.recovery.model.config import (
            CandidateModelSpec,
            GeneratingModelSpec,
            ModelRecoveryConfig,
        )

        gen_a = MagicMock(spec=GeneratingModelSpec)
        gen_a.name = "ModelA"
        gen_b = MagicMock(spec=GeneratingModelSpec)
        gen_b.name = "ModelB"
        cand_a = MagicMock(spec=CandidateModelSpec)
        cand_a.name = "ModelA"
        cand_b = MagicMock(spec=CandidateModelSpec)
        cand_b.name = "ModelB"
        config = MagicMock(spec=ModelRecoveryConfig)
        config.generating_models = [gen_a, gen_b]
        config.candidate_models = [cand_a, cand_b]

        return ModelRecoveryResult(config=config, replications=tuple(replications))

    def test_diagonal_correct_recovery(self) -> None:
        reps = [
            _make_replication(0, "ModelA", "ModelA"),
            _make_replication(1, "ModelA", "ModelA"),
            _make_replication(0, "ModelB", "ModelB"),
            _make_replication(1, "ModelB", "ModelA"),
        ]
        result = self._make_result(reps)
        matrix = compute_confusion_matrix(result)

        assert matrix["ModelA"]["ModelA"] == 2
        assert matrix["ModelA"]["ModelB"] == 0
        assert matrix["ModelB"]["ModelB"] == 1
        assert matrix["ModelB"]["ModelA"] == 1

    def test_zero_filled_for_all_pairs(self) -> None:
        result = self._make_result([_make_replication(0, "ModelA", "ModelA")])
        matrix = compute_confusion_matrix(result)
        assert "ModelA" in matrix
        assert "ModelB" in matrix
        assert matrix["ModelB"]["ModelA"] == 0
        assert matrix["ModelB"]["ModelB"] == 0


# ---------------------------------------------------------------------------
# analysis: recovery_rates
# ---------------------------------------------------------------------------


class TestRecoveryRates:
    def _make_result(self, replications: list[ReplicationResult]) -> ModelRecoveryResult:
        from unittest.mock import MagicMock

        from comp_model.recovery.model.config import GeneratingModelSpec, ModelRecoveryConfig

        gen_a = MagicMock(spec=GeneratingModelSpec)
        gen_a.name = "ModelA"
        gen_b = MagicMock(spec=GeneratingModelSpec)
        gen_b.name = "ModelB"
        config = MagicMock(spec=ModelRecoveryConfig)
        config.generating_models = [gen_a, gen_b]

        return ModelRecoveryResult(config=config, replications=tuple(replications))

    def test_perfect_recovery(self) -> None:
        reps = [_make_replication(i, "ModelA", "ModelA") for i in range(5)] + [
            _make_replication(i, "ModelB", "ModelB") for i in range(5)
        ]
        result = self._make_result(reps)
        rates = compute_recovery_rates(result)
        assert rates["ModelA"] == pytest.approx(1.0)
        assert rates["ModelB"] == pytest.approx(1.0)

    def test_partial_recovery(self) -> None:
        reps = [
            _make_replication(0, "ModelA", "ModelA"),
            _make_replication(1, "ModelA", "ModelB"),
            _make_replication(0, "ModelB", "ModelA"),
            _make_replication(1, "ModelB", "ModelA"),
        ]
        result = self._make_result(reps)
        rates = compute_recovery_rates(result)
        assert rates["ModelA"] == pytest.approx(0.5)
        assert rates["ModelB"] == pytest.approx(0.0)

    def test_nan_for_zero_reps(self) -> None:
        result = self._make_result([_make_replication(0, "ModelA", "ModelA")])
        rates = compute_recovery_rates(result)
        assert rates["ModelA"] == pytest.approx(1.0)
        assert rates["ModelB"] != rates["ModelB"]  # nan check


# ---------------------------------------------------------------------------
# display
# ---------------------------------------------------------------------------


class TestDisplay:
    def _make_matrix(self) -> dict[str, dict[str, int]]:
        return {
            "ModelA": {"ModelA": 8, "ModelB": 2},
            "ModelB": {"ModelA": 3, "ModelB": 7},
        }

    def test_confusion_matrix_table_contains_counts(self) -> None:
        matrix = self._make_matrix()
        table = model_recovery_confusion_table(matrix, ["ModelA", "ModelB"])
        assert "8" in table
        assert "7" in table
        assert "ModelA" in table
        assert "ModelB" in table

    def test_confusion_matrix_table_has_header_and_separator(self) -> None:
        matrix = self._make_matrix()
        table = model_recovery_confusion_table(matrix, ["ModelA", "ModelB"])
        lines = table.split("\n")
        assert len(lines) >= 3

    def test_recovery_rate_table_contains_rates(self) -> None:
        from unittest.mock import MagicMock

        from comp_model.recovery.model.config import GeneratingModelSpec, ModelRecoveryConfig

        gen_a = MagicMock(spec=GeneratingModelSpec)
        gen_a.name = "ModelA"
        gen_b = MagicMock(spec=GeneratingModelSpec)
        gen_b.name = "ModelB"
        config = MagicMock(spec=ModelRecoveryConfig)
        config.generating_models = [gen_a, gen_b]
        reps = [
            _make_replication(0, "ModelA", "ModelA"),
            _make_replication(1, "ModelA", "ModelB"),
            _make_replication(0, "ModelB", "ModelB"),
        ]
        result = ModelRecoveryResult(config=config, replications=tuple(reps))
        rates = {"ModelA": 0.5, "ModelB": 1.0}
        table = model_recovery_rate_table(rates, result)
        assert "ModelA" in table
        assert "ModelB" in table
        assert "0.500" in table
        assert "1.000" in table
