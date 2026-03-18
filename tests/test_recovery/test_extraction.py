"""Tests for recovery.extraction — estimate extraction from fit results."""

from __future__ import annotations

import numpy as np
import pytest

from comp_model.inference.bayes.result import BayesFitResult
from comp_model.inference.config import HierarchyStructure
from comp_model.inference.mle.optimize import MleFitResult
from comp_model.recovery.extraction import extract_bayes_estimates, extract_mle_estimates

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mle_result(
    subject_id: str,
    constrained: dict[str, float],
    *,
    converged: bool = True,
    params_by_condition: dict[str, dict[str, float]] | None = None,
) -> MleFitResult:
    """Build a minimal MleFitResult for testing."""
    return MleFitResult(
        subject_id=subject_id,
        model_id="test",
        log_likelihood=-10.0,
        n_params=len(constrained),
        raw_params={k: 0.0 for k in constrained},
        constrained_params=constrained,
        aic=20.0,
        bic=20.0,
        n_trials=100,
        converged=converged,
        n_restarts=1,
        all_candidates=(constrained,),
        all_log_likelihoods=(-10.0,),
        params_by_condition=params_by_condition,
    )


def _make_bayes_result(
    posterior: dict[str, np.ndarray],
    hierarchy: HierarchyStructure = HierarchyStructure.STUDY_SUBJECT,
) -> BayesFitResult:
    """Build a minimal BayesFitResult for testing."""
    return BayesFitResult(
        model_id="test",
        hierarchy=hierarchy,
        posterior_samples=posterior,
        log_lik=np.zeros((10, 1)),
        subject_params=None,
        diagnostics={},
    )


# ---------------------------------------------------------------------------
# MLE extraction
# ---------------------------------------------------------------------------


class TestExtractMleEstimates:
    def test_basic_extraction(self) -> None:
        results = [
            _make_mle_result("s0", {"alpha": 0.3, "beta": 2.0}),
            _make_mle_result("s1", {"alpha": 0.5, "beta": 3.0}, converged=False),
        ]
        estimates = extract_mle_estimates(results)

        assert len(estimates) == 2
        assert estimates[0].subject_id == "s0"
        assert estimates[0].point_estimates == {"alpha": 0.3, "beta": 2.0}
        assert estimates[0].posterior_samples is None
        assert estimates[0].converged is True

        assert estimates[1].subject_id == "s1"
        assert estimates[1].converged is False

    def test_condition_aware_extraction(self) -> None:
        from comp_model.models.condition.shared_delta import SharedDeltaLayout
        from comp_model.models.kernels import AsocialQLearningKernel

        layout = SharedDeltaLayout(
            kernel_spec=AsocialQLearningKernel.spec(),
            conditions=("easy", "hard"),
            baseline_condition="easy",
        )
        results = [
            _make_mle_result(
                "s0",
                {"alpha": 0.3, "beta": 2.0},
                params_by_condition={
                    "easy": {"alpha": 0.4, "beta": 2.0},
                    "hard": {"alpha": 0.2, "beta": 2.0},
                },
            ),
        ]
        estimates = extract_mle_estimates(results, layout=layout)

        assert "alpha__easy" in estimates[0].point_estimates
        assert "alpha__hard" in estimates[0].point_estimates
        assert estimates[0].point_estimates["alpha__easy"] == pytest.approx(0.4)
        assert estimates[0].point_estimates["alpha__hard"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Bayesian extraction
# ---------------------------------------------------------------------------


class TestExtractBayesEstimates:
    def test_2d_per_subject_samples(self) -> None:
        """Standard hierarchical case: (n_draws, n_subjects)."""
        rng = np.random.default_rng(0)
        n_draws, n_subjects = 100, 3
        posterior = {
            "alpha": rng.normal(0.3, 0.01, size=(n_draws, n_subjects)),
            "beta": rng.normal(2.0, 0.1, size=(n_draws, n_subjects)),
        }
        result = _make_bayes_result(posterior)
        subject_ids = ["s0", "s1", "s2"]

        estimates = extract_bayes_estimates(result, subject_ids, ("alpha", "beta"))

        assert len(estimates) == 3
        for i, est in enumerate(estimates):
            assert est.subject_id == subject_ids[i]
            assert est.converged is None
            assert est.posterior_samples is not None
            assert est.posterior_samples["alpha"].shape == (n_draws,)
            assert est.point_estimates["alpha"] == pytest.approx(
                float(np.mean(posterior["alpha"][:, i])), abs=1e-10
            )

    def test_1d_shared_samples(self) -> None:
        """SUBJECT_SHARED case: (n_draws,) — all subjects get same draws."""
        rng = np.random.default_rng(1)
        n_draws = 50
        alpha_draws = rng.normal(0.3, 0.01, size=n_draws)
        beta_draws = rng.normal(2.0, 0.1, size=n_draws)
        posterior = {"alpha": alpha_draws, "beta": beta_draws}
        result = _make_bayes_result(posterior, HierarchyStructure.SUBJECT_SHARED)
        subject_ids = ["s0", "s1", "s2"]

        estimates = extract_bayes_estimates(result, subject_ids, ("alpha", "beta"))

        assert len(estimates) == 3
        # All subjects should get the same draws and same point estimate
        expected_alpha_mean = float(np.mean(alpha_draws))
        for est in estimates:
            assert est.posterior_samples is not None
            np.testing.assert_array_equal(est.posterior_samples["alpha"], alpha_draws)
            assert est.point_estimates["alpha"] == pytest.approx(expected_alpha_mean, abs=1e-10)

    def test_3d_condition_aware_samples(self) -> None:
        """Condition-aware case: (n_draws, n_subjects, n_conditions)."""
        from comp_model.models.condition.shared_delta import SharedDeltaLayout
        from comp_model.models.kernels import AsocialQLearningKernel

        rng = np.random.default_rng(2)
        n_draws, n_subjects, n_conditions = 80, 2, 2
        posterior = {
            "alpha": rng.normal(0.3, 0.01, size=(n_draws, n_subjects, n_conditions)),
        }
        result = _make_bayes_result(posterior)
        layout = SharedDeltaLayout(
            kernel_spec=AsocialQLearningKernel.spec(),
            conditions=("easy", "hard"),
            baseline_condition="easy",
        )

        estimates = extract_bayes_estimates(
            result, ["s0", "s1"], ("alpha",), layout=layout
        )

        assert len(estimates) == 2
        est0 = estimates[0]
        assert "alpha__easy" in est0.point_estimates
        assert "alpha__hard" in est0.point_estimates
        assert est0.posterior_samples is not None
        assert est0.posterior_samples["alpha__easy"].shape == (n_draws,)
        np.testing.assert_array_equal(
            est0.posterior_samples["alpha__easy"], posterior["alpha"][:, 0, 0]
        )

    def test_single_subject_2d(self) -> None:
        """Edge case: 2D samples with only one subject."""
        rng = np.random.default_rng(3)
        n_draws = 40
        posterior = {"alpha": rng.normal(0.3, 0.01, size=(n_draws, 1))}
        result = _make_bayes_result(posterior)

        estimates = extract_bayes_estimates(result, ["s0"], ("alpha",))

        assert len(estimates) == 1
        assert estimates[0].posterior_samples is not None
        assert estimates[0].posterior_samples["alpha"].shape == (n_draws,)
