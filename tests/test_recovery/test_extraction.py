"""Tests for recovery.extraction — estimate extraction from fit results."""

from __future__ import annotations

import numpy as np
import pytest

from comp_model.inference.bayes.result import BayesFitResult
from comp_model.inference.config import HierarchyStructure
from comp_model.inference.mle.optimize import MleFitResult
from comp_model.recovery.parameter.extraction import (
    extract_bayes_subject_records,
    extract_mle_subject_records,
    extract_population_records,
)
from comp_model.recovery.parameter.result import PopulationRecord, SubjectRecord

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


class TestExtractMleSubjectRecords:
    """Tests for extract_mle_subject_records."""

    def test_basic_extraction(self) -> None:
        """Each subject/param combination produces a SubjectRecord."""
        results = [
            _make_mle_result("s0", {"alpha": 0.3, "beta": 2.0}),
            _make_mle_result("s1", {"alpha": 0.5, "beta": 3.0}, converged=False),
        ]
        true_params = {
            "s0": {"alpha": 0.25, "beta": 1.8},
            "s1": {"alpha": 0.55, "beta": 2.9},
        }
        records = extract_mle_subject_records(results, true_params)

        assert len(records) == 4  # 2 subjects * 2 params
        assert all(isinstance(r, SubjectRecord) for r in records)

        # Check first subject's alpha record
        s0_alpha = [r for r in records if r.subject_id == "s0" and r.param_name == "alpha"]
        assert len(s0_alpha) == 1
        assert s0_alpha[0].true_value == 0.25
        assert s0_alpha[0].estimated_value == 0.3
        assert s0_alpha[0].condition is None
        assert s0_alpha[0].posterior_draws is None

    def test_condition_aware_extraction(self) -> None:
        """Condition-aware MLE produces records with condition set."""
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
        true_params = {
            "s0": {
                "alpha__easy": 0.35,
                "alpha__hard": 0.15,
                "beta__easy": 1.9,
                "beta__hard": 1.9,
            },
        }
        records = extract_mle_subject_records(results, true_params, layout=layout)

        easy_alpha = [r for r in records if r.param_name == "alpha" and r.condition == "easy"]
        assert len(easy_alpha) == 1
        assert easy_alpha[0].estimated_value == pytest.approx(0.4)
        assert easy_alpha[0].true_value == pytest.approx(0.35)

        hard_alpha = [r for r in records if r.param_name == "alpha" and r.condition == "hard"]
        assert len(hard_alpha) == 1
        assert hard_alpha[0].estimated_value == pytest.approx(0.2)
        assert hard_alpha[0].true_value == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Bayesian extraction
# ---------------------------------------------------------------------------


class TestExtractBayesSubjectRecords:
    """Tests for extract_bayes_subject_records."""

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
        true_params = {
            "s0": {"alpha": 0.25, "beta": 1.8},
            "s1": {"alpha": 0.30, "beta": 2.0},
            "s2": {"alpha": 0.35, "beta": 2.2},
        }

        records = extract_bayes_subject_records(result, subject_ids, ("alpha", "beta"), true_params)

        assert len(records) == 6  # 3 subjects * 2 params
        assert all(isinstance(r, SubjectRecord) for r in records)

        s0_alpha = [r for r in records if r.subject_id == "s0" and r.param_name == "alpha"]
        assert len(s0_alpha) == 1
        assert s0_alpha[0].condition is None
        assert s0_alpha[0].posterior_draws is not None
        assert s0_alpha[0].posterior_draws.shape == (n_draws,)
        assert s0_alpha[0].estimated_value == pytest.approx(
            float(np.mean(posterior["alpha"][:, 0])), abs=1e-10
        )
        assert s0_alpha[0].true_value == 0.25

    def test_1d_shared_samples(self) -> None:
        """SUBJECT_SHARED case: (n_draws,) — all subjects get same draws."""
        rng = np.random.default_rng(1)
        n_draws = 50
        alpha_draws = rng.normal(0.3, 0.01, size=n_draws)
        beta_draws = rng.normal(2.0, 0.1, size=n_draws)
        posterior = {"alpha": alpha_draws, "beta": beta_draws}
        result = _make_bayes_result(posterior, HierarchyStructure.SUBJECT_SHARED)
        subject_ids = ["s0", "s1", "s2"]
        true_params = {
            "s0": {"alpha": 0.25, "beta": 1.8},
            "s1": {"alpha": 0.30, "beta": 2.0},
            "s2": {"alpha": 0.35, "beta": 2.2},
        }

        records = extract_bayes_subject_records(result, subject_ids, ("alpha", "beta"), true_params)

        assert len(records) == 6
        # All subjects should get the same draws for shared params
        expected_alpha_mean = float(np.mean(alpha_draws))
        s0_alpha = next(r for r in records if r.subject_id == "s0" and r.param_name == "alpha")
        s1_alpha = next(r for r in records if r.subject_id == "s1" and r.param_name == "alpha")
        assert s0_alpha.posterior_draws is not None
        np.testing.assert_array_equal(s0_alpha.posterior_draws, alpha_draws)
        np.testing.assert_array_equal(s1_alpha.posterior_draws, alpha_draws)
        assert s0_alpha.estimated_value == pytest.approx(expected_alpha_mean, abs=1e-10)

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
        true_params = {
            "s0": {"alpha__easy": 0.25, "alpha__hard": 0.20},
            "s1": {"alpha__easy": 0.30, "alpha__hard": 0.28},
        }

        records = extract_bayes_subject_records(
            result, ["s0", "s1"], ("alpha",), true_params, layout=layout
        )

        assert len(records) == 4  # 2 subjects * 2 conditions
        s0_easy = [r for r in records if r.subject_id == "s0" and r.condition == "easy"]
        assert len(s0_easy) == 1
        assert s0_easy[0].param_name == "alpha"
        assert s0_easy[0].posterior_draws is not None
        assert s0_easy[0].posterior_draws.shape == (n_draws,)
        np.testing.assert_array_equal(s0_easy[0].posterior_draws, posterior["alpha"][:, 0, 0])
        assert s0_easy[0].true_value == 0.25

    def test_2d_condition_aware_samples(self) -> None:
        """SUBJECT_BLOCK_CONDITION case: (n_draws, n_conditions) with layout."""
        from comp_model.models.condition.shared_delta import SharedDeltaLayout
        from comp_model.models.kernels import AsocialQLearningKernel

        rng = np.random.default_rng(4)
        n_draws, n_conditions = 80, 2
        posterior = {
            "alpha": rng.normal(0.3, 0.01, size=(n_draws, n_conditions)),
        }
        result = _make_bayes_result(posterior, HierarchyStructure.SUBJECT_BLOCK_CONDITION)
        layout = SharedDeltaLayout(
            kernel_spec=AsocialQLearningKernel.spec(),
            conditions=("easy", "hard"),
            baseline_condition="easy",
        )
        true_params = {
            "s0": {"alpha__easy": 0.25, "alpha__hard": 0.20},
        }

        records = extract_bayes_subject_records(
            result, ["s0"], ("alpha",), true_params, layout=layout
        )

        assert len(records) == 2  # 1 subject * 2 conditions
        easy = [r for r in records if r.condition == "easy"]
        assert len(easy) == 1
        assert easy[0].param_name == "alpha"
        assert easy[0].true_value == 0.25
        assert easy[0].posterior_draws is not None
        assert easy[0].posterior_draws.shape == (n_draws,)
        np.testing.assert_array_equal(easy[0].posterior_draws, posterior["alpha"][:, 0])

        hard = [r for r in records if r.condition == "hard"]
        assert len(hard) == 1
        assert hard[0].true_value == 0.20
        np.testing.assert_array_equal(hard[0].posterior_draws, posterior["alpha"][:, 1])

    def test_single_subject_2d(self) -> None:
        """Edge case: 2D samples with only one subject."""
        rng = np.random.default_rng(3)
        n_draws = 40
        posterior = {"alpha": rng.normal(0.3, 0.01, size=(n_draws, 1))}
        result = _make_bayes_result(posterior)
        true_params = {"s0": {"alpha": 0.28}}

        records = extract_bayes_subject_records(result, ["s0"], ("alpha",), true_params)

        assert len(records) == 1
        assert records[0].posterior_draws is not None
        assert records[0].posterior_draws.shape == (n_draws,)


# ---------------------------------------------------------------------------
# Population extraction
# ---------------------------------------------------------------------------


class TestExtractPopulationRecords:
    """Tests for extract_population_records."""

    def test_pop_key_extraction(self) -> None:
        """Constrained-scale population mean keys are extracted."""
        rng = np.random.default_rng(10)
        n_draws = 100
        posterior = {
            "alpha": rng.normal(0.3, 0.01, size=(n_draws, 3)),
            "alpha_pop": rng.normal(0.3, 0.005, size=n_draws),
        }
        result = _make_bayes_result(posterior)
        true_pop = {"alpha_pop": 0.30}

        records = extract_population_records(result, true_pop)

        assert len(records) == 1
        assert isinstance(records[0], PopulationRecord)
        assert records[0].param_name == "alpha_pop"
        assert records[0].true_value == 0.30
        assert records[0].estimated_value == pytest.approx(float(np.mean(posterior["alpha_pop"])))
        assert records[0].posterior_draws is not None

    def test_unconstrained_keys_are_extracted(self) -> None:
        """Unconstrained-scale mu/sd keys are extracted as population records."""
        rng = np.random.default_rng(11)
        n_draws = 100
        posterior = {
            "alpha": rng.normal(0.3, 0.01, size=(n_draws, 3)),
            "mu_alpha_z": rng.normal(0.0, 0.1, size=n_draws),
            "sd_alpha_z": rng.normal(1.0, 0.1, size=n_draws),
        }
        result = _make_bayes_result(posterior)
        true_pop = {"mu_alpha_z": 0.0, "sd_alpha_z": 1.0}

        records = extract_population_records(result, true_pop)

        assert len(records) == 2
        names = {r.param_name for r in records}
        assert names == {"mu_alpha_z", "sd_alpha_z"}

    def test_missing_true_pop_skipped(self) -> None:
        """Keys present in posterior but missing from true_pop are skipped."""
        rng = np.random.default_rng(12)
        n_draws = 100
        posterior = {
            "alpha": rng.normal(0.3, 0.01, size=(n_draws, 3)),
            "alpha_pop": rng.normal(0.3, 0.005, size=n_draws),
        }
        result = _make_bayes_result(posterior)
        true_pop: dict[str, float] = {}  # no true values

        records = extract_population_records(result, true_pop)

        assert len(records) == 0

    def test_condition_aware_extracts_all_population_keys(self) -> None:
        """Condition-aware population extraction keeps constrained and unconstrained keys."""
        rng = np.random.default_rng(13)
        n_draws = 100
        posterior = {
            "alpha_shared_pop": rng.normal(0.3, 0.005, size=n_draws),
            "mu_alpha_shared_z": rng.normal(0.0, 0.1, size=n_draws),
            "sd_alpha_shared_z": rng.normal(1.0, 0.1, size=n_draws),
        }
        result = _make_bayes_result(posterior)
        true_pop = {
            "alpha_shared_pop": 0.30,
            "mu_alpha_shared_z": 0.0,
            "sd_alpha_shared_z": 1.0,
        }

        records = extract_population_records(result, true_pop)

        assert len(records) == 3
        names = {r.param_name for r in records}
        assert names == {"alpha_shared_pop", "mu_alpha_shared_z", "sd_alpha_shared_z"}

    def test_condition_aware_vector_delta_keys_are_split_per_condition(self) -> None:
        """Vector-valued unconstrained delta keys are split per non-baseline condition."""
        from comp_model.models.condition.shared_delta import SharedDeltaLayout
        from comp_model.models.kernels import AsocialQLearningKernel

        rng = np.random.default_rng(14)
        n_draws = 100
        posterior = {
            "mu_alpha_delta_z": rng.normal(0.0, 0.1, size=(n_draws, 2)),
            "sd_alpha_delta_z": rng.normal(0.5, 0.1, size=(n_draws, 2)),
        }
        result = _make_bayes_result(
            posterior,
            HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
        )
        layout = SharedDeltaLayout(
            kernel_spec=AsocialQLearningKernel.spec(),
            conditions=("baseline", "social", "transfer"),
            baseline_condition="baseline",
        )
        true_pop = {
            "mu_alpha_delta_z": 0.0,
            "sd_alpha_delta_z": 0.5,
        }

        records = extract_population_records(result, true_pop, layout=layout)

        assert len(records) == 4  # 2 keys * 2 non-baseline conditions
        record_keys = {(r.param_name, r.condition) for r in records}
        assert record_keys == {
            ("mu_alpha_delta_z", "social"),
            ("mu_alpha_delta_z", "transfer"),
            ("sd_alpha_delta_z", "social"),
            ("sd_alpha_delta_z", "transfer"),
        }
