"""Integration smoke test for the model recovery runner."""

from __future__ import annotations

import pytest
from scipy import stats

from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.models.kernels import AsocialQLearningKernel
from comp_model.recovery import ParamDist
from comp_model.recovery.model import (
    CandidateModelSpec,
    GeneratingModelSpec,
    ModelRecoveryConfig,
    compute_confusion_matrix,
    compute_recovery_rates,
    run_model_recovery,
)
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_ACTIONS = 2
N_TRIALS = 30  # small for fast tests


@pytest.fixture()
def bandit_task() -> TaskSpec:
    return TaskSpec(
        task_id="test_bandit",
        blocks=(
            BlockSpec(
                condition="default",
                n_trials=N_TRIALS,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": N_ACTIONS},
            ),
        ),
    )


def _env_factory() -> StationaryBanditEnvironment:
    return StationaryBanditEnvironment(n_actions=N_ACTIONS, reward_probs=(0.7, 0.3))


def _mle_config(n_restarts: int = 1) -> InferenceConfig:
    return InferenceConfig(
        hierarchy=HierarchyStructure.SUBJECT_SHARED,
        backend="mle",
        mle_config=MleOptimizerConfig(n_restarts=n_restarts, seed=0),
    )


_PARAM_DISTS = (
    ParamDist("alpha", stats.uniform(0.1, 0.8)),
    ParamDist("beta", stats.uniform(1.0, 9.0)),
)

# ---------------------------------------------------------------------------
# Smoke test: single generating model, two candidate specs
# ---------------------------------------------------------------------------


class TestRunModelRecovery:
    def test_basic_run_produces_expected_replications(self, bandit_task: TaskSpec) -> None:
        """Runner completes and produces one ReplicationResult per rep x gen model."""

        kernel = AsocialQLearningKernel()
        config = ModelRecoveryConfig(
            generating_models=(
                GeneratingModelSpec(
                    name="Q-Learning",
                    kernel=kernel,
                    param_dists=_PARAM_DISTS,
                ),
            ),
            candidate_models=(
                CandidateModelSpec(
                    name="CandA",
                    kernel=AsocialQLearningKernel(),
                    inference_config=_mle_config(),
                ),
                CandidateModelSpec(
                    name="CandB",
                    kernel=AsocialQLearningKernel(),
                    inference_config=_mle_config(),
                ),
            ),
            n_replications=2,
            n_subjects=3,
            task=bandit_task,
            env_factory=_env_factory,
            schema=ASOCIAL_BANDIT_SCHEMA,
            criterion="aic",
            simulation_base_seed=0,
            max_workers=1,
        )

        result = run_model_recovery(config)

        assert len(result.replications) == 2
        for rep in result.replications:
            assert rep.generating_model == "Q-Learning"
            assert rep.selected_model in ("CandA", "CandB")
            assert "CandA" in rep.candidate_scores
            assert "CandB" in rep.candidate_scores
            assert rep.winner_score == max(rep.candidate_scores.values())
            assert rep.second_best_model in ("CandA", "CandB")
            assert rep.delta_to_second is not None

    def test_bic_criterion_works(self, bandit_task: TaskSpec) -> None:
        """Runner runs end-to-end with BIC criterion."""

        kernel = AsocialQLearningKernel()
        config = ModelRecoveryConfig(
            generating_models=(
                GeneratingModelSpec(name="Q", kernel=kernel, param_dists=_PARAM_DISTS),
            ),
            candidate_models=(
                CandidateModelSpec(
                    name="Fit", kernel=AsocialQLearningKernel(), inference_config=_mle_config()
                ),
            ),
            n_replications=1,
            n_subjects=2,
            task=bandit_task,
            env_factory=_env_factory,
            schema=ASOCIAL_BANDIT_SCHEMA,
            criterion="bic",
            simulation_base_seed=99,
            max_workers=1,
        )

        result = run_model_recovery(config)
        assert len(result.replications) == 1
        assert result.replications[0].selected_model == "Fit"

    def test_log_likelihood_criterion_works(self, bandit_task: TaskSpec) -> None:
        """Runner runs end-to-end with log_likelihood criterion."""

        kernel = AsocialQLearningKernel()
        config = ModelRecoveryConfig(
            generating_models=(
                GeneratingModelSpec(name="Q", kernel=kernel, param_dists=_PARAM_DISTS),
            ),
            candidate_models=(
                CandidateModelSpec(
                    name="Fit", kernel=AsocialQLearningKernel(), inference_config=_mle_config()
                ),
            ),
            n_replications=1,
            n_subjects=2,
            task=bandit_task,
            env_factory=_env_factory,
            schema=ASOCIAL_BANDIT_SCHEMA,
            criterion="log_likelihood",
            simulation_base_seed=77,
            max_workers=1,
        )

        result = run_model_recovery(config)
        assert len(result.replications) == 1

    def test_confusion_matrix_and_rates_shape(self, bandit_task: TaskSpec) -> None:
        """Confusion matrix and recovery rates have expected keys."""

        kernel = AsocialQLearningKernel()
        config = ModelRecoveryConfig(
            generating_models=(
                GeneratingModelSpec(name="Q", kernel=kernel, param_dists=_PARAM_DISTS),
            ),
            candidate_models=(
                CandidateModelSpec(
                    name="Q",
                    kernel=AsocialQLearningKernel(),
                    inference_config=_mle_config(),
                ),
            ),
            n_replications=2,
            n_subjects=2,
            task=bandit_task,
            env_factory=_env_factory,
            schema=ASOCIAL_BANDIT_SCHEMA,
            criterion="aic",
            simulation_base_seed=1,
            max_workers=1,
        )

        result = run_model_recovery(config)
        matrix = compute_confusion_matrix(result)
        rates = compute_recovery_rates(result)

        assert "Q" in matrix
        assert "Q" in matrix["Q"]
        assert matrix["Q"]["Q"] == 2  # both reps → correct (only one candidate)

        assert "Q" in rates
        assert rates["Q"] == pytest.approx(1.0)

    def test_multiple_generating_models(self, bandit_task: TaskSpec) -> None:
        """Runner iterates over multiple generating models correctly."""

        kernel = AsocialQLearningKernel()
        config = ModelRecoveryConfig(
            generating_models=(
                GeneratingModelSpec(name="GenA", kernel=kernel, param_dists=_PARAM_DISTS),
                GeneratingModelSpec(name="GenB", kernel=kernel, param_dists=_PARAM_DISTS),
            ),
            candidate_models=(
                CandidateModelSpec(
                    name="Cand",
                    kernel=AsocialQLearningKernel(),
                    inference_config=_mle_config(),
                ),
            ),
            n_replications=1,
            n_subjects=2,
            task=bandit_task,
            env_factory=_env_factory,
            schema=ASOCIAL_BANDIT_SCHEMA,
            criterion="aic",
            simulation_base_seed=5,
            max_workers=1,
        )

        result = run_model_recovery(config)
        assert len(result.replications) == 2  # 2 generating models x 1 rep each
        gen_names = {rep.generating_model for rep in result.replications}
        assert gen_names == {"GenA", "GenB"}
