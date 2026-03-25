"""Tests for the parameter recovery runner."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from comp_model.data.schema import Dataset
from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference.bayes.result import BayesFitResult
from comp_model.inference.bayes.stan import AsocialQLearningStanAdapter
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.models.condition.shared_delta import SharedDeltaLayout
from comp_model.models.kernels import AsocialQLearningKernel
from comp_model.recovery import ParamDist
from comp_model.recovery.parameter import runner as runner_module
from comp_model.recovery.parameter.config import ParameterRecoveryConfig
from comp_model.recovery.parameter.runner import run_parameter_recovery
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec


def _env_factory() -> StationaryBanditEnvironment:
    """Create a minimal environment for parameter-runner tests.

    Returns
    -------
    StationaryBanditEnvironment
        Two-action bandit environment for deterministic runner tests.
    """

    return StationaryBanditEnvironment(n_actions=2, reward_probs=(0.7, 0.3))


def _condition_task() -> TaskSpec:
    """Create a two-condition task for within-subject recovery tests.

    Returns
    -------
    TaskSpec
        Task specification with baseline and social blocks.
    """

    return TaskSpec(
        task_id="parameter-recovery-condition-aware",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=1,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
            BlockSpec(
                condition="social",
                n_trials=1,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
        ),
    )


class TestRunParameterRecovery:
    """Tests for population-record generation in parameter recovery."""

    def test_condition_aware_stan_pop_records_are_emitted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Condition-aware Stan recovery must emit shared/delta population records.

        Parameters
        ----------
        monkeypatch
            Pytest monkeypatch fixture used to isolate runner dependencies.
        """

        kernel = AsocialQLearningKernel()
        layout = SharedDeltaLayout(
            kernel_spec=kernel.spec(),
            conditions=("baseline", "social"),
            baseline_condition="baseline",
        )
        config = ParameterRecoveryConfig(
            n_replications=1,
            n_subjects=2,
            param_dists=(
                ParamDist("alpha", stats.norm(0.1, 0.2), scale="unconstrained"),
                ParamDist("beta", stats.norm(1.0, 0.3), scale="unconstrained"),
                ParamDist("alpha__delta", stats.norm(-0.4, 0.1), scale="unconstrained"),
                ParamDist("beta__delta", stats.norm(0.5, 0.2), scale="unconstrained"),
            ),
            task=_condition_task(),
            env_factory=_env_factory,
            kernel=kernel,
            schema=ASOCIAL_BANDIT_SCHEMA,
            inference_config=InferenceConfig(
                hierarchy=HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
                backend="stan",
            ),
            layout=layout,
            adapter=AsocialQLearningStanAdapter(),
            simulation_base_seed=7,
            max_workers=1,
        )

        true_table = {
            "sub_00": {
                "alpha__baseline": 0.40,
                "alpha__social": 0.60,
                "beta__baseline": 1.00,
                "beta__social": 2.00,
            },
            "sub_01": {
                "alpha__baseline": 0.50,
                "alpha__social": 0.70,
                "beta__baseline": 1.50,
                "beta__social": 2.50,
            },
        }

        def _fake_sample_true_params(
            *args: object,
            **kwargs: object,
        ) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
            """Return a fixed condition-aware truth table for the runner test.

            Parameters
            ----------
            *args
                Unused positional arguments from the runner.
            **kwargs
                Unused keyword arguments from the runner.

            Returns
            -------
            tuple[dict[str, dict[str, float]], dict[str, object]]
                Fixed true-table and dummy parsed parameters.
            """
            return true_table, {}

        def _fake_simulate_dataset(
            cfg: ParameterRecoveryConfig,
            params_per_subject: dict[str, object],
            seed: int,
        ) -> Dataset:
            """Return a dummy dataset because fit is mocked in this test.

            Parameters
            ----------
            cfg
                Recovery configuration.
            params_per_subject
                Parsed subject parameters.
            seed
                Simulation seed.

            Returns
            -------
            Dataset
                Empty dataset placeholder for the mocked fit.
            """
            del cfg, params_per_subject, seed
            return Dataset(subjects=())

        def _fake_fit(*args: object, **kwargs: object) -> BayesFitResult:
            """Return a Bayesian fit result with condition-aware population keys.

            Parameters
            ----------
            *args
                Unused positional arguments from the runner.
            **kwargs
                Unused keyword arguments from the runner.

            Returns
            -------
            BayesFitResult
                Mock posterior samples with condition-aware population keys.
            """
            del args, kwargs
            n_draws = 8
            posterior = {
                "alpha": np.full((n_draws, 2, 2), 0.55),
                "beta": np.full((n_draws, 2, 2), 1.75),
                "alpha_shared_pop": np.full(n_draws, 0.45),
                "beta_shared_pop": np.full(n_draws, 1.25),
                "mu_alpha_shared_z": np.full(n_draws, 0.1),
                "sd_alpha_shared_z": np.full(n_draws, 0.2),
                "mu_beta_shared_z": np.full(n_draws, 1.0),
                "sd_beta_shared_z": np.full(n_draws, 0.3),
                "mu_alpha_delta_z": np.full((n_draws, 1), -0.4),
                "sd_alpha_delta_z": np.full((n_draws, 1), 0.1),
                "mu_beta_delta_z": np.full((n_draws, 1), 0.5),
                "sd_beta_delta_z": np.full((n_draws, 1), 0.2),
            }
            return BayesFitResult(
                model_id="asocial_q_learning",
                hierarchy=HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
                posterior_samples=posterior,
                log_lik=np.zeros((n_draws, 2)),
                subject_params=None,
                diagnostics={},
            )

        monkeypatch.setattr(runner_module, "sample_true_params", _fake_sample_true_params)
        monkeypatch.setattr(runner_module, "_simulate_dataset", _fake_simulate_dataset)
        monkeypatch.setattr(runner_module, "fit", _fake_fit)

        result = run_parameter_recovery(config)

        assert len(result.replications) == 1
        population_level = result.replications[0].population_level
        assert population_level is not None

        records = {
            (record.param_name, record.condition): record for record in population_level.records
        }
        # Constrained-scale shared population means
        assert ("alpha_shared_pop", None) in records
        assert ("beta_shared_pop", None) in records
        assert records[("alpha_shared_pop", None)].true_value == pytest.approx(0.45)
        assert records[("beta_shared_pop", None)].true_value == pytest.approx(1.25)

        # Unconstrained-scale shared mu/sd
        assert ("mu_alpha_shared_z", None) in records
        assert ("sd_alpha_shared_z", None) in records
        assert ("mu_beta_shared_z", None) in records
        assert ("sd_beta_shared_z", None) in records

        # Unconstrained-scale delta mu/sd (split per non-baseline condition)
        assert ("mu_alpha_delta_z", "social") in records
        assert ("sd_alpha_delta_z", "social") in records
        assert ("mu_beta_delta_z", "social") in records
        assert ("sd_beta_delta_z", "social") in records

        # True values for unconstrained-scale shared params come from ParamDist
        assert records[("mu_alpha_shared_z", None)].true_value == pytest.approx(0.1)
        assert records[("sd_alpha_shared_z", None)].true_value == pytest.approx(0.2)

        # True values for unconstrained-scale delta params come from ParamDist
        assert records[("mu_alpha_delta_z", "social")].true_value == pytest.approx(-0.4)
        assert records[("sd_alpha_delta_z", "social")].true_value == pytest.approx(0.1)
