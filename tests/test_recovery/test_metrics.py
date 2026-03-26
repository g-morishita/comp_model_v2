"""Tests for parameter recovery metrics."""

from __future__ import annotations

import numpy as np
import pytest

from comp_model.recovery.parameter.metrics import (
    compute_parameter_recovery_metrics,
    compute_population_metrics,
)
from comp_model.recovery.parameter.result import (
    ParameterRecoveryResult,
    PopulationLevelResult,
    PopulationRecord,
    ReplicationResult,
    SubjectLevelResult,
)


class TestComputeParameterRecoveryMetrics:
    """Tests for pooled recovery metric computation."""

    def test_population_metrics_include_condition_specific_constrained_records(self) -> None:
        """Condition-specific constrained population records should be reported."""

        result = ParameterRecoveryResult(
            config=None,  # type: ignore[arg-type]
            replications=(
                ReplicationResult(
                    replication_index=0,
                    subject_level=SubjectLevelResult(records=()),
                    population_level=PopulationLevelResult(
                        records=(
                            PopulationRecord(
                                param_name="alpha_pop",
                                condition="baseline",
                                true_value=0.4,
                                estimated_value=0.42,
                                posterior_draws=np.linspace(0.3, 0.5, 101),
                            ),
                            PopulationRecord(
                                param_name="alpha_pop",
                                condition="social",
                                true_value=0.6,
                                estimated_value=0.58,
                                posterior_draws=np.linspace(0.5, 0.7, 101),
                            ),
                        )
                    ),
                ),
                ReplicationResult(
                    replication_index=1,
                    subject_level=SubjectLevelResult(records=()),
                    population_level=PopulationLevelResult(
                        records=(
                            PopulationRecord(
                                param_name="alpha_pop",
                                condition="baseline",
                                true_value=0.5,
                                estimated_value=0.52,
                                posterior_draws=np.linspace(0.4, 0.6, 101),
                            ),
                            PopulationRecord(
                                param_name="alpha_pop",
                                condition="social",
                                true_value=0.7,
                                estimated_value=0.68,
                                posterior_draws=np.linspace(0.6, 0.8, 101),
                            ),
                        )
                    ),
                ),
            ),
        )

        pooled_metrics = compute_parameter_recovery_metrics(result)
        population_metrics = compute_population_metrics(result)

        expected_keys = {"alpha_pop__baseline", "alpha_pop__social"}
        assert set(pooled_metrics.per_parameter) == expected_keys
        assert set(population_metrics.per_parameter) == expected_keys
        metric = population_metrics.per_parameter["alpha_pop__social"]
        assert metric.n_observations == 2
        assert metric.coverage_90 == pytest.approx(1.0)
        assert metric.coverage_95 == pytest.approx(1.0)
