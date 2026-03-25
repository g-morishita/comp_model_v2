"""Tests for parameter recovery metrics."""

from __future__ import annotations

import numpy as np
import pytest

from comp_model.recovery.parameter.metrics import compute_parameter_recovery_metrics
from comp_model.recovery.parameter.result import (
    ParameterRecoveryResult,
    PopulationLevelResult,
    PopulationRecord,
    ReplicationResult,
    SubjectLevelResult,
)


class TestComputeParameterRecoveryMetrics:
    """Tests for pooled recovery metric computation."""

    def test_population_condition_keys_are_scored_without_hdi_crashes(self) -> None:
        """Condition-specific population records should be keyed and scored safely."""

        result = ParameterRecoveryResult(
            config=None,  # type: ignore[arg-type]
            replications=(
                ReplicationResult(
                    replication_index=0,
                    subject_level=SubjectLevelResult(records=()),
                    population_level=PopulationLevelResult(
                        records=(
                            PopulationRecord(
                                param_name="mu_alpha_delta_z",
                                condition="social",
                                true_value=-0.4,
                                estimated_value=-0.39,
                                posterior_draws=np.linspace(-0.5, -0.3, 101),
                            ),
                        )
                    ),
                ),
            ),
        )

        metrics = compute_parameter_recovery_metrics(result)

        assert set(metrics.per_parameter) == {"mu_alpha_delta_z__social"}
        metric = metrics.per_parameter["mu_alpha_delta_z__social"]
        assert metric.n_observations == 1
        assert metric.coverage_90 == pytest.approx(1.0)
        assert metric.coverage_95 == pytest.approx(1.0)
