"""Tests for recovery display formatting."""

from __future__ import annotations

import numpy as np

from comp_model.recovery.parameter.display import parameter_recovery_tables
from comp_model.recovery.parameter.result import (
    ParameterRecoveryResult,
    PopulationLevelResult,
    PopulationRecord,
    ReplicationResult,
    SubjectLevelResult,
    SubjectRecord,
)


class TestParameterRecoveryTables:
    """Tests for parameter_recovery_tables."""

    def test_separate_subject_and_population_sections(self) -> None:
        """Output should contain both subject-level and population-level sections."""
        result = ParameterRecoveryResult(
            config=None,  # type: ignore[arg-type]
            replications=(
                ReplicationResult(
                    replication_index=0,
                    subject_level=SubjectLevelResult(
                        records=(
                            SubjectRecord(
                                subject_id="s0",
                                param_name="alpha",
                                condition=None,
                                true_value=0.3,
                                estimated_value=0.35,
                                posterior_draws=None,
                            ),
                        )
                    ),
                    population_level=PopulationLevelResult(
                        records=(
                            PopulationRecord(
                                param_name="alpha_pop",
                                condition=None,
                                true_value=0.3,
                                estimated_value=0.32,
                                posterior_draws=np.linspace(0.25, 0.35, 50),
                            ),
                        )
                    ),
                ),
            ),
        )

        output = parameter_recovery_tables(result)

        assert "Subject-level metrics" in output
        assert "Population-level metrics" in output
        # alpha should appear in subject section, alpha_pop in population section
        assert "alpha" in output
        assert "alpha_pop" in output

    def test_population_section_omitted_when_no_population_records(self) -> None:
        """Population section should not appear when there are no population records."""
        result = ParameterRecoveryResult(
            config=None,  # type: ignore[arg-type]
            replications=(
                ReplicationResult(
                    replication_index=0,
                    subject_level=SubjectLevelResult(
                        records=(
                            SubjectRecord(
                                subject_id="s0",
                                param_name="alpha",
                                condition=None,
                                true_value=0.3,
                                estimated_value=0.35,
                                posterior_draws=None,
                            ),
                        )
                    ),
                    population_level=None,
                ),
            ),
        )

        output = parameter_recovery_tables(result)

        assert "Subject-level metrics" in output
        assert "Population-level metrics" not in output

    def test_population_only_output_has_no_leading_blank_line(self) -> None:
        """Population-only output should start with the section label."""
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
                                condition=None,
                                true_value=0.3,
                                estimated_value=0.32,
                                posterior_draws=np.linspace(0.25, 0.35, 50),
                            ),
                        )
                    ),
                ),
            ),
        )

        output = parameter_recovery_tables(result)

        assert output.startswith("Population-level metrics")

    def test_population_tables_show_condition_specific_population_metrics(self) -> None:
        """Population table should show condition-specific constrained metrics."""
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
                                true_value=0.3,
                                estimated_value=0.32,
                                posterior_draws=np.linspace(0.25, 0.35, 50),
                            ),
                            PopulationRecord(
                                param_name="alpha_pop",
                                condition="social",
                                true_value=0.5,
                                estimated_value=0.48,
                                posterior_draws=np.linspace(0.4, 0.6, 50),
                            ),
                        )
                    ),
                ),
            ),
        )

        output = parameter_recovery_tables(result)

        assert "alpha_pop__baseline" in output
        assert "alpha_pop__social" in output
