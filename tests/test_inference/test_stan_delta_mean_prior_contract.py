"""Regression tests for conditioned hierarchy delta-mean prior wiring in Stan."""

from __future__ import annotations

import re
from pathlib import Path

_PROGRAMS_DIR = Path(__file__).resolve().parents[2] / "src/comp_model/inference/bayes/stan/programs"
_CONDITIONED_STUDY_PROGRAMS = sorted(
    _PROGRAMS_DIR.glob("*__study_subject_block_condition_hierarchy.stan")
)
_HARDCODED_DELTA_MEAN = re.compile(r"mu_[a-z0-9_]+_delta_z ~ normal\(0, 1\);")
_CONFIGURABLE_DELTA_MEAN = re.compile(
    r"prior_lpdf\(mu_[a-z0-9_]+_delta_z\[d\] \| [a-z0-9_]+_delta_prior_family,"
)


def test_conditioned_study_programs_use_configurable_delta_mean_priors() -> None:
    """Conditioned study hierarchies should not hardcode delta-mean priors."""

    assert _CONDITIONED_STUDY_PROGRAMS

    for program in _CONDITIONED_STUDY_PROGRAMS:
        source = program.read_text()

        assert _HARDCODED_DELTA_MEAN.search(source) is None
        assert _CONFIGURABLE_DELTA_MEAN.search(source) is not None
