"""Tests for the concrete asocial Q-learning example."""

import math

from example.asocial_q_learning import format_example_summary, run_example


def test_run_example_returns_finite_fit_and_summary() -> None:
    """Ensure the example workflow produces runnable simulation and fitting output.

    Returns
    -------
    None
        This test exercises the example module end to end.
    """

    result = run_example(
        n_trials=24,
        simulation_seed=5,
        optimizer_seed=2,
        n_restarts=2,
        max_iter=75,
    )

    assert result.subject.subject_id == "demo_subject"
    assert len(result.subject.blocks) == 1
    assert len(result.subject.blocks[0].trials) == 24
    assert result.fit_result.n_trials == 24
    assert len(result.fit_result.all_candidates) == 2
    assert math.isfinite(result.fit_result.log_likelihood)
    assert 0.0 < result.fit_result.constrained_params["alpha"] < 1.0
    assert result.fit_result.constrained_params["beta"] > 0.0

    summary = format_example_summary(result)
    assert "Generating parameters" in summary
    assert "Recovered parameters" in summary
