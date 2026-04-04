"""Regression tests for vectorized action-tendency updates in Stan mixture programs."""

from __future__ import annotations

from pathlib import Path

_PROGRAMS_DIR = Path(__file__).resolve().parents[2] / "src/comp_model/inference/bayes/stan/programs"
_MIXTURE_PROGRAMS = sorted(_PROGRAMS_DIR.glob("social_rl_*mixture*.stan"))
_EXPECTED_MIXTURE_FAMILIES = {
    "social_rl_demo_mixture",
    "social_rl_self_reward_demo_action_mixture",
    "social_rl_self_reward_demo_action_mixture_sticky",
    "social_rl_self_reward_demo_mixture",
    "social_rl_self_reward_demo_mixture_sticky",
}


def _expected_vectorized_lines(program_name: str) -> tuple[str, str]:
    """Return the vectorized action-tendency update expected for a program."""

    if program_name.endswith("__subject_shared.stan"):
        return (
            "T = (1 - alpha_other_action) * T;",
            "T[sa] = T[sa] + alpha_other_action;",
        )
    if program_name.endswith("__subject_block_condition_hierarchy.stan"):
        return (
            "T = (1 - alpha_other_action[cc]) * T;",
            "T[sa] = T[sa] + alpha_other_action[cc];",
        )
    if program_name.endswith("__study_subject_hierarchy.stan"):
        return (
            "T[n] = (1 - alpha_other_action[n]) * T[n];",
            "T[n][sa] = T[n][sa] + alpha_other_action[n];",
        )
    if program_name.endswith("__study_subject_block_condition_hierarchy.stan"):
        return (
            "T[n] = (1 - alpha_other_action[n][cc]) * T[n];",
            "T[n][sa] = T[n][sa] + alpha_other_action[n][cc];",
        )
    raise AssertionError(f"Unhandled mixture Stan program: {program_name}")


def test_all_mixture_programs_use_vectorized_action_tendency_updates() -> None:
    """Mixture-family Stan programs should use the shared vectorized tendency update."""

    assert {
        program.name.split("__")[0] for program in _MIXTURE_PROGRAMS
    } == _EXPECTED_MIXTURE_FAMILIES
    assert len(_MIXTURE_PROGRAMS) == len(_EXPECTED_MIXTURE_FAMILIES) * 4

    for program in _MIXTURE_PROGRAMS:
        source = program.read_text()
        expected_decay, expected_increment = _expected_vectorized_lines(program.name)

        assert "for (a in 1:A) if (sa != a) T" not in source
        assert source.count(expected_decay) == 2
        assert source.count(expected_increment) == 2
