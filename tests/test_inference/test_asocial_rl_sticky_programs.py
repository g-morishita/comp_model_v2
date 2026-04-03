"""Regression tests for choice-history handling in asocial sticky Stan programs."""

from __future__ import annotations

from pathlib import Path

_PROGRAMS_DIR = Path(__file__).resolve().parents[2] / "src/comp_model/inference/bayes/stan/programs"
_PROGRAMS = sorted(_PROGRAMS_DIR.glob("asocial_rl_sticky__*.stan"))


def _choice_history_assignment(program_name: str) -> str:
    """Return the expected choice-history update line for a program."""

    if program_name in {
        "asocial_rl_sticky__subject_shared.stan",
        "asocial_rl_sticky__subject_block_condition_hierarchy.stan",
    }:
        return "last_self_choice = step_choice[e];"
    if program_name in {
        "asocial_rl_sticky__study_subject_hierarchy.stan",
        "asocial_rl_sticky__study_subject_block_condition_hierarchy.stan",
    }:
        return "last_self_choice[n] = step_choice[e];"
    raise AssertionError(f"Unhandled asocial sticky Stan program: {program_name}")


def test_asocial_sticky_programs_preserve_choice_history_without_feedback() -> None:
    """Choice rows should update stickiness history even when no feedback row is emitted."""

    assert len(_PROGRAMS) == 4

    for program in _PROGRAMS:
        source = program.read_text()
        expected_assignment = _choice_history_assignment(program.name)

        assert source.count(expected_assignment) == 2
