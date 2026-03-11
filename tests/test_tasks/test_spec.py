"""Tests for task-level design specifications."""

from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA, SOCIAL_PRE_CHOICE_SCHEMA
from comp_model.tasks.spec import BlockSpec, TaskSpec


def test_task_spec_reports_number_of_blocks() -> None:
    """Ensure task specs report block counts correctly.

    Returns
    -------
    None
        This test asserts the `n_blocks` property.
    """

    task = TaskSpec(
        task_id="bandit",
        blocks=(BlockSpec(condition="a", n_trials=10, schema=ASOCIAL_BANDIT_SCHEMA),),
    )

    assert task.n_blocks == 1


def test_task_spec_conditions_preserve_first_seen_order() -> None:
    """Ensure duplicate conditions are collapsed in order.

    Returns
    -------
    None
        This test asserts the ordered unique conditions.
    """

    task = TaskSpec(
        task_id="mixed",
        blocks=(
            BlockSpec(condition="baseline", n_trials=5, schema=ASOCIAL_BANDIT_SCHEMA),
            BlockSpec(condition="social", n_trials=5, schema=SOCIAL_PRE_CHOICE_SCHEMA),
            BlockSpec(condition="baseline", n_trials=5, schema=ASOCIAL_BANDIT_SCHEMA),
        ),
    )

    assert task.conditions == ("baseline", "social")
