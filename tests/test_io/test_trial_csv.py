"""Tests for schema-specific trial CSV import and export."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import pytest

from comp_model.data import (
    Block,
    Dataset,
    Event,
    EventPhase,
    SubjectData,
    Trial,
    replay_trial_steps,
)
from comp_model.io import (
    get_trial_csv_converter,
    load_dataset_from_csv,
    register_trial_csv_converter,
    save_dataset_to_csv,
)
from comp_model.tasks import (
    ASOCIAL_BANDIT_SCHEMA,
    SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA,
    SOCIAL_POST_OUTCOME_DEMO_LEARNS_SCHEMA,
    SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA,
    SOCIAL_POST_OUTCOME_SCHEMA,
    SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
    SOCIAL_PRE_CHOICE_DEMO_LEARNS_SCHEMA,
    SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA,
    SOCIAL_PRE_CHOICE_SCHEMA,
    TrialSchema,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_get_trial_csv_converter_returns_registered_builtin() -> None:
    """Ensure built-in schemas resolve to registered trial CSV converters.

    Returns
    -------
    None
        This test asserts registry lookup behavior only.
    """

    converter = get_trial_csv_converter(ASOCIAL_BANDIT_SCHEMA)

    assert converter.schema_id == ASOCIAL_BANDIT_SCHEMA.schema_id
    assert converter.fieldnames == (
        "subject_id",
        "block_index",
        "condition",
        "schema_id",
        "trial_index",
        "available_actions",
        "choice",
        "reward",
    )


def test_save_and_load_dataset_round_trip_preserves_fitting_views(tmp_path: Path) -> None:
    """Ensure schema-specific CSV round-trips preserve fitting-relevant views.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test compares extracted fitting-relevant values after round-trip.
    """

    cases = (
        (ASOCIAL_BANDIT_SCHEMA, _make_asocial_dataset()),
        (SOCIAL_PRE_CHOICE_SCHEMA, _make_social_pre_choice_dataset()),
        (
            SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
            _dataset_with_schema_id(
                _make_social_pre_choice_dataset(),
                SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA.schema_id,
            ),
        ),
        (
            SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA,
            _make_social_pre_choice_no_self_outcome_dataset(),
        ),
        (
            SOCIAL_PRE_CHOICE_DEMO_LEARNS_SCHEMA,
            _make_social_pre_choice_demo_learns_dataset(),
        ),
        (SOCIAL_POST_OUTCOME_SCHEMA, _make_social_post_outcome_dataset()),
        (
            SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA,
            _dataset_with_schema_id(
                _make_social_post_outcome_dataset(),
                SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA.schema_id,
            ),
        ),
        (
            SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA,
            _make_social_post_outcome_no_self_outcome_dataset(),
        ),
        (
            SOCIAL_POST_OUTCOME_DEMO_LEARNS_SCHEMA,
            _make_social_post_outcome_demo_learns_dataset(),
        ),
    )

    for schema, dataset in cases:
        csv_path = tmp_path / f"{schema.schema_id}.csv"
        save_dataset_to_csv(dataset, schema=schema, path=csv_path)
        loaded_dataset = load_dataset_from_csv(csv_path, schema=schema)

        assert _fitting_view_signatures(dataset, schema) == _fitting_view_signatures(
            loaded_dataset,
            schema,
        )


def test_load_social_pre_choice_reconstructs_demonstrator_before_decision(
    tmp_path: Path,
) -> None:
    """Ensure the pre-choice converter restores demonstrator input timing.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test inspects canonical event order after loading.
    """

    dataset = _make_social_pre_choice_dataset()
    csv_path = tmp_path / "social_pre_choice.csv"

    save_dataset_to_csv(dataset, schema=SOCIAL_PRE_CHOICE_SCHEMA, path=csv_path)
    loaded_dataset = load_dataset_from_csv(csv_path, schema=SOCIAL_PRE_CHOICE_SCHEMA)

    trial = loaded_dataset.subjects[0].blocks[0].trials[0]
    # New schema: INPUT(demo) first, social UPDATE at index 4, subject DECISION at 6
    assert trial.events[0].phase == EventPhase.INPUT
    assert trial.events[0].actor_id == "demonstrator"
    assert trial.events[4].phase == EventPhase.UPDATE
    assert trial.events[6].phase == EventPhase.DECISION


def test_load_social_post_outcome_reconstructs_demonstrator_after_outcome(
    tmp_path: Path,
) -> None:
    """Ensure the post-outcome converter restores demonstrator input timing.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test inspects canonical event order after loading.
    """

    dataset = _make_social_post_outcome_dataset()
    csv_path = tmp_path / "social_post_outcome.csv"

    save_dataset_to_csv(dataset, schema=SOCIAL_POST_OUTCOME_SCHEMA, path=csv_path)
    loaded_dataset = load_dataset_from_csv(csv_path, schema=SOCIAL_POST_OUTCOME_SCHEMA)

    trial = loaded_dataset.subjects[0].blocks[0].trials[0]
    assert trial.events[2].phase == EventPhase.OUTCOME
    assert trial.events[4].phase == EventPhase.INPUT
    assert trial.events[4].actor_id == "demonstrator"


def test_save_action_only_schema_preserves_demonstrator_reward_in_csv(tmp_path: Path) -> None:
    """Ensure action-only export keeps the demonstrator's latent reward in CSV.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test checks the serialized demonstrator reward cells directly.
    """

    cases = (
        (
            SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
            _dataset_with_schema_id(
                _make_social_pre_choice_dataset(),
                SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA.schema_id,
            ),
            "1.0",
        ),
        (
            SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA,
            _dataset_with_schema_id(
                _make_social_post_outcome_dataset(),
                SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA.schema_id,
            ),
            "0.0",
        ),
    )

    for schema, dataset, expected_demo_reward in cases:
        csv_path = tmp_path / f"{schema.schema_id}.csv"
        save_dataset_to_csv(dataset, schema=schema, path=csv_path)

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 1
        assert rows[0]["demonstrator_reward"] == expected_demo_reward


def test_load_action_only_schema_keeps_social_reward_hidden_from_subject(
    tmp_path: Path,
) -> None:
    """Ensure action-only replay still hides reward after CSV round-trip.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test verifies the subject-facing social update still has
        ``reward is None`` after export and import.
    """

    cases = (
        (
            SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
            _dataset_with_schema_id(
                _make_social_pre_choice_dataset(),
                SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA.schema_id,
            ),
        ),
        (
            SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA,
            _dataset_with_schema_id(
                _make_social_post_outcome_dataset(),
                SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA.schema_id,
            ),
        ),
    )

    for schema, dataset in cases:
        csv_path = tmp_path / f"{schema.schema_id}.csv"
        save_dataset_to_csv(dataset, schema=schema, path=csv_path)
        loaded_dataset = load_dataset_from_csv(csv_path, schema=schema)

        trial = loaded_dataset.subjects[0].blocks[0].trials[0]
        subject_social_updates = [
            view
            for event_type, learner_id, view in replay_trial_steps(trial, schema)
            if event_type == EventPhase.UPDATE
            and learner_id == "subject"
            and view.actor_id == "demonstrator"
        ]

        assert len(subject_social_updates) == 1
        assert subject_social_updates[0].reward is None


def test_load_social_pre_choice_demo_learns_preserves_demo_feedback_to_both_agents(
    tmp_path: Path,
) -> None:
    """Ensure pre-choice demo-learns reconstruction preserves both update sides.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test inspects the canonical demo-facing and subject-facing update
        payloads after CSV round-trip.
    """

    dataset = _make_social_pre_choice_demo_learns_dataset()
    csv_path = tmp_path / "social_pre_choice_demo_learns.csv"

    save_dataset_to_csv(dataset, schema=SOCIAL_PRE_CHOICE_DEMO_LEARNS_SCHEMA, path=csv_path)
    loaded_dataset = load_dataset_from_csv(csv_path, schema=SOCIAL_PRE_CHOICE_DEMO_LEARNS_SCHEMA)

    trial = loaded_dataset.subjects[0].blocks[0].trials[0]
    assert len(trial.events) == 10
    assert trial.events[4].actor_id == "demonstrator"
    assert trial.events[4].payload == {"choice": 0, "reward": 1.0}
    assert trial.events[9].actor_id == "subject"
    assert trial.events[9].payload == {"choice": 1, "reward": 0.0}


def test_load_social_post_outcome_demo_learns_preserves_demo_feedback_to_both_agents(
    tmp_path: Path,
) -> None:
    """Ensure post-outcome demo-learns reconstruction preserves both update sides.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test inspects the canonical demo-facing and subject-facing update
        payloads after CSV round-trip.
    """

    dataset = _make_social_post_outcome_demo_learns_dataset()
    csv_path = tmp_path / "social_post_outcome_demo_learns.csv"

    save_dataset_to_csv(dataset, schema=SOCIAL_POST_OUTCOME_DEMO_LEARNS_SCHEMA, path=csv_path)
    loaded_dataset = load_dataset_from_csv(
        csv_path,
        schema=SOCIAL_POST_OUTCOME_DEMO_LEARNS_SCHEMA,
    )

    trial = loaded_dataset.subjects[0].blocks[0].trials[0]
    assert len(trial.events) == 10
    assert trial.events[4].actor_id == "subject"
    assert trial.events[4].payload == {"choice": 0, "reward": 1.0}
    assert trial.events[9].actor_id == "demonstrator"
    assert trial.events[9].payload == {"choice": 1, "reward": 0.0}


def test_save_no_self_outcome_schema_writes_blank_subject_reward(tmp_path: Path) -> None:
    """Ensure no-self-outcome schemas export an empty subject reward cell.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test checks the serialized CSV row contents directly.
    """

    cases = (
        (
            SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA,
            _make_social_pre_choice_no_self_outcome_dataset(),
        ),
        (
            SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA,
            _make_social_post_outcome_no_self_outcome_dataset(),
        ),
    )

    for schema, dataset in cases:
        csv_path = tmp_path / f"{schema.schema_id}.csv"
        save_dataset_to_csv(dataset, schema=schema, path=csv_path)

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 1
        assert rows[0]["reward"] == ""


def test_load_social_pre_choice_no_self_outcome_preserves_missing_subject_outcome(
    tmp_path: Path,
) -> None:
    """Ensure pre-choice no-self-outcome reconstruction does not add reward events.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test inspects canonical event order after loading.
    """

    dataset = _make_social_pre_choice_no_self_outcome_dataset()
    csv_path = tmp_path / "social_pre_choice_no_self_outcome.csv"

    save_dataset_to_csv(dataset, schema=SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA, path=csv_path)
    loaded_dataset = load_dataset_from_csv(
        csv_path,
        schema=SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA,
    )

    trial = loaded_dataset.subjects[0].blocks[0].trials[0]
    assert len(trial.events) == 7
    assert [(event.phase, event.actor_id) for event in trial.events] == [
        (EventPhase.INPUT, "demonstrator"),
        (EventPhase.DECISION, "demonstrator"),
        (EventPhase.OUTCOME, "demonstrator"),
        (EventPhase.UPDATE, "demonstrator"),
        (EventPhase.UPDATE, "demonstrator"),
        (EventPhase.INPUT, "subject"),
        (EventPhase.DECISION, "subject"),
    ]


def test_load_social_post_outcome_no_self_outcome_preserves_missing_subject_outcome(
    tmp_path: Path,
) -> None:
    """Ensure post-outcome no-self-outcome reconstruction does not add reward events.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test inspects canonical event order after loading.
    """

    dataset = _make_social_post_outcome_no_self_outcome_dataset()
    csv_path = tmp_path / "social_post_outcome_no_self_outcome.csv"

    save_dataset_to_csv(dataset, schema=SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA, path=csv_path)
    loaded_dataset = load_dataset_from_csv(
        csv_path,
        schema=SOCIAL_POST_OUTCOME_NO_SELF_OUTCOME_SCHEMA,
    )

    trial = loaded_dataset.subjects[0].blocks[0].trials[0]
    assert len(trial.events) == 7
    assert [(event.phase, event.actor_id) for event in trial.events] == [
        (EventPhase.INPUT, "subject"),
        (EventPhase.DECISION, "subject"),
        (EventPhase.INPUT, "demonstrator"),
        (EventPhase.DECISION, "demonstrator"),
        (EventPhase.OUTCOME, "demonstrator"),
        (EventPhase.UPDATE, "demonstrator"),
        (EventPhase.UPDATE, "demonstrator"),
    ]


def test_load_no_self_outcome_rejects_non_empty_subject_reward(tmp_path: Path) -> None:
    """Ensure no-self-outcome schemas reject rows that invent subject rewards.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test raises on non-empty ``reward`` for a no-self-outcome schema.
    """

    csv_path = tmp_path / "bad_no_self_outcome.csv"
    csv_path.write_text(
        "\n".join(
            [
                (
                    "subject_id,block_index,condition,schema_id,trial_index,"
                    "available_actions,choice,reward,demonstrator_choice,demonstrator_reward"
                ),
                "s1,0,social,social_pre_choice_no_self_outcome,0,0|1,1,1.0,0,0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Field 'reward' must be empty"):
        load_dataset_from_csv(csv_path, schema=SOCIAL_PRE_CHOICE_NO_SELF_OUTCOME_SCHEMA)


@pytest.mark.parametrize("missing_value", ("", "NA"))
def test_load_dataset_from_csv_accepts_missing_asocial_choice_as_timeout(
    tmp_path: Path,
    missing_value: str,
) -> None:
    """Ensure blank/NA asocial choices load as timeout-style canonical events.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.
    missing_value
        Missing-value marker written into the ``choice`` and ``reward`` cells.

    Returns
    -------
    None
        This test checks canonical reconstruction and replay behavior.
    """

    csv_path = tmp_path / "asocial_timeout.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,block_index,condition,schema_id,trial_index,available_actions,choice,reward",
                f"s1,0,A,asocial_bandit,0,0|1,{missing_value},{missing_value}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset_from_csv(csv_path, schema=ASOCIAL_BANDIT_SCHEMA)

    trial = dataset.subjects[0].blocks[0].trials[0]
    assert trial.events[1].payload["action"] is None
    assert trial.events[2].payload["reward"] is None
    assert dict(trial.events[3].payload) == {"choice": None, "reward": None}

    subject_steps = [
        (event_type, view.action, view.reward)
        for event_type, learner_id, view in replay_trial_steps(trial, ASOCIAL_BANDIT_SCHEMA)
        if learner_id == "subject"
    ]
    assert subject_steps == [(EventPhase.UPDATE, None, None)]


@pytest.mark.parametrize("missing_value", ("", "NA"))
def test_load_social_pre_choice_missing_subject_choice_preserves_social_update(
    tmp_path: Path,
    missing_value: str,
) -> None:
    """Ensure timeout rows still replay demonstrator learning before self update.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.
    missing_value
        Missing-value marker written into the subject ``choice``/``reward`` cells.

    Returns
    -------
    None
        This test checks subject-facing replay behavior after CSV import.
    """

    csv_path = tmp_path / "social_pre_choice_timeout.csv"
    csv_path.write_text(
        "\n".join(
            [
                (
                    "subject_id,block_index,condition,schema_id,trial_index,"
                    "available_actions,choice,reward,demonstrator_choice,demonstrator_reward"
                ),
                f"s1,0,social,social_pre_choice,0,0|1,{missing_value},{missing_value},1,0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset_from_csv(csv_path, schema=SOCIAL_PRE_CHOICE_SCHEMA)

    trial = dataset.subjects[0].blocks[0].trials[0]
    assert trial.events[6].payload["action"] is None
    assert trial.events[7].payload["reward"] is None
    assert dict(trial.events[8].payload) == {"choice": None, "reward": None}

    subject_steps = [
        (event_type, view.actor_id, view.action, view.reward)
        for event_type, learner_id, view in replay_trial_steps(trial, SOCIAL_PRE_CHOICE_SCHEMA)
        if learner_id == "subject"
    ]
    assert subject_steps == [
        (EventPhase.UPDATE, "demonstrator", 1, 0.5),
        (EventPhase.UPDATE, "subject", None, None),
    ]


def test_load_dataset_from_csv_rejects_duplicate_trial_keys(tmp_path: Path) -> None:
    """Ensure duplicate subject-block-trial keys are rejected.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test raises on duplicate keys.
    """

    csv_path = tmp_path / "duplicates.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,block_index,condition,schema_id,trial_index,available_actions,choice,reward",
                "s1,0,A,asocial_bandit,0,0|1,1,1.0",
                "s1,0,A,asocial_bandit,0,0|1,0,0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate trial key"):
        load_dataset_from_csv(csv_path, schema=ASOCIAL_BANDIT_SCHEMA)


def test_load_dataset_from_csv_rejects_inconsistent_block_conditions(
    tmp_path: Path,
) -> None:
    """Ensure one block cannot carry multiple condition labels.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test raises on inconsistent block conditions.
    """

    csv_path = tmp_path / "conditions.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,block_index,condition,schema_id,trial_index,available_actions,choice,reward",
                "s1,0,A,asocial_bandit,0,0|1,1,1.0",
                "s1,0,B,asocial_bandit,1,0|1,0,0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Inconsistent condition"):
        load_dataset_from_csv(csv_path, schema=ASOCIAL_BANDIT_SCHEMA)


def test_load_dataset_from_csv_rejects_missing_required_columns(
    tmp_path: Path,
) -> None:
    """Ensure missing required columns (not optional ones) are rejected.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test raises on missing header columns.
    """

    csv_path = tmp_path / "missing_columns.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,block_index,condition,schema_id,trial_index,available_actions",
                "s1,0,A,asocial_bandit,0,0|1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        load_dataset_from_csv(csv_path, schema=ASOCIAL_BANDIT_SCHEMA)


def test_load_dataset_from_csv_rejects_invalid_available_actions(
    tmp_path: Path,
) -> None:
    """Ensure malformed available-action encoding is rejected.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test raises on invalid action encoding.
    """

    csv_path = tmp_path / "invalid_actions.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,block_index,condition,schema_id,trial_index,available_actions,choice,reward",
                "s1,0,A,asocial_bandit,0,0||1,1,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="available_actions"):
        load_dataset_from_csv(csv_path, schema=ASOCIAL_BANDIT_SCHEMA)


def test_load_dataset_from_csv_rejects_wrong_schema_converter(
    tmp_path: Path,
) -> None:
    """Ensure a file cannot be loaded with a mismatched schema converter.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test raises on converter-header mismatch.
    """

    dataset = _make_social_pre_choice_dataset()
    csv_path = tmp_path / "social.csv"

    save_dataset_to_csv(dataset, schema=SOCIAL_PRE_CHOICE_SCHEMA, path=csv_path)

    with pytest.raises(ValueError, match="Unknown columns"):
        load_dataset_from_csv(csv_path, schema=ASOCIAL_BANDIT_SCHEMA)


def test_register_trial_csv_converter_raises_on_duplicate_schema_id() -> None:
    """Ensure registering a converter for an already-registered schema_id raises.

    Returns
    -------
    None
        This test raises on duplicate registration.
    """

    existing_converter = get_trial_csv_converter(ASOCIAL_BANDIT_SCHEMA)

    with pytest.raises(ValueError, match="already registered"):
        register_trial_csv_converter(existing_converter)


def test_load_dataset_from_csv_rejects_duplicate_available_actions(
    tmp_path: Path,
) -> None:
    """Ensure duplicate values in available_actions are rejected.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test raises on duplicate action values.
    """

    csv_path = tmp_path / "dup_actions.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,block_index,condition,schema_id,trial_index,available_actions,choice,reward",
                "s1,0,A,asocial_bandit,0,1|0|1,1,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_dataset_from_csv(csv_path, schema=ASOCIAL_BANDIT_SCHEMA)


def test_load_dataset_from_csv_subjects_ordered_by_subject_id(
    tmp_path: Path,
) -> None:
    """Ensure subjects in the loaded dataset are sorted by subject_id.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test checks subject ordering in the reconstructed dataset.
    """

    csv_path = tmp_path / "multi_subject.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,block_index,condition,schema_id,trial_index,available_actions,choice,reward",
                "s3,0,A,asocial_bandit,0,0|1,1,1.0",
                "s1,0,A,asocial_bandit,0,0|1,0,0.0",
                "s2,0,A,asocial_bandit,0,0|1,1,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset_from_csv(csv_path, schema=ASOCIAL_BANDIT_SCHEMA)

    assert [s.subject_id for s in dataset.subjects] == ["s1", "s2", "s3"]


def _make_asocial_dataset() -> Dataset:
    """Create a small asocial dataset for CSV round-trip tests.

    Returns
    -------
    Dataset
        Dataset with one subject, one block, and two asocial trials.
    """

    return Dataset(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(
                    Block(
                        block_index=0,
                        condition="A",
                        schema_id="asocial_bandit",
                        trials=(
                            Trial(
                                trial_index=0,
                                events=(
                                    Event(
                                        phase=EventPhase.INPUT,
                                        event_index=0,
                                        node_id="main",
                                        payload={"available_actions": (0, 1)},
                                    ),
                                    Event(
                                        phase=EventPhase.DECISION,
                                        event_index=1,
                                        node_id="main",
                                        payload={"action": 1},
                                    ),
                                    Event(
                                        phase=EventPhase.OUTCOME,
                                        event_index=2,
                                        node_id="main",
                                        payload={"reward": 1.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=3,
                                        node_id="main",
                                        payload={"choice": 1, "reward": 1.0},
                                    ),
                                ),
                            ),
                            Trial(
                                trial_index=1,
                                events=(
                                    Event(
                                        phase=EventPhase.INPUT,
                                        event_index=0,
                                        node_id="main",
                                        payload={"available_actions": (0, 1)},
                                    ),
                                    Event(
                                        phase=EventPhase.DECISION,
                                        event_index=1,
                                        node_id="main",
                                        payload={"action": 0},
                                    ),
                                    Event(
                                        phase=EventPhase.OUTCOME,
                                        event_index=2,
                                        node_id="main",
                                        payload={"reward": 0.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=3,
                                        node_id="main",
                                        payload={"choice": 0, "reward": 0.0},
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def _dataset_with_schema_id(dataset: Dataset, schema_id: str) -> Dataset:
    """Copy a dataset while replacing every block schema identifier.

    Parameters
    ----------
    dataset
        Source dataset whose trials should be preserved.
    schema_id
        Schema identifier to stamp onto every copied block.

    Returns
    -------
    Dataset
        Dataset with identical subjects, blocks, and trials except for the
        replaced block ``schema_id`` values.
    """

    return Dataset(
        subjects=tuple(
            SubjectData(
                subject_id=subject.subject_id,
                blocks=tuple(
                    Block(
                        block_index=block.block_index,
                        condition=block.condition,
                        schema_id=schema_id,
                        trials=block.trials,
                    )
                    for block in subject.blocks
                ),
            )
            for subject in dataset.subjects
        )
    )


def _make_social_pre_choice_dataset() -> Dataset:
    """Create a small pre-choice social dataset for CSV round-trip tests.

    Returns
    -------
    Dataset
        Dataset with one subject and one pre-choice social block.
    """

    # SOCIAL_PRE_CHOICE_SCHEMA: INPUT(demo) DECISION(demo) OUTCOME(demo) UPDATE(demo→demo)
    #   UPDATE(demo→subj) INPUT(subj) DECISION(subj) OUTCOME(subj) UPDATE(subj→subj)
    return Dataset(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(
                    Block(
                        block_index=0,
                        condition="social_pre",
                        schema_id="social_pre_choice",
                        trials=(
                            Trial(
                                trial_index=0,
                                events=(
                                    Event(
                                        phase=EventPhase.INPUT,
                                        event_index=0,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"available_actions": (0, 1)},
                                    ),
                                    Event(
                                        phase=EventPhase.DECISION,
                                        event_index=1,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"action": 0},
                                    ),
                                    Event(
                                        phase=EventPhase.OUTCOME,
                                        event_index=2,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"reward": 1.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=3,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"choice": 0, "reward": 1.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=4,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"choice": 0, "reward": 1.0},
                                    ),
                                    Event(
                                        phase=EventPhase.INPUT,
                                        event_index=5,
                                        node_id="main",
                                        payload={"available_actions": (0, 1)},
                                    ),
                                    Event(
                                        phase=EventPhase.DECISION,
                                        event_index=6,
                                        node_id="main",
                                        payload={"action": 1},
                                    ),
                                    Event(
                                        phase=EventPhase.OUTCOME,
                                        event_index=7,
                                        node_id="main",
                                        payload={"reward": 0.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=8,
                                        node_id="main",
                                        payload={"choice": 1, "reward": 0.0},
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def _make_social_post_outcome_dataset() -> Dataset:
    """Create a small post-outcome social dataset for CSV round-trip tests.

    Returns
    -------
    Dataset
        Dataset with one subject and one post-outcome social block.
    """

    # SOCIAL_POST_OUTCOME_SCHEMA: INPUT(subj) DECISION(subj) OUTCOME(subj) UPDATE(subj→subj)
    #   INPUT(demo) DECISION(demo) OUTCOME(demo) UPDATE(demo→demo) UPDATE(demo→subj)
    return Dataset(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(
                    Block(
                        block_index=0,
                        condition="social_post",
                        schema_id="social_post_outcome",
                        trials=(
                            Trial(
                                trial_index=0,
                                events=(
                                    Event(
                                        phase=EventPhase.INPUT,
                                        event_index=0,
                                        node_id="main",
                                        payload={"available_actions": (0, 1)},
                                    ),
                                    Event(
                                        phase=EventPhase.DECISION,
                                        event_index=1,
                                        node_id="main",
                                        payload={"action": 0},
                                    ),
                                    Event(
                                        phase=EventPhase.OUTCOME,
                                        event_index=2,
                                        node_id="main",
                                        payload={"reward": 1.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=3,
                                        node_id="main",
                                        payload={"choice": 0, "reward": 1.0},
                                    ),
                                    Event(
                                        phase=EventPhase.INPUT,
                                        event_index=4,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"available_actions": (0, 1)},
                                    ),
                                    Event(
                                        phase=EventPhase.DECISION,
                                        event_index=5,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"action": 1},
                                    ),
                                    Event(
                                        phase=EventPhase.OUTCOME,
                                        event_index=6,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"reward": 0.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=7,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"choice": 1, "reward": 0.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=8,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"choice": 1, "reward": 0.0},
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def _make_social_pre_choice_no_self_outcome_dataset() -> Dataset:
    """Create a small pre-choice no-self-outcome dataset for CSV tests.

    Returns
    -------
    Dataset
        Dataset with one subject and one demonstrator-first no-self-outcome
        social block.
    """

    return Dataset(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(
                    Block(
                        block_index=0,
                        condition="social_pre_no_self",
                        schema_id="social_pre_choice_no_self_outcome",
                        trials=(
                            Trial(
                                trial_index=0,
                                events=(
                                    Event(
                                        phase=EventPhase.INPUT,
                                        event_index=0,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"available_actions": (0, 1)},
                                    ),
                                    Event(
                                        phase=EventPhase.DECISION,
                                        event_index=1,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"action": 0},
                                    ),
                                    Event(
                                        phase=EventPhase.OUTCOME,
                                        event_index=2,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"reward": 1.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=3,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"choice": 0, "reward": 1.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=4,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"choice": 0, "reward": 1.0},
                                    ),
                                    Event(
                                        phase=EventPhase.INPUT,
                                        event_index=5,
                                        node_id="main",
                                        payload={"available_actions": (0, 1)},
                                    ),
                                    Event(
                                        phase=EventPhase.DECISION,
                                        event_index=6,
                                        node_id="main",
                                        payload={"action": 1},
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def _make_social_post_outcome_no_self_outcome_dataset() -> Dataset:
    """Create a small post-outcome no-self-outcome dataset for CSV tests.

    Returns
    -------
    Dataset
        Dataset with one subject and one subject-first no-self-outcome social
        block.
    """

    return Dataset(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(
                    Block(
                        block_index=0,
                        condition="social_post_no_self",
                        schema_id="social_post_outcome_no_self_outcome",
                        trials=(
                            Trial(
                                trial_index=0,
                                events=(
                                    Event(
                                        phase=EventPhase.INPUT,
                                        event_index=0,
                                        node_id="main",
                                        payload={"available_actions": (0, 1)},
                                    ),
                                    Event(
                                        phase=EventPhase.DECISION,
                                        event_index=1,
                                        node_id="main",
                                        payload={"action": 0},
                                    ),
                                    Event(
                                        phase=EventPhase.INPUT,
                                        event_index=2,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"available_actions": (0, 1)},
                                    ),
                                    Event(
                                        phase=EventPhase.DECISION,
                                        event_index=3,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"action": 1},
                                    ),
                                    Event(
                                        phase=EventPhase.OUTCOME,
                                        event_index=4,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"reward": 0.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=5,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"choice": 1, "reward": 0.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=6,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"choice": 1, "reward": 0.0},
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def _make_social_pre_choice_demo_learns_dataset() -> Dataset:
    """Create a small pre-choice demo-learns dataset for CSV tests.

    Returns
    -------
    Dataset
        Dataset with one subject and one bidirectional demonstrator-first
        social block.
    """

    return Dataset(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(
                    Block(
                        block_index=0,
                        condition="social_pre_demo_learns",
                        schema_id="social_pre_choice_demo_learns",
                        trials=(
                            Trial(
                                trial_index=0,
                                events=(
                                    Event(
                                        phase=EventPhase.INPUT,
                                        event_index=0,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"available_actions": (0, 1)},
                                    ),
                                    Event(
                                        phase=EventPhase.DECISION,
                                        event_index=1,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"action": 0},
                                    ),
                                    Event(
                                        phase=EventPhase.OUTCOME,
                                        event_index=2,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"reward": 1.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=3,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"choice": 0, "reward": 1.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=4,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"choice": 0, "reward": 1.0},
                                    ),
                                    Event(
                                        phase=EventPhase.INPUT,
                                        event_index=5,
                                        node_id="main",
                                        payload={"available_actions": (0, 1)},
                                    ),
                                    Event(
                                        phase=EventPhase.DECISION,
                                        event_index=6,
                                        node_id="main",
                                        payload={"action": 1},
                                    ),
                                    Event(
                                        phase=EventPhase.OUTCOME,
                                        event_index=7,
                                        node_id="main",
                                        payload={"reward": 0.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=8,
                                        node_id="main",
                                        payload={"choice": 1, "reward": 0.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=9,
                                        node_id="main",
                                        actor_id="subject",
                                        payload={"choice": 1, "reward": 0.0},
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def _make_social_post_outcome_demo_learns_dataset() -> Dataset:
    """Create a small post-outcome demo-learns dataset for CSV tests.

    Returns
    -------
    Dataset
        Dataset with one subject and one bidirectional subject-first social
        block.
    """

    return Dataset(
        subjects=(
            SubjectData(
                subject_id="s1",
                blocks=(
                    Block(
                        block_index=0,
                        condition="social_post_demo_learns",
                        schema_id="social_post_outcome_demo_learns",
                        trials=(
                            Trial(
                                trial_index=0,
                                events=(
                                    Event(
                                        phase=EventPhase.INPUT,
                                        event_index=0,
                                        node_id="main",
                                        payload={"available_actions": (0, 1)},
                                    ),
                                    Event(
                                        phase=EventPhase.DECISION,
                                        event_index=1,
                                        node_id="main",
                                        payload={"action": 0},
                                    ),
                                    Event(
                                        phase=EventPhase.OUTCOME,
                                        event_index=2,
                                        node_id="main",
                                        payload={"reward": 1.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=3,
                                        node_id="main",
                                        payload={"choice": 0, "reward": 1.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=4,
                                        node_id="main",
                                        actor_id="subject",
                                        payload={"choice": 0, "reward": 1.0},
                                    ),
                                    Event(
                                        phase=EventPhase.INPUT,
                                        event_index=5,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"available_actions": (0, 1)},
                                    ),
                                    Event(
                                        phase=EventPhase.DECISION,
                                        event_index=6,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"action": 1},
                                    ),
                                    Event(
                                        phase=EventPhase.OUTCOME,
                                        event_index=7,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"reward": 0.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=8,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"choice": 1, "reward": 0.0},
                                    ),
                                    Event(
                                        phase=EventPhase.UPDATE,
                                        event_index=9,
                                        node_id="main",
                                        actor_id="demonstrator",
                                        payload={"choice": 1, "reward": 0.0},
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def test_load_csv_without_available_actions_infers_from_choices_asocial(
    tmp_path: Path,
) -> None:
    """Ensure omitting ``available_actions`` infers from ``choice`` column.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test verifies inferred available actions match observed choices.
    """

    csv_path = tmp_path / "no_actions.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,block_index,condition,schema_id,trial_index,choice,reward",
                "s1,0,A,asocial_bandit,0,1,1.0",
                "s1,0,A,asocial_bandit,1,0,0.0",
                "s1,0,A,asocial_bandit,2,2,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset_from_csv(csv_path, schema=ASOCIAL_BANDIT_SCHEMA)

    trial = dataset.subjects[0].blocks[0].trials[0]
    input_event = trial.events[0]
    assert input_event.payload["available_actions"] == (0, 1, 2)


def test_load_csv_without_available_actions_infers_from_social_columns(
    tmp_path: Path,
) -> None:
    """Ensure omitting ``available_actions`` includes ``demonstrator_choice``.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test verifies the inferred set includes demonstrator actions.
    """

    csv_path = tmp_path / "no_actions_social.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,block_index,condition,schema_id,trial_index,choice,reward,demonstrator_choice,demonstrator_reward",
                "s1,0,social,social_pre_choice,0,0,1.0,1,0.0",
                "s1,0,social,social_pre_choice,1,0,0.0,2,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset_from_csv(csv_path, schema=SOCIAL_PRE_CHOICE_SCHEMA)

    trial = dataset.subjects[0].blocks[0].trials[0]
    input_event = trial.events[0]
    assert input_event.payload["available_actions"] == (0, 1, 2)


def test_load_csv_with_available_actions_still_works(
    tmp_path: Path,
) -> None:
    """Ensure explicit ``available_actions`` column is still honoured.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test verifies backward compatibility when the column is present.
    """

    csv_path = tmp_path / "with_actions.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,block_index,condition,schema_id,trial_index,available_actions,choice,reward",
                "s1,0,A,asocial_bandit,0,0|1|2,1,1.0",
                "s1,0,A,asocial_bandit,1,0|1|2,0,0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset_from_csv(csv_path, schema=ASOCIAL_BANDIT_SCHEMA)

    trial = dataset.subjects[0].blocks[0].trials[0]
    input_event = trial.events[0]
    assert input_event.payload["available_actions"] == (0, 1, 2)


def test_load_csv_without_schema_id_uses_schema_argument(
    tmp_path: Path,
) -> None:
    """Ensure omitting ``schema_id`` column fills it from the schema argument.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test verifies schema_id is injected from the function argument.
    """

    csv_path = tmp_path / "no_schema_id.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,block_index,condition,trial_index,available_actions,choice,reward",
                "s1,0,A,0,0|1,1,1.0",
                "s1,0,A,1,0|1,0,0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset_from_csv(csv_path, schema=ASOCIAL_BANDIT_SCHEMA)

    assert dataset.subjects[0].blocks[0].schema_id == "asocial_bandit"


def test_load_csv_without_schema_id_and_available_actions(
    tmp_path: Path,
) -> None:
    """Ensure both ``schema_id`` and ``available_actions`` can be omitted.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test verifies both columns are inferred simultaneously.
    """

    csv_path = tmp_path / "minimal.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,block_index,condition,trial_index,choice,reward,demonstrator_choice,demonstrator_reward",
                "s1,0,social,0,0,1.0,1,0.0",
                "s1,0,social,1,1,0.0,2,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset_from_csv(csv_path, schema=SOCIAL_PRE_CHOICE_SCHEMA)

    block = dataset.subjects[0].blocks[0]
    assert block.schema_id == "social_pre_choice"
    trial = block.trials[0]
    input_event = trial.events[0]
    assert input_event.payload["available_actions"] == (0, 1, 2)


def _fitting_view_signatures(
    dataset: Dataset,
    schema: TrialSchema,
) -> tuple[tuple[object, ...], ...]:
    """Extract fitting-relevant view signatures from a dataset.

    Parameters
    ----------
    dataset
        Dataset to summarize.
    schema
        Schema used to extract flat decision views.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Stable view signatures ignoring non-exported observation payloads.
    """

    signatures: list[tuple[object, ...]] = []
    for subject in dataset.subjects:
        for block in subject.blocks:
            for trial in block.trials:
                last_subject_update = None
                for event_type, learner_id, view in replay_trial_steps(trial, schema):
                    if event_type == EventPhase.UPDATE and learner_id == "subject":
                        last_subject_update = view
                if last_subject_update is not None:
                    signatures.append(
                        (
                            subject.subject_id,
                            block.block_index,
                            block.condition,
                            last_subject_update.trial_index,
                            last_subject_update.available_actions,
                            last_subject_update.actor_id,
                            last_subject_update.learner_id,
                            last_subject_update.action,
                            last_subject_update.reward,
                        )
                    )
    return tuple(signatures)
