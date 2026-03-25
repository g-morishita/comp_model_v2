"""Tests for schema-specific trial CSV import and export."""

from __future__ import annotations

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
    SOCIAL_POST_OUTCOME_SCHEMA,
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
        (SOCIAL_POST_OUTCOME_SCHEMA, _make_social_post_outcome_dataset()),
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
                "subject_id,block_index,condition,trial_index,available_actions,choice,reward",
                "s1,0,A,0,0|1,1,1.0",
                "s1,0,A,0,0|1,0,0.0",
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
                "subject_id,block_index,condition,trial_index,available_actions,choice,reward",
                "s1,0,A,0,0|1,1,1.0",
                "s1,0,B,1,0|1,0,0.0",
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
    """Ensure missing converter columns are rejected at header validation.

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
                "subject_id,block_index,condition,trial_index,choice,reward",
                "s1,0,A,0,1,1.0",
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
                "subject_id,block_index,condition,trial_index,available_actions,choice,reward",
                "s1,0,A,0,0||1,1,1.0",
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
                "subject_id,block_index,condition,trial_index,available_actions,choice,reward",
                "s1,0,A,0,1|0|1,1,1.0",
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
                "subject_id,block_index,condition,trial_index,available_actions,choice,reward",
                "s3,0,A,0,0|1,1,1.0",
                "s1,0,A,0,0|1,0,0.0",
                "s2,0,A,0,0|1,1,1.0",
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
