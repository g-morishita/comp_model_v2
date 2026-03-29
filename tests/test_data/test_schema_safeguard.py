"""Tests for the schema provenance safeguard.

These tests verify that validation functions and inference entrypoints reject
data whose ``schema_id`` does not match the expected trial schema.
"""

from pathlib import Path

import pytest

from comp_model.data.schema import Block, Dataset, Event, EventPhase, SubjectData, Trial
from comp_model.data.validation import validate_block, validate_dataset, validate_subject
from comp_model.tasks.schemas import (
    ASOCIAL_BANDIT_SCHEMA,
    SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
    SOCIAL_PRE_CHOICE_SCHEMA,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _asocial_trial(trial_index: int = 0) -> Trial:
    """Build a minimal valid asocial trial.

    Parameters
    ----------
    trial_index
        Index for the trial.

    Returns
    -------
    Trial
        A structurally valid asocial trial.
    """
    return Trial(
        trial_index=trial_index,
        events=(
            Event(
                phase=EventPhase.INPUT,
                event_index=0,
                node_id="main",
                actor_id="subject",
                payload={"available_actions": (0, 1)},
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=1,
                node_id="main",
                actor_id="subject",
                payload={"action": 0},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=2,
                node_id="main",
                actor_id="subject",
                payload={"reward": 1.0},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=3,
                node_id="main",
                actor_id="subject",
                payload={"choice": 0, "reward": 1.0},
            ),
        ),
    )


def _block(schema_id: str, block_index: int = 0) -> Block:
    """Build a single-trial block with the given schema_id.

    Parameters
    ----------
    schema_id
        Schema identifier stamped on the block.
    block_index
        Index for the block.

    Returns
    -------
    Block
        A block containing one valid asocial trial.
    """
    return Block(
        block_index=block_index,
        condition="default",
        schema_id=schema_id,
        trials=(_asocial_trial(),),
    )


def _subject(schema_id: str, subject_id: str = "sub_00") -> SubjectData:
    """Build a single-block subject with the given schema_id.

    Parameters
    ----------
    schema_id
        Schema identifier stamped on the block.
    subject_id
        Subject identifier.

    Returns
    -------
    SubjectData
        Subject data with one block.
    """
    return SubjectData(subject_id=subject_id, blocks=(_block(schema_id),))


# ---------------------------------------------------------------------------
# validate_block: schema_id mismatch
# ---------------------------------------------------------------------------


class TestValidateBlockSchemaId:
    """Tests for schema_id checking in validate_block."""

    def test_matching_schema_id_passes(self) -> None:
        """Block with matching schema_id passes validation."""
        block = _block(ASOCIAL_BANDIT_SCHEMA.schema_id)
        validate_block(block, ASOCIAL_BANDIT_SCHEMA)

    def test_mismatched_schema_id_raises(self) -> None:
        """Block with wrong schema_id is rejected."""
        block = _block("asocial_bandit")
        with pytest.raises(ValueError, match="schema_id mismatch"):
            validate_block(block, SOCIAL_PRE_CHOICE_SCHEMA)

    def test_no_schema_skips_check(self) -> None:
        """Without a schema, schema_id is not checked."""
        block = _block("anything_goes")
        validate_block(block)


# ---------------------------------------------------------------------------
# validate_subject: schema_id mismatch
# ---------------------------------------------------------------------------


class TestValidateSubjectSchemaId:
    """Tests for schema_id checking in validate_subject."""

    def test_matching_schema_id_passes(self) -> None:
        """Subject with matching schema_id passes validation."""
        subject = _subject(ASOCIAL_BANDIT_SCHEMA.schema_id)
        validate_subject(subject, ASOCIAL_BANDIT_SCHEMA)

    def test_mismatched_schema_id_raises(self) -> None:
        """Subject with wrong schema_id is rejected at block level."""
        subject = _subject("asocial_bandit")
        with pytest.raises(ValueError, match="schema_id mismatch"):
            validate_subject(subject, SOCIAL_PRE_CHOICE_SCHEMA)


# ---------------------------------------------------------------------------
# validate_dataset: schema_id mismatch
# ---------------------------------------------------------------------------


class TestValidateDatasetSchemaId:
    """Tests for schema_id checking in validate_dataset."""

    def test_matching_schema_id_passes(self) -> None:
        """Dataset with matching schema_id passes validation."""
        dataset = Dataset(
            subjects=(
                _subject(ASOCIAL_BANDIT_SCHEMA.schema_id, "sub_00"),
                _subject(ASOCIAL_BANDIT_SCHEMA.schema_id, "sub_01"),
            )
        )
        validate_dataset(dataset, ASOCIAL_BANDIT_SCHEMA)

    def test_mismatched_schema_id_raises(self) -> None:
        """Dataset with wrong schema_id is rejected."""
        dataset = Dataset(subjects=(_subject("asocial_bandit"),))
        with pytest.raises(ValueError, match="schema_id mismatch"):
            validate_dataset(dataset, SOCIAL_PRE_CHOICE_SCHEMA)


# ---------------------------------------------------------------------------
# Regression: similar but distinct schemas must be distinguished
# ---------------------------------------------------------------------------


class TestSimilarSchemaRegression:
    """social_pre_choice vs social_pre_choice_action_only must not pass each other."""

    def test_social_pre_choice_rejected_by_action_only_schema(self) -> None:
        """Data stamped with social_pre_choice fails against action_only schema."""
        block = _block(SOCIAL_PRE_CHOICE_SCHEMA.schema_id)
        with pytest.raises(ValueError, match="schema_id mismatch"):
            validate_block(block, SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA)

    def test_action_only_rejected_by_social_pre_choice_schema(self) -> None:
        """Data stamped with action_only fails against social_pre_choice schema."""
        block = _block(SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA.schema_id)
        with pytest.raises(ValueError, match="schema_id mismatch"):
            validate_block(block, SOCIAL_PRE_CHOICE_SCHEMA)


# ---------------------------------------------------------------------------
# MLE fitting entrypoints: schema_id mismatch
# ---------------------------------------------------------------------------


class TestMleFittingSchemaGuard:
    """MLE fit functions reject data with mismatched schema_id."""

    def test_fit_mle_simple_rejects_mismatch(self) -> None:
        """fit_mle_simple raises before optimisation on schema-id mismatch."""
        from comp_model.inference.mle.optimize import fit_mle_simple
        from comp_model.models.kernels import AsocialQLearningKernel

        subject = _subject("asocial_bandit")
        kernel = AsocialQLearningKernel()
        with pytest.raises(ValueError, match="schema_id mismatch"):
            fit_mle_simple(kernel, subject, SOCIAL_PRE_CHOICE_SCHEMA)

    def test_fit_mle_conditioned_rejects_mismatch(self) -> None:
        """fit_mle_conditioned raises before optimisation on schema-id mismatch.

        The validate_subject call is the first thing in fit_mle_conditioned,
        so we only need to verify the schema guard fires — no real layout needed.
        """
        from comp_model.inference.mle.optimize import fit_mle_conditioned
        from comp_model.models.kernels import AsocialQLearningKernel

        subject = _subject("asocial_bandit")
        kernel = AsocialQLearningKernel()
        # layout=None is invalid for a real call but the schema guard fires first
        with pytest.raises(ValueError, match="schema_id mismatch"):
            fit_mle_conditioned(
                kernel,
                None,  # type: ignore[arg-type]
                subject,
                SOCIAL_PRE_CHOICE_SCHEMA,
            )


# ---------------------------------------------------------------------------
# Stan data builder: schema_id mismatch
# ---------------------------------------------------------------------------


class TestStanDataBuilderSchemaGuard:
    """Stan data builders reject data with mismatched schema_id."""

    def test_subject_to_stan_data_rejects_mismatch(self) -> None:
        """subject_to_stan_data raises on schema-id mismatch."""
        from comp_model.inference.bayes.stan.data_builder import subject_to_stan_data

        subject = _subject("asocial_bandit")
        with pytest.raises(ValueError, match="schema_id mismatch"):
            subject_to_stan_data(subject, SOCIAL_PRE_CHOICE_SCHEMA)

    def test_dataset_to_stan_data_rejects_mismatch(self) -> None:
        """dataset_to_stan_data raises on schema-id mismatch."""
        from comp_model.inference.bayes.stan.data_builder import dataset_to_stan_data

        dataset = Dataset(subjects=(_subject("asocial_bandit"),))
        with pytest.raises(ValueError, match="schema_id mismatch"):
            dataset_to_stan_data(dataset, SOCIAL_PRE_CHOICE_SCHEMA)


# ---------------------------------------------------------------------------
# Recovery runners: preflight schema check
# ---------------------------------------------------------------------------


class TestRecoveryRunnerPreflightCheck:
    """Recovery runners reject mismatched task/schema before simulation."""

    def test_parameter_recovery_rejects_mismatch(self) -> None:
        """run_parameter_recovery raises before any simulation on mismatch."""
        from comp_model.recovery.parameter.runner import _check_schema_consistency
        from comp_model.tasks.spec import BlockSpec, TaskSpec

        task = TaskSpec(
            task_id="test",
            blocks=(
                BlockSpec(
                    condition="default",
                    n_trials=10,
                    schema=ASOCIAL_BANDIT_SCHEMA,
                ),
            ),
        )
        with pytest.raises(ValueError, match="schema"):
            _check_schema_consistency(task, SOCIAL_PRE_CHOICE_SCHEMA)

    def test_model_recovery_rejects_mismatch(self) -> None:
        """run_model_recovery raises before any simulation on mismatch."""
        from comp_model.recovery.model.runner import _check_schema_consistency
        from comp_model.tasks.spec import BlockSpec, TaskSpec

        task = TaskSpec(
            task_id="test",
            blocks=(
                BlockSpec(
                    condition="default",
                    n_trials=10,
                    schema=ASOCIAL_BANDIT_SCHEMA,
                ),
            ),
        )
        with pytest.raises(ValueError, match="schema"):
            _check_schema_consistency(task, SOCIAL_PRE_CHOICE_SCHEMA)

    def test_parameter_recovery_accepts_matching(self) -> None:
        """Matching schemas pass the preflight check without error."""
        from comp_model.recovery.parameter.runner import _check_schema_consistency
        from comp_model.tasks.spec import BlockSpec, TaskSpec

        task = TaskSpec(
            task_id="test",
            blocks=(
                BlockSpec(
                    condition="default",
                    n_trials=10,
                    schema=ASOCIAL_BANDIT_SCHEMA,
                ),
            ),
        )
        _check_schema_consistency(task, ASOCIAL_BANDIT_SCHEMA)

    def test_model_recovery_accepts_matching(self) -> None:
        """Matching schemas pass the preflight check without error."""
        from comp_model.recovery.model.runner import _check_schema_consistency
        from comp_model.tasks.spec import BlockSpec, TaskSpec

        task = TaskSpec(
            task_id="test",
            blocks=(
                BlockSpec(
                    condition="default",
                    n_trials=10,
                    schema=ASOCIAL_BANDIT_SCHEMA,
                ),
            ),
        )
        _check_schema_consistency(task, ASOCIAL_BANDIT_SCHEMA)


# ---------------------------------------------------------------------------
# CSV round-trip: schema_id persisted and validated
# ---------------------------------------------------------------------------


class TestCsvSchemaIdRoundTrip:
    """CSV export persists schema_id; load rejects mismatched schemas."""

    def test_load_rejects_csv_with_wrong_schema(self, tmp_path: Path) -> None:
        """A CSV exported under one schema cannot be loaded with a different one."""
        from comp_model.io.trial_csv import load_dataset_from_csv

        # Hand-craft a CSV stamped with social_pre_choice schema_id.
        csv_path = tmp_path / "social.csv"
        header = (
            "subject_id,block_index,condition,schema_id,"
            "trial_index,available_actions,choice,reward,"
            "demonstrator_choice,demonstrator_reward"
        )
        row = "sub_00,0,default,social_pre_choice,0,0|1,1,1.0,0,0.5"
        csv_path.write_text(f"{header}\n{row}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="schema_id mismatch"):
            load_dataset_from_csv(csv_path, schema=SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA)

    def test_round_trip_with_matching_schema_passes(self, tmp_path: Path) -> None:
        """A CSV exported and loaded with the same schema succeeds."""
        from comp_model.io.trial_csv import load_dataset_from_csv, save_dataset_to_csv

        dataset = Dataset(subjects=(_subject(ASOCIAL_BANDIT_SCHEMA.schema_id, "sub_00"),))
        csv_path = tmp_path / "asocial.csv"
        save_dataset_to_csv(dataset, schema=ASOCIAL_BANDIT_SCHEMA, path=csv_path)

        loaded = load_dataset_from_csv(csv_path, schema=ASOCIAL_BANDIT_SCHEMA)
        assert len(loaded.subjects) == 1
        assert loaded.subjects[0].blocks[0].schema_id == ASOCIAL_BANDIT_SCHEMA.schema_id

    def test_export_rejects_block_with_wrong_schema_id(self, tmp_path: Path) -> None:
        """save_dataset_to_csv rejects blocks stamped with a different schema_id."""
        from comp_model.io.trial_csv import save_dataset_to_csv

        # Block stamped with wrong schema_id but containing valid asocial trials.
        dataset = Dataset(subjects=(_subject("wrong_schema", "sub_00"),))
        csv_path = tmp_path / "bad_export.csv"

        with pytest.raises(ValueError, match="schema_id mismatch"):
            save_dataset_to_csv(dataset, schema=ASOCIAL_BANDIT_SCHEMA, path=csv_path)
