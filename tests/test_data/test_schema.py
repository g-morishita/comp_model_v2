"""Tests for canonical hierarchical schema types."""

from comp_model.data.schema import Block, Dataset, Event, EventPhase, SubjectData, Trial


def test_subject_trial_helpers_flatten_in_order() -> None:
    """Check flattened trial ordering on a subject.

    Returns
    -------
    None
        This test asserts ordering only.
    """

    trial_0 = Trial(
        trial_index=0,
        events=(
            Event(
                phase=EventPhase.INPUT,
                event_index=0,
                payload={"available_actions": (0, 1)},
            ),
        ),
    )
    trial_1 = Trial(
        trial_index=1,
        events=(
            Event(
                phase=EventPhase.INPUT,
                event_index=0,
                payload={"available_actions": (0, 1)},
            ),
        ),
    )
    subject = SubjectData(
        subject_id="s1",
        blocks=(
            Block(block_index=0, condition="a", schema_id="test", trials=(trial_0,)),
            Block(block_index=1, condition="b", schema_id="test", trials=(trial_1,)),
        ),
    )

    assert subject.trials == (trial_0, trial_1)
    assert tuple(subject.iter_block_trials()) == (
        (subject.blocks[0], trial_0),
        (subject.blocks[1], trial_1),
    )


def test_dataset_flattening_helpers_collect_blocks_and_trials() -> None:
    """Check dataset flattening helpers.

    Returns
    -------
    None
        This test asserts the flattened tuples.
    """

    trial = Trial(
        trial_index=0,
        events=(
            Event(
                phase=EventPhase.INPUT,
                event_index=0,
                payload={"available_actions": (0, 1)},
            ),
        ),
    )
    block = Block(block_index=0, condition="a", schema_id="test", trials=(trial,))
    dataset = Dataset(subjects=(SubjectData(subject_id="s1", blocks=(block,)),))

    assert dataset.blocks == (block,)
    assert dataset.trials == (trial,)
