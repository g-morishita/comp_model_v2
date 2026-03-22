"""Tests for the social observed-outcome Q-learning Stan adapter."""

from pathlib import Path

from comp_model.data.schema import Block, Event, EventPhase, SubjectData, Trial
from comp_model.inference.bayes.stan.adapters.social_rl_self_reward_demo_reward import (
    SocialRlSelfRewardDemoRewardStanAdapter,
)
from comp_model.inference.config import HierarchyStructure
from comp_model.models.condition.shared_delta import SharedDeltaLayout
from comp_model.models.kernels.social_rl_self_reward_demo_reward import (
    SocialRlSelfRewardDemoRewardKernel,
)
from comp_model.tasks.schemas import SOCIAL_PRE_CHOICE_SCHEMA


def _social_trial(
    trial_index: int, action: int, reward: float, social_action: int, social_reward: float
) -> Trial:
    """Create a minimal social trial for adapter tests.

    Parameters
    ----------
    trial_index
        Trial index within the current block.
    action
        Chosen action value.
    reward
        Observed reward.
    social_action
        Demonstrator's chosen action.
    social_reward
        Demonstrator's observed reward.

    Returns
    -------
    Trial
        Event-based trial matching ``SOCIAL_PRE_CHOICE_SCHEMA``.
    """

    return Trial(
        trial_index=trial_index,
        events=(
            Event(
                phase=EventPhase.INPUT,
                event_index=0,
                node_id="main",
                payload={"available_actions": (0, 1)},
            ),
            Event(
                phase=EventPhase.INPUT,
                event_index=1,
                node_id="main",
                actor_id="demonstrator",
                payload={
                    "available_actions": (0, 1),
                    "observation": {"social_action": social_action, "social_reward": social_reward},
                },
            ),
            Event(
                phase=EventPhase.DECISION,
                event_index=2,
                node_id="main",
                actor_id="demonstrator",
                payload={"action": social_action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=3,
                node_id="main",
                actor_id="demonstrator",
                payload={"reward": social_reward},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=4,
                node_id="main",
                actor_id="demonstrator",
                payload={},
            ),
            Event(phase=EventPhase.UPDATE, event_index=5, node_id="main", payload={}),
            Event(
                phase=EventPhase.DECISION, event_index=6, node_id="main", payload={"action": action}
            ),
            Event(
                phase=EventPhase.OUTCOME, event_index=7, node_id="main", payload={"reward": reward}
            ),
            Event(phase=EventPhase.UPDATE, event_index=8, node_id="main", payload={}),
        ),
    )


def _social_subject() -> SubjectData:
    """Create a social subject with two trials for adapter tests.

    Returns
    -------
    SubjectData
        Subject with social trial data.
    """

    return SubjectData(
        subject_id="social-s1",
        blocks=(
            Block(
                block_index=0,
                condition="baseline",
                trials=(
                    _social_trial(0, action=0, reward=1.0, social_action=1, social_reward=0.0),
                    _social_trial(1, action=1, reward=0.0, social_action=0, social_reward=1.0),
                ),
            ),
        ),
    )


def test_social_adapter_program_paths_exist() -> None:
    """Ensure the social adapter resolves existing Stan program files.

    Returns
    -------
    None
        This test asserts program path existence for all hierarchies.
    """

    adapter = SocialRlSelfRewardDemoRewardStanAdapter()

    for hierarchy in HierarchyStructure:
        path = Path(adapter.stan_program_path(hierarchy))
        assert path.exists(), f"Missing Stan program: {path}"


def test_social_adapter_builds_subject_stan_data() -> None:
    """Ensure the social adapter exports Stan data with social fields.

    Returns
    -------
    None
        This test asserts required Stan data keys.
    """

    adapter = SocialRlSelfRewardDemoRewardStanAdapter()
    subject = _social_subject()

    stan_data = adapter.build_stan_data(
        subject,
        SOCIAL_PRE_CHOICE_SCHEMA,
        HierarchyStructure.SUBJECT_SHARED,
    )

    assert stan_data["A"] == 2
    # PRE_CHOICE: 3 subject steps per trial x 2 trials = 6 total steps
    assert stan_data["E"] == 6
    assert stan_data["D"] == 2
    assert "step_social_action" in stan_data
    assert "step_social_reward" in stan_data
    assert len(stan_data["step_social_action"]) == 6
    assert len(stan_data["step_social_reward"]) == 6
    assert "alpha_self_prior_family" in stan_data
    assert "alpha_other_prior_family" in stan_data
    assert "beta_prior_family" in stan_data
    assert stan_data["q_init"] == 0.5
    assert "reset_on_block" in stan_data


def test_social_adapter_adds_condition_data() -> None:
    """Ensure the social adapter augments Stan data with condition indices.

    Returns
    -------
    None
        This test asserts conditioned Stan export fields.
    """

    subject = SubjectData(
        subject_id="social-cond",
        blocks=(
            Block(
                block_index=0,
                condition="baseline",
                trials=(
                    _social_trial(0, action=0, reward=1.0, social_action=1, social_reward=0.0),
                ),
            ),
            Block(
                block_index=1,
                condition="social",
                trials=(
                    _social_trial(0, action=1, reward=0.0, social_action=0, social_reward=1.0),
                ),
            ),
        ),
    )
    adapter = SocialRlSelfRewardDemoRewardStanAdapter()
    kernel = SocialRlSelfRewardDemoRewardKernel()
    layout = SharedDeltaLayout(
        kernel_spec=kernel.spec(),
        conditions=("baseline", "social"),
        baseline_condition="baseline",
    )

    stan_data = adapter.build_stan_data(
        subject,
        SOCIAL_PRE_CHOICE_SCHEMA,
        HierarchyStructure.SUBJECT_BLOCK_CONDITION,
        layout=layout,
    )

    assert stan_data["C"] == 2
    # PRE_CHOICE: 3 subject steps per trial x 1 trial per block = [1,1,1, 2,2,2]
    assert stan_data["step_condition"] == [1, 1, 1, 2, 2, 2]
    assert stan_data["baseline_cond"] == 1


def test_social_adapter_subject_param_names() -> None:
    """Ensure the social adapter reports correct subject parameter names.

    Returns
    -------
    None
        This test asserts subject-level parameter names.
    """

    adapter = SocialRlSelfRewardDemoRewardStanAdapter()
    assert adapter.subject_param_names() == ("alpha_self", "alpha_other", "beta")
