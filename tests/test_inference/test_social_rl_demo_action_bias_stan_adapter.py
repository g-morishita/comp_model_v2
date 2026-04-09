"""Tests for the non-sticky social demo-action bias Stan adapter."""

from pathlib import Path

from comp_model.data.schema import Block, Event, EventPhase, SubjectData, Trial
from comp_model.inference.bayes.stan.adapters.social_rl_demo_action_bias import (
    SocialRlDemoActionBiasStanAdapter,
)
from comp_model.inference.config import HierarchyStructure
from comp_model.models.condition.shared_delta import SharedDeltaLayout
from comp_model.models.kernels.social_rl_demo_action_bias import SocialRlDemoActionBiasKernel
from comp_model.tasks.schemas import SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA


def _social_trial(
    trial_index: int, action: int, reward: float, social_action: int, social_reward: float
) -> Trial:
    """Create a minimal social trial for action-bias adapter tests."""

    return Trial(
        trial_index=trial_index,
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
                payload={"action": social_action},
            ),
            Event(
                phase=EventPhase.OUTCOME,
                event_index=2,
                node_id="main",
                actor_id="demonstrator",
                payload={"reward": social_reward},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=3,
                node_id="main",
                actor_id="demonstrator",
                payload={"choice": social_action, "reward": social_reward},
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=4,
                node_id="main",
                actor_id="demonstrator",
                payload={"choice": social_action, "reward": social_reward},
            ),
            Event(
                phase=EventPhase.INPUT,
                event_index=5,
                node_id="main",
                payload={"available_actions": (0, 1)},
            ),
            Event(
                phase=EventPhase.DECISION, event_index=6, node_id="main", payload={"action": action}
            ),
            Event(
                phase=EventPhase.OUTCOME, event_index=7, node_id="main", payload={"reward": reward}
            ),
            Event(
                phase=EventPhase.UPDATE,
                event_index=8,
                node_id="main",
                payload={"choice": action, "reward": reward},
            ),
        ),
    )


def _social_subject() -> SubjectData:
    """Create a social subject with two trials for action-bias adapter tests."""

    return SubjectData(
        subject_id="social-action-bias-s1",
        blocks=(
            Block(
                block_index=0,
                condition="baseline",
                schema_id="social_pre_choice_action_only",
                trials=(
                    _social_trial(0, action=0, reward=1.0, social_action=1, social_reward=0.0),
                    _social_trial(1, action=1, reward=0.0, social_action=0, social_reward=1.0),
                ),
            ),
        ),
    )


def test_action_bias_adapter_program_paths_exist() -> None:
    """Ensure the non-sticky action-bias adapter resolves all Stan programs."""

    adapter = SocialRlDemoActionBiasStanAdapter()

    for hierarchy in HierarchyStructure:
        path = Path(adapter.stan_program_path(hierarchy))
        assert path.exists(), f"Missing Stan program: {path}"


def test_action_bias_adapter_builds_subject_stan_data() -> None:
    """Ensure the action-bias adapter exports the expected Stan data."""

    adapter = SocialRlDemoActionBiasStanAdapter()
    subject = _social_subject()

    stan_data = adapter.build_stan_data(
        subject,
        SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
        HierarchyStructure.SUBJECT_SHARED,
    )

    assert stan_data["A"] == 2
    assert stan_data["E"] == 6
    assert stan_data["D"] == 2
    assert "step_social_action" in stan_data
    assert "step_social_reward" in stan_data
    assert "demo_bias_prior_family" in stan_data
    assert "stickiness_prior_family" not in stan_data
    assert "reset_on_block" in stan_data
    assert "q_init" not in stan_data
    assert all(reward == 0.0 for reward in stan_data["step_social_reward"])


def test_action_bias_adapter_adds_condition_data() -> None:
    """Ensure the action-bias adapter augments conditioned Stan data."""

    subject = SubjectData(
        subject_id="social-action-bias-cond",
        blocks=(
            Block(
                block_index=0,
                condition="baseline",
                schema_id="social_pre_choice_action_only",
                trials=(
                    _social_trial(0, action=0, reward=1.0, social_action=1, social_reward=0.0),
                ),
            ),
            Block(
                block_index=1,
                condition="social",
                schema_id="social_pre_choice_action_only",
                trials=(
                    _social_trial(0, action=1, reward=0.0, social_action=0, social_reward=1.0),
                ),
            ),
        ),
    )
    adapter = SocialRlDemoActionBiasStanAdapter()
    kernel = SocialRlDemoActionBiasKernel()
    layout = SharedDeltaLayout(
        kernel_spec=kernel.spec(),
        conditions=("baseline", "social"),
        baseline_condition="baseline",
    )

    stan_data = adapter.build_stan_data(
        subject,
        SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
        HierarchyStructure.SUBJECT_BLOCK_CONDITION,
        layout=layout,
    )

    assert stan_data["C"] == 2
    assert stan_data["step_condition"] == [1, 1, 1, 2, 2, 2]
    assert stan_data["baseline_cond"] == 1


def test_action_bias_adapter_subject_param_names() -> None:
    """Ensure the action-bias adapter reports the expected subject parameters."""

    adapter = SocialRlDemoActionBiasStanAdapter()
    assert adapter.subject_param_names() == ("demo_bias",)
