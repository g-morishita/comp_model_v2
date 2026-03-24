"""Tests for Stan data export builders."""

from dataclasses import replace

from comp_model.data.schema import Block, Event, EventPhase, SubjectData, Trial
from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.inference.bayes.stan.data_builder import (
    add_condition_data,
    add_prior_data,
    add_state_reset_data,
    dataset_to_stan_data,
    dataset_to_step_data,
    subject_to_stan_data,
    subject_to_step_data,
)
from comp_model.models.condition.shared_delta import SharedDeltaLayout
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.runtime.engine import SimulationConfig, simulate_dataset, simulate_subject
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA
from comp_model.tasks.spec import BlockSpec, TaskSpec


def _task() -> TaskSpec:
    """Create a simple one-block task for Stan export tests.

    Returns
    -------
    TaskSpec
        One-block asocial bandit task.
    """

    return TaskSpec(
        task_id="stan-export",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=4,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
        ),
    )


def test_subject_to_stan_data_exports_expected_shapes() -> None:
    """Ensure subject export produces expected trial counts and indexing.

    Returns
    -------
    None
        This test asserts Stan array structure.
    """

    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    subject = simulate_subject(
        task=_task(),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=params,
        config=SimulationConfig(seed=13),
        subject_id="s1",
    )

    stan_data = subject_to_stan_data(subject, ASOCIAL_BANDIT_SCHEMA)

    assert stan_data["A"] == 2
    assert stan_data["T"] == 4
    assert stan_data["block_of_trial"] == [1, 1, 1, 1]
    assert all(choice in (1, 2) for choice in stan_data["choice"])


def test_dataset_to_stan_data_adds_subject_indices() -> None:
    """Ensure dataset export includes hierarchical subject indexing.

    Returns
    -------
    None
        This test asserts hierarchical export fields.
    """

    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    dataset = simulate_dataset(
        task=_task(),
        env_factory=lambda: StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params_per_subject={"s1": params, "s2": params},
        config=SimulationConfig(seed=17),
    )

    stan_data = dataset_to_stan_data(dataset, ASOCIAL_BANDIT_SCHEMA)

    assert stan_data["N"] == 2
    assert len(stan_data["subj"]) == stan_data["T"]
    assert set(stan_data["subj"]) == {1, 2}


def test_condition_prior_and_reset_data_are_added() -> None:
    """Ensure condition, prior, and reset metadata extend the Stan data dictionary.

    Returns
    -------
    None
        This test asserts condition, prior, and reset exports.
    """

    task = TaskSpec(
        task_id="stan-conditioned",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=2,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
            BlockSpec(
                condition="social",
                n_trials=2,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
        ),
    )
    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    subject = simulate_subject(
        task=task,
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=params,
        config=SimulationConfig(seed=19),
        subject_id="s1",
    )
    stan_data = subject_to_stan_data(subject, ASOCIAL_BANDIT_SCHEMA)
    layout = SharedDeltaLayout(
        kernel_spec=kernel.spec(),
        conditions=("baseline", "social"),
        baseline_condition="baseline",
    )

    add_condition_data(stan_data, subject, layout)
    add_prior_data(stan_data, kernel.spec())
    add_state_reset_data(stan_data, kernel.spec())

    assert stan_data["C"] == 2
    assert stan_data["baseline_cond"] == 1
    assert stan_data["cond"] == [1, 1, 2, 2]
    assert "alpha_prior_family" in stan_data
    assert "beta_prior_p2" in stan_data
    assert stan_data["reset_on_block"] == 1


def test_add_state_reset_data_exports_per_block_policy() -> None:
    """Ensure per-block kernels export the Stan reset flag.

    Returns
    -------
    None
        This test asserts per-block reset metadata.
    """

    kernel = AsocialQLearningKernel()
    per_block_spec = replace(kernel.spec(), state_reset_policy="per_block")
    stan_data: dict[str, int] = {}

    add_state_reset_data(stan_data, per_block_spec)

    assert stan_data["reset_on_block"] == 1


def test_subject_to_stan_data_remaps_noncontiguous_actions() -> None:
    """Ensure Stan export remaps sparse action identifiers to contiguous indices.

    Returns
    -------
    None
        This test asserts contiguous Stan action encoding.
    """

    subject = SubjectData(
        subject_id="sparse-actions",
        blocks=(
            Block(
                block_index=0,
                condition="baseline",
                trials=(
                    Trial(
                        trial_index=0,
                        events=(
                            Event(
                                phase=EventPhase.INPUT,
                                event_index=0,
                                node_id="main",
                                payload={"available_actions": (2, 5)},
                            ),
                            Event(
                                phase=EventPhase.DECISION,
                                event_index=1,
                                node_id="main",
                                payload={"action": 5},
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
                                payload={"choice": 5, "reward": 1.0},
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    stan_data = subject_to_stan_data(subject, ASOCIAL_BANDIT_SCHEMA)

    assert stan_data["A"] == 2
    assert stan_data["choice"] == [2]
    assert stan_data["avail_mask"] == [[1.0, 1.0]]


# ---------------------------------------------------------------------------
# Step-based builder tests
# ---------------------------------------------------------------------------


def test_subject_to_step_data_exports_step_stream() -> None:
    """Ensure step-based subject export produces expected step-stream fields.

    Returns
    -------
    None
        This test asserts step-stream array structure.
    """

    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    subject = simulate_subject(
        task=_task(),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=params,
        config=SimulationConfig(seed=13),
        subject_id="s1",
    )

    stan_data = subject_to_step_data(subject, ASOCIAL_BANDIT_SCHEMA, kernel_spec=kernel.spec())

    assert stan_data["A"] == 2
    # 4 trials x 2 steps each (action + self-update) = 8 total steps
    assert stan_data["E"] == 8
    assert stan_data["D"] == 4
    assert len(stan_data["step_choice"]) == 8
    assert len(stan_data["step_update_action"]) == 8
    assert len(stan_data["step_reward"]) == 8
    assert len(stan_data["step_avail_mask"]) == 8
    assert len(stan_data["step_block"]) == 8
    assert stan_data["step_block"] == [1] * 8
    # Action steps carry the choice; update steps have step_choice == 0
    choices = [c for c in stan_data["step_choice"] if c > 0]
    assert len(choices) == 4
    assert all(c in (1, 2) for c in choices)
    # Self-update steps carry the same action as the preceding choice
    update_actions = [c for c in stan_data["step_update_action"] if c > 0]
    assert len(update_actions) == 4
    assert all(c in (1, 2) for c in update_actions)


def test_dataset_to_step_data_adds_subject_indices() -> None:
    """Ensure step-based dataset export includes hierarchical subject indexing.

    Returns
    -------
    None
        This test asserts hierarchical step-stream export fields.
    """

    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    dataset = simulate_dataset(
        task=_task(),
        env_factory=lambda: StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params_per_subject={"s1": params, "s2": params},
        config=SimulationConfig(seed=17),
    )

    stan_data = dataset_to_step_data(dataset, ASOCIAL_BANDIT_SCHEMA, kernel_spec=kernel.spec())

    assert stan_data["N"] == 2
    assert len(stan_data["step_subject"]) == stan_data["E"]
    assert set(stan_data["step_subject"]) == {1, 2}


def test_subject_to_step_data_with_conditions() -> None:
    """Ensure step-based export includes condition indices when condition map provided.

    Returns
    -------
    None
        This test asserts condition-aware step-stream fields.
    """

    task = TaskSpec(
        task_id="stan-conditioned",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=2,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
            BlockSpec(
                condition="social",
                n_trials=2,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
        ),
    )
    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    subject = simulate_subject(
        task=task,
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=params,
        config=SimulationConfig(seed=19),
        subject_id="s1",
    )
    condition_map = {"baseline": 1, "social": 2}

    stan_data = subject_to_step_data(
        subject,
        ASOCIAL_BANDIT_SCHEMA,
        kernel_spec=kernel.spec(),
        condition_map=condition_map,
    )

    # 2 trials x 2 steps per trial per condition = 4 steps per condition
    assert stan_data["step_condition"] == [1, 1, 1, 1, 2, 2, 2, 2]


def test_step_data_remaps_noncontiguous_actions() -> None:
    """Ensure step-based export remaps sparse action identifiers.

    Returns
    -------
    None
        This test asserts contiguous Stan action encoding in step format.
    """

    subject = SubjectData(
        subject_id="sparse-actions",
        blocks=(
            Block(
                block_index=0,
                condition="baseline",
                trials=(
                    Trial(
                        trial_index=0,
                        events=(
                            Event(
                                phase=EventPhase.INPUT,
                                event_index=0,
                                node_id="main",
                                payload={"available_actions": (2, 5)},
                            ),
                            Event(
                                phase=EventPhase.DECISION,
                                event_index=1,
                                node_id="main",
                                payload={"action": 5},
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
                                payload={"choice": 5, "reward": 1.0},
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    kernel = AsocialQLearningKernel()

    stan_data = subject_to_step_data(subject, ASOCIAL_BANDIT_SCHEMA, kernel_spec=kernel.spec())

    assert stan_data["A"] == 2
    # 1 trial x 2 steps (action + self-update)
    assert stan_data["step_choice"] == [2, 0]
    assert stan_data["step_update_action"] == [0, 2]
    assert stan_data["step_avail_mask"] == [[1.0, 1.0], [1.0, 1.0]]


def test_step_data_includes_social_fields_when_requested() -> None:
    """Ensure step-based export includes social arrays when include_social=True.

    Returns
    -------
    None
        This test asserts social step-stream fields.
    """

    # SOCIAL_PRE_CHOICE_SCHEMA: INPUT(demo) DECISION(demo) OUTCOME(demo) UPDATE(demo→demo)
    #   UPDATE(demo→subj) INPUT(subj) DECISION(subj) OUTCOME(subj) UPDATE(subj→subj)
    subject = SubjectData(
        subject_id="social-test",
        blocks=(
            Block(
                block_index=0,
                condition="baseline",
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
                                payload={"action": 1},
                            ),
                            Event(
                                phase=EventPhase.OUTCOME,
                                event_index=2,
                                node_id="main",
                                actor_id="demonstrator",
                                payload={"reward": 0.5},
                            ),
                            Event(
                                phase=EventPhase.UPDATE,
                                event_index=3,
                                node_id="main",
                                actor_id="demonstrator",
                                payload={"choice": 1, "reward": 0.5},
                            ),
                            Event(
                                phase=EventPhase.UPDATE,
                                event_index=4,
                                node_id="main",
                                actor_id="demonstrator",
                                payload={"choice": 1, "reward": 0.5},
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
                                payload={"action": 0},
                            ),
                            Event(
                                phase=EventPhase.OUTCOME,
                                event_index=7,
                                node_id="main",
                                payload={"reward": 1.0},
                            ),
                            Event(
                                phase=EventPhase.UPDATE,
                                event_index=8,
                                node_id="main",
                                payload={"choice": 0, "reward": 1.0},
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    from comp_model.tasks.schemas import SOCIAL_PRE_CHOICE_SCHEMA

    kernel = AsocialQLearningKernel()
    stan_data = subject_to_step_data(
        subject,
        SOCIAL_PRE_CHOICE_SCHEMA,
        kernel_spec=kernel.spec(),
        include_social=True,
    )

    assert "step_social_action" in stan_data
    assert "step_social_reward" in stan_data
    # PRE_CHOICE: 3 subject steps per trial — social-update, action, self-update
    # Action index mapping (first-seen order): demo action 1 → index 1, subject action 0 → index 2
    # step 0 (social-update): social_action=1 (action 1 → index 1), social_reward=0.5
    # step 1 (action):        choice=2 (action 0 → index 2)
    # step 2 (self-update):   update_action=2, reward=1.0
    assert stan_data["E"] == 3
    assert stan_data["D"] == 1
    assert stan_data["step_choice"] == [0, 2, 0]
    assert stan_data["step_update_action"] == [0, 0, 2]
    assert stan_data["step_social_action"] == [1, 0, 0]
    assert stan_data["step_social_reward"] == [0.5, 0.0, 0.0]


def test_step_data_excludes_social_fields_by_default() -> None:
    """Ensure step-based export omits social arrays by default.

    Returns
    -------
    None
        This test asserts backward compatibility.
    """

    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    subject = simulate_subject(
        task=_task(),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=params,
        config=SimulationConfig(seed=13),
        subject_id="s1",
    )

    stan_data = subject_to_step_data(subject, ASOCIAL_BANDIT_SCHEMA, kernel_spec=kernel.spec())

    assert "step_social_action" not in stan_data
    assert "step_social_reward" not in stan_data
