"""Tests for inference configuration and dispatch."""

from comp_model.data.schema import Dataset
from comp_model.environments.bandit import StationaryBanditEnvironment
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.dispatch import fit
from comp_model.inference.mle.optimize import MleFitResult
from comp_model.models.condition.shared_delta import SharedDeltaLayout
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.runtime.engine import SimulationConfig, simulate_subject
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA
from comp_model.tasks.spec import BlockSpec, TaskSpec


def _single_subject() -> object:
    """Create a simulated single-subject dataset fixture.

    Returns
    -------
    object
        Simulated subject data for dispatch tests.
    """

    kernel = AsocialQLearningKernel()
    return simulate_subject(
        task=TaskSpec(
            task_id="dispatch-simple",
            blocks=(
                BlockSpec(
                    condition="baseline",
                    n_trials=4,
                    schema=ASOCIAL_BANDIT_SCHEMA,
                    metadata={"n_actions": 2},
                ),
            ),
        ),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=kernel.parse_params({"alpha": 0.0, "beta": 1.0}),
        config=SimulationConfig(seed=11),
        subject_id="s1",
    )


def _two_condition_task() -> TaskSpec:
    """Create a two-block task with distinct conditions.

    Returns
    -------
    TaskSpec
        Two-condition task for conditioned dispatch tests.
    """

    return TaskSpec(
        task_id="dispatch-conditioned",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=6,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
            BlockSpec(
                condition="social",
                n_trials=6,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": 2},
            ),
        ),
    )


def test_inference_config_defaults_match_expected_backend_settings() -> None:
    """Ensure inference config defaults match the planning contract.

    Returns
    -------
    None
        This test asserts configuration defaults.
    """

    config = InferenceConfig(hierarchy=HierarchyStructure.SUBJECT_SHARED)

    assert config.backend == "stan"
    assert config.sampler == "nuts"


def test_dispatch_routes_mle_simple_fits() -> None:
    """Ensure dispatch routes simple MLE fits through the MLE backend.

    Returns
    -------
    None
        This test asserts the returned fit result type.
    """

    kernel = AsocialQLearningKernel()
    subject = _single_subject()
    result = fit(
        InferenceConfig(
            hierarchy=HierarchyStructure.SUBJECT_SHARED,
            backend="mle",
        ),
        kernel,
        subject,
        ASOCIAL_BANDIT_SCHEMA,
    )

    assert isinstance(result, MleFitResult)
    assert result.subject_id == "s1"


def test_dispatch_rejects_dataset_for_mle_backend() -> None:
    """Ensure MLE dispatch rejects dataset-level fits.

    Returns
    -------
    None
        This test raises on unsupported MLE data scope.
    """

    kernel = AsocialQLearningKernel()
    subject = _single_subject()
    dataset = Dataset(subjects=(subject,))

    try:
        fit(
            InferenceConfig(
                hierarchy=HierarchyStructure.SUBJECT_SHARED,
                backend="mle",
            ),
            kernel,
            dataset,
            ASOCIAL_BANDIT_SCHEMA,
        )
    except ValueError as exc:
        assert "single-subject" in str(exc)
    else:
        raise AssertionError("Expected MLE dispatch to reject dataset-level data")


def test_dispatch_routes_conditioned_mle_fits() -> None:
    """Ensure dispatch routes conditioned MLE fits when a layout is provided.

    Returns
    -------
    None
        This test asserts conditioned fit metadata exists.
    """

    kernel = AsocialQLearningKernel()
    params = kernel.parse_params({"alpha": 0.0, "beta": 1.0})
    subject = simulate_subject(
        task=_two_condition_task(),
        env=StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2)),
        kernel=kernel,
        params=params,
        config=SimulationConfig(seed=7),
        subject_id="s1",
    )
    layout = SharedDeltaLayout(
        kernel_spec=kernel.spec(),
        conditions=("baseline", "social"),
        baseline_condition="baseline",
    )

    result = fit(
        InferenceConfig(
            hierarchy=HierarchyStructure.SUBJECT_BLOCK_CONDITION,
            backend="mle",
        ),
        kernel,
        subject,
        ASOCIAL_BANDIT_SCHEMA,
        layout=layout,
    )

    assert isinstance(result, MleFitResult)
    assert result.params_by_condition is not None
