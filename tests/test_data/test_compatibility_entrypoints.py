"""Integration tests: entrypoints reject incompatible kernel+schema combinations.

These tests verify that the compatibility check fires at every public
entrypoint before any real computation begins.
"""

from __future__ import annotations

import pytest

from comp_model.data.schema import Block, Event, EventPhase, SubjectData, Trial
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    SocialRlSelfRewardDemoRewardKernel,
)
from comp_model.tasks.schemas import ASOCIAL_BANDIT_SCHEMA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SOCIAL_KERNEL = SocialRlSelfRewardDemoRewardKernel()
_ASOCIAL_SCHEMA = ASOCIAL_BANDIT_SCHEMA


def _dummy_subject(schema_id: str = "asocial_bandit") -> SubjectData:
    """Build a minimal valid subject with one asocial trial."""
    return SubjectData(
        subject_id="sub_00",
        blocks=(
            Block(
                block_index=0,
                condition="default",
                schema_id=schema_id,
                trials=(
                    Trial(
                        trial_index=0,
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
                    ),
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# fit() dispatch entrypoint
# ---------------------------------------------------------------------------


class TestFitDispatchRejectsIncompatible:
    """fit() must reject social kernel + asocial schema before fitting."""

    def test_mle_rejects_social_kernel_on_asocial_schema(self) -> None:
        """MLE path raises before optimisation."""
        from comp_model.inference.dispatch import fit

        config = InferenceConfig(
            hierarchy=HierarchyStructure.SUBJECT_SHARED,
            backend="mle",
        )
        subject = _dummy_subject()

        with pytest.raises(ValueError, match="requires social information"):
            fit(config, _SOCIAL_KERNEL, subject, _ASOCIAL_SCHEMA)


# ---------------------------------------------------------------------------
# simulate_subject() entrypoint
# ---------------------------------------------------------------------------


class TestSimulateSubjectRejectsIncompatible:
    """simulate_subject() must reject social kernel + asocial schema."""

    def test_simulate_rejects_social_kernel_on_asocial_task(self) -> None:
        """Simulation raises before the trial loop."""
        from comp_model.runtime.engine import SimulationConfig, simulate_subject
        from comp_model.tasks.spec import BlockSpec, TaskSpec

        task = TaskSpec(
            task_id="test",
            blocks=(
                BlockSpec(
                    condition="default",
                    n_trials=5,
                    schema=_ASOCIAL_SCHEMA,
                ),
            ),
        )
        params = _SOCIAL_KERNEL.parse_params({"alpha_self": 0.0, "alpha_other": 0.0, "beta": 1.0})

        with pytest.raises(ValueError, match="requires social information"):
            simulate_subject(
                task=task,
                env=_make_dummy_env(),
                kernel=_SOCIAL_KERNEL,
                params=params,
                config=SimulationConfig(seed=0),
            )


# ---------------------------------------------------------------------------
# run_parameter_recovery() entrypoint
# ---------------------------------------------------------------------------


class TestParameterRecoveryRejectsIncompatible:
    """run_parameter_recovery() must reject social kernel + asocial schema."""

    def test_rejects_social_kernel_on_asocial_schema(self) -> None:
        """Recovery raises before any simulation."""
        from scipy import stats

        from comp_model.recovery.parameter.config import ParamDist, ParameterRecoveryConfig
        from comp_model.recovery.parameter.runner import run_parameter_recovery
        from comp_model.tasks.spec import BlockSpec, TaskSpec

        task = TaskSpec(
            task_id="test",
            blocks=(
                BlockSpec(
                    condition="default",
                    n_trials=5,
                    schema=_ASOCIAL_SCHEMA,
                ),
            ),
        )
        config = ParameterRecoveryConfig(
            n_replications=1,
            n_subjects=1,
            param_dists=(
                ParamDist("alpha_self", stats.uniform(0.1, 0.8)),
                ParamDist("alpha_other", stats.uniform(0.1, 0.8)),
                ParamDist("beta", stats.uniform(0.5, 4.0)),
            ),
            task=task,
            env_factory=_make_dummy_env,
            kernel=_SOCIAL_KERNEL,
            schema=_ASOCIAL_SCHEMA,
            inference_config=InferenceConfig(
                hierarchy=HierarchyStructure.SUBJECT_SHARED,
                backend="mle",
            ),
        )

        with pytest.raises(ValueError, match="requires social information"):
            run_parameter_recovery(config)


# ---------------------------------------------------------------------------
# run_model_recovery() entrypoint
# ---------------------------------------------------------------------------


class TestModelRecoveryRejectsIncompatible:
    """run_model_recovery() must reject incompatible generating/candidate kernels."""

    def test_rejects_social_generating_model_on_asocial_schema(self) -> None:
        """Social generating model + asocial schema raises before simulation."""
        from scipy import stats

        from comp_model.recovery.model.config import (
            CandidateModelSpec,
            GeneratingModelSpec,
            ModelRecoveryConfig,
        )
        from comp_model.recovery.model.runner import run_model_recovery
        from comp_model.recovery.parameter.config import ParamDist
        from comp_model.tasks.spec import BlockSpec, TaskSpec

        task = TaskSpec(
            task_id="test",
            blocks=(
                BlockSpec(
                    condition="default",
                    n_trials=5,
                    schema=_ASOCIAL_SCHEMA,
                ),
            ),
        )
        config = ModelRecoveryConfig(
            generating_models=(
                GeneratingModelSpec(
                    name="social",
                    kernel=_SOCIAL_KERNEL,
                    param_dists=(
                        ParamDist("alpha_self", stats.uniform(0.1, 0.8)),
                        ParamDist("alpha_other", stats.uniform(0.1, 0.8)),
                        ParamDist("beta", stats.uniform(0.5, 4.0)),
                    ),
                ),
            ),
            candidate_models=(
                CandidateModelSpec(
                    name="asocial",
                    kernel=AsocialQLearningKernel(),
                    inference_config=InferenceConfig(
                        hierarchy=HierarchyStructure.SUBJECT_SHARED,
                        backend="mle",
                    ),
                ),
            ),
            n_replications=1,
            n_subjects=1,
            task=task,
            env_factory=_make_dummy_env,
            schema=_ASOCIAL_SCHEMA,
        )

        with pytest.raises(ValueError, match="requires social information"):
            run_model_recovery(config)

    def test_rejects_social_candidate_on_asocial_schema(self) -> None:
        """Social candidate model + asocial schema raises before fitting."""
        from scipy import stats

        from comp_model.recovery.model.config import (
            CandidateModelSpec,
            GeneratingModelSpec,
            ModelRecoveryConfig,
        )
        from comp_model.recovery.model.runner import run_model_recovery
        from comp_model.recovery.parameter.config import ParamDist
        from comp_model.tasks.spec import BlockSpec, TaskSpec

        task = TaskSpec(
            task_id="test",
            blocks=(
                BlockSpec(
                    condition="default",
                    n_trials=5,
                    schema=_ASOCIAL_SCHEMA,
                ),
            ),
        )
        config = ModelRecoveryConfig(
            generating_models=(
                GeneratingModelSpec(
                    name="asocial",
                    kernel=AsocialQLearningKernel(),
                    param_dists=(
                        ParamDist("alpha", stats.uniform(0.1, 0.8)),
                        ParamDist("beta", stats.uniform(0.5, 4.0)),
                    ),
                ),
            ),
            candidate_models=(
                CandidateModelSpec(
                    name="social",
                    kernel=_SOCIAL_KERNEL,
                    inference_config=InferenceConfig(
                        hierarchy=HierarchyStructure.SUBJECT_SHARED,
                        backend="mle",
                    ),
                ),
            ),
            n_replications=1,
            n_subjects=1,
            task=task,
            env_factory=_make_dummy_env,
            schema=_ASOCIAL_SCHEMA,
        )

        with pytest.raises(ValueError, match="requires social information"):
            run_model_recovery(config)


# ---------------------------------------------------------------------------
# Dummy environment
# ---------------------------------------------------------------------------


def _make_dummy_env():
    """Create a minimal bandit environment for testing."""
    from comp_model.environments.bandit import StationaryBanditEnvironment

    return StationaryBanditEnvironment(n_actions=2, reward_probs=(0.8, 0.2))
