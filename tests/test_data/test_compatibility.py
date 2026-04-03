"""Tests for kernel + schema compatibility validation."""

import pytest

from comp_model.data.compatibility import (
    check_kernel_schema_compatibility,
    check_spec_schema_compatibility,
)
from comp_model.data.schema import EventPhase
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    AsocialRlAsymmetricKernel,
    AsocialRlStickyKernel,
    SocialRlDemoMixtureKernel,
    SocialRlDemoRewardKernel,
    SocialRlSelfRewardDemoActionMixtureKernel,
    SocialRlSelfRewardDemoMixtureKernel,
    SocialRlSelfRewardDemoMixtureStickyKernel,
    SocialRlSelfRewardDemoRewardKernel,
)
from comp_model.tasks.schemas import (
    ASOCIAL_BANDIT_SCHEMA,
    SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA,
    SOCIAL_POST_OUTCOME_SCHEMA,
    SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
    SOCIAL_PRE_CHOICE_SCHEMA,
)

# ---------------------------------------------------------------------------
# Asocial kernel — always compatible
# ---------------------------------------------------------------------------


class TestAsocialKernelCompatibility:
    """Asocial kernels are compatible with any schema."""

    @pytest.mark.parametrize(
        "schema",
        [
            ASOCIAL_BANDIT_SCHEMA,
            SOCIAL_PRE_CHOICE_SCHEMA,
            SOCIAL_POST_OUTCOME_SCHEMA,
            SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
            SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA,
        ],
        ids=[
            "asocial",
            "social_pre_choice",
            "social_post_outcome",
            "social_pre_choice_action_only",
            "social_post_outcome_action_only",
        ],
    )
    def test_asocial_q_learning_accepts_any_schema(self, schema):
        """AsocialQLearningKernel passes on all schemas."""
        check_kernel_schema_compatibility(AsocialQLearningKernel(), schema)

    @pytest.mark.parametrize(
        "schema",
        [ASOCIAL_BANDIT_SCHEMA, SOCIAL_PRE_CHOICE_SCHEMA],
        ids=["asocial", "social"],
    )
    def test_asocial_asymmetric_accepts_any_schema(self, schema):
        """AsocialRlAsymmetricKernel passes on all schemas."""
        check_kernel_schema_compatibility(AsocialRlAsymmetricKernel(), schema)

    @pytest.mark.parametrize(
        "schema",
        [ASOCIAL_BANDIT_SCHEMA, SOCIAL_PRE_CHOICE_SCHEMA],
        ids=["asocial", "social"],
    )
    def test_asocial_sticky_accepts_any_schema(self, schema):
        """AsocialRlStickyKernel passes on all schemas."""
        check_kernel_schema_compatibility(AsocialRlStickyKernel(), schema)


# ---------------------------------------------------------------------------
# Social kernel + asocial schema — must reject
# ---------------------------------------------------------------------------


class TestSocialKernelOnAsocialSchema:
    """Social kernels must be rejected when paired with an asocial schema."""

    def test_demo_reward_kernel_on_asocial_raises(self) -> None:
        """SocialRlDemoRewardKernel fails on asocial schema."""
        with pytest.raises(ValueError, match="requires social information"):
            check_kernel_schema_compatibility(
                SocialRlDemoRewardKernel(),
                ASOCIAL_BANDIT_SCHEMA,
            )

    def test_self_reward_demo_reward_kernel_on_asocial_raises(self) -> None:
        """SocialRlSelfRewardDemoRewardKernel fails on asocial schema."""
        with pytest.raises(ValueError, match="requires social information"):
            check_kernel_schema_compatibility(
                SocialRlSelfRewardDemoRewardKernel(),
                ASOCIAL_BANDIT_SCHEMA,
            )

    def test_demo_mixture_kernel_on_asocial_raises(self) -> None:
        """SocialRlSelfRewardDemoMixtureKernel fails on asocial schema."""
        with pytest.raises(ValueError, match="requires social information"):
            check_kernel_schema_compatibility(
                SocialRlSelfRewardDemoMixtureKernel(),
                ASOCIAL_BANDIT_SCHEMA,
            )

    def test_demo_mixture_sticky_kernel_on_asocial_raises(self) -> None:
        """SocialRlSelfRewardDemoMixtureStickyKernel fails on asocial schema."""
        with pytest.raises(ValueError, match="requires social information"):
            check_kernel_schema_compatibility(
                SocialRlSelfRewardDemoMixtureStickyKernel(),
                ASOCIAL_BANDIT_SCHEMA,
            )

    def test_demo_mixture_no_self_reward_on_asocial_raises(self) -> None:
        """SocialRlDemoMixtureKernel fails on asocial schema."""
        with pytest.raises(ValueError, match="requires social information"):
            check_kernel_schema_compatibility(
                SocialRlDemoMixtureKernel(),
                ASOCIAL_BANDIT_SCHEMA,
            )

    def test_self_reward_demo_action_mixture_on_asocial_raises(self) -> None:
        """SocialRlSelfRewardDemoActionMixtureKernel fails on asocial schema."""
        with pytest.raises(ValueError, match="requires social information"):
            check_kernel_schema_compatibility(
                SocialRlSelfRewardDemoActionMixtureKernel(),
                ASOCIAL_BANDIT_SCHEMA,
            )


# ---------------------------------------------------------------------------
# Social kernel + action-only schema — must reject (needs reward)
# ---------------------------------------------------------------------------


class TestSocialKernelOnActionOnlySchema:
    """Social kernels requiring reward must be rejected on action-only schemas."""

    @pytest.mark.parametrize(
        "schema",
        [
            SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
            SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA,
        ],
        ids=["pre_choice_action_only", "post_outcome_action_only"],
    )
    def test_demo_reward_kernel_on_action_only_raises(self, schema) -> None:
        """SocialRlDemoRewardKernel needs reward, action-only lacks it."""
        with pytest.raises(ValueError, match=r"Missing.*reward"):
            check_kernel_schema_compatibility(SocialRlDemoRewardKernel(), schema)

    @pytest.mark.parametrize(
        "schema",
        [
            SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
            SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA,
        ],
        ids=["pre_choice_action_only", "post_outcome_action_only"],
    )
    def test_self_reward_demo_reward_kernel_on_action_only_raises(self, schema) -> None:
        """SocialRlSelfRewardDemoRewardKernel needs reward, action-only lacks it."""
        with pytest.raises(ValueError, match=r"Missing.*reward"):
            check_kernel_schema_compatibility(SocialRlSelfRewardDemoRewardKernel(), schema)

    @pytest.mark.parametrize(
        "schema",
        [
            SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
            SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA,
        ],
        ids=["pre_choice_action_only", "post_outcome_action_only"],
    )
    def test_demo_mixture_kernel_on_action_only_raises(self, schema) -> None:
        """SocialRlSelfRewardDemoMixtureKernel needs reward, action-only lacks it."""
        with pytest.raises(ValueError, match=r"Missing.*reward"):
            check_kernel_schema_compatibility(SocialRlSelfRewardDemoMixtureKernel(), schema)

    @pytest.mark.parametrize(
        "schema",
        [
            SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
            SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA,
        ],
        ids=["pre_choice_action_only", "post_outcome_action_only"],
    )
    def test_demo_mixture_sticky_kernel_on_action_only_raises(self, schema) -> None:
        """SocialRlSelfRewardDemoMixtureStickyKernel needs reward, action-only lacks it."""
        with pytest.raises(ValueError, match=r"Missing.*reward"):
            check_kernel_schema_compatibility(SocialRlSelfRewardDemoMixtureStickyKernel(), schema)

    @pytest.mark.parametrize(
        "schema",
        [
            SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
            SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA,
        ],
        ids=["pre_choice_action_only", "post_outcome_action_only"],
    )
    def test_demo_mixture_no_self_reward_on_action_only_raises(self, schema) -> None:
        """SocialRlDemoMixtureKernel needs reward, action-only lacks it."""
        with pytest.raises(ValueError, match=r"Missing.*reward"):
            check_kernel_schema_compatibility(SocialRlDemoMixtureKernel(), schema)


# ---------------------------------------------------------------------------
# Social kernel + full-observation schema — must pass
# ---------------------------------------------------------------------------


class TestSocialKernelOnFullObservationSchema:
    """Social kernels pass when the schema provides all required fields."""

    @pytest.mark.parametrize(
        "schema",
        [SOCIAL_PRE_CHOICE_SCHEMA, SOCIAL_POST_OUTCOME_SCHEMA],
        ids=["pre_choice", "post_outcome"],
    )
    def test_demo_reward_kernel_on_full_observation_passes(self, schema) -> None:
        """SocialRlDemoRewardKernel is compatible with full-observation schemas."""
        check_kernel_schema_compatibility(SocialRlDemoRewardKernel(), schema)

    @pytest.mark.parametrize(
        "schema",
        [SOCIAL_PRE_CHOICE_SCHEMA, SOCIAL_POST_OUTCOME_SCHEMA],
        ids=["pre_choice", "post_outcome"],
    )
    def test_self_reward_demo_reward_kernel_on_full_observation_passes(self, schema) -> None:
        """SocialRlSelfRewardDemoRewardKernel is compatible with full-observation schemas."""
        check_kernel_schema_compatibility(SocialRlSelfRewardDemoRewardKernel(), schema)

    @pytest.mark.parametrize(
        "schema",
        [SOCIAL_PRE_CHOICE_SCHEMA, SOCIAL_POST_OUTCOME_SCHEMA],
        ids=["pre_choice", "post_outcome"],
    )
    def test_demo_mixture_kernel_on_full_observation_passes(self, schema) -> None:
        """SocialRlSelfRewardDemoMixtureKernel is compatible with full-observation schemas."""
        check_kernel_schema_compatibility(SocialRlSelfRewardDemoMixtureKernel(), schema)

    @pytest.mark.parametrize(
        "schema",
        [SOCIAL_PRE_CHOICE_SCHEMA, SOCIAL_POST_OUTCOME_SCHEMA],
        ids=["pre_choice", "post_outcome"],
    )
    def test_demo_mixture_sticky_kernel_on_full_observation_passes(self, schema) -> None:
        """SocialRlSelfRewardDemoMixtureStickyKernel is compatible with full-observation schemas."""
        check_kernel_schema_compatibility(SocialRlSelfRewardDemoMixtureStickyKernel(), schema)

    @pytest.mark.parametrize(
        "schema",
        [SOCIAL_PRE_CHOICE_SCHEMA, SOCIAL_POST_OUTCOME_SCHEMA],
        ids=["pre_choice", "post_outcome"],
    )
    def test_demo_mixture_no_self_reward_on_full_observation_passes(self, schema) -> None:
        """SocialRlDemoMixtureKernel is compatible with full-observation schemas."""
        check_kernel_schema_compatibility(SocialRlDemoMixtureKernel(), schema)


# ---------------------------------------------------------------------------
# Action-only kernel — passes on action-only and full-observation schemas
# ---------------------------------------------------------------------------


class TestActionOnlyKernelCompatibility:
    """SocialRlSelfRewardDemoActionMixtureKernel only requires action, not reward."""

    @pytest.mark.parametrize(
        "schema",
        [
            SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
            SOCIAL_POST_OUTCOME_ACTION_ONLY_SCHEMA,
        ],
        ids=["pre_choice_action_only", "post_outcome_action_only"],
    )
    def test_self_reward_demo_action_mixture_on_action_only_passes(self, schema) -> None:
        """SocialRlSelfRewardDemoActionMixtureKernel passes on action-only schemas."""
        check_kernel_schema_compatibility(SocialRlSelfRewardDemoActionMixtureKernel(), schema)

    @pytest.mark.parametrize(
        "schema",
        [SOCIAL_PRE_CHOICE_SCHEMA, SOCIAL_POST_OUTCOME_SCHEMA],
        ids=["pre_choice", "post_outcome"],
    )
    def test_self_reward_demo_action_mixture_on_full_observation_passes(self, schema) -> None:
        """SocialRlSelfRewardDemoActionMixtureKernel passes on full-observation schemas."""
        check_kernel_schema_compatibility(SocialRlSelfRewardDemoActionMixtureKernel(), schema)


# ---------------------------------------------------------------------------
# Per-step validation — split fields across steps must still be rejected
# ---------------------------------------------------------------------------


class TestPerStepFieldValidation:
    """Verify that per-step checking catches split observable fields."""

    def test_split_fields_across_steps_rejected(self) -> None:
        """A schema where one step provides action and another provides reward must be rejected.

        The union would be {action, reward} (passing), but neither step
        individually provides both. The per-step check catches this.
        """
        from comp_model.tasks.schemas import TrialSchema, TrialSchemaStep

        split_schema = TrialSchema(
            schema_id="split_fields_test",
            steps=(
                TrialSchemaStep(
                    phase=EventPhase.INPUT,
                    node_id="main",
                    actor_id="subject",
                ),
                TrialSchemaStep(
                    phase=EventPhase.DECISION,
                    node_id="demo",
                    actor_id="demonstrator",
                ),
                TrialSchemaStep(
                    phase=EventPhase.OUTCOME,
                    node_id="demo",
                    actor_id="demonstrator",
                ),
                # Social UPDATE step 1: only action observable
                TrialSchemaStep(
                    phase=EventPhase.UPDATE,
                    node_id="demo",
                    actor_id="demonstrator",
                    learner_id="subject",
                    observable_fields=frozenset({"action"}),
                ),
                # Social UPDATE step 2: only reward observable
                TrialSchemaStep(
                    phase=EventPhase.UPDATE,
                    node_id="demo_reward",
                    actor_id="demonstrator",
                    learner_id="subject",
                    observable_fields=frozenset({"reward"}),
                ),
                TrialSchemaStep(
                    phase=EventPhase.DECISION,
                    node_id="main",
                    actor_id="subject",
                ),
                TrialSchemaStep(
                    phase=EventPhase.OUTCOME,
                    node_id="main",
                    actor_id="subject",
                ),
                TrialSchemaStep(
                    phase=EventPhase.UPDATE,
                    node_id="main",
                    actor_id="subject",
                    learner_id="subject",
                ),
            ),
        )
        with pytest.raises(ValueError, match=r"Missing.*reward"):
            check_kernel_schema_compatibility(SocialRlSelfRewardDemoRewardKernel(), split_schema)


# ---------------------------------------------------------------------------
# check_spec_schema_compatibility — direct spec-based check
# ---------------------------------------------------------------------------


class TestSpecSchemaCompatibility:
    """Verify that the spec-based check works identically to the kernel-based one."""

    def test_spec_rejects_social_on_asocial(self) -> None:
        """Passing a spec directly to check_spec_schema_compatibility works."""
        spec = SocialRlSelfRewardDemoRewardKernel().spec()
        with pytest.raises(ValueError, match="requires social information"):
            check_spec_schema_compatibility(spec, ASOCIAL_BANDIT_SCHEMA)

    def test_spec_accepts_compatible(self) -> None:
        """Spec-based check passes for compatible combinations."""
        spec = SocialRlSelfRewardDemoRewardKernel().spec()
        check_spec_schema_compatibility(spec, SOCIAL_PRE_CHOICE_SCHEMA)
