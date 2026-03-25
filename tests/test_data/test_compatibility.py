"""Tests for kernel + schema compatibility validation."""

import pytest

from comp_model.data.compatibility import check_kernel_schema_compatibility
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    AsocialRlAsymmetricKernel,
    SocialRlSelfRewardDemoMixtureKernel,
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


# ---------------------------------------------------------------------------
# Social kernel + asocial schema — must reject
# ---------------------------------------------------------------------------


class TestSocialKernelOnAsocialSchema:
    """Social kernels must be rejected when paired with an asocial schema."""

    def test_demo_reward_kernel_on_asocial_raises(self) -> None:
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
