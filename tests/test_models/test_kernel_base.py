"""Tests for kernel metadata structures."""

from dataclasses import asdict
from typing import Any

import pytest

from comp_model.models.kernels.base import ModelKernelSpec, ParameterSpec
from comp_model.models.kernels.social_rl_demo_action import SocialRlDemoActionKernel
from comp_model.models.kernels.social_rl_demo_action_bias import (
    SocialRlDemoActionBiasKernel,
)
from comp_model.models.kernels.social_rl_demo_action_bias_sticky import (
    SocialRlDemoActionBiasStickyKernel,
)
from comp_model.models.kernels.social_rl_demo_mixture import SocialRlDemoMixtureKernel
from comp_model.models.kernels.social_rl_demo_mixture_sticky import (
    SocialRlDemoMixtureStickyKernel,
)
from comp_model.models.kernels.social_rl_demo_reward import SocialRlDemoRewardKernel
from comp_model.models.kernels.social_rl_demo_reward_sticky import (
    SocialRlDemoRewardStickyKernel,
)
from comp_model.models.kernels.social_rl_self_reward_demo_action_mixture import (
    SocialRlSelfRewardDemoActionMixtureKernel,
)
from comp_model.models.kernels.social_rl_self_reward_demo_action_mixture_sticky import (
    SocialRlSelfRewardDemoActionMixtureStickyKernel,
)
from comp_model.models.kernels.social_rl_self_reward_demo_mixture import (
    SocialRlSelfRewardDemoMixtureKernel,
)
from comp_model.models.kernels.social_rl_self_reward_demo_mixture_sticky import (
    SocialRlSelfRewardDemoMixtureStickyKernel,
)
from comp_model.models.kernels.social_rl_self_reward_demo_reward import (
    SocialRlSelfRewardDemoRewardKernel,
)
from comp_model.models.kernels.social_rl_self_reward_demo_reward_sticky import (
    SocialRlSelfRewardDemoRewardStickyKernel,
)
from comp_model.models.kernels.transforms import get_transform


def test_parameter_spec_captures_bounds() -> None:
    """Ensure parameter metadata stores constrained bounds.

    Returns
    -------
    None
        This test asserts stored dataclass values.
    """

    parameter = ParameterSpec(
        name="alpha",
        transform_id="sigmoid",
        description="learning rate",
        bounds=(0.0, 1.0),
    )

    assert parameter.bounds == (0.0, 1.0)


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        pytest.param((None, None), "at least one side", id="both_open"),
        pytest.param((1.0, 1.0), "smaller", id="equal_sides"),
        pytest.param((2.0, 1.0), "smaller", id="reversed"),
    ],
)
def test_parameter_spec_rejects_invalid_bounds(
    bounds: tuple[float | None, float | None],
    message: str,
) -> None:
    """ParameterSpec rejects malformed constrained bounds."""

    with pytest.raises(ValueError, match=message):
        ParameterSpec(name="beta", transform_id="softplus", bounds=bounds)


def test_model_kernel_spec_defaults_match_plan() -> None:
    """Ensure kernel spec defaults align with the implementation plan.

    Returns
    -------
    None
        This test asserts the metadata defaults.
    """

    spec = ModelKernelSpec(model_id="demo", parameter_specs=())

    assert spec.requires_social is False
    assert spec.required_social_fields == frozenset()
    assert spec.n_actions is None
    assert spec.state_reset_policy == "per_block"


class TestRequiredSocialFields:
    """Tests for required_social_fields on concrete kernels."""

    @pytest.mark.parametrize(
        "kernel_cls, expected",
        [
            pytest.param(
                "AsocialQLearningKernel",
                frozenset(),
                id="asocial_q_learning",
            ),
            pytest.param(
                "AsocialRlAsymmetricKernel",
                frozenset(),
                id="asocial_rl_asymmetric",
            ),
            pytest.param(
                "AsocialRlStickyKernel",
                frozenset(),
                id="asocial_rl_sticky",
            ),
            pytest.param(
                "SocialRlDemoActionBiasKernel",
                frozenset({"action"}),
                id="social_demo_action_bias",
            ),
            pytest.param(
                "SocialRlDemoActionKernel",
                frozenset({"action"}),
                id="social_demo_action",
            ),
            pytest.param(
                "SocialRlDemoActionBiasStickyKernel",
                frozenset({"action"}),
                id="social_demo_action_bias_sticky",
            ),
            pytest.param(
                "SocialRlDemoRewardKernel",
                frozenset({"action", "reward"}),
                id="social_demo_reward",
            ),
            pytest.param(
                "SocialRlDemoRewardStickyKernel",
                frozenset({"action", "reward"}),
                id="social_demo_reward_sticky",
            ),
            pytest.param(
                "SocialRlSelfRewardDemoRewardKernel",
                frozenset({"action", "reward"}),
                id="social_self_reward_demo_reward",
            ),
            pytest.param(
                "SocialRlSelfRewardDemoRewardStickyKernel",
                frozenset({"action", "reward"}),
                id="social_self_reward_demo_reward_sticky",
            ),
            pytest.param(
                "SocialRlSelfRewardDemoMixtureKernel",
                frozenset({"action", "reward"}),
                id="social_self_reward_demo_mixture",
            ),
            pytest.param(
                "SocialRlSelfRewardDemoMixtureStickyKernel",
                frozenset({"action", "reward"}),
                id="social_self_reward_demo_mixture_sticky",
            ),
            pytest.param(
                "SocialRlDemoMixtureKernel",
                frozenset({"action", "reward"}),
                id="social_demo_mixture",
            ),
            pytest.param(
                "SocialRlDemoMixtureStickyKernel",
                frozenset({"action", "reward"}),
                id="social_demo_mixture_sticky",
            ),
            pytest.param(
                "SocialRlSelfRewardDemoActionMixtureKernel",
                frozenset({"action"}),
                id="social_self_reward_demo_action_mixture",
            ),
            pytest.param(
                "SocialRlSelfRewardDemoActionMixtureStickyKernel",
                frozenset({"action"}),
                id="social_self_reward_demo_action_mixture_sticky",
            ),
        ],
    )
    def test_required_social_fields(self, kernel_cls: str, expected: frozenset[str]) -> None:
        """Each kernel declares the correct required_social_fields."""
        import comp_model.models.kernels as kernels_mod

        cls = getattr(kernels_mod, kernel_cls)
        spec = cls.spec()
        assert spec.required_social_fields == expected

    @pytest.mark.parametrize(
        "kernel_cls",
        [
            "SocialRlDemoActionKernel",
            "SocialRlDemoActionBiasStickyKernel",
            "SocialRlDemoActionBiasKernel",
            "SocialRlSelfRewardDemoRewardKernel",
            "SocialRlSelfRewardDemoRewardStickyKernel",
            "SocialRlDemoRewardKernel",
            "SocialRlDemoRewardStickyKernel",
            "SocialRlSelfRewardDemoMixtureKernel",
            "SocialRlSelfRewardDemoMixtureStickyKernel",
            "SocialRlDemoMixtureKernel",
            "SocialRlDemoMixtureStickyKernel",
            "SocialRlSelfRewardDemoActionMixtureKernel",
            "SocialRlSelfRewardDemoActionMixtureStickyKernel",
        ],
    )
    def test_social_kernels_set_requires_social(self, kernel_cls: str) -> None:
        """Social kernels must have requires_social=True."""
        import comp_model.models.kernels as kernels_mod

        cls = getattr(kernels_mod, kernel_cls)
        spec = cls.spec()
        assert spec.requires_social is True

    @pytest.mark.parametrize(
        "kernel_cls",
        [
            "AsocialQLearningKernel",
            "AsocialRlAsymmetricKernel",
            "AsocialRlStickyKernel",
        ],
    )
    def test_asocial_kernels_unset_requires_social(self, kernel_cls: str) -> None:
        """Asocial kernels must have requires_social=False."""
        import comp_model.models.kernels as kernels_mod

        cls = getattr(kernels_mod, kernel_cls)
        spec = cls.spec()
        assert spec.requires_social is False


@pytest.mark.parametrize(
    "kernel_cls",
    [
        SocialRlDemoActionBiasKernel,
        SocialRlDemoActionKernel,
        SocialRlDemoActionBiasKernel,
        SocialRlDemoActionBiasStickyKernel,
        SocialRlSelfRewardDemoRewardKernel,
        SocialRlSelfRewardDemoRewardStickyKernel,
        SocialRlDemoRewardKernel,
        SocialRlDemoRewardStickyKernel,
        SocialRlDemoMixtureKernel,
        SocialRlDemoMixtureStickyKernel,
        SocialRlSelfRewardDemoActionMixtureKernel,
        SocialRlSelfRewardDemoActionMixtureStickyKernel,
        SocialRlSelfRewardDemoMixtureKernel,
        SocialRlSelfRewardDemoMixtureStickyKernel,
    ],
)
def test_social_kernel_transform_lookup_is_cached_per_class(kernel_cls: type[Any]) -> None:
    """Social kernels should reuse one transform map per kernel class."""

    first = kernel_cls._parameter_transforms()
    second = kernel_cls()._parameter_transforms()

    assert first is second


@pytest.mark.parametrize(
    ("kernel", "raw"),
    [
        pytest.param(
            SocialRlDemoActionBiasKernel(),
            {
                "demo_bias": 0.3,
            },
            id="social_demo_action_bias",
        ),
        pytest.param(
            SocialRlDemoActionKernel(),
            {
                "alpha_other_action": 0.4,
                "beta": 0.8,
            },
            id="social_demo_action",
        ),
        pytest.param(
            SocialRlDemoActionBiasStickyKernel(),
            {
                "demo_bias": 0.3,
                "stickiness": -0.5,
            },
            id="social_demo_action_bias_sticky",
        ),
        pytest.param(
            SocialRlDemoRewardKernel(),
            {"alpha_other": -0.3, "beta": 1.2},
            id="social_demo_reward",
        ),
        pytest.param(
            SocialRlDemoRewardStickyKernel(),
            {"alpha_other": -0.3, "beta": 1.2, "stickiness": -0.8},
            id="social_demo_reward_sticky",
        ),
        pytest.param(
            SocialRlSelfRewardDemoRewardKernel(),
            {"alpha_self": 0.1, "alpha_other": -0.3, "beta": 1.2},
            id="social_self_reward_demo_reward",
        ),
        pytest.param(
            SocialRlSelfRewardDemoRewardStickyKernel(),
            {
                "alpha_self": 0.1,
                "alpha_other": -0.3,
                "beta": 1.2,
                "stickiness": -0.8,
            },
            id="social_self_reward_demo_reward_sticky",
        ),
        pytest.param(
            SocialRlDemoMixtureKernel(),
            {
                "alpha_other_outcome": 0.2,
                "alpha_other_action": -0.4,
                "w_imitation": 0.7,
                "beta": 1.1,
            },
            id="social_demo_mixture",
        ),
        pytest.param(
            SocialRlDemoMixtureStickyKernel(),
            {
                "alpha_other_outcome": -0.6,
                "alpha_other_action": 0.5,
                "w_imitation": -0.2,
                "beta": 1.1,
                "stickiness": 0.8,
            },
            id="social_demo_mixture_sticky",
        ),
        pytest.param(
            SocialRlSelfRewardDemoActionMixtureKernel(),
            {
                "alpha_self": -0.2,
                "alpha_other_action": 0.4,
                "w_imitation": -0.6,
                "beta": 0.8,
            },
            id="social_self_reward_demo_action_mixture",
        ),
        pytest.param(
            SocialRlSelfRewardDemoActionMixtureStickyKernel(),
            {
                "alpha_self": -0.2,
                "alpha_other_action": 0.4,
                "w_imitation": -0.6,
                "beta": 0.8,
                "stickiness": -0.5,
            },
            id="social_self_reward_demo_action_mixture_sticky",
        ),
        pytest.param(
            SocialRlSelfRewardDemoMixtureKernel(),
            {
                "alpha_self": 0.3,
                "alpha_other_outcome": -0.2,
                "alpha_other_action": 0.1,
                "w_imitation": -0.5,
                "beta": 1.4,
            },
            id="social_self_reward_demo_mixture",
        ),
        pytest.param(
            SocialRlSelfRewardDemoMixtureStickyKernel(),
            {
                "alpha_self": -0.1,
                "alpha_other_outcome": 0.6,
                "alpha_other_action": -0.7,
                "w_imitation": 0.2,
                "beta": 1.3,
                "stickiness": -0.9,
            },
            id="social_self_reward_demo_mixture_sticky",
        ),
    ],
)
def test_social_kernel_parse_params_matches_declared_transforms(
    kernel: Any,
    raw: dict[str, float],
) -> None:
    """Social-kernel parameter parsing should preserve declared transform semantics."""

    params = kernel.parse_params(raw)
    expected = {
        parameter.name: get_transform(parameter.transform_id).forward(raw[parameter.name])
        for parameter in kernel.spec().parameter_specs
    }

    assert asdict(params) == expected
