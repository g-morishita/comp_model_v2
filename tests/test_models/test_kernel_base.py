"""Tests for kernel metadata structures."""

import pytest

from comp_model.models.kernels.base import InitSpec, ModelKernelSpec, ParameterSpec


def test_parameter_spec_captures_init_metadata() -> None:
    """Ensure parameter metadata stores init specification.

    Returns
    -------
    None
        This test asserts stored dataclass values.
    """

    init = InitSpec(strategy="fixed", kwargs={}, default_unconstrained=0.5)
    parameter = ParameterSpec(
        name="alpha",
        transform_id="sigmoid",
        description="learning rate",
        mle_init=init,
    )

    assert parameter.mle_init == init


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
                "SocialRlSelfRewardDemoRewardKernel",
                frozenset({"action", "reward"}),
                id="social_self_reward_demo_reward",
            ),
            pytest.param(
                "SocialRlSelfRewardDemoMixtureKernel",
                frozenset({"action", "reward"}),
                id="social_self_reward_demo_mixture",
            ),
            pytest.param(
                "SocialRlDemoMixtureKernel",
                frozenset({"action", "reward"}),
                id="social_demo_mixture",
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
            "SocialRlSelfRewardDemoRewardKernel",
            "SocialRlSelfRewardDemoMixtureKernel",
            "SocialRlDemoMixtureKernel",
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
        ],
    )
    def test_asocial_kernels_unset_requires_social(self, kernel_cls: str) -> None:
        """Asocial kernels must have requires_social=False."""
        import comp_model.models.kernels as kernels_mod

        cls = getattr(kernels_mod, kernel_cls)
        spec = cls.spec()
        assert spec.requires_social is False
