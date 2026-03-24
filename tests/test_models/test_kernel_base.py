"""Tests for kernel metadata structures."""

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
    assert spec.n_actions is None
    assert spec.state_reset_policy == "per_block"
