"""Tests for the shared-plus-delta condition layout."""

from comp_model.models.condition.shared_delta import SharedDeltaLayout
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel


def test_shared_delta_layout_reconstructs_baseline_and_delta_conditions() -> None:
    """Ensure shared and delta parameters reconstruct per condition correctly.

    Returns
    -------
    None
        This test asserts layout reconstruction behavior.
    """

    layout = SharedDeltaLayout(
        kernel_spec=AsocialQLearningKernel.spec(),
        conditions=("baseline", "social"),
        baseline_condition="baseline",
    )

    reconstructed = layout.reconstruct_all(
        {
            "alpha__shared_z": 0.1,
            "beta__shared_z": 1.2,
            "alpha__delta_z__social": 0.3,
            "beta__delta_z__social": -0.5,
        }
    )

    assert reconstructed["baseline"] == {"alpha": 0.1, "beta": 1.2}
    assert reconstructed["social"] == {"alpha": 0.4, "beta": 0.7}
