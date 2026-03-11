"""Smoke tests for the initial project scaffold."""

from comp_model import __version__


def test_package_exposes_version() -> None:
    """Verify the top-level package is importable.

    Returns
    -------
    None
        This test only asserts that the package exposes a version string.
    """
    assert __version__ == "3.0.0"
