"""Smoke tests for the structured example workflow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "example" / "complete_workflow"
DEFAULT_MODEL_ID = "asocial_q_learning"
MODEL_IDS = (
    "asocial_q_learning",
    "asocial_rl_asymmetric",
    "asocial_rl_sticky",
    "social_rl_demo_reward",
    "social_rl_demo_reward_sticky",
    "social_rl_demo_mixture",
    "social_rl_demo_mixture_sticky",
    "social_rl_self_reward_demo_reward",
    "social_rl_self_reward_demo_reward_sticky",
    "social_rl_self_reward_demo_action_mixture",
    "social_rl_self_reward_demo_action_mixture_sticky",
    "social_rl_self_reward_demo_mixture",
    "social_rl_self_reward_demo_mixture_sticky",
)

NON_STAN_SCRIPTS = (
    "01_model_and_task.py",
    "02_priors_and_parameter_sampling.py",
    "05_model_comparison.py",
    "06_parameter_comparison.py",
)


@pytest.mark.parametrize("script_name", NON_STAN_SCRIPTS)
def test_complete_workflow_script_smoke(script_name: str, tmp_path: Path) -> None:
    """Non-Stan example scripts should run end to end in quick mode."""

    script = EXAMPLE_DIR / script_name
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--quick",
            "--model",
            DEFAULT_MODEL_ID,
            "--output-dir",
            str(tmp_path / script.stem),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_complete_workflow_mle_for_each_model(model_id: str, tmp_path: Path) -> None:
    """Each model should complete the quick MLE workflow."""

    script = EXAMPLE_DIR / "03_fit_with_mle.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--quick",
            "--model",
            model_id,
            "--output-dir",
            str(tmp_path / model_id),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.stan
def test_complete_workflow_stan_script_smoke(tmp_path: Path) -> None:
    """The Stan example should run in quick mode when CmdStan is available."""

    cmdstanpy = pytest.importorskip("cmdstanpy")
    try:
        cmdstan_path = Path(cmdstanpy.cmdstan_path())
    except (RuntimeError, ValueError):
        pytest.skip("CmdStan is not configured")
    if not (cmdstan_path / "bin" / "diagnose").exists():
        pytest.skip("CmdStan diagnose executable is missing")

    script = EXAMPLE_DIR / "04_fit_with_stan.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--quick",
            "--model",
            DEFAULT_MODEL_ID,
            "--output-dir",
            str(tmp_path / "stan"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
