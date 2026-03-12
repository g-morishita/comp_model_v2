"""Concrete simulation and fitting example for an asocial bandit task."""

from __future__ import annotations

from dataclasses import dataclass

from comp_model.data import EventPhase, SubjectData, validate_subject
from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import HierarchyStructure, InferenceConfig, fit
from comp_model.inference.mle import MleFitResult, MleOptimizerConfig
from comp_model.models.kernels.asocial_q_learning import AsocialQLearningKernel
from comp_model.models.kernels.transforms import get_transform
from comp_model.runtime import SimulationConfig, simulate_subject
from comp_model.tasks import ASOCIAL_BANDIT_SCHEMA, BlockSpec, TaskSpec

DEFAULT_REWARD_PROBS = (0.8, 0.2)


@dataclass(frozen=True, slots=True)
class ExampleRunResult:
    """Outputs produced by the concrete asocial Q-learning example.

    Attributes
    ----------
    subject
        Simulated subject data generated from the task and environment.
    generating_params
        Constrained parameters used to simulate the subject.
    fit_result
        Maximum-likelihood fit recovered from the simulated data.
    """

    subject: SubjectData
    generating_params: dict[str, float]
    fit_result: MleFitResult


def build_bandit_task(*, n_trials: int, n_actions: int) -> TaskSpec:
    """Build the one-block task used by the example workflow.

    Parameters
    ----------
    n_trials
        Number of trials to include in the single task block.
    n_actions
        Number of legal bandit actions.

    Returns
    -------
    TaskSpec
        Task specification for the example bandit experiment.

    Raises
    ------
    ValueError
        Raised when the task dimensions are not strictly positive.
    """

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if n_actions <= 0:
        raise ValueError("n_actions must be positive")

    return TaskSpec(
        task_id="example-asocial-bandit",
        blocks=(
            BlockSpec(
                condition="baseline",
                n_trials=n_trials,
                schema=ASOCIAL_BANDIT_SCHEMA,
                metadata={"n_actions": n_actions},
            ),
        ),
    )


def constrained_to_raw_params(*, alpha: float, beta: float) -> dict[str, float]:
    """Convert constrained Q-learning parameters to raw optimizer space.

    Parameters
    ----------
    alpha
        Learning rate in ``(0, 1)``.
    beta
        Inverse temperature in ``(0, +inf)``.

    Returns
    -------
    dict[str, float]
        Unconstrained parameter dictionary compatible with the kernel parser.
    """

    return {
        "alpha": get_transform("sigmoid").inverse(alpha),
        "beta": get_transform("softplus").inverse(beta),
    }


def simulate_example_subject(
    *,
    n_trials: int = 80,
    reward_probs: tuple[float, ...] = DEFAULT_REWARD_PROBS,
    alpha: float = 0.35,
    beta: float = 3.0,
    seed: int = 21,
    subject_id: str = "demo_subject",
) -> tuple[SubjectData, dict[str, float]]:
    """Simulate a concrete subject for the example workflow.

    Parameters
    ----------
    n_trials
        Number of task trials for the simulated subject.
    reward_probs
        Reward probabilities for each bandit action.
    alpha
        Constrained learning-rate value used to generate behavior.
    beta
        Constrained inverse-temperature value used to generate behavior.
    seed
        Random seed for the simulation engine.
    subject_id
        Identifier assigned to the generated subject.

    Returns
    -------
    tuple[SubjectData, dict[str, float]]
        Simulated subject together with the constrained generating parameters.

    Raises
    ------
    ValueError
        Raised when fewer than two reward probabilities are provided.
    """

    if len(reward_probs) < 2:
        raise ValueError("reward_probs must define at least two actions")

    kernel = AsocialQLearningKernel()
    raw_params = constrained_to_raw_params(alpha=alpha, beta=beta)
    subject = simulate_subject(
        task=build_bandit_task(n_trials=n_trials, n_actions=len(reward_probs)),
        env=StationaryBanditEnvironment(
            n_actions=len(reward_probs),
            reward_probs=reward_probs,
        ),
        kernel=kernel,
        params=kernel.parse_params(raw_params),
        config=SimulationConfig(seed=seed),
        subject_id=subject_id,
    )
    validate_subject(subject, schema=ASOCIAL_BANDIT_SCHEMA)

    return subject, {"alpha": alpha, "beta": beta}


def fit_example_subject(
    subject: SubjectData,
    *,
    n_restarts: int = 8,
    optimizer_seed: int = 0,
    max_iter: int = 200,
) -> MleFitResult:
    """Fit the example subject with the public inference dispatcher.

    Parameters
    ----------
    subject
        Simulated subject data to fit.
    n_restarts
        Number of optimizer restarts for MLE.
    optimizer_seed
        Random seed used to generate restart points.
    max_iter
        Maximum number of optimizer iterations per restart.

    Returns
    -------
    MleFitResult
        Single-subject MLE fit result.

    Raises
    ------
    TypeError
        Raised when the dispatcher returns a non-MLE result unexpectedly.
    """

    kernel = AsocialQLearningKernel()
    result = fit(
        InferenceConfig(
            hierarchy=HierarchyStructure.SUBJECT_SHARED,
            backend="mle",
            mle_config=MleOptimizerConfig(
                n_restarts=n_restarts,
                seed=optimizer_seed,
                max_iter=max_iter,
            ),
        ),
        kernel,
        subject,
        ASOCIAL_BANDIT_SCHEMA,
    )
    if not isinstance(result, MleFitResult):
        raise TypeError("Expected an MLE fit result from the example configuration")
    return result


def run_example(
    *,
    n_trials: int = 80,
    reward_probs: tuple[float, ...] = DEFAULT_REWARD_PROBS,
    alpha: float = 0.35,
    beta: float = 3.0,
    simulation_seed: int = 21,
    optimizer_seed: int = 0,
    n_restarts: int = 8,
    max_iter: int = 200,
    subject_id: str = "demo_subject",
) -> ExampleRunResult:
    """Execute the full simulate-then-fit workflow for the example.

    Parameters
    ----------
    n_trials
        Number of task trials for the simulated subject.
    reward_probs
        Reward probabilities for each bandit action.
    alpha
        Constrained generating learning rate.
    beta
        Constrained generating inverse temperature.
    simulation_seed
        Random seed for stochastic simulation.
    optimizer_seed
        Random seed for optimizer restarts.
    n_restarts
        Number of optimizer restart points to evaluate.
    max_iter
        Maximum optimizer iterations per restart.
    subject_id
        Identifier assigned to the simulated subject.

    Returns
    -------
    ExampleRunResult
        Bundle containing the simulated data, generating parameters, and fit result.
    """

    subject, generating_params = simulate_example_subject(
        n_trials=n_trials,
        reward_probs=reward_probs,
        alpha=alpha,
        beta=beta,
        seed=simulation_seed,
        subject_id=subject_id,
    )
    fit_result = fit_example_subject(
        subject,
        n_restarts=n_restarts,
        optimizer_seed=optimizer_seed,
        max_iter=max_iter,
    )
    return ExampleRunResult(
        subject=subject,
        generating_params=generating_params,
        fit_result=fit_result,
    )


def format_example_summary(result: ExampleRunResult) -> str:
    """Format a human-readable summary of the example output.

    Parameters
    ----------
    result
        Example outputs to summarize.

    Returns
    -------
    str
        Multi-line report covering the simulation and recovered parameters.
    """

    reward_total = sum(
        float(event.payload["reward"])
        for block in result.subject.blocks
        for trial in block.trials
        for event in trial.events
        if event.phase == EventPhase.OUTCOME
    )

    recovered_params = result.fit_result.constrained_params
    return "\n".join(
        (
            f"Simulated subject: {result.subject.subject_id}",
            f"Trials: {result.fit_result.n_trials}",
            f"Total reward: {reward_total:.0f}",
            "",
            "Generating parameters",
            f"  alpha={result.generating_params['alpha']:.3f}",
            f"  beta={result.generating_params['beta']:.3f}",
            "",
            "Recovered parameters",
            f"  alpha={recovered_params['alpha']:.3f}",
            f"  beta={recovered_params['beta']:.3f}",
            "",
            "Fit summary",
            f"  log_likelihood={result.fit_result.log_likelihood:.3f}",
            f"  aic={result.fit_result.aic:.3f}",
            f"  bic={result.fit_result.bic:.3f}",
            f"  converged={result.fit_result.converged}",
        )
    )


def main() -> None:
    """Run the default example and print its summary.

    Returns
    -------
    None
        This function prints the example output summary.
    """

    print(format_example_summary(run_example()))


if __name__ == "__main__":
    main()
