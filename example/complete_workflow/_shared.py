"""Shared helpers for the complete workflow examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scipy import stats

from comp_model.data import EventPhase, replay_trial_steps
from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import HierarchyStructure, InferenceConfig, PriorSpec
from comp_model.inference.bayes.stan import (
    AsocialQLearningStanAdapter,
    AsocialRlAsymmetricStanAdapter,
    AsocialRlStickyStanAdapter,
    SocialRlDemoMixtureStanAdapter,
    SocialRlDemoMixtureStickyStanAdapter,
    SocialRlDemoRewardStanAdapter,
    SocialRlDemoRewardStickyStanAdapter,
    SocialRlSelfRewardDemoActionMixtureStanAdapter,
    SocialRlSelfRewardDemoActionMixtureStickyStanAdapter,
    SocialRlSelfRewardDemoMixtureStanAdapter,
    SocialRlSelfRewardDemoMixtureStickyStanAdapter,
    SocialRlSelfRewardDemoRewardStanAdapter,
    SocialRlSelfRewardDemoRewardStickyStanAdapter,
    StanFitConfig,
)
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.io import save_dataset_to_csv
from comp_model.models.kernels import (
    AsocialQLearningKernel,
    AsocialRlAsymmetricKernel,
    AsocialRlStickyKernel,
    SocialRlDemoMixtureKernel,
    SocialRlDemoMixtureStickyKernel,
    SocialRlDemoRewardKernel,
    SocialRlDemoRewardStickyKernel,
    SocialRlSelfRewardDemoActionMixtureKernel,
    SocialRlSelfRewardDemoActionMixtureStickyKernel,
    SocialRlSelfRewardDemoMixtureKernel,
    SocialRlSelfRewardDemoMixtureStickyKernel,
    SocialRlSelfRewardDemoRewardKernel,
    SocialRlSelfRewardDemoRewardStickyKernel,
    get_transform,
)
from comp_model.recovery import FlatParamDist, HierarchicalParamDist
from comp_model.tasks import (
    ASOCIAL_BANDIT_SCHEMA,
    SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
    SOCIAL_PRE_CHOICE_SCHEMA,
    BlockSpec,
    TaskSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from comp_model.data import Dataset, SubjectData
    from comp_model.inference.bayes.stan.adapters import StanAdapter
    from comp_model.models.kernels.base import ModelKernel, ParameterSpec
    from comp_model.tasks import TrialSchema


@dataclass(frozen=True, slots=True)
class ExampleSettings:
    """Shared runtime sizes for the workflow examples."""

    n_actions: int
    n_trials: int
    n_subjects: int
    reward_probs: tuple[float, ...]
    simulation_seed: int
    mle_restarts: int
    mle_max_iter: int
    stan_warmup: int
    stan_samples: int
    stan_chains: int
    model_recovery_replications: int
    parameter_recovery_replications: int


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """One model family exposed through the workflow examples."""

    model_id: str
    kernel_factory: Callable[[], ModelKernel[Any, Any]]
    adapter_factory: Callable[[], StanAdapter]
    schema: TrialSchema
    comparison_peer: str
    manual_values: Mapping[str, float]


DEFAULT_MODEL_ID = "asocial_q_learning"
_DEMONSTRATOR_VALUES = {"alpha": 0.30, "beta": 3.50}


def get_settings(*, quick: bool) -> ExampleSettings:
    """Return either the default or smoke-sized example settings."""

    if quick:
        return ExampleSettings(
            n_actions=2,
            n_trials=24,
            n_subjects=4,
            reward_probs=(0.75, 0.25),
            simulation_seed=7,
            mle_restarts=3,
            mle_max_iter=80,
            stan_warmup=150,
            stan_samples=150,
            stan_chains=2,
            model_recovery_replications=1,
            parameter_recovery_replications=2,
        )

    return ExampleSettings(
        n_actions=2,
        n_trials=80,
        n_subjects=8,
        reward_probs=(0.75, 0.25),
        simulation_seed=7,
        mle_restarts=8,
        mle_max_iter=300,
        stan_warmup=400,
        stan_samples=400,
        stan_chains=4,
        model_recovery_replications=4,
        parameter_recovery_replications=4,
    )


def _default_manual_value(parameter: ParameterSpec) -> float:
    """Choose one readable constrained value for a parameter."""

    if parameter.name == "alpha_pos":
        return 0.55
    if parameter.name == "alpha_neg":
        return 0.15
    if parameter.name == "alpha_self":
        return 0.30
    if parameter.name == "alpha_other":
        return 0.45
    if parameter.name == "alpha_other_outcome":
        return 0.45
    if parameter.name == "alpha_other_action":
        return 0.35
    if parameter.name == "w_imitation":
        return 0.65
    if parameter.transform_id == "sigmoid":
        return 0.35
    if parameter.transform_id == "softplus":
        return 2.50
    if "stickiness" in parameter.name:
        return 1.00
    if "bias" in parameter.name:
        return 1.00
    return 0.50


def _build_manual_values(kernel: ModelKernel[Any, Any]) -> dict[str, float]:
    """Build manual constrained values for every parameter of a kernel."""

    return {
        parameter.name: _default_manual_value(parameter)
        for parameter in kernel.spec().parameter_specs
    }


def _make_profile(
    kernel_factory: Callable[[], ModelKernel[Any, Any]],
    adapter_factory: Callable[[], StanAdapter],
    schema: TrialSchema,
    comparison_peer: str,
) -> ModelProfile:
    """Create one model workflow profile."""

    kernel = kernel_factory()
    return ModelProfile(
        model_id=kernel.spec().model_id,
        kernel_factory=kernel_factory,
        adapter_factory=adapter_factory,
        schema=schema,
        comparison_peer=comparison_peer,
        manual_values=_build_manual_values(kernel),
    )


_PROFILE_LIST = (
    _make_profile(
        AsocialQLearningKernel,
        AsocialQLearningStanAdapter,
        ASOCIAL_BANDIT_SCHEMA,
        "asocial_rl_sticky",
    ),
    _make_profile(
        AsocialRlAsymmetricKernel,
        AsocialRlAsymmetricStanAdapter,
        ASOCIAL_BANDIT_SCHEMA,
        "asocial_q_learning",
    ),
    _make_profile(
        AsocialRlStickyKernel,
        AsocialRlStickyStanAdapter,
        ASOCIAL_BANDIT_SCHEMA,
        "asocial_q_learning",
    ),
    _make_profile(
        SocialRlDemoRewardKernel,
        SocialRlDemoRewardStanAdapter,
        SOCIAL_PRE_CHOICE_SCHEMA,
        "social_rl_demo_reward_sticky",
    ),
    _make_profile(
        SocialRlDemoRewardStickyKernel,
        SocialRlDemoRewardStickyStanAdapter,
        SOCIAL_PRE_CHOICE_SCHEMA,
        "social_rl_demo_reward",
    ),
    _make_profile(
        SocialRlDemoMixtureKernel,
        SocialRlDemoMixtureStanAdapter,
        SOCIAL_PRE_CHOICE_SCHEMA,
        "social_rl_demo_mixture_sticky",
    ),
    _make_profile(
        SocialRlDemoMixtureStickyKernel,
        SocialRlDemoMixtureStickyStanAdapter,
        SOCIAL_PRE_CHOICE_SCHEMA,
        "social_rl_demo_mixture",
    ),
    _make_profile(
        SocialRlSelfRewardDemoRewardKernel,
        SocialRlSelfRewardDemoRewardStanAdapter,
        SOCIAL_PRE_CHOICE_SCHEMA,
        "social_rl_self_reward_demo_reward_sticky",
    ),
    _make_profile(
        SocialRlSelfRewardDemoRewardStickyKernel,
        SocialRlSelfRewardDemoRewardStickyStanAdapter,
        SOCIAL_PRE_CHOICE_SCHEMA,
        "social_rl_self_reward_demo_reward",
    ),
    _make_profile(
        SocialRlSelfRewardDemoActionMixtureKernel,
        SocialRlSelfRewardDemoActionMixtureStanAdapter,
        SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
        "social_rl_self_reward_demo_action_mixture_sticky",
    ),
    _make_profile(
        SocialRlSelfRewardDemoActionMixtureStickyKernel,
        SocialRlSelfRewardDemoActionMixtureStickyStanAdapter,
        SOCIAL_PRE_CHOICE_ACTION_ONLY_SCHEMA,
        "social_rl_self_reward_demo_action_mixture",
    ),
    _make_profile(
        SocialRlSelfRewardDemoMixtureKernel,
        SocialRlSelfRewardDemoMixtureStanAdapter,
        SOCIAL_PRE_CHOICE_SCHEMA,
        "social_rl_self_reward_demo_mixture_sticky",
    ),
    _make_profile(
        SocialRlSelfRewardDemoMixtureStickyKernel,
        SocialRlSelfRewardDemoMixtureStickyStanAdapter,
        SOCIAL_PRE_CHOICE_SCHEMA,
        "social_rl_self_reward_demo_mixture",
    ),
)

MODEL_PROFILES = {profile.model_id: profile for profile in _PROFILE_LIST}


def all_model_ids() -> tuple[str, ...]:
    """Return the supported model ids in stable display order."""

    return tuple(profile.model_id for profile in _PROFILE_LIST)


def get_profile(model_id: str) -> ModelProfile:
    """Return the workflow profile for one model id."""

    if model_id not in MODEL_PROFILES:
        raise ValueError(f"Unknown model {model_id!r}. Available: {all_model_ids()}")
    return MODEL_PROFILES[model_id]


def make_task(settings: ExampleSettings, profile: ModelProfile) -> TaskSpec:
    """Build the shared bandit task for one workflow profile."""

    return TaskSpec(
        task_id=f"example_{profile.model_id}",
        blocks=(
            BlockSpec(
                condition="learning",
                n_trials=settings.n_trials,
                schema=profile.schema,
                metadata={"n_actions": settings.n_actions},
            ),
        ),
    )


def make_env_factory(
    settings: ExampleSettings,
) -> Callable[[], StationaryBanditEnvironment]:
    """Return the shared stationary bandit environment factory."""

    return lambda: StationaryBanditEnvironment(
        n_actions=settings.n_actions,
        reward_probs=settings.reward_probs,
    )


def make_kernel(profile: ModelProfile) -> ModelKernel[Any, Any]:
    """Instantiate the focal kernel for a workflow profile."""

    return profile.kernel_factory()


def _manual_values_to_params(
    kernel: ModelKernel[Any, Any],
    manual_values: Mapping[str, float],
) -> Any:
    """Convert constrained manual values into the parsed params object."""

    raw: dict[str, float] = {}
    for parameter in kernel.spec().parameter_specs:
        value = manual_values[parameter.name]
        raw[parameter.name] = get_transform(parameter.transform_id).inverse(value)
    return kernel.parse_params(raw)


def make_manual_params(profile: ModelProfile) -> Any:
    """Return one readable hand-picked parameter object for the focal model."""

    kernel = make_kernel(profile)
    return _manual_values_to_params(kernel, profile.manual_values)


def make_demonstrator_setup(profile: ModelProfile) -> dict[str, Any]:
    """Return demonstrator arguments when the focal model is social."""

    kernel = make_kernel(profile)
    if not kernel.spec().requires_social:
        return {}

    demonstrator_kernel = AsocialQLearningKernel()
    demonstrator_params = _manual_values_to_params(demonstrator_kernel, _DEMONSTRATOR_VALUES)
    return {
        "demonstrator_kernel": demonstrator_kernel,
        "demonstrator_params": demonstrator_params,
    }


def _flat_distribution_for_parameter(
    parameter: ParameterSpec,
    manual_value: float,
):
    """Build one flat constrained-scale sampling distribution."""

    if parameter.transform_id == "sigmoid":
        lower = max(0.05, manual_value - 0.20)
        upper = min(0.95, manual_value + 0.20)
        return stats.uniform(loc=lower, scale=max(0.05, upper - lower))

    if parameter.transform_id == "softplus":
        lower = max(0.25, manual_value - 1.00)
        upper = manual_value + 1.50
        return stats.uniform(loc=lower, scale=max(0.25, upper - lower))

    return stats.norm(loc=manual_value, scale=0.35)


def make_flat_param_dists(profile: ModelProfile) -> tuple[FlatParamDist, ...]:
    """Return flat per-subject sampling distributions for the focal model."""

    kernel = make_kernel(profile)
    return tuple(
        FlatParamDist(
            parameter.name,
            _flat_distribution_for_parameter(parameter, profile.manual_values[parameter.name]),
        )
        for parameter in kernel.spec().parameter_specs
    )


def _hierarchical_center(parameter: ParameterSpec, manual_value: float) -> float:
    """Return the unconstrained population center for a parameter."""

    return get_transform(parameter.transform_id).inverse(manual_value)


def _hierarchical_mu_scale(parameter: ParameterSpec) -> float:
    """Choose a mean-prior width for a parameter family."""

    if parameter.transform_id == "softplus":
        return 0.60
    if parameter.transform_id == "sigmoid":
        return 0.50
    return 0.50


def _hierarchical_sd_scale(parameter: ParameterSpec) -> float:
    """Choose an SD prior width for a parameter family."""

    if parameter.transform_id == "softplus":
        return 0.30
    if parameter.transform_id == "sigmoid":
        return 0.30
    return 0.25


def make_hierarchical_param_dists(profile: ModelProfile) -> tuple[HierarchicalParamDist, ...]:
    """Return hierarchical parameter distributions for the focal model."""

    kernel = make_kernel(profile)
    return tuple(
        HierarchicalParamDist(
            parameter.name,
            mu_prior=stats.norm(
                loc=_hierarchical_center(parameter, profile.manual_values[parameter.name]),
                scale=_hierarchical_mu_scale(parameter),
            ),
            sd_prior=stats.halfnorm(scale=_hierarchical_sd_scale(parameter)),
        )
        for parameter in kernel.spec().parameter_specs
    )


def make_prior_specs(profile: ModelProfile) -> dict[str, PriorSpec]:
    """Return simple hierarchical Stan priors for the focal model."""

    kernel = make_kernel(profile)
    priors: dict[str, PriorSpec] = {}
    for parameter in kernel.spec().parameter_specs:
        priors[parameter.name] = PriorSpec(
            "normal",
            {
                "mu": _hierarchical_center(parameter, profile.manual_values[parameter.name]),
                "sigma": 1.0,
            },
        )
        priors[f"sd_{parameter.name}"] = PriorSpec(
            "normal",
            {"mu": 0.0, "sigma": 0.5},
        )
    return priors


def make_mle_config(settings: ExampleSettings) -> InferenceConfig:
    """Return the shared MLE configuration used by the examples."""

    return InferenceConfig(
        hierarchy=HierarchyStructure.SUBJECT_SHARED,
        backend="mle",
        mle_config=MleOptimizerConfig(
            n_restarts=settings.mle_restarts,
            seed=settings.simulation_seed,
            max_iter=settings.mle_max_iter,
        ),
    )


def make_stan_config(
    settings: ExampleSettings,
    *,
    prior_specs: dict[str, PriorSpec],
) -> InferenceConfig:
    """Return the shared Stan configuration used by the examples."""

    return InferenceConfig(
        hierarchy=HierarchyStructure.STUDY_SUBJECT,
        backend="stan",
        stan_config=StanFitConfig(
            n_warmup=settings.stan_warmup,
            n_samples=settings.stan_samples,
            n_chains=settings.stan_chains,
            seed=settings.simulation_seed,
            show_console=False,
        ),
        prior_specs=prior_specs,
    )


def make_adapter(profile: ModelProfile) -> StanAdapter:
    """Instantiate the Stan adapter for a workflow profile."""

    return profile.adapter_factory()


def population_truth_from_latent(
    profile: ModelProfile,
    pop_params: Mapping[str, float],
) -> dict[str, float]:
    """Convert sampled latent population parameters to the constrained scale."""

    kernel = make_kernel(profile)
    truth: dict[str, float] = {}
    for parameter in kernel.spec().parameter_specs:
        mu_key = f"mu_{parameter.name}_z"
        if mu_key not in pop_params:
            continue
        transform = get_transform(parameter.transform_id).forward
        truth[f"{parameter.name}_pop"] = transform(pop_params[mu_key])
    return truth


def format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Format a small plain-text table."""

    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    header_line = "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    separator = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in string_rows]
    return "\n".join([header_line, separator, *body])


def fmt(value: float, digits: int = 3) -> str:
    """Format a float with fixed precision."""

    return f"{value:.{digits}f}"


def save_dataset_if_requested(
    dataset: Dataset,
    *,
    schema: TrialSchema,
    output_dir: Path | None,
    filename: str,
) -> Path | None:
    """Write a dataset CSV when the caller requested artifacts."""

    if output_dir is None:
        return None

    if schema.schema_id.endswith("action_only"):
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    save_dataset_to_csv(dataset, schema=schema, path=path)
    return path


def summarize_parameter_specs(kernel: ModelKernel[Any, Any]) -> list[tuple[str, str, str, str]]:
    """Return rows describing the kernel's free parameters."""

    rows: list[tuple[str, str, str, str]] = []
    for parameter in kernel.spec().parameter_specs:
        if parameter.bounds is None:
            bounds = "None"
        else:
            lower, upper = parameter.bounds
            lower_text = "-inf" if lower is None else str(lower)
            upper_text = "+inf" if upper is None else str(upper)
            bounds = f"[{lower_text}, {upper_text}]"
        rows.append(
            (
                parameter.name,
                parameter.transform_id,
                bounds,
                parameter.description or "-",
            )
        )
    return rows


def preview_subject_trials(
    subject: SubjectData,
    *,
    schema: TrialSchema,
    limit: int = 5,
) -> list[tuple[str, str, str, str]]:
    """Return a compact preview of the first few subject trials."""

    rows: list[tuple[str, str, str, str]] = []
    for block in subject.blocks:
        for trial in block.trials:
            choice_text = "-"
            reward_text = "-"
            for phase, learner_id, view in replay_trial_steps(trial, schema):
                if learner_id != "subject":
                    continue
                if phase == EventPhase.DECISION and view.action is not None:
                    choice_text = str(view.action)
                if (
                    phase == EventPhase.UPDATE
                    and view.actor_id == view.learner_id
                    and view.reward is not None
                ):
                    reward_text = fmt(view.reward, digits=1)
            rows.append(
                (
                    str(block.block_index),
                    str(trial.trial_index),
                    choice_text,
                    reward_text,
                )
            )
            if len(rows) >= limit:
                return rows
    return rows


def require_stan() -> None:
    """Raise a friendly error when the Stan toolchain is unavailable."""

    try:
        import cmdstanpy
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in user runs
        raise SystemExit(
            "This example requires `pip install .[stan]` and a working CmdStan installation."
        ) from exc

    try:
        cmdstan_path = Path(cmdstanpy.cmdstan_path())
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - exercised in user runs
        raise SystemExit(
            "CmdStan is not configured. Install CmdStan before running the Stan example."
        ) from exc

    diagnose = cmdstan_path / "bin" / "diagnose"
    if not diagnose.exists():  # pragma: no cover - exercised in user runs
        raise SystemExit(
            "CmdStan is incomplete: missing the `diagnose` executable. "
            "Reinstall CmdStan before running the Stan example."
        )
