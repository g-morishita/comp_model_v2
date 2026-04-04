"""Inspect the core pieces of a comp_model workflow and simulate one subject."""

from __future__ import annotations

import argparse
from pathlib import Path

from _shared import (
    DEFAULT_MODEL_ID,
    all_model_ids,
    fmt,
    format_table,
    get_profile,
    get_settings,
    make_demonstrator_setup,
    make_env_factory,
    make_kernel,
    make_manual_params,
    make_task,
    preview_subject_trials,
    save_dataset_if_requested,
    summarize_parameter_specs,
)

from comp_model.data import Dataset
from comp_model.runtime import SimulationConfig, simulate_subject


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a smaller smoke-sized demo.")
    parser.add_argument(
        "--model",
        choices=all_model_ids(),
        default=DEFAULT_MODEL_ID,
        help="Model id to demonstrate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for a CSV export of the simulated subject.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the introductory model-and-task walkthrough."""

    args = parse_args()
    profile = get_profile(args.model)
    settings = get_settings(quick=args.quick)
    task = make_task(settings, profile)
    kernel = make_kernel(profile)

    subject = simulate_subject(
        task=task,
        env=make_env_factory(settings)(),
        kernel=kernel,
        params=make_manual_params(profile),
        config=SimulationConfig(seed=settings.simulation_seed),
        subject_id="example_subject",
        **make_demonstrator_setup(profile),
    )

    dataset_path = save_dataset_if_requested(
        Dataset(subjects=(subject,)),
        schema=profile.schema,
        output_dir=args.output_dir,
        filename=f"{profile.model_id}_intro_subject.csv",
    )

    print("Task")
    print(
        format_table(
            ("task_id", "condition", "schema_id", "requires_social", "n_trials", "reward_probs"),
            [
                (
                    task.task_id,
                    "learning",
                    profile.schema.schema_id,
                    str(kernel.spec().requires_social),
                    str(settings.n_trials),
                    str(settings.reward_probs),
                )
            ],
        )
    )

    print("\nKernel Parameters")
    print(
        format_table(
            ("name", "transform", "bounds", "description"),
            summarize_parameter_specs(kernel),
        )
    )

    param_names = [parameter.name for parameter in kernel.spec().parameter_specs]
    print("\nManual Subject Parameters")
    print(
        format_table(
            tuple(param_names),
            [[fmt(profile.manual_values[name]) for name in param_names]],
        )
    )

    print("\nFirst Simulated Trials")
    print(
        format_table(
            ("block", "trial", "choice", "reward"),
            preview_subject_trials(subject, schema=profile.schema),
        )
    )

    if dataset_path is not None:
        print(f"\nSaved CSV: {dataset_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
