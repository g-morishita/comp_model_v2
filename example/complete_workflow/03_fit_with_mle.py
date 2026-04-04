"""Simulate a dataset and recover subject parameters with per-subject MLE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _shared import (
    DEFAULT_MODEL_ID,
    all_model_ids,
    fmt,
    format_table,
    get_profile,
    get_settings,
    make_demonstrator_setup,
    make_env_factory,
    make_flat_param_dists,
    make_kernel,
    make_mle_config,
    make_task,
    save_dataset_if_requested,
)

from comp_model.inference import fit
from comp_model.recovery import sample_true_params
from comp_model.runtime import SimulationConfig, simulate_dataset


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
        help="Optional directory for the simulated dataset CSV.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the MLE fitting example."""

    args = parse_args()
    profile = get_profile(args.model)
    settings = get_settings(quick=args.quick)
    task = make_task(settings, profile)
    kernel = make_kernel(profile)

    true_table, params_per_subject, _ = sample_true_params(
        make_flat_param_dists(profile),
        kernel,
        settings.n_subjects,
        np.random.default_rng(settings.simulation_seed),
    )

    dataset = simulate_dataset(
        task=task,
        env_factory=make_env_factory(settings),
        kernel=kernel,
        params_per_subject=params_per_subject,
        config=SimulationConfig(seed=settings.simulation_seed),
        **make_demonstrator_setup(profile),
    )

    dataset_path = save_dataset_if_requested(
        dataset,
        schema=profile.schema,
        output_dir=args.output_dir,
        filename=f"{profile.model_id}_mle_dataset.csv",
    )

    inference_config = make_mle_config(settings)
    param_names = [parameter.name for parameter in kernel.spec().parameter_specs]
    rows: list[tuple[str, ...]] = []

    for subject in dataset.subjects:
        result = fit(inference_config, kernel, subject, task.blocks[0].schema)
        row: list[str] = [subject.subject_id]
        for name in param_names:
            row.extend(
                [
                    fmt(true_table[subject.subject_id][name]),
                    fmt(result.constrained_params[name]),
                ]
            )
        row.extend(
            [
                fmt(result.log_likelihood, digits=2),
                fmt(result.aic, digits=2),
                fmt(result.bic, digits=2),
            ]
        )
        rows.append(tuple(row))

    headers = ["subject"]
    for name in param_names:
        headers.extend([f"true {name}", f"fit {name}"])
    headers.extend(["loglik", "AIC", "BIC"])

    print("Subject-Level Parameter Comparison")
    print(format_table(tuple(headers), rows))

    if dataset_path is not None:
        print(f"\nSaved CSV: {dataset_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
