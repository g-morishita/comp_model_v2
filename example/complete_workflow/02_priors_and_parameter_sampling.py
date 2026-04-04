"""Define Stan priors and sample hierarchical ground-truth parameters."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from _shared import (
    DEFAULT_MODEL_ID,
    all_model_ids,
    fmt,
    format_table,
    get_profile,
    get_settings,
    make_hierarchical_param_dists,
    make_kernel,
    make_prior_specs,
    population_truth_from_latent,
)

from comp_model.recovery import sample_true_params


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
        help="Optional directory for CSV exports of the sampled parameters.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the prior-and-parameter sampling walkthrough."""

    args = parse_args()
    profile = get_profile(args.model)
    settings = get_settings(quick=args.quick)
    kernel = make_kernel(profile)
    prior_specs = make_prior_specs(profile)

    true_table, _params_per_subject, pop_params = sample_true_params(
        make_hierarchical_param_dists(profile),
        kernel,
        settings.n_subjects,
        np.random.default_rng(settings.simulation_seed),
    )

    prior_rows = [
        (
            name,
            spec.family,
            ", ".join(f"{key}={value}" for key, value in spec.kwargs.items()),
            spec.parameterization,
        )
        for name, spec in sorted(prior_specs.items())
    ]

    population_truth = population_truth_from_latent(profile, pop_params)
    population_rows = []
    for parameter in kernel.spec().parameter_specs:
        population_rows.append(
            (
                parameter.name,
                fmt(pop_params[f"mu_{parameter.name}_z"]),
                fmt(pop_params[f"sd_{parameter.name}_z"]),
                fmt(population_truth[f"{parameter.name}_pop"]),
            )
        )

    param_names = [parameter.name for parameter in kernel.spec().parameter_specs]
    subject_rows = [
        tuple([subject_id, *[fmt(values[name]) for name in param_names]])
        for subject_id, values in sorted(true_table.items())
    ]

    print("Stan Prior Specs")
    print(format_table(("parameter", "family", "kwargs", "scale"), prior_rows))

    print("\nSampled Population Parameters")
    print(
        format_table(
            ("parameter", "mu_z", "sd_z", "constrained population mean"),
            population_rows,
        )
    )

    print("\nSampled Subject Parameters")
    print(format_table(("subject", *param_names), subject_rows))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        with (args.output_dir / f"{profile.model_id}_sampled_population_parameters.csv").open(
            "w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["parameter", "mu_z", "sd_z", "constrained_population_mean"])
            writer.writerows(population_rows)

        with (args.output_dir / f"{profile.model_id}_sampled_subject_parameters.csv").open(
            "w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["subject_id", *param_names])
            writer.writerows(subject_rows)

        print(f"\nSaved CSVs to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
