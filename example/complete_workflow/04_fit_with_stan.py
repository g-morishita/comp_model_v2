"""Simulate a hierarchical dataset, set priors, and fit with Stan.

Requires: `pip install .[stan]` and a working CmdStan installation.
"""

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
    make_adapter,
    make_demonstrator_setup,
    make_env_factory,
    make_hierarchical_param_dists,
    make_kernel,
    make_prior_specs,
    make_stan_config,
    make_task,
    population_truth_from_latent,
    require_stan,
    save_dataset_if_requested,
)

from comp_model.inference import HierarchyStructure, fit
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
    """Run the hierarchical Stan fitting example."""

    require_stan()

    args = parse_args()
    profile = get_profile(args.model)
    settings = get_settings(quick=args.quick)
    task = make_task(settings, profile)
    kernel = make_kernel(profile)
    prior_specs = make_prior_specs(profile)

    true_table, params_per_subject, pop_params = sample_true_params(
        make_hierarchical_param_dists(profile),
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
        filename=f"{profile.model_id}_stan_dataset.csv",
    )

    adapter = make_adapter(profile)
    result = fit(
        make_stan_config(settings, prior_specs=prior_specs),
        kernel,
        dataset,
        task.blocks[0].schema,
        adapter=adapter,
    )

    population_truth = population_truth_from_latent(profile, pop_params)
    population_param_names = [
        name
        for name in adapter.population_param_names(HierarchyStructure.STUDY_SUBJECT)
        if name.endswith("_pop") and name in population_truth
    ]
    population_rows = [
        (
            name,
            fmt(population_truth[name]),
            fmt(float(np.mean(result.posterior_samples[name]))),
        )
        for name in population_param_names
    ]

    subject_param_names = adapter.subject_param_names()
    subject_rows = []
    for index, subject in enumerate(dataset.subjects):
        row = [subject.subject_id]
        for name in subject_param_names:
            posterior = np.asarray(result.posterior_samples[name])
            row.extend(
                [
                    fmt(true_table[subject.subject_id][name]),
                    fmt(float(np.mean(posterior[:, index]))),
                ]
            )
        subject_rows.append(tuple(row))

    print("Population Posterior Means")
    print(format_table(("parameter", "true", "posterior mean"), population_rows))

    subject_headers = ["subject"]
    for name in subject_param_names:
        subject_headers.extend([f"true {name}", f"post {name}"])

    print("\nSubject Posterior Means")
    print(format_table(tuple(subject_headers), subject_rows))

    print(f"\nStan divergences: {result.diagnostics['n_divergences']}")

    if dataset_path is not None:
        print(f"Saved CSV: {dataset_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
