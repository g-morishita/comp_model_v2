"""Run a parameter recovery study to compare true and recovered parameters."""

from __future__ import annotations

import argparse
from pathlib import Path

from _shared import (
    DEFAULT_MODEL_ID,
    all_model_ids,
    get_profile,
    get_settings,
    make_demonstrator_setup,
    make_env_factory,
    make_flat_param_dists,
    make_kernel,
    make_mle_config,
    make_task,
)

from comp_model.recovery import (
    ParameterRecoveryConfig,
    parameter_recovery_summary,
    parameter_recovery_tables,
    run_parameter_recovery,
    save_subject_csv,
)


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
        help="Optional directory for subject-level recovery CSV artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the parameter comparison example."""

    args = parse_args()
    profile = get_profile(args.model)
    settings = get_settings(quick=args.quick)
    task = make_task(settings, profile)

    result = run_parameter_recovery(
        ParameterRecoveryConfig(
            n_replications=settings.parameter_recovery_replications,
            n_subjects=settings.n_subjects,
            param_dists=make_flat_param_dists(profile),
            task=task,
            env_factory=make_env_factory(settings),
            kernel=make_kernel(profile),
            schema=task.blocks[0].schema,
            inference_config=make_mle_config(settings),
            simulation_base_seed=settings.simulation_seed,
            max_workers=1,
            **make_demonstrator_setup(profile),
        )
    )

    print("Subject-Level Comparison")
    print(parameter_recovery_summary(result))

    print("\nRecovery Metrics")
    print(parameter_recovery_tables(result))

    if args.output_dir is not None:
        save_subject_csv(
            result,
            args.output_dir / f"{profile.model_id}_parameter_recovery_subjects.csv",
        )
        print(f"\nSaved CSVs to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
