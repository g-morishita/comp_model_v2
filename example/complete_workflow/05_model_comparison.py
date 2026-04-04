"""Run a model recovery study to compare candidate models."""

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
    make_flat_param_dists,
    make_kernel,
    make_mle_config,
    make_task,
)

from comp_model.recovery import (
    CandidateModelSpec,
    GeneratingModelSpec,
    ModelRecoveryConfig,
    compute_confusion_matrix,
    compute_recovery_rates,
    model_recovery_confusion_table,
    model_recovery_rate_table,
    run_model_recovery,
    save_confusion_matrix_csv,
    save_replication_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a smaller smoke-sized demo.")
    parser.add_argument(
        "--model",
        choices=all_model_ids(),
        default=DEFAULT_MODEL_ID,
        help="Focal model id to compare against its paired peer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for confusion-matrix CSV artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the model comparison example."""

    args = parse_args()
    profile = get_profile(args.model)
    peer = get_profile(profile.comparison_peer)
    settings = get_settings(quick=args.quick)
    task = make_task(settings, profile)

    result = run_model_recovery(
        ModelRecoveryConfig(
            generating_models=(
                GeneratingModelSpec(
                    name=profile.model_id,
                    kernel=make_kernel(profile),
                    param_dists=make_flat_param_dists(profile),
                ),
                GeneratingModelSpec(
                    name=peer.model_id,
                    kernel=make_kernel(peer),
                    param_dists=make_flat_param_dists(peer),
                ),
            ),
            candidate_models=(
                CandidateModelSpec(
                    name=profile.model_id,
                    kernel=make_kernel(profile),
                    inference_config=make_mle_config(settings),
                ),
                CandidateModelSpec(
                    name=peer.model_id,
                    kernel=make_kernel(peer),
                    inference_config=make_mle_config(settings),
                ),
            ),
            n_replications=settings.model_recovery_replications,
            n_subjects=settings.n_subjects,
            task=task,
            env_factory=make_env_factory(settings),
            schema=task.blocks[0].schema,
            criterion="aic",
            simulation_base_seed=settings.simulation_seed,
            max_workers=1,
            **make_demonstrator_setup(profile),
        )
    )

    matrix = compute_confusion_matrix(result)
    rates = compute_recovery_rates(result)
    model_names = [profile.model_id, peer.model_id]

    print("Confusion Matrix")
    print(model_recovery_confusion_table(matrix, model_names))

    print("\nRecovery Rates")
    print(model_recovery_rate_table(rates, result))

    replication_rows = [
        (
            rep.generating_model,
            rep.replication_index,
            rep.selected_model,
            fmt(rep.winner_score, digits=2),
            "-" if rep.delta_to_second is None else fmt(rep.delta_to_second, digits=2),
        )
        for rep in result.replications
    ]
    print("\nPer-Replication Winners")
    print(
        format_table(
            ("generating", "rep", "selected", "winner score", "delta to second"),
            replication_rows,
        )
    )

    if args.output_dir is not None:
        save_confusion_matrix_csv(
            result,
            args.output_dir / f"{profile.model_id}_model_confusion.csv",
        )
        save_replication_csv(
            result,
            args.output_dir / f"{profile.model_id}_model_recovery_replications.csv",
        )
        print(f"\nSaved CSVs to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
