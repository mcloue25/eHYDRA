'''
Script to run the multi method tsCaptum comparison.

Usage examples
    Moved examples to README
'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.captum_comparison import CaptumComparison
from utils.cli import select_datasets_from_closest


def parse_args():
    ''' For handling args passed in the CLI, look at README for instructions 
    '''
    parser = argparse.ArgumentParser(description="Compare HYDRA, tsCaptum Shapley Sampling, Feature Ablation, and MrSQM.")

    # NOTE - Dataset selection
    parser.add_argument("--datasets", nargs="*", default=None, help="Explicit dataset names. Overrides --all-datasets and --datasets-per-cluster.")
    parser.add_argument("--all-datasets", action="store_true", help="Run on every dataset in --cluster-csv. Overrides --datasets-per-cluster.")
    parser.add_argument("--datasets-per-cluster", type=int, default=None, help="Take N closest-to-centroid datasets per cluster (10 → 38-dataset subset).")

    # CSVs used for dataset selection and cluster level analysis
    parser.add_argument("--closest-csv", default="outputs/clustering/csv/closest_20_datasets_per_cluster.csv", help="Pre-computed closest-to-centroid CSV used with --datasets-per-cluster.")
    parser.add_argument("--cluster-csv", default="outputs/clustering/csv/ucr_dataset_clusters_k4_with_types.csv", help="Full cluster assignment CSV. Used with --all-datasets and for cluster analysis.")
    parser.add_argument("--output-dir", default="outputs/saliency/captum_comparison")

    # Method parameters
    parser.add_argument("--n-segments", type=int, default=20,
        help=(
            "tsCaptum segmentation granularity passed to Shapley Sampling and "
            "Feature Ablation. -1 = point-wise (slow, matches Turbé et al.); "
            "20 = 20 equal segments (fast). Default: 20."
        ),
    )
    parser.add_argument("--max-samples-per-dataset", type=int, default=20, help="Cap on correctly-classified test samples evaluated per dataset.")
    parser.add_argument("--no-mrsqm", action="store_true", help="Skip MrSQM fitting and explanation.")

    # TSHAP (tshap_centroid, tshap_zero always run; tshap_train is optional/expensive)
    parser.add_argument("--no-tshap", action="store_true", help="Skip TSHAP entirely (tshap_centroid, tshap_zero, tshap_train).")
    parser.add_argument("--tshap-train-background", action="store_true",
        help=(
            "Additionally run TSHAP with a train-sample background "
            "(tshap_train). Closest to the paper's own real-dataset setup "
            "but the most expensive condition, (cost scales with "
            "--tshap-train-background-samples). Recommended: run on the "
            "38-dataset subset first before enabling on the full 128."
        ),
    )
    parser.add_argument("--tshap-train-background-samples", type=int, default=20, help="Number of training samples used for the tshap_train background.")
    parser.add_argument("--tshap-window-fraction", type=float, default=0.10, help="TSHAP window_length as a fraction of series length (paper: 0.10).")
    parser.add_argument("--tshap-max-stride", type=int, default=5, help="Upper bound on TSHAP stride (paper: 5); scaled down for short series.")
    parser.add_argument("--no-deletion-curves", action="store_true",
        help=(
            "Skip Turbé AUCS̃_top / F1S̃ deletion curves. "
            "Use this for quick exploratory runs, deletion curves roughly "
            "double the number of model calls per sample."
        ),
    )
    parser.add_argument("--deletion-n-steps", type=int, default=20, help="Number of equally-spaced fractions on the deletion curve.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
        help=(
            "Bypass the per-dataset skip-cache and recompute even if "
            "captum_{dataset}_pairwise.csv / _deletion.csv already exist. "
            "Use this whenever you've changed methods or flags (e.g. added "
            "TSHAP, toggled deletion curves) and are rerunning a dataset "
            "you already tested — otherwise the stale cached CSVs are "
            "silently reloaded instead of recomputed."
        ),
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device string ('cpu', 'cuda', 'mps'). Default: cpu.")
    return parser.parse_args()


def main():
    ''' Main function for running the comparison pipeline
    '''
    args = parse_args()
    # Resolve dataset list explicit list takes highest priority
    if args.datasets:
        datasets = args.datasets
    elif args.all_datasets:
        # Load every dataset from the cluster CSV
        clusters = pd.read_csv(PROJECT_ROOT / args.cluster_csv)
        datasets = clusters["dataset"].dropna().unique().tolist()
        print(f"Loaded {len(datasets)} datasets from {args.cluster_csv}")
    else:
        # Default is set to stratified closest to centroid selection
        datasets = select_datasets_from_closest(PROJECT_ROOT / args.closest_csv, datasets_per_cluster=args.datasets_per_cluster)

    print(f"Datasets: {len(datasets)}")
    print(f"n_segments: {args.n_segments}  (-1 = point-wise)")
    print(f"MrSQM included: {not args.no_mrsqm}")
    print(f"Deletion curves: {not args.no_deletion_curves}")
    print(f"Max samples/ds: {args.max_samples_per_dataset}")
    print(f"TSHAP included: {not args.no_tshap}")
    if not args.no_tshap:
        print(f"TSHAP train background: {args.tshap_train_background} " f"(n={args.tshap_train_background_samples})")
        print(f"TSHAP window_fraction={args.tshap_window_fraction}, " f"max_stride={args.tshap_max_stride}")

    # NOTE - Running the Captum comparison
    comparison = CaptumComparison(
        datasets=datasets,
        output_dir=PROJECT_ROOT / args.output_dir,
        n_segments=args.n_segments,
        max_samples_per_dataset=args.max_samples_per_dataset,
        include_mrsqm=not args.no_mrsqm,
        compute_deletion_curves=not args.no_deletion_curves,
        deletion_n_steps=args.deletion_n_steps,
        include_tshap=not args.no_tshap,
        include_tshap_train_background=args.tshap_train_background,
        tshap_train_background_samples=args.tshap_train_background_samples,
        tshap_window_fraction=args.tshap_window_fraction,
        tshap_max_stride=args.tshap_max_stride,
        force=args.force,
        seed=args.seed,
        device=args.device,
    )
    results = comparison.run()

    # NOTE - Attach cluster labels and print the key cross-cluster breakdown
    cluster_csv = PROJECT_ROOT / args.cluster_csv
    if cluster_csv.exists():
        cluster_results = comparison.cluster_level_analysis(cluster_csv)

        print("\nCluster overlap: HYDRA vs Shapley Sampling @ 10% masking:")
        mask = (
            (cluster_results["cluster_overlap"]["method_a"] == "hydra")
            & (cluster_results["cluster_overlap"]["method_b"] == "shapley_sampling")
            & (cluster_results["cluster_overlap"]["fraction"] == 0.10)
        )
        print(cluster_results["cluster_overlap"][mask].to_string(index=False))
        print("\nCluster deletion curve summary:")
        print(cluster_results["cluster_deletion"].to_string(index=False))
    else:
        print(f"\n[FileNotFound] Cluster CSV not found at {cluster_csv}. Skipping cluster analysis.", "Have you run the cluster generation script yet?")

    print(f"\nDone. Outputs saved to: {PROJECT_ROOT / args.output_dir}")


if __name__ == "__main__":
    main()