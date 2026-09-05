from pathlib import Path
import argparse

import pandas as pd

from classes.clustering import SignalMorphologyClusterer
from classes.cluster_analysis import ClusterAnalysis
from classes.saliency_analysis import SaliencyResultsAnalysis
from classes.windowshap import WindowSHAPComparison
from scripts.plotting.plot_saliency_figures import (
    plot_hydra_flip_and_score_bars_side_by_side,
    plot_hydra_cluster_score_drop_heatmap,
)


SEED = 42
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


def load_dataset_names():
    '''Read the full UCR-128 dataset list used for clustering and full-archive runs
    '''
    summary = pd.read_csv(DATA_DIR / "summary.csv")
    return sorted(summary["dataset"].dropna().unique())


def run_clustering_analysis(n_clusters=4, seed=SEED):
    '''Build signal-morphology clusters and the accompanying diagnostic tables
    '''
    datasets = load_dataset_names()

    clusterer = SignalMorphologyClusterer(
        datasets=datasets,
        output_dir=OUTPUT_DIR / "clustering",
        n_clusters=n_clusters,
        seed=seed,
    )
    clustered_df = clusterer.run()

    analysis = ClusterAnalysis(clusterer)
    cluster_results = analysis.run()

    ucr_types = pd.read_csv(DATA_DIR / "ucr_dataset_types.csv")
    type_metrics = analysis.compare_to_ucr_types(ucr_types)
    analysis.type_contingency(ucr_types)
    analysis.type_summary(ucr_types)

    print(f"\nClustered {len(clustered_df)} datasets into {n_clusters} clusters.")
    print("\nCluster sizes:")
    print(clustered_df["cluster"].value_counts().sort_index())
    print("\nUCR Type agreement (ARI / NMI / Purity):")
    print(type_metrics.round(3))

    return clustered_df, cluster_results, type_metrics


def cmd_cluster(args):
    ''' Main function for clustering the UCR dataset
    '''
    run_clustering_analysis(n_clusters=args.n_clusters, seed=args.seed)



def cmd_analyse_saliency(args):
    ''' Main function for analysing saliency
    '''
    cluster_csv = OUTPUT_DIR / "clustering" / "csv" / "ucr_dataset_clusters_k4.csv"

    analysis = SaliencyResultsAnalysis(
        saliency_output_dir=Path(args.saliency_output_dir),
        cluster_csv_path=cluster_csv if cluster_csv.exists() else None,
        output_dir=Path(args.analysis_dir),
    )

    results = analysis.run(fraction=args.fraction)

    print("\nFlip-rate table (%):")
    print(results["flip_table"].round(2))

    print("\nBounded relative score-drop table:")
    print(results["bounded_table"].round(3))

    print("\nHYDRA paired tests:")
    print(results["paired_tests"].to_string(index=False, float_format="{:.3e}".format))

    if "cluster_gap" in results:
        print("\nHYDRA cluster gap table:")
        print(results["cluster_gap"].round(3))

    return results




def cmd_plot_saliency(args):
    ''' Main plot for plotting saliency
    '''
    plot_hydra_flip_and_score_bars_side_by_side(analysis_dir=args.analysis_dir, output_dir=args.figure_dir)
    plot_hydra_cluster_score_drop_heatmap(analysis_dir=args.analysis_dir, output_dir=args.figure_dir)
    print(f"\nFigures written to: {Path(args.figure_dir).resolve()}")



def cmd_windowshap(args):
    ''' Main function for running windowshap comparison
    '''
    comparison = WindowSHAPComparison(
        datasets=[args.dataset],
        output_dir=Path(args.output_dir),
        n_segments=args.n_segments,
        shap_nsamples=args.shap_nsamples,
        max_samples_per_dataset=args.max_samples,
        seed=args.seed,
    )
    results = comparison.run()

    # Print results
    print("\nWindowSHAP overlap summary:")
    print(results["summary"].round(3))
    print("\nTiming summary:")
    print(results["timing"].round(3))
    return results


def build_parser():
    ''' Main function for building the arg parser
    '''
    parser = argparse.ArgumentParser(description="HYDRA saliency project entry point. See README.md for the full command list.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_cluster = subparsers.add_parser("cluster", help="Build signal-morphology clusters.")
    p_cluster.add_argument("--n-clusters", type=int, default=4)
    p_cluster.add_argument("--seed", type=int, default=SEED)
    p_cluster.set_defaults(func=cmd_cluster)

    p_analyse = subparsers.add_parser("analyse-saliency", help="Regenerate flip-rate/score-drop tables from a saliency output directory.")
    p_analyse.add_argument("--saliency-output-dir", required=True)
    p_analyse.add_argument("--analysis-dir", required=True)
    p_analyse.add_argument("--fraction", type=float, default=0.10)
    p_analyse.set_defaults(func=cmd_analyse_saliency)

    p_plot = subparsers.add_parser("plot-saliency", help="Regenerate the main saliency figures.")
    p_plot.add_argument("--analysis-dir", required=True)
    p_plot.add_argument("--figure-dir", required=True)
    p_plot.set_defaults(func=cmd_plot_saliency)

    p_windowshap = subparsers.add_parser("windowshap", help="Run a quick single-dataset HYDRA-vs-WindowSHAP validation.")
    p_windowshap.add_argument("--dataset", required=True)
    p_windowshap.add_argument("--output-dir", default="outputs/saliency/windowshap_validation")
    p_windowshap.add_argument("--n-segments", type=int, default=100)
    p_windowshap.add_argument("--shap-nsamples", type=int, default=500)
    p_windowshap.add_argument("--max-samples", type=int, default=20)
    p_windowshap.add_argument("--seed", type=int, default=SEED)
    p_windowshap.set_defaults(func=cmd_windowshap)
    return parser


def main():
    ''' Main function for building parser and handling passed args
    '''
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()