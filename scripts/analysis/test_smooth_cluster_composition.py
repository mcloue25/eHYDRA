'''
Tests the Smooth-cluster subset-vs-full-archive magnitude where every method's mean AUCS̃_top 
drops between subset clkuster and the full 37-dataset Smooth cluster

Usage:
python scripts/analysis/test_smooth_cluster_composition.py \
    --full128-dir outputs/saliency/captum_comparison_tshap_full128 \
    --cluster-csv outputs/clustering/csv/ucr_dataset_clusters_k4_with_types.csv \
    --closest-csv outputs/clustering/csv/closest_20_datasets_per_cluster.csv

'''

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np

SMOOTH_CLUSTER_NAME = "Smooth / low-complexity"


def plot_composition(smooth: pd.DataFrame, subset_datasets: list[str], output_fig: Path):
    '''Scatter of distance_to_centroid vs. mean AUCS̃_top for the Smooth cluster, coloured by subset membership
    '''
    in_subset = smooth["dataset"].isin(subset_datasets)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(
        smooth.loc[~in_subset, "distance_to_centroid"],
        smooth.loc[~in_subset, "mean_aucs_top_all_methods"],
        color="#4C72B0", label=f"Not in subset run (n={(~in_subset).sum()})",
        edgecolor="black", linewidth=0.4, s=50, zorder=3,
    )
    ax.scatter(
        smooth.loc[in_subset, "distance_to_centroid"],
        smooth.loc[in_subset, "mean_aucs_top_all_methods"],
        color="#DD8452", label=f"In subset run (n={in_subset.sum()})",
        edgecolor="black", linewidth=0.4, s=70, zorder=4, marker="D",
    )

    # NOTE - trend line + Spearman annotation. Theil-Sen (not OLS) is used for the line 
    rho, p = stats.spearmanr(smooth["distance_to_centroid"], smooth["mean_aucs_top_all_methods"])
    slope, intercept, _, _ = stats.theilslopes(smooth["mean_aucs_top_all_methods"], smooth["distance_to_centroid"])
    x_line = np.linspace(smooth["distance_to_centroid"].min(), smooth["distance_to_centroid"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="gray", linestyle="--", linewidth=1.5, zorder=2, label=f"Theil-Sen trend (Spearman ρ={rho:.3f}, p={p:.3f})")

    # Formatting
    ax.set_xlabel("Distance to cluster centroid")
    ax.set_ylabel("Mean AUCS̃$_{top}$ (across 7 methods)")
    ax.set_title("Smooth/low-complexity cluster: centroid distance vs. faithfulness")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_fig, dpi=200)
    plt.close(fig)
    print(f"Figure saved to {output_fig}")


def select_subset_datasets(closest_csv: Path, cluster_name: str, n: int = 10) -> list[str]:
    ''' Reproduces run_captum_comparison.py's select_datasets_from_closest() logic for a single cluster
    '''
    closest = pd.read_csv(closest_csv)
    cluster_rows = closest[closest["cluster_name"] == cluster_name] if "cluster_name" in closest.columns else closest

    sort_cols = ["cluster"] if "cluster" in cluster_rows.columns else []
    if "rank_within_cluster" in cluster_rows.columns:
        sort_cols.append("rank_within_cluster")
    elif "distance_to_centroid" in cluster_rows.columns:
        sort_cols.append("distance_to_centroid")

    selected = cluster_rows.sort_values(sort_cols).head(n) if sort_cols else cluster_rows.head(n)
    return selected["dataset"].dropna().unique().tolist()



def parse_args() -> argparse.Namespace:
    ''' Init arg parser
    '''
    parser = argparse.ArgumentParser(description="Re-test the Smooth-cluster subset-vs-full swing.")
    parser.add_argument("--full128-dir", type=Path, required=True)
    parser.add_argument("--cluster-csv", type=Path, required=True)
    parser.add_argument("--closest-csv", type=Path, required=True)
    parser.add_argument("--cluster-name", default=SMOOTH_CLUSTER_NAME)
    parser.add_argument("--output-fig", type=Path, default=None, help="Optional path to save the distance-vs-faithfulness scatter figure.")
    return parser.parse_args()


def main() -> None:
    ''' Main function for running the compositional bias test
    '''
    args = parse_args()

    deletion = pd.read_csv(args.full128_dir / "captum_deletion_samples.csv")
    clusters = pd.read_csv(args.cluster_csv)

    # Per-dataset, per-method mean AUCS̃_top across all evaluated samples
    per_dataset_method = (deletion.groupby(["dataset", "method"])["aucs_top"].mean().reset_index())
    # Also the mean across all seven methods, per dataset
    per_dataset_mean = (per_dataset_method.groupby("dataset")["aucs_top"].mean().reset_index().rename(columns={"aucs_top": "mean_aucs_top_all_methods"}))

    merged = per_dataset_mean.merge(clusters[["dataset", "cluster_name", "distance_to_centroid"]], on="dataset", how="left")
    smooth = merged[merged["cluster_name"] == args.cluster_name].copy()
    print(f"Smooth cluster: {len(smooth)} datasets found in {args.cluster_csv.name}\n")

    # NOTE - Test Spearman correlation of distance_to_centroid vs AUCS̃_top
    print()
    print("Spearman(distance_to_centroid, AUCS̃_top), Smooth cluster only:")
    print()

    def spearman_report(label: str, df: pd.DataFrame, value_col: str):
        rho, p = stats.spearmanr(df["distance_to_centroid"], df[value_col])
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"{label:30s} rho={rho:+.3f}  p={p:.4f}  ({sig}, n={len(df)})")

    spearman_report("Mean across 7 methods", smooth, "mean_aucs_top_all_methods")

    for method in ["hydra", "shapley_sampling"]:
        method_df = per_dataset_method[per_dataset_method["method"] == method].merge(clusters[["dataset", "cluster_name", "distance_to_centroid"]], on="dataset")
        method_df = method_df[method_df["cluster_name"] == args.cluster_name]
        spearman_report(method, method_df, "aucs_top")

    print("Negative slope = compositional bias (datasets farther from centroid score lower & the subset really was atypical) otherwise result is because of staistical power")



    # NOTE - subset 10 vs remaining-27, Mann-Whitney U
    print()
    print("\n\n\nTesting: Mann-Whitney U, subset 10 vs remaining 27 (mean AUCS̃_top, all methods):")
    print()
    subset_datasets = select_subset_datasets(args.closest_csv, args.cluster_name, n=10)
    subset_datasets = [d for d in subset_datasets if d in smooth["dataset"].values]
    print(f"  Identified {len(subset_datasets)} subset datasets: {subset_datasets}\n")

    in_subset = smooth[smooth["dataset"].isin(subset_datasets)]
    not_in_subset = smooth[~smooth["dataset"].isin(subset_datasets)]

    print(f"Subset (n={len(in_subset)}):     mean={in_subset['mean_aucs_top_all_methods'].mean():.4f}"
          f"median={in_subset['mean_aucs_top_all_methods'].median():.4f}")
    print(f"Remaining (n={len(not_in_subset)}):  mean={not_in_subset['mean_aucs_top_all_methods'].mean():.4f}"
          f"median={not_in_subset['mean_aucs_top_all_methods'].median():.4f}")


    if len(in_subset) >= 3 and len(not_in_subset) >= 3:
        u_stat, p_val = stats.mannwhitneyu(in_subset["mean_aucs_top_all_methods"], not_in_subset["mean_aucs_top_all_methods"], alternative="greater")
        sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"
        print(f"\n  Mann-Whitney U (subset > remaining): U={u_stat:.1f}  p={p_val:.4f}  ({sig})")
        print("Significant result = the 10 datasets used were genuinely, measurably higher-scoring than the rest of the Smooth cluster, otherwise driven by variance or statistical power")
    else:
        print("\n  [WARN] Too few datasets in one group for a meaningful Mann-Whitney U test.")

    if args.output_fig is not None:
        args.output_fig.parent.mkdir(parents=True, exist_ok=True)
        plot_composition(smooth, subset_datasets, args.output_fig)


if __name__ == "__main__":
    main()
