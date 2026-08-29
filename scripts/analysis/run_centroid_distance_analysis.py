'''
Centroid distance vs Shapley-HYDRA score drop gap analysis.

Tests whether the significance reversal between the 38-dataset stratified subset and the full 128-dataset run is a compositional effect 
    (datasets closer to their cluster centroid happen to favour eHYDRA) or simply a statistical power effect.
Usage: 
    python scripts/analysis/run_centroid_distance_analysis.py
'''

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.globals_config import CLUSTER_CSV

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.plot_config import (
    CLUSTER_COLOURS,
    CLUSTER_SHORT,
    CLUSTER_ORDER,
    METHOD_COLOURS,
    apply_thesis_style,
)

apply_thesis_style()

PAIRWISE_CSV = PROJECT_ROOT / "outputs/saliency/captum_comparison_full128/captum_pairwise_samples.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/plots/tscaptum"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_analysis(pairwise_csv: Path = PAIRWISE_CSV, cluster_csv: Path = CLUSTER_CSV, output_dir: Path = OUTPUT_DIR):
    ''' Correlates centroid distance against the Shapley-eHYDRA score drop gap overall and per cluster
    Returns:
        Saves the scatter plot & results CSV
    '''
    pairwise = pd.read_csv(pairwise_csv)
    clusters = pd.read_csv(cluster_csv)

    print(f"Datasets in pairwise CSV: {pairwise.dataset.nunique()}")
    gap = (
        pairwise[(pairwise.method_a == "hydra") & (pairwise.method_b == "shapley_sampling") & (pairwise.fraction == 0.10)]
        .groupby("dataset", as_index=False)
        .agg(mean_hydra_drop=("score_drop_a", "mean"), mean_shapley_drop=("score_drop_b", "mean"))
    )
    gap["shapley_minus_hydra"] = gap["mean_shapley_drop"] - gap["mean_hydra_drop"]

    cluster_cols = clusters.columns.tolist()
    if "distance_to_centroid" not in cluster_cols:
        print("[INFO] distance_to_centroid not found — computing from PCA coords")
        if "pca1" in cluster_cols and "pca2" in cluster_cols:
            centroids = clusters.groupby("cluster")[["pca1", "pca2"]].mean().rename(columns={"pca1": "c1", "pca2": "c2"})
            clusters = clusters.merge(centroids, on="cluster", how="left")
            clusters["distance_to_centroid"] = np.sqrt((clusters.pca1 - clusters.c1) ** 2 + (clusters.pca2 - clusters.c2) ** 2)
        else:
            print("[ERROR] Neither distance_to_centroid nor pca1/pca2 found in cluster CSV.")
            print("Re-run clustering with the updated SignalMorphologyClusterer.")
            return pd.DataFrame()

    keep = ["dataset", "cluster", "cluster_name", "distance_to_centroid"]
    keep = [c for c in keep if c in clusters.columns]
    merged = gap.merge(clusters[keep].drop_duplicates(), on="dataset", how="inner")

    print(f"\nGap stats (Shapley - eHYDRA score drop @ 10%):")
    print(merged.shapley_minus_hydra.describe().round(3).to_string())
    print(f"\nDistance stats:")
    print(merged.distance_to_centroid.describe().round(3).to_string())

    # Calculate & print correlations 
    rho, p = spearmanr(merged.distance_to_centroid, merged.shapley_minus_hydra)
    print(f"\n Overall Spearman correlation: ")
    print(f"ρ = {rho:.3f},  p = {p:.4f},  n = {len(merged)}")
    direction = "positive" if rho > 0 else "negative"
    grows = "gap grows" if rho > 0 else "gap shrinks"
    print(f"Interpretation: {direction} correlation — {grows} as datasets move away from centroid")

    print(f"\nPer-cluster Spearman: ")
    for cluster_name, group in merged.groupby("cluster_name"):
        if len(group) >= 5:
            rho_c, p_c = spearmanr(group.distance_to_centroid, group.shapley_minus_hydra)
            print(f"{cluster_name}: ρ={rho_c:.3f}, p={p_c:.4f}, n={len(group)}")
        else:
            print(f"{cluster_name}: n={len(group)} (too small to test)")

    # Save results
    out_csv = output_dir / "centroid_distance_gap_results.csv"
    merged.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}")
    plot(merged, rho, p, output_dir)
    return merged


def plot(merged: pd.DataFrame, rho: float, p: float, output_dir: Path):
    ''' Plot all datasets with an overall trend line & per-cluster trend lines
    '''
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for cluster_name, group in merged.groupby("cluster_name"):
        colour = CLUSTER_COLOURS.get(cluster_name, "#AAAAAA")
        label = CLUSTER_SHORT.get(cluster_name, cluster_name)
        ax.scatter(group.distance_to_centroid, group.shapley_minus_hydra, color=colour, alpha=0.7, s=40, label=label)

    m, b = np.polyfit(merged.distance_to_centroid, merged.shapley_minus_hydra, 1)  # linear trend, Theil-Sen approx
    xs = np.linspace(merged.distance_to_centroid.min(), merged.distance_to_centroid.max(), 100)
    ax.plot(xs, m * xs + b, color="black", linewidth=1.5, linestyle="--", alpha=0.7, label=f"Overall trend (ρ={rho:.2f}, p={p:.3f})")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

    ax.set_xlabel("Distance from cluster centroid\n(standardised feature space)")
    ax.set_ylabel("Shapley − eHYDRA score drop (10% masking)\nPositive = Shapley leads")
    ax.set_title("Shapley−eHYDRA gap vs centroid distance")
    ax.legend(fontsize=8, loc="upper left")

    
    ax2 = axes[1]
    for cluster_name, group in merged.groupby("cluster_name"):
        colour = CLUSTER_COLOURS.get(cluster_name, "#AAAAAA")
        label = CLUSTER_SHORT.get(cluster_name, cluster_name)
        ax2.scatter(group.distance_to_centroid, group.shapley_minus_hydra, color=colour, alpha=0.5, s=30)
        if len(group) >= 5:
            rho_c, p_c = spearmanr(group.distance_to_centroid, group.shapley_minus_hydra)
            m_c, b_c = np.polyfit(group.distance_to_centroid, group.shapley_minus_hydra, 1)
            xs_c = np.linspace(group.distance_to_centroid.min(), group.distance_to_centroid.max(), 50)
            ax2.plot(xs_c, m_c * xs_c + b_c, color=colour, linewidth=1.8, label=f"{label} (ρ={rho_c:.2f}, p={p_c:.3f})")

    # Formatting 
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Distance from cluster centroid\n(standardised feature space)")
    ax2.set_ylabel("Shapley − eHYDRA score drop (10% masking)")
    ax2.set_title("Per-cluster trend lines")
    ax2.legend(fontsize=8)

    fig.suptitle("Does the Shapley−eHYDRA gap grow as datasets move further from their centroid?", fontsize=11)
    fig.tight_layout()

    out = output_dir / "centroid_distance_gap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Plot saved to: {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Centroid distance vs Shapley-eHYDRA score drop gap analysis.")
    parser.add_argument("--pairwise-csv", default=str(PAIRWISE_CSV), help="captum_pairwise_samples.csv from the full 128-dataset run.")
    parser.add_argument("--cluster-csv", default=str(CLUSTER_CSV), help="ucr_dataset_clusters_k4_with_types.csv with distance_to_centroid column.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory to write plot and results CSV into.")
    args = parser.parse_args()

    run_analysis(
        pairwise_csv=Path(args.pairwise_csv),
        cluster_csv=Path(args.cluster_csv),
        output_dir=Path(args.output_dir),
    )