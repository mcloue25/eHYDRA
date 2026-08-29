'''
Analyses AUCS̃_top deletion curves by signal-morphology cluster: 
    - Where each method's deletion ordering is strongest, 
    - Whether AUCS̃_top and score drop faithfulness agree 
    - Which morphology features predict eHYDRA-Shapley region agreement (IoU).

REQUIRES: captum_pairwise_samples.csv, captum_cluster_deletion_summary.csv,
captum_cluster_overlap_summary.csv, ucr_dataset_clusters_k4_with_types.csv
OPTIONAL: captum_deletion_samples.csv (enables dataset-level analysis 2)

TO RUN: 
    python scripts/analysis/analyse_deletion_curves_by_cluster.py
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
    METHOD_COLOURS,
    METHOD_LABELS,
    CLUSTER_COLOURS,
    CLUSTER_SHORT,
    CLUSTER_ORDER,
    apply_thesis_style,
)

apply_thesis_style()

RUN_DIR = PROJECT_ROOT / "outputs/saliency/captum_comparison_full128"
PLOT_DIR = PROJECT_ROOT / "outputs/plots/tscaptum"  # all plots go here
CSV_DIR = RUN_DIR  # CSVs stay alongside the data

PLOT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)




def analysis_1_aucs_by_cluster(cd: pd.DataFrame):
    ''' Bar chart and printed table of AUCS̃_top per method per cluster
    '''
    print("\n\n\n")
    print("ANALYSIS 1: AUCS̃_top by signal-morphology cluster")
    print(cd[["cluster_name", "method", "mean_aucs_top", "n_datasets"]].to_string(index=False))

    pivot = cd.pivot_table(index="cluster_name", columns="method", values="mean_aucs_top")
    if "hydra" in pivot.columns and "shapley_sampling" in pivot.columns:
        pivot["hydra_minus_shapley"] = pivot["hydra"] - pivot["shapley_sampling"]
        print("\n\n\neHYDRA vs Shapley AUCS̃_top per cluster:")
        print(pivot[["hydra", "shapley_sampling", "hydra_minus_shapley"]].round(4).to_string())
        print("\nPositive = eHYDRA has better deletion ordering in that cluster")
        print("Negative = Shapley has better deletion ordering in that cluster")

    methods = ["hydra", "shapley_sampling", "feature_ablation", "mrsqm"]
    clusters = [c for c in CLUSTER_ORDER if c in cd.cluster_name.values]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(clusters))
    width = 0.18

    for i, method in enumerate(methods):
        sub = cd[cd.method == method].set_index("cluster_name")
        vals = [float(sub.loc[c, "mean_aucs_top"]) if c in sub.index else np.nan for c in clusters]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=METHOD_LABELS.get(method, method), color=METHOD_COLOURS.get(method, "#AAAAAA"), alpha=0.85)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([CLUSTER_SHORT.get(c, c) for c in clusters])
    ax.set_ylabel("Mean AUCS̃$_{top}$")
    ax.set_title("Deletion curve faithfulness by signal-morphology cluster\nHigher = method's ranking causes faster score degradation")
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(0, 0.60)

    # NOTE - Annotating the two clusters where eHYDRA leads
    ax.annotate("eHYDRA leads\nin this cluster",
                xy=(0 + (-1.5) * width, 0.285),
                xytext=(0.6, 0.44),
                fontsize=8, color=METHOD_COLOURS.get("hydra", "#4C72B0"),
                arrowprops=dict(arrowstyle="->", color=METHOD_COLOURS.get("hydra", "#4C72B0")))
    
    ax.annotate("eHYDRA leads\nin this cluster",
                xy=(3 + (-1.5) * width, 0.428),
                xytext=(2.1, 0.54),
                fontsize=8, color=METHOD_COLOURS.get("hydra", "#4C72B0"),
                arrowprops=dict(arrowstyle="->", color=METHOD_COLOURS.get("hydra", "#4C72B0")))

    fig.tight_layout()
    out = PLOT_DIR / "aucs_top_by_cluster.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved: {out}")



def analysis_2a_dataset_level(deletion_samples: pd.DataFrame, pairwise: pd.DataFrame, clusters: pd.DataFrame):
    ''' Dataset-level scatter of AUCS̃_top diff vs score drop gap
    '''
    print("\n\n\n")
    print("ANALYSIS 2a: Dataset level AUCS̃_top diff vs score drop gap:")
    cluster_cols = [c for c in ["dataset", "cluster", "cluster_name"] if c in clusters.columns]
    cl = clusters[cluster_cols].drop_duplicates()
    deletion_samples = deletion_samples.merge(cl, on="dataset", how="left")
    pairwise = pairwise.merge(cl, on="dataset", how="left")

    aucs = (
        deletion_samples[deletion_samples.method.isin(["hydra", "shapley_sampling"])]
        .groupby(["dataset", "method", "cluster_name"], as_index=False)
        .agg(mean_aucs_top=("aucs_top", "mean"))
    )
    aucs_wide = aucs.pivot_table(index=["dataset", "cluster_name"], columns="method", values="mean_aucs_top").reset_index()
    aucs_wide.columns.name = None

    if "hydra" not in aucs_wide.columns or "shapley_sampling" not in aucs_wide.columns:
        print("[WARN] Missing hydra or shapley_sampling in deletion samples — skipping 2a")
        return
    
    aucs_wide["aucs_diff"] = aucs_wide["hydra"] - aucs_wide["shapley_sampling"]
    gap = (
        pairwise[(pairwise.method_a == "hydra") & (pairwise.method_b == "shapley_sampling") & (pairwise.fraction == 0.10)]
        .groupby(["dataset", "cluster_name"], as_index=False)
        .agg(mean_hydra_drop=("score_drop_a", "mean"), mean_shapley_drop=("score_drop_b", "mean"))
    )
    gap["score_drop_gap"] = gap["mean_shapley_drop"] - gap["mean_hydra_drop"]
    merged = aucs_wide.merge(gap[["dataset", "score_drop_gap"]], on="dataset", how="inner").dropna(subset=["aucs_diff", "score_drop_gap"])

    print(f"\nDatasets available: {len(merged)}")
    print(f"eHYDRA AUCS̃_top > Shapley: {(merged.aucs_diff > 0).sum()}")
    print(f"Shapley AUCS̃_top > eHYDRA:  {(merged.aucs_diff < 0).sum()}")
    print()
    print("Full dataset level table:")
    print(merged[["dataset", "cluster_name", "hydra", "shapley_sampling", "aucs_diff", "score_drop_gap"]].sort_values("aucs_diff", ascending=False).to_string(index=False))

    rho, p = spearmanr(merged["aucs_diff"], merged["score_drop_gap"])
    print(f"\nOverall Spearman: ρ={rho:.3f}, p={p:.4f}, n={len(merged)}")

    # NOTE - Calculate spearman correlation 
    print("\nPer-cluster Spearman:")
    for cluster, group in merged.groupby("cluster_name"):
        if len(group) >= 5:
            r, pv = spearmanr(group["aucs_diff"], group["score_drop_gap"])
            print(f"  {cluster}: ρ={r:.3f}, p={pv:.4f}, n={len(group)}")

    # Plot the results
    plot_scatter(merged, rho, p, dataset_level=True)
    out_csv = CSV_DIR / "aucs_gap_vs_score_drop_gap_datasets.csv"
    merged.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")




def analysis_2b_cluster_level(cd: pd.DataFrame, co: pd.DataFrame):
    ''' 4 point cluster level fallback if deletion_samples.csv isn't generated
    '''
    print("\n\n\n")
    print("ANALYSIS 2b: Cluster level AUCS̃_top diff vs score drop gap:")
    print("(n=4 clusters — no statistical power, descriptive only)")
    aucs_wide = cd.pivot_table(index="cluster_name", columns="method", values="mean_aucs_top").reset_index()
    aucs_wide.columns.name = None
    if "hydra" not in aucs_wide.columns or "shapley_sampling" not in aucs_wide.columns:
        print("[WARN] Missing methods in cluster deletion summary")
        return

    aucs_wide["aucs_diff"] = aucs_wide["hydra"] - aucs_wide["shapley_sampling"]
    hs = (co[(co.method_a == "hydra") & (co.method_b == "shapley_sampling") & (co.fraction == 0.10)] [["cluster_name", "mean_score_drop_a", "mean_score_drop_b", "n_datasets"]].copy())
    hs["score_drop_gap"] = hs["mean_score_drop_b"] - hs["mean_score_drop_a"]
    merged = aucs_wide.merge(hs, on="cluster_name", how="inner")

    print("\nCluster-level table:")
    print(merged[["cluster_name", "hydra", "shapley_sampling", "aucs_diff", "mean_score_drop_a", "mean_score_drop_b", "score_drop_gap", "n_datasets"]].round(4).to_string(index=False))

    print("\nKey observations:")
    for _, row in merged.iterrows():
        direction_aucs = "eHYDRA leads on AUCS̃_top" if row.aucs_diff > 0 else "Shapley leads on AUCS̃_top"
        direction_drop = "Shapley leads on score drop" if row.score_drop_gap > 0 else "eHYDRA leads on score drop"
        consistent = (row.aucs_diff < 0) == (row.score_drop_gap > 0)
        flag = "CONSISTENT" if consistent else "INCONSISTENT — metrics disagree"
        print(f"{row.cluster_name}:")
        print(f"{direction_aucs} (Δ={row.aucs_diff:+.3f})  |  {direction_drop} (Δ={row.score_drop_gap:+.3f})  →  {flag}")

    plot_scatter(merged, rho=None, p=None, dataset_level=False)
    out_csv = CSV_DIR / "aucs_gap_vs_score_drop_gap_clusters.csv"
    merged.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")



def plot_scatter(df: pd.DataFrame, rho, p, dataset_level: bool):
    ''' Shared scatter plot for analysis 2a (per-dataset) and 2b (per-cluster)
    '''
    fig, ax = plt.subplots(figsize=(7, 5))

    if dataset_level:
        for cluster_name, group in df.groupby("cluster_name"):
            ax.scatter(group["aucs_diff"], group["score_drop_gap"], color=CLUSTER_COLOURS.get(cluster_name, "#AAAAAA"), alpha=0.65, s=35, label=CLUSTER_SHORT.get(cluster_name, cluster_name))
        m, b = np.polyfit(df["aucs_diff"], df["score_drop_gap"], 1)  # trend line
        xs = np.linspace(df["aucs_diff"].min(), df["aucs_diff"].max(), 100)
        ax.plot(xs, m * xs + b, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
        title = f"Dataset-level: AUCS̃$_{{top}}$ diff vs score drop gap\nSpearman ρ={rho:.3f}, p={p:.4f}"
        fname = "aucs_gap_vs_score_drop_gap_datasets.png"
    else:
        for _, row in df.iterrows():
            c = CLUSTER_COLOURS.get(row.cluster_name, "#AAAAAA")
            ax.scatter(row["aucs_diff"], row["score_drop_gap"], color=c, s=120, zorder=3)
            ax.annotate(CLUSTER_SHORT.get(row.cluster_name, row.cluster_name), xy=(row["aucs_diff"], row["score_drop_gap"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        title = "Cluster-level: AUCS̃$_{top}$ diff vs score drop gap\n(n=4, descriptive only)"
        fname = "aucs_gap_vs_score_drop_gap_clusters.png"

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("AUCS̃$_{top}$ (eHYDRA) − AUCS̃$_{top}$ (Shapley)\n→ positive = eHYDRA has better deletion ordering")
    ax.set_ylabel("Score drop gap: Shapley − eHYDRA @ 10% masking\n→ positive = Shapley selects more sensitive regions")
    ax.set_title(title)
    if dataset_level:
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = PLOT_DIR / fname
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")



def analysis_3_morphology_vs_iou(pairwise: pd.DataFrame, clusters: pd.DataFrame):
    ''' Correlates continuous morphology features against per-dataset mean eHYDRA-Shapley IoU at 10% masking
    '''
    print("\n\n\n")
    print("ANALYSIS 3: Morphology features vs eHYDRA-Shapley region agreement (IoU):")
    iou = (
        pairwise[(pairwise.method_a == "hydra") & (pairwise.method_b == "shapley_sampling") & (pairwise.fraction == 0.10)]
        .groupby("dataset", as_index=False)
        .agg(mean_iou=("iou", "mean"))
    )

    merged = iou.merge(clusters, on="dataset", how="inner")
    print(f"\nDatasets: {len(merged)}")
    print(f"Mean IoU (eHYDRA vs Shapley @ 10%): {merged.mean_iou.mean():.3f}")
    print(f"Std IoU: {merged.mean_iou.std():.3f}")
    print(f"Range: [{merged.mean_iou.min():.3f}, {merged.mean_iou.max():.3f}]")

    morphology_features = [
        "spectral_entropy",
        "mean_abs_diff",
        "mean_abs_second_diff",
        "spike_ratio",
        "zero_crossing_rate",
        "acf_lag1",
        "low_freq_energy_ratio",
        "log_series_length",
        "n_classes",
    ]

    print("\nSpearman correlations: morphology feature vs eHYDRA-Shapley IoU")
    print(f"{'Feature':<35}  {'ρ':>6}  {'p':>8}  Sig")
    print("  " + "-" * 55)

    results = []
    for feat in morphology_features:
        if feat not in merged.columns:
            print(f"{feat:<35}  [column not found]")
            continue
        valid = merged[[feat, "mean_iou"]].dropna()
        rho, p = spearmanr(valid[feat], valid["mean_iou"])
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{feat:<35}  {rho:+.3f}  {p:.4f}  {sig}")
        results.append({"feature": feat, "rho": rho, "p": p, "sig": sig})

    results_df = pd.DataFrame(results).sort_values("rho")

    sig_results = results_df[results_df.p < 0.05]
    if not sig_results.empty:
        print(f"\nSignificant predictors (p < 0.05): {len(sig_results)}/{len(results_df)}")
        for _, row in sig_results.iterrows():
            direction = "higher IoU (more agreement)" if row.rho > 0 else "lower IoU (less agreement)"
            print(f"{row.feature}: ρ={row.rho:+.3f} --> higher {row.feature} predicts {direction}")
    else:
        print("\nNo features reach p < 0.05 individually.")

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = ["#2a78d6" if r > 0 else "#e34948" for r in results_df.rho]
    bars = ax.barh(results_df.feature, results_df.rho, color=colours, alpha=0.85)

    for bar, (_, row) in zip(bars, results_df.iterrows()):
        label = "**" if row.p < 0.01 else "*" if row.p < 0.05 else ""
        if label:
            x = row.rho + (0.01 if row.rho >= 0 else -0.01)
            ha = "left" if row.rho >= 0 else "right"
            ax.text(x, bar.get_y() + bar.get_height() / 2, label, va="center", ha=ha, fontsize=10, color="black")

    # Formatting & save results
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Spearman ρ (positive = feature predicts higher eHYDRA-Shapley IoU)")
    ax.set_title("Which morphology features predict eHYDRA-Shapley region agreement?\n* p<0.05   ** p<0.01")
    ax.set_xlim(-0.6, 0.6)
    fig.tight_layout()
    out = PLOT_DIR / "morphology_vs_iou.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved: {out}")
    out_csv = CSV_DIR / "morphology_vs_iou.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


def main():
    ''' Main function for running all 4 analysis tests
    '''
    print("Loading files...")

    pairwise = pd.read_csv(RUN_DIR / "captum_pairwise_samples.csv")
    cd = pd.read_csv(RUN_DIR / "captum_cluster_deletion_summary.csv")
    co = pd.read_csv(RUN_DIR / "captum_cluster_overlap_summary.csv")
    clusters = pd.read_csv(CLUSTER_CSV)

    print(f"Pairwise rows: {len(pairwise):,}  |  datasets: {pairwise.dataset.nunique()}")
    print(f"Cluster deletion rows: {len(cd)}")
    print(f"Cluster overlap rows: {len(co)}")

    # NOTE - AUCS by cluster test 
    analysis_1_aucs_by_cluster(cd)

    # NOTE - dataset-level analysis if the raw deletion CSV exists, else cluster-level fallback
    deletion_path = RUN_DIR / "captum_deletion_samples.csv"
    if deletion_path.exists() and deletion_path.stat().st_size > 0:
        print(f"\nFound {deletion_path.name} — running dataset-level analysis")
        deletion_samples = pd.read_csv(deletion_path)
        print(f"Deletion sample rows: {len(deletion_samples):,}")
        analysis_2a_dataset_level(deletion_samples, pairwise, clusters)
    else:
        print(f"\n{deletion_path.name} not found — running cluster-level fallback")
        analysis_2b_cluster_level(cd, co)

    # NOTE - Morphology vs iou
    analysis_3_morphology_vs_iou(pairwise, clusters)  # always runs

    print(f"\nAll plots saved to: {PLOT_DIR}")
    print(f"All CSVs saved to: {CSV_DIR}")


if __name__ == "__main__":
    main()