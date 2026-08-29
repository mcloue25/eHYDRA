'''
Figures for the TSHAP integration results (38-dataset subset).

Usage:
    python scripts/plotting/plot_tshap_comparison.py \
        --input-dir outputs/saliency/captum_comparison_tshap \
        --output-dir outputs/figures/tshap_comparison
'''

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



TSHAP_METHOD_COLORS = {
    "hydra": "#4C72B0",
    "shapley_sampling": "#55A868",
    "feature_ablation": "#8172B2",
    "mrsqm": "#8C8C8C",
    "tshap_centroid": "#DD8452",
    "tshap_zero": "#C44E52",
    "tshap_train": "#937860",
}

TSHAP_METHOD_LABELS = {
    "hydra": "HYDRA",
    "shapley_sampling": "Shapley Sampling",
    "feature_ablation": "Feature Ablation",
    "mrsqm": "MrSQM",
    "tshap_centroid": "TSHAP (centroid)",
    "tshap_zero": "TSHAP (zero)",
    "tshap_train": "TSHAP (train)",
}


def plot_overall_aucs_top(deletion_summary: pd.DataFrame, output_dir: Path):
    ''' Used to plot the overall AUCS top 
    '''
    df = deletion_summary.sort_values("mean_aucs_top", ascending=False)
    colors = [TSHAP_METHOD_COLORS[m] for m in df["method"]]
    labels = [TSHAP_METHOD_LABELS[m] for m in df["method"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, df["mean_aucs_top"], yerr=df["std_aucs_top"], color=colors, capsize=4, edgecolor="black", linewidth=0.6)
    ax.set_ylabel(r"Mean AUCS̃$_{top}$ (higher = more faithful)")
    ax.set_title(f"Method comparison across {int(df['n_datasets'].iloc[0])} datasets")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=30, ha="right")
    for bar, val in zip(bars, df["mean_aucs_top"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "overall_aucs_top.png", dpi=200)
    plt.close(fig)



def plot_cluster_aucs_top(cluster_deletion: pd.DataFrame, output_dir: Path):
    ''' Plots the AUCS top for one cluster  
    '''
    clusters = cluster_deletion[["cluster", "cluster_name"]].drop_duplicates().sort_values("cluster")
    methods = list(TSHAP_METHOD_COLORS.keys())

    fig, ax = plt.subplots(figsize=(11, 6))
    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    x = np.arange(len(clusters))

    for i, method in enumerate(methods):
        vals = []
        for cluster_id in clusters["cluster"]:
            row = cluster_deletion[(cluster_deletion["cluster"] == cluster_id) & (cluster_deletion["method"] == method)]
            vals.append(row["mean_aucs_top"].iloc[0] if len(row) else np.nan)
        ax.bar(x + i * bar_width, vals, width=bar_width, label=TSHAP_METHOD_LABELS[method], color=TSHAP_METHOD_COLORS[method], edgecolor="black", linewidth=0.4)

    # Formatting
    ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels(clusters["cluster_name"], rotation=15, ha="right")
    ax.set_ylabel(r"Mean AUCS̃$_{top}$")
    ax.set_title("Method faithfulness by morphology cluster")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_aucs_top.png", dpi=200)
    plt.close(fig)


def plot_background_significance(deletion_summary: pd.DataFrame, paired_tests: pd.DataFrame, output_dir: Path):
    ''' TSHAP background comparison with paired test p-values for centroid VS zero and train VS zero comparison
    '''
    bg_methods = ["tshap_zero", "tshap_centroid", "tshap_train"]
    df = deletion_summary[deletion_summary["method"].isin(bg_methods)]
    df = df.set_index("method").loc[bg_methods].reset_index()

    def get_p(a: str, b: str, fraction: float = 0.10) -> float:
        '''p-value for a > b, using whichever row direction is in the file.'''
        row = paired_tests[(paired_tests.method_a == a) & (paired_tests.method_b == b) & (paired_tests.metric == "score_drop") & (paired_tests.fraction == fraction)]
        if len(row):
            return 1 - row["wilcoxon_p_b_gt_a"].iloc[0]
        row = paired_tests[(paired_tests.method_a == b) & (paired_tests.method_b == a) & (paired_tests.metric == "score_drop") & (paired_tests.fraction == fraction)]
        return row["wilcoxon_p_b_gt_a"].iloc[0] if len(row) else float("nan")

    p_centroid_v_zero = get_p("tshap_centroid", "tshap_zero")
    p_train_v_zero = get_p("tshap_train", "tshap_zero")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = [TSHAP_METHOD_COLORS[m] for m in bg_methods]
    labels = [TSHAP_METHOD_LABELS[m] for m in bg_methods]
    bars = ax.bar(labels, df["mean_aucs_top"], yerr=df["std_aucs_top"], color=colors, capsize=4, edgecolor="black", linewidth=0.6)
    ax.set_ylabel(r"Mean AUCS̃$_{top}$")
    ax.set_title("TSHAP: effect of background choice\n(p-values: score_drop @ 10% masking, Wilcoxon)")

    def sig_stars(p: float) -> str:
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    y_top = df["mean_aucs_top"].max() + df["std_aucs_top"].max()
    ax.plot([0, 1], [y_top + 0.05] * 2, color="black", linewidth=1)
    ax.text(0.5, y_top + 0.06, f"p={p_centroid_v_zero:.1e} {sig_stars(p_centroid_v_zero)}", ha="center", fontsize=9)
    ax.plot([0, 2], [y_top + 0.15] * 2, color="black", linewidth=1)
    ax.text(1.0, y_top + 0.16, f"p={p_train_v_zero:.1e} {sig_stars(p_train_v_zero)}", ha="center", fontsize=9)
    ax.set_ylim(top=y_top + 0.28)
    fig.tight_layout()
    fig.savefig(output_dir / "background_significance.png", dpi=200)
    plt.close(fig)


def plot_method_iou_heatmap(overlap_summary: pd.DataFrame, output_dir: Path, fraction: float = 0.10):
    ''' Plots heatmap of region of IoU
    '''
    df = overlap_summary[overlap_summary["fraction"] == fraction]
    methods = list(TSHAP_METHOD_COLORS.keys())
    n = len(methods)
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)

    for _, row in df.iterrows():
        if row["method_a"] in methods and row["method_b"] in methods:
            i, j = methods.index(row["method_a"]), methods.index(row["method_b"])
            mat[i, j] = mat[j, i] = row["mean_iou"]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([TSHAP_METHOD_LABELS[m] for m in methods], rotation=45, ha="right")
    ax.set_yticklabels([TSHAP_METHOD_LABELS[m] for m in methods])
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                color = "white" if mat[i, j] < 0.5 else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_title(f"Pairwise attribution IoU at {int(fraction*100)}% masking")
    fig.colorbar(im, ax=ax, label="Mean IoU", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "method_iou_heatmap.png", dpi=200)
    plt.close(fig)


def plot_subset_vs_full_comparison(deletion_summary_38: pd.DataFrame, deletion_summary_128: pd.DataFrame, output_dir: Path) -> None:
    ''' Grouped bars comparing mean AUCS̃_top between the 38 dataset subset and full 128-dataset archive per method 
    '''
    methods = list(TSHAP_METHOD_COLORS.keys())
    d38 = deletion_summary_38.set_index("method")
    d128 = deletion_summary_128.set_index("method")
    methods_sorted = d128.loc[methods].sort_values("mean_aucs_top", ascending=False).index.tolist()  # order by full-128 ranking

    x = np.arange(len(methods_sorted))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals_38 = [d38.loc[m, "mean_aucs_top"] for m in methods_sorted]
    vals_128 = [d128.loc[m, "mean_aucs_top"] for m in methods_sorted]
    ax.bar(x - width / 2, vals_38, width, label="38-dataset subset", color=[TSHAP_METHOD_COLORS[m] for m in methods_sorted], alpha=0.55, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, vals_128, width, label="Full 128-dataset archive", color=[TSHAP_METHOD_COLORS[m] for m in methods_sorted], alpha=1.0, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([TSHAP_METHOD_LABELS[m] for m in methods_sorted], rotation=30, ha="right")
    ax.set_ylabel(r"Mean AUCS̃$_{top}$")
    ax.set_title("38-dataset subset vs. full 128-dataset archive")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "subset_vs_full_comparison.png", dpi=200)
    plt.close(fig)




def parse_args() -> argparse.Namespace:
    ''' Init arg parser 
    '''
    parser = argparse.ArgumentParser(description="Generate TSHAP comparison figures.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compare-dir", type=Path, default=None,
        help=(
            "Optional second results directory (e.g. the 38-dataset subset) "
            "to compare against --input-dir. Produces "
            "subset_vs_full_comparison.png in addition to the standard four."
        ),
    )
    return parser.parse_args()



def main():
    ''' Main function for plotting AUCS results
    '''
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    deletion_summary = pd.read_csv(args.input_dir / "captum_deletion_summary.csv")
    cluster_deletion = pd.read_csv(args.input_dir / "captum_cluster_deletion_summary.csv")
    overlap_summary = pd.read_csv(args.input_dir / "captum_overlap_summary.csv")
    paired_tests = pd.read_csv(args.input_dir / "captum_paired_tests.csv")

    plot_overall_aucs_top(deletion_summary, args.output_dir)
    plot_cluster_aucs_top(cluster_deletion, args.output_dir)
    plot_background_significance(deletion_summary, paired_tests, args.output_dir)
    plot_method_iou_heatmap(overlap_summary, args.output_dir)

    if args.compare_dir is not None:
        deletion_summary_compare = pd.read_csv(args.compare_dir / "captum_deletion_summary.csv")
        plot_subset_vs_full_comparison(deletion_summary_compare, deletion_summary, args.output_dir)

    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()