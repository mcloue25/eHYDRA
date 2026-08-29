'''
HYDRA saliency figures: cluster/mode score-drop heatmap and flip-rate/score-drop bar charts
'''
from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.globals_config import MODE_ORDER, CLUSTER_NAMES
from utils.plot_config import MODE_COLOURS, MODE_LABELS


def save_png_pdf(fig, output_dir, stem, dpi=300):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", dpi=dpi, bbox_inches="tight")


def get_score_drop_column(df):
    if "mean_bounded_relative_score_drop" in df.columns:
        return "mean_bounded_relative_score_drop"
    if "bounded_relative_score_drop" in df.columns:
        return "bounded_relative_score_drop"
    raise KeyError(
        "Could not find a bounded score-drop column. Expected either "
        "'mean_bounded_relative_score_drop' or 'bounded_relative_score_drop'."
    )


def plot_hydra_cluster_score_drop_heatmap(analysis_dir="outputs/saliency/analysis", output_dir="outputs/saliency/imgs"):
    analysis_dir = Path(analysis_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_summary_path = analysis_dir / "combined_saliency_summary_with_clusters.csv"
    if not cluster_summary_path.exists():
        raise FileNotFoundError(f"Missing {cluster_summary_path}. Run `python main.py analyse-saliency` first.")

    cluster_summary = pd.read_csv(cluster_summary_path)

    required_cols = {"model", "fraction", "mode", "cluster", "mean_score_drop"}
    missing_cols = required_cols - set(cluster_summary.columns)
    if missing_cols:
        raise KeyError(f"{cluster_summary_path} is missing required columns: {sorted(missing_cols)}")

    # dataset-level rows aggregated to one row per cluster/fraction/mode
    cluster_summary = cluster_summary.groupby(["model", "fraction", "mode", "cluster"], as_index=False).agg(mean_score_drop=("mean_score_drop", "mean"))

    cluster_order = sorted(CLUSTER_NAMES.keys())
    cluster_name_order = [CLUSTER_NAMES[c] for c in cluster_order]
    fractions_to_plot = [0.05, 0.10, 0.20]

    hydra_rows = cluster_summary[(cluster_summary["model"] == "HYDRA") & (cluster_summary["fraction"].isin(fractions_to_plot))]
    if hydra_rows.empty:
        raise ValueError(f"No HYDRA rows found for fractions 0.05, 0.10 and 0.20 in {cluster_summary_path}.")

    vmin = hydra_rows["mean_score_drop"].min()
    vmax = hydra_rows["mean_score_drop"].max()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    for ax, frac in zip(axes, fractions_to_plot):
        heatmap_df = cluster_summary[(cluster_summary["model"] == "HYDRA") & (cluster_summary["fraction"] == frac)].copy()
        heatmap_df["cluster_name"] = heatmap_df["cluster"].map(CLUSTER_NAMES)
        heatmap_df["cluster_name"] = pd.Categorical(heatmap_df["cluster_name"], categories=cluster_name_order, ordered=True)
        heatmap_df["mode"] = pd.Categorical(heatmap_df["mode"], categories=MODE_ORDER, ordered=True)

        # NOTE Plotting the actual heatmap
        heatmap_pivot = heatmap_df.pivot(index="cluster_name", columns="mode", values="mean_score_drop").reindex(index=cluster_name_order, columns=MODE_ORDER)
        sns.heatmap(
            heatmap_pivot, annot=True, fmt=".2f", cmap="coolwarm", vmin=vmin, vmax=vmax,
            linewidths=0.5, linecolor="white", cbar=ax is axes[-1],
            annot_kws={"fontsize": 10}, ax=ax,
        )

        # Formatting
        ax.set_title(f"{int(frac * 100)}% masking", fontsize=12, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("" if ax is not axes[0] else "Cluster", fontsize=11)
        ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER], rotation=20, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle("HYDRA: Mean score drop by cluster and masking strategy", fontsize=14, y=1.03)
    save_png_pdf(fig, output_dir, "hydra_cluster_score_drop_heatmap_expansive")
    save_png_pdf(fig, output_dir, "mean_score_drop_by_cluster")
    plt.close(fig)


def plot_hydra_flip_and_score_bars_side_by_side(analysis_dir="outputs/saliency/analysis", output_dir="outputs/saliency/imgs"):
    ''' 
    '''
    analysis_dir = Path(analysis_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = analysis_dir / "combined_saliency_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run `python main.py analyse-saliency` first.")

    summary = pd.read_csv(summary_path)

    required_cols = {"model", "fraction", "mode"}
    missing_cols = required_cols - set(summary.columns)
    if missing_cols:
        raise KeyError(f"{summary_path} is missing required columns: {sorted(missing_cols)}")

    flip_col = "mean_flip_rate" if "mean_flip_rate" in summary.columns else "flip_rate"
    score_col = get_score_drop_column(summary)

    # dataset-level rows aggregated to one row per fraction/mode
    df = summary[summary["model"] == "HYDRA"].groupby(["fraction", "mode"], as_index=False).agg(
        mean_flip_rate=(flip_col, "mean"),
        mean_bounded_relative_score_drop=(score_col, "mean"),
    )
    if df.empty:
        raise ValueError(f"No HYDRA rows found in {summary_path}.")

    fractions = sorted(df["fraction"].unique())
    x = np.arange(len(fractions))
    width = 0.24
    offsets = [-width, 0, width]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    ax_flip, ax_score = axes

    legend_handles = []
    legend_labels = []

    for offset, mode in zip(offsets, MODE_ORDER):
        sub = df[df["mode"] == mode].sort_values("fraction").set_index("fraction").reindex(fractions).reset_index()

        y_flip = sub["mean_flip_rate"].to_numpy()
        bars_flip = ax_flip.bar(x + offset, y_flip, width=width, alpha=0.90, color=MODE_COLOURS[mode], label=MODE_LABELS[mode])
        legend_handles.append(bars_flip[0])
        legend_labels.append(MODE_LABELS[mode])

        for bar, val in zip(bars_flip, y_flip):
            if pd.isna(val):
                continue
            ax_flip.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1%}", ha="center", va="bottom", fontsize=9)

        y_score = sub["mean_bounded_relative_score_drop"].to_numpy()
        bars_score = ax_score.bar(x + offset, y_score, width=width, alpha=0.90, color=MODE_COLOURS[mode])

        for bar, val in zip(bars_score, y_score):
            if pd.isna(val):
                continue
            ax_score.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax_flip.set_title("HYDRA: Prediction flip rate")
    ax_flip.set_xlabel("Masked fraction of time steps")
    ax_flip.set_ylabel("")
    ax_flip.set_xticks(x)
    ax_flip.set_xticklabels([f"{f:.0%}" for f in fractions])
    ax_flip.set_yticks([])
    ax_flip.spines["left"].set_visible(False)
    ax_flip.spines["top"].set_visible(False)
    ax_flip.spines["right"].set_visible(False)

    ax_score.set_title("HYDRA: Bounded relative score drop")
    ax_score.set_xlabel("Masked fraction of time steps")
    ax_score.set_ylabel("")
    ax_score.set_xticks(x)
    ax_score.set_xticklabels([f"{f:.0%}" for f in fractions])
    ax_score.set_yticks([])
    ax_score.spines["left"].set_visible(False)
    ax_score.spines["top"].set_visible(False)
    ax_score.spines["right"].set_visible(False)

    ymin, ymax = ax_flip.get_ylim()
    if ymax > 0:
        ax_flip.set_ylim(ymin, ymax * 1.18)

    ymin, ymax = ax_score.get_ylim()
    if ymax > 0:
        ax_score.set_ylim(ymin, ymax * 1.18)

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    # notebook-era filename and report-era filename
    save_png_pdf(fig, output_dir, "hydra_flip_and_bounded_score_bars_expansive")
    save_png_pdf(fig, output_dir, "updated_prediction_bounded_score_drop")

    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="outputs/saliency/analysis")
    parser.add_argument("--output-dir", default="outputs/saliency/imgs")
    return parser.parse_args()


def main():
    args = parse_args()
    plot_hydra_flip_and_score_bars_side_by_side(analysis_dir=args.analysis_dir, output_dir=args.output_dir)
    plot_hydra_cluster_score_drop_heatmap(analysis_dir=args.analysis_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()