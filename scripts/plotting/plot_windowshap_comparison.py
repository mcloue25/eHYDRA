'''
Plot eHYDRA vs WindowSHAP comparison figures

Usage:
    python scripts/plotting/plot_windowshap_comparison.py \
        --summary-csv outputs/saliency/windowshap/hydra_windowshap_summary.csv \
        --by-dataset-csv outputs/saliency/windowshap/hydra_windowshap_by_dataset.csv \
        --output-dir outputs/plots/windowshap
'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from utils.plot_config import (
    METHOD_COLOURS,
    FRACTIONS,
    FRACTION_LABELS,
    apply_thesis_style,
)

apply_thesis_style()

EHYDRA_COLOUR = METHOD_COLOURS["ehydra"]
WSHAP_COLOUR = METHOD_COLOURS["windowshap"]


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    '''3-panel summary: score drop, flip rate, region agreement (IoU/overlap).'''
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    x = np.arange(len(FRACTIONS))
    width = 0.35

    # NOTE - score drop
    ax = axes[0]
    hydra_drops = [float(summary.loc[summary.fraction == f, "mean_hydra_score_drop"]) for f in FRACTIONS]
    shap_drops = [float(summary.loc[summary.fraction == f, "mean_shap_score_drop"]) for f in FRACTIONS]

    # Formatting
    ax.bar(x - width/2, hydra_drops, width, label="eHYDRA", color=EHYDRA_COLOUR, alpha=0.85)
    ax.bar(x + width/2, shap_drops, width, label="WindowSHAP", color=WSHAP_COLOUR, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(FRACTION_LABELS)
    ax.set_xlabel("Masking fraction"); ax.set_ylabel("Mean score drop")
    ax.set_title("Score drop"); ax.legend(); ax.set_ylim(0, max(shap_drops) * 1.35)
    for i, (h, s) in enumerate(zip(hydra_drops, shap_drops)):
        ax.text(i - width/2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, s + 0.005, f"{s:.3f}", ha="center", va="bottom", fontsize=8)

    # NOTE - Flip rate
    ax = axes[1]
    hydra_flips = [float(summary.loc[summary.fraction == f, "hydra_flip_rate"]) for f in FRACTIONS]
    shap_flips = [float(summary.loc[summary.fraction == f, "shap_flip_rate"]) for f in FRACTIONS]

    # Formatting
    ax.bar(x - width/2, hydra_flips, width, label="eHYDRA", color=EHYDRA_COLOUR, alpha=0.85)
    ax.bar(x + width/2, shap_flips, width, label="WindowSHAP", color=WSHAP_COLOUR, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(FRACTION_LABELS)
    ax.set_xlabel("Masking fraction"); ax.set_ylabel("Flip rate")
    ax.set_title("Prediction flip rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, max(shap_flips) * 1.35); ax.legend()
    for i, (h, s) in enumerate(zip(hydra_flips, shap_flips)):
        ax.text(i - width/2, h + 0.005, f"{h:.1%}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, s + 0.005, f"{s:.1%}", ha="center", va="bottom", fontsize=8)

    # NOTE - IoU
    ax = axes[2]
    ious = [float(summary.loc[summary.fraction == f, "mean_iou"]) for f in FRACTIONS]
    overlaps = [float(summary.loc[summary.fraction == f, "mean_overlap_fraction"]) for f in FRACTIONS]

    ax.plot(FRACTION_LABELS, ious, marker="o", color=EHYDRA_COLOUR, linewidth=2, markersize=7, label="Mean IoU")
    ax.plot(FRACTION_LABELS, overlaps, marker="s", color=WSHAP_COLOUR, linewidth=2, markersize=7, linestyle="--", label="Mean overlap")
    ax.set_xlabel("Masking fraction"); ax.set_ylabel("Region agreement")
    ax.set_title("Region agreement"); ax.legend(); ax.set_ylim(0, max(overlaps) * 1.6)
    for frac_label, iou, ov in zip(FRACTION_LABELS, ious, overlaps):
        ax.annotate(f"{iou:.3f}", xy=(frac_label, iou), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=EHYDRA_COLOUR)
        ax.annotate(f"{ov:.3f}", xy=(frac_label, ov), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8, color=WSHAP_COLOUR)

    fig.suptitle("eHYDRA vs WindowSHAP: perturbation faithfulness and region agreement", fontsize=12)
    fig.tight_layout()
    out = output_dir / "windowshap_vs_ehydra_summary.png"
    fig.savefig(out); plt.close(fig); print(f"Saved: {out}")




def plot_advantage_gap(summary: pd.DataFrame, output_dir: Path) -> None:
    ''' Bar charts of WindowSHAP - eHYDRA gap for score drop and flip rate, per fraction
    '''
    score_gaps = [float(summary.loc[summary.fraction == f, "mean_shap_score_drop"]) - float(summary.loc[summary.fraction == f, "mean_hydra_score_drop"]) for f in FRACTIONS]
    flip_gaps = [float(summary.loc[summary.fraction == f, "shap_flip_rate"]) - float(summary.loc[summary.fraction == f, "hydra_flip_rate"]) for f in FRACTIONS]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    x = np.arange(len(FRACTIONS))

    axes_tuple_data = [
        (axes[0], score_gaps, "WindowSHAP \u2212 eHYDRA score drop", "Score drop advantage\n(positive\u202f=\u202fWindowSHAP leads)", "{:+.3f}"),
        (axes[1], flip_gaps, "WindowSHAP \u2212 eHYDRA flip rate", "Flip rate advantage\n(positive\u202f=\u202fWindowSHAP leads)", "{:+.1%}"),
    ]

    for ax, gaps, ylabel, title, fmt in axes_tuple_data:
        # Map colours and bars
        colours = [WSHAP_COLOUR if g > 0 else EHYDRA_COLOUR for g in gaps]
        bars = ax.bar(x, gaps, color=colours, alpha=0.85)

        # Formatting
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(FRACTION_LABELS)
        ax.set_xlabel("Masking fraction"); ax.set_ylabel(ylabel); ax.set_title(title)
        if "%" in fmt:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

        for bar, v in zip(bars, gaps):
            offset = 0.003 if v >= 0 else -0.006
            va = "bottom" if v >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset, fmt.format(v), ha="center", va=va, fontsize=9)

    # save fig
    fig.suptitle("WindowSHAP advantage over eHYDRA", fontsize=12)
    fig.tight_layout()
    out = output_dir / "windowshap_advantage_gap.png"
    fig.savefig(out); plt.close(fig); print(f"Saved: {out}")




def plot_per_dataset_scatter(by_dataset: pd.DataFrame, output_dir: Path):
    ''' Per-dataset eHYDRA vs WindowSHAP score drop at 10% masking
    '''
    sub = by_dataset[by_dataset.fraction == 0.10].copy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sub.hydra_score_drop, sub.shap_score_drop, alpha=0.65, s=40, color=WSHAP_COLOUR, edgecolors="white", linewidths=0.5)

    lim_max = max(sub.hydra_score_drop.max(), sub.shap_score_drop.max()) * 1.08
    ax.plot([0, lim_max], [0, lim_max], color="gray", linewidth=1, linestyle="--", label="Equal performance")

    n_wshap = (sub.shap_score_drop > sub.hydra_score_drop).sum()
    n_ehydra = (sub.shap_score_drop <= sub.hydra_score_drop).sum()

    ax.text(0.05, 0.93, f"WindowSHAP leads: {n_wshap}/{len(sub)} datasets", transform=ax.transAxes, fontsize=9, color=WSHAP_COLOUR)
    ax.text(0.05, 0.87, f"eHYDRA leads: {n_ehydra}/{len(sub)} datasets", transform=ax.transAxes, fontsize=9, color=EHYDRA_COLOUR)
    ax.set_xlabel("eHYDRA score drop"); ax.set_ylabel("WindowSHAP score drop")
    ax.set_title("Per-dataset score drop at 10% masking\neHYDRA vs WindowSHAP")
    ax.set_xlim(0, lim_max); ax.set_ylim(0, lim_max); ax.legend(fontsize=9)
    fig.tight_layout()
    out = output_dir / "windowshap_per_dataset_scatter.png"
    fig.savefig(out); plt.close(fig); print(f"Saved: {out}")



def parse_args():
    ''' Define the argparser
    '''
    parser = argparse.ArgumentParser(description="Plot eHYDRA vs WindowSHAP figures.")
    parser.add_argument("--summary-csv", default="outputs/saliency/windowshap/hydra_windowshap_summary.csv")
    parser.add_argument("--by-dataset-csv", default="outputs/saliency/windowshap/hydra_windowshap_by_dataset.csv")
    parser.add_argument("--output-dir", default="outputs/plots/windowshap")
    return parser.parse_args()



def main():
    ''' Main function for running the windowSHAP comparison
    '''
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary_csv)
    by_dataset = pd.read_csv(args.by_dataset_csv)

    print(f"Loaded summary: {len(summary)} rows")
    print(f"Loaded by-dataset: {len(by_dataset)} rows ({by_dataset.dataset.nunique()} datasets)")

    plot_summary(summary, out)
    plot_advantage_gap(summary, out)
    plot_per_dataset_scatter(by_dataset, out)
    print(f"All figures written to: {out.resolve()}")


if __name__ == "__main__":
    main()