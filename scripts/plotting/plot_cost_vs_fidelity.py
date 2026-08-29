#!/usr/bin/env python3
"""
Cost-vs-fidelity scatter: AUCS~_top (faithfulness) against median per-sample
explanation time (cost, log scale), one point per method, for the
Computational Complexity section.

All numbers transcribed directly from thesis tables already in the document
(tab:tscaptum-timing-38, tab:tshap-timing, tab:tshap-overall-128,
tab:tscaptum-128-aucs). WindowSHAP has no AUCS~_top reported in this thesis
(evaluated with score drop / flip rate / IoU instead), so it is excluded
from this plot rather than shown with a placeholder y-value.

Label placement uses adjustText (pip install adjustText) to automatically
resolve overlaps and draw thin leader lines from crowded points to their
labels -- several TSHAP/Shapley points sit within ~6% of the x-axis's log
range of each other, too close for fixed manual offsets to reliably avoid
collision as the exact numbers get tweaked.

Usage:
    python3 scripts/plotting/plot_cost_vs_fidelity.py
"""
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from adjustText import adjust_text

from utils.plot_config import apply_slide_style

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "plots",
                        "complexity")

apply_slide_style({
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.frameon": False,
    "legend.fontsize": 9,
})

# (method, AUCS~_top [full 128-dataset archive], median time (s), needs_background)
METHODS = [
    ("eHYDRA", 0.358, 0.185, False),
    ("MrSQM", 0.246, 0.001, False),
    ("Feature Ablation", 0.363, 0.041, True),
    ("Shapley Sampling", 0.370, 0.970, True),
    ("TSHAP (zero)", 0.366, 0.590, True),
    ("TSHAP (centroid)", 0.407, 0.620, True),
    ("TSHAP (train)", 0.423, 0.840, True),
]

COLOR_EHYDRA = "#1f4e8c"
COLOR_NO_BG = "#2e7d32"
COLOR_BG = "#a83232"


def main():
    ''' 
    '''
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    texts = []
    for name, aucs, time_s, needs_bg in METHODS:
        is_ehydra = name == "eHYDRA"
        color = COLOR_EHYDRA if is_ehydra else (COLOR_BG if needs_bg else COLOR_NO_BG)
        size = 140 if is_ehydra else 100
        ax.scatter(time_s, aucs, s=size, marker="o", color=color, edgecolor="white", linewidth=0.9, zorder=3)
        t = ax.text(time_s, aucs, f"  {name}", fontsize=9.5, fontweight="bold" if is_ehydra else "normal", color=color, zorder=4)
        texts.append(t)

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 5], numticks=10))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel("Median explanation time per sample (s, log scale, CUDA)")
    ax.set_ylabel(r"Mean AUCS$\widetilde{\ }_{\mathrm{top}}$ (full 128-dataset archive)")
    ax.set_title("Cost versus fidelity across explanation methods")

    ax.set_xlim(6e-4, 3.5)
    ax.set_ylim(0.20, 0.46)

    ax.xaxis.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    adjust_text(texts, ax=ax, expand_points=(1.4, 1.6), expand_text=(1.2, 1.4), force_text=(0.6, 0.8))

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_EHYDRA, markersize=10, label='eHYDRA'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_NO_BG, markersize=9, label='No background required'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_BG, markersize=9, label='Requires background / coalition sampling'),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=False)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "cost_vs_fidelity.png")
    fig.savefig(out_path, dpi=220)
    print(f"-> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()