'''
Plots the Synthetic Ground Truth grouped bar chart of cosine similarity against ground-truth Phi for eHYDRA vs. TSHAP 

Usage:
    python3 scripts/plotting/plot_synthetic_ground_truth_comparison.py
'''
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.plot_config import METHOD_COLOURS, apply_slide_style

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "plots", "viva_slides")
METHOD_COLOURS = {**METHOD_COLOURS, "tshap": "#8172B2"} 

apply_slide_style({
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11,
    "legend.frameon": False,
    "legend.fontsize": 12,
})

# Experiment results
TSHAP_CONDITIONS = {
    "High-frequency": [0.762, 0.593, 0.717, 0.575],
    "Short/rough": [0.615, 0.377, 0.552, 0.325],
    "Smooth": [0.427, 0.496, 0.350, 0.338],
    "Spiky": [-0.080, -0.130, -0.108, -0.130],
}

EHYDRA = {
    "High-frequency": 0.125,
    "Short/rough": 0.291,
    "Smooth": 0.092,
    "Spiky": -0.011,
}

REGIMES = ["High-frequency", "Short/rough", "Smooth", "Spiky"]


def main():
    '''
    '''
    os.makedirs(OUT_DIR, exist_ok=True)
    ehydra_vals = [EHYDRA[r] for r in REGIMES]
    tshap_means = [np.mean(TSHAP_CONDITIONS[r]) for r in REGIMES]
    tshap_mins = [min(TSHAP_CONDITIONS[r]) for r in REGIMES]
    tshap_maxs = [max(TSHAP_CONDITIONS[r]) for r in REGIMES]

    x = np.arange(len(REGIMES))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    bars_e = ax.bar(x - width / 2, ehydra_vals, width, label="eHYDRA", color=METHOD_COLOURS["ehydra"], edgecolor="white", linewidth=1.0, zorder=3)
    bars_t = ax.bar(x + width / 2, tshap_means, width, label="TSHAP (mean of 4 conditions)", color=METHOD_COLOURS["tshap"], edgecolor="white", linewidth=1.0, zorder=3)

    # thin range bars showing the spread across the 4 TSHAP conditions, so the mean doesn't hide the variation
    for xi, lo, hi in zip(x + width / 2, tshap_mins, tshap_maxs):
        ax.plot([xi, xi], [lo, hi], color="#4a3f66", linewidth=1.6, alpha=0.4, zorder=2)
        ax.plot([xi - 0.05, xi + 0.05], [lo, lo], color="#4a3f66", linewidth=1.6, alpha=0.4, zorder=2)
        ax.plot([xi - 0.05, xi + 0.05], [hi, hi], color="#4a3f66", linewidth=1.6, alpha=0.4, zorder=2)


    for bar, val in zip(bars_e, ehydra_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + (0.03 if val >= 0 else -0.05),
                f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=11, fontweight="bold", color=METHOD_COLOURS["ehydra"], zorder=6,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    
    for bar, val in zip(bars_t, tshap_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + (0.03 if val >= 0 else -0.05),
                f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=11, fontweight="bold", color=METHOD_COLOURS["tshap"], zorder=6,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    ax.axhline(0, color="#444444", linewidth=1.0, zorder=2)

    # flag the Spiky reversal directly on the chart
    spiky_idx = REGIMES.index("Spiky")
    ax.annotate("eHYDRA leads\n(reversal)",
               xy=(spiky_idx, max(ehydra_vals[spiky_idx], tshap_means[spiky_idx]) + 0.05),
               xytext=(spiky_idx - 0.65, 0.55),
               fontsize=11, fontweight="bold", color="#333333", ha="center",
               arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2,
                               connectionstyle="arc3,rad=-0.2"))

    ax.set_xticks(x)
    ax.set_xticklabels(REGIMES)
    ax.set_title("eHYDRA vs. TSHAP: sign/shape agreement by signal regime", pad=16)
    ax.set_ylim(-0.28, 0.98)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#444444")
    ax.get_yaxis().set_visible(False)
    ax.tick_params(axis="x", length=0)

    ax.legend(loc="upper right")
    fig.text(0.5, -0.02, "Cosine similarity with ground truth $\\Phi$, $n=90$ per regime. TSHAP bar: mean of 4 conditions", ha="center", va="top", fontsize=14, color="#666666", style="italic")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "synthetic_ground_truth_comparison.png")
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    print(f"-> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()