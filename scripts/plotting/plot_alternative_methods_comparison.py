'''
Plot comparison against alternbative explainer methods as a horizontal bar chart of mean 
AUCS_top on the full 128-dataset archive, eHYDRA vs. the three tsCaptum baselines. 
Usage:
    python3 scripts/plotting/plot_alternative_methods_comparison.py
'''
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.plot_config import METHOD_COLOURS, apply_slide_style

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "plots", "viva_slides")

apply_slide_style({
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 13,
})

# Values copied from the tscaptum results
METHODS = [
    ("Shapley Sampling", 0.370, 0.267),
    ("Feature Ablation", 0.363, 0.265),
    ("eHYDRA", 0.358, 0.289),
    ("MrSQM", 0.246, 0.228),
]

NAME_TO_KEY = {
    "eHYDRA": "ehydra",
    "Shapley Sampling": "shapley_sampling",
    "Feature Ablation": "feature_ablation",
    "MrSQM": "mrsqm",
}


def main():
    ''' 
    '''
    os.makedirs(OUT_DIR, exist_ok=True)
    ordered = sorted(METHODS, key=lambda m: m[1], reverse=True)
    names = [m[0] for m in ordered]
    values = [m[1] for m in ordered]
    stds = [m[2] for m in ordered]

    fig, ax = plt.subplots(figsize=(9, 5))

    y_pos = list(range(len(names)))
    bar_color = [METHOD_COLOURS[NAME_TO_KEY[n]] for n in names]

    for y, val, std, bc, name in zip(y_pos, values, stds, bar_color, names):
        ax.barh(y, val, color=bc, edgecolor="white", height=0.58, linewidth=1.0, zorder=3)

    # eHYDRA gets bold/highlighted labels so it stands out against the three baselines
    for y, val, std, name in zip(y_pos, values, stds, names):
        weight = "bold" if name == "eHYDRA" else "normal"
        color = METHOD_COLOURS["ehydra"] if name == "eHYDRA" else "#333333"
        ax.text(val + 0.015, y, f"{val:.3f}", va="center", ha="left", fontsize=13, fontweight=weight, color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    for tick, name in zip(ax.get_yticklabels(), names):
        tick.set_fontweight("bold" if name == "eHYDRA" else "normal")
        tick.set_color(METHOD_COLOURS["ehydra"] if name == "eHYDRA" else "#333333")

    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.16)
    ax.set_xlabel("Mean AUCS$_{top}$", labelpad=8)
    ax.set_title("eHYDRA vs. alternative attribution methods", pad=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color="#444444")
    ax.xaxis.grid(True, alpha=0.18, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.5, -0.02, "Full 128-dataset UCR archive. WindowSHAP omitted (evaluated with a different metric).", ha="center", va="top", fontsize=9, color="#666666", style="italic")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "alternative_methods_comparison.png")
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    print(f"-> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()