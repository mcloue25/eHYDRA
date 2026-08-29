'''
Bar chart of UCR dataset counts per signal-morphology cluster
Usage:
    python3 scripts/plotting/plot_cluster_size_disparity.py
'''
import os

import matplotlib.pyplot as plt

from utils.plot_config import apply_slide_style

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "plots", "viva_slides")

COLOR_FLAG = "#C44E52"     # High-frequency: the small, flagged cluster
COLOR_NORMAL = "#8C9BB5"   # the other three clusters

apply_slide_style({
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 16,
    "xtick.labelsize": 12.5,
})

# Cluster sizes
CLUSTERS = [
    ("High-frequency", 8),
    ("Smooth", 37),
    ("Short/rough", 42),
    ("Spiky", 41),
]


def main():
    ''' 
    '''
    os.makedirs(OUT_DIR, exist_ok=True)

    names = [c[0] for c in CLUSTERS]
    sizes = [c[1] for c in CLUSTERS]
    colors = [COLOR_FLAG if n == "High-frequency" else COLOR_NORMAL for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    bars = ax.bar(names, sizes, color=colors, edgecolor="white", linewidth=1.2, width=0.6, zorder=3)

    for bar, n in zip(bars, sizes):
        weight = "bold" if bar.get_height() == 8 else "normal"
        color = COLOR_FLAG if bar.get_height() == 8 else "#333333"
        label_gap = 1.5
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + label_gap,
                f"n = {n}", ha="center", va="bottom", fontsize=13,
                fontweight=weight, color=color, zorder=5)

    ax.set_title("Signal-morphology cluster sizes", pad=16)
    ax.set_ylim(0, 50)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#444444")
    ax.get_yaxis().set_visible(False)
    ax.tick_params(axis="x", length=0)

    for tick, name in zip(ax.get_xticklabels(), names):
        if name == "High-frequency":
            tick.set_fontweight("bold")
            tick.set_color(COLOR_FLAG)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "cluster_size_disparity.png")
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    print(f"-> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()