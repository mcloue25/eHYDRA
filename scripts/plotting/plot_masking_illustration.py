'''
Used to create the plot in the "How We Test Faithfulness" viva slide: 

Usage:
    python3 scripts/plotting/plot_masking_illustration.py
'''
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from utils.plot_config import MODE_COLOURS, apply_slide_style

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "plots", "viva_slides")
SALIENCY_CMAP = "jet" 
SHOW_SALIENCY_PANEL = True  

apply_slide_style({
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
})


def make_synthetic_example(L=150, seed=7):
    ''' Builds a series with one clear localised bump and an otherwise flat, low-saliency background
    '''
    rng = np.random.RandomState(seed)
    t = np.arange(L)

    baseline = 0.15 * np.sin(2 * np.pi * t / 40)
    noise = rng.normal(0, 0.05, size=L)

    bump_center, bump_width = 55, 18
    bump = 1.3 * np.exp(-0.5 * ((t - bump_center) / (bump_width / 2.5)) ** 2)

    x = baseline + bump + noise

    raw_importance = np.abs(bump) + 0.03
    kernel = np.ones(7) / 7
    importance = np.convolve(raw_importance, kernel, mode="same")  # smooth so the curve doesn't look jagged
    importance += rng.normal(0, 0.01, size=L).clip(min=0)
    importance = np.clip(importance, 0, None)
    return t, x, importance



def select_window(importance, width, mode, rng=None, exclude=None):
    ''' Slides a fixed-width window across the series; argmax/argmin of cumulative importance for top/bottom, uniform random
    '''
    L = len(importance)
    starts = np.arange(0, L - width + 1)
    cum = np.array([importance[s:s + width].sum() for s in starts])

    if mode == "top":
        start = starts[np.argmax(cum)]
    elif mode == "bottom":
        start = starts[np.argmin(cum)]
    elif mode == "random":
        candidates = starts
        if exclude is not None:
            # keep the random window from landing adjacent to an excluded one, not just overlapping it
            buffer = max(6, width // 3)
            mask = np.ones(len(starts), dtype=bool)
            for (e_start, e_width) in exclude:
                too_close = ~((starts + width + buffer <= e_start) |
                              (starts >= e_start + e_width + buffer))
                mask &= ~too_close
            candidates = starts[mask] if mask.any() else starts
        start = rng.choice(candidates)
    else:
        raise ValueError(mode)
    return start



def colour_coded_line(ax, t, x, values, cmap=SALIENCY_CMAP, vmin=0, vmax=100, linewidth=2.4, zorder=3):
    ''' Draws (t, x) as a line coloured by values at each point, via LineCollection. Returns the LineCollection for the colorbar
    '''
    points = np.array([t, x]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=vmin, vmax=vmax)
    seg_values = (values[:-1] + values[1:]) / 2  # colour each segment by its midpoint value
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, zorder=zorder, capstyle="round")
    lc.set_array(seg_values)
    ax.add_collection(lc)
    return lc


def main():
    ''' Main function for saliency viva plot
    '''
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(42)

    t, x, importance = make_synthetic_example()
    L = len(t)
    width = max(1, round(0.10 * L))

    top_start = select_window(importance, width, "top")
    bottom_start = select_window(importance, width, "bottom")
    random_start = select_window(importance, width, "random", rng=rng, exclude=[(top_start, width), (bottom_start, width)])
    windows = [("top", top_start), ("random", random_start), ("bottom", bottom_start)]
    normalised_saliency = 100 * importance / importance.max()  # 0-100 scale, matches the heatmap convention

    if SHOW_SALIENCY_PANEL:
        fig = plt.figure(figsize=(10, 6.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[30, 1], hspace=0.14, wspace=0.04)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
        cax = fig.add_subplot(gs[0, 1])
        ax_spacer = fig.add_subplot(gs[1, 1])
        ax_spacer.axis("off")  # empty cell, just keeps column widths aligned between rows
    else:
        fig = plt.figure(figsize=(10, 4.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[30, 1], wspace=0.04)
        ax_top = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax_bot = None

    # NOTE - top panel: series coloured by saliency
    lc = colour_coded_line(ax_top, t, x, normalised_saliency)
    ax_top.set_xlim(t.min(), t.max())
    ax_top.set_ylim(x.min() - 0.15, x.max() * 1.30)

    # boundary lines + label for each window, centred on its own midpoint
    label_y = x.max() * 1.10
    for mode, start in windows:
        color = MODE_COLOURS[mode]
        ax_top.axvline(start, color=color, linewidth=1.4, linestyle="--", zorder=4)
        ax_top.axvline(start + width, color=color, linewidth=1.4, linestyle="--", zorder=4)
        ax_top.text(start + width / 2, label_y, mode.capitalize(),
                    color=color, fontsize=11, fontweight="bold",
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, linewidth=1.1, alpha=0.95))

    ax_top.set_ylabel("Signal value")
    ax_top.set_title("Masking strategies: top-saliency, random, and bottom-saliency windows", pad=14)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    if SHOW_SALIENCY_PANEL:
        ax_top.set_xlabel("")
    else:
        ax_top.set_xlabel("Timestep $t$")

    cbar = fig.colorbar(lc, cax=cax)
    cbar.set_label("Normalised\nsaliency", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # NOTE - bottom panel: the saliency curve driving the window choice
    if SHOW_SALIENCY_PANEL:
        ax_bot.fill_between(t, importance, color="#555555", alpha=0.12, zorder=1)
        ax_bot.plot(t, importance, color="#333333", linewidth=1.6, zorder=3)

        for mode, start in windows:
            color = MODE_COLOURS[mode]
            ax_bot.axvspan(start, start + width, color=color, alpha=0.20, zorder=2)

        ax_bot.set_xlabel("Timestep $t$")
        ax_bot.set_ylabel(r"Saliency $I_t^{(c)}(x)$")
        ax_bot.spines["top"].set_visible(False)
        ax_bot.spines["right"].set_visible(False)
        ax_bot.set_ylim(0, importance.max() * 1.25)
        ax_bot.set_xlim(t.min(), t.max())

    fig.savefig(os.path.join(OUT_DIR, "masking_strategy_illustration.png"), dpi=220, bbox_inches="tight")
    print(f"Saved to : {os.path.join(OUT_DIR, 'masking_strategy_illustration.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()