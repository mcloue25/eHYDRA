#!/usr/bin/env python3
"""

Script for comparison the saliency of eHYDRA, MrSQM & LR

Usage:
    python3 scripts/plotting/plot_ehydra_lr_mrsqm_comparison.py
    python3 scripts/plotting/plot_ehydra_lr_mrsqm_comparison.py --dataset GunPoint --sample-index 0
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, REPO_ROOT)

from utils.data_utils import load_dataset
from classes.models.hydra_explainable import HydraModelExplainable
from classes.models.lr_explainable import LRRawExplainableModel
from classes.models.mrsqm_explainable import MrSQMExplainableModel
from utils.plot_config import METHOD_COLOURS, METHOD_LABELS, apply_thesis_style

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "plots", "saliency_comparison")

SALIENCY_CMAP = "jet"
LR_COLOUR = "#DD8452"

apply_thesis_style()


def colour_coded_line(ax, t, x, values, cmap=SALIENCY_CMAP, vmin=0, vmax=100, linewidth=2.2, zorder=3):
    points = np.array([t, x]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=vmin, vmax=vmax)
    seg_values = (values[:-1] + values[1:]) / 2
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, zorder=zorder, capstyle="round")
    lc.set_array(seg_values)
    ax.add_collection(lc)
    return lc


def normalise_0_100(v):
    v = np.abs(v)
    if v.max() > 0:
        return 100 * v / v.max()
    return v


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="GunPoint")
    parser.add_argument("--sample-index", type=int, default=0, help="index among correctly-classified test samples to show")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # load_dataset() already returns plain 2D (N, T) arrays -- no
    # adapt_hydra_input_shape() conversion needed here (that helper is for
    # going the other way, 3D (N, 1, T) generator output down to 2D, which
    # doesn't apply to data coming straight out of load_dataset()).
    x_train, y_train, x_test, y_test, le = load_dataset(args.dataset)
    L = x_train.shape[-1]

    # NOTE - Fit eHYDRA, MrSQM & LR models
    hydra = HydraModelExplainable(input_dim=L, seed=args.seed)
    hydra.fit(x_train, y_train)

    lr = LRRawExplainableModel()
    lr.fit(x_train, y_train)

    mrsqm = MrSQMExplainableModel()
    mrsqm.fit(x_train, y_train)

    # DEBUG -- remove once diagnosed
    mrsqm_test_acc = (mrsqm.predict(x_test) == y_test).mean()
    print(f"[MrSQM debug] test accuracy: {mrsqm_test_acc:.4f}")

    # pick a correctly-classified test sample, per the evaluation protocol's restriction
    hydra_preds = hydra.predict(x_test)
    correct_idx = np.where(hydra_preds == y_test)[0]
    if len(correct_idx) == 0:
        raise RuntimeError("No correctly classified test samples found.")
    sample_idx = correct_idx[args.sample_index]
    x_single = x_test[sample_idx]
    true_label = y_test[sample_idx]

    # Each explainer has its own explain() signature: eHYDRA takes
    # class_index (None -> predicted class), LR takes y (None -> predicted
    # class), and MrSQM takes no class-selection argument at all (it
    # explains whatever the wrapped MrSQMClassifier itself predicted).
    hydra_saliency = normalise_0_100(hydra.explain(x_single, class_index=None))
    lr_saliency = normalise_0_100(lr.explain(x_single, y=None))
    mrsqm_saliency = normalise_0_100(mrsqm.explain(x_single))

    t = np.arange(L)

    fig, axes = plt.subplots(4, 1, figsize=(9, 9.5), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1.1], "hspace": 0.35})

    panels = [
        (METHOD_LABELS["ehydra"], hydra_saliency, axes[0]),
        ("LR", lr_saliency, axes[1]),
        (METHOD_LABELS["mrsqm"], mrsqm_saliency, axes[2]),
    ]

    lc_for_cbar = None
    for name, saliency, ax in panels:
        lc = colour_coded_line(ax, t, x_single, saliency)
        lc_for_cbar = lc
        ax.set_xlim(t.min(), t.max())
        ax.set_ylim(x_single.min() - 0.15, x_single.max() * 1.15 if x_single.max() > 0 else x_single.max() + 0.15)
        ax.set_title(f"GunPoint \u2014 {name}", loc="left", fontsize=12)
        ax.set_ylabel("Signal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if name == METHOD_LABELS["ehydra"]:
            ax.text(0.02, 0.90, f"True class: {le.inverse_transform([true_label])[0]}", transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round", fc="#eef2f7", ec="#888888", alpha=0.9))

    cbar = fig.colorbar(lc_for_cbar, ax=axes[:3], location="right", fraction=0.025, pad=0.02)
    cbar.set_label("Normalised saliency", fontsize=10)

    # NOTE - bottom panel for saliency profile comparison
    ax_profile = axes[3]
    ax_profile.plot(t, hydra_saliency, color=METHOD_COLOURS["ehydra"], linewidth=1.8, label=METHOD_LABELS["ehydra"])
    ax_profile.plot(t, lr_saliency, color=LR_COLOUR, linewidth=1.8, label="LR")
    ax_profile.plot(t, mrsqm_saliency, color=METHOD_COLOURS["mrsqm"], linewidth=1.8, label=METHOD_LABELS["mrsqm"])
    ax_profile.set_title(f"{args.dataset} \u2014 saliency profile comparison", loc="left", fontsize=12)
    ax_profile.set_xlabel("Time step")
    ax_profile.set_ylabel("Saliency")
    ax_profile.set_ylim(0, 105)
    ax_profile.legend(loc="upper right", ncol=3, fontsize=9)
    ax_profile.spines["top"].set_visible(False)
    ax_profile.spines["right"].set_visible(False)

    fig.suptitle(f"{args.dataset}: eHYDRA vs LR vs MrSQM saliency comparison (sample {sample_idx})", fontsize=13, fontweight="bold", y=0.995)
    out_path = os.path.join(OUT_DIR, f"{args.dataset}_ehydra_lr_mrsqm_{sample_idx}.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()