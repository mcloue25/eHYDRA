'''
Qualitative saliency overlay for HYDRA vs TSHAP (all three backgrounds) on a single example sample

Usage:
    python scripts/plotting/plot_tshap_qualitative.py \
        --dataset GunPoint \
        --sample-index 0 \
        --output outputs/figures/tshap_qualitative/GunPoint_example.png
'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.captum_comparison import (
    CaptumComparison,
    HydraTsCaptumAdapter,
    explain_one_sample_tshap,
    import_tshap_explainer,
    tshap_backgrounds,
    tshap_window_stride,
)
from classes.models.hydra_explainable import HydraModelExplainable
from classes.qualitative_plots import clean_saliency, normalize_saliency_0_100, smooth_curve
from utils.data_utils import load_dataset


def interpolate_series(ts: np.ndarray, interp_points: int = 5000):
    ''' Dense-interpolates the raw signal & returns the original x-grid so each method's saliency can be interpolated on the same basis
    '''
    ts = np.asarray(ts, dtype=float).squeeze()
    x_orig = np.linspace(0, len(ts) - 1, num=len(ts))
    x_dense = np.linspace(0, len(ts) - 1, num=interp_points)
    ts_interp = interp1d(x_orig, ts, kind="linear")
    return x_orig, x_dense, ts_interp(x_dense)


def interpolate_saliency(saliency: np.ndarray, x_orig: np.ndarray, x_dense: np.ndarray) -> np.ndarray:
    ''' Dense interpolates one method's saliency onto the signal's x-grid, after normalising to [0, 100]
    '''
    saliency = np.asarray(saliency, dtype=float).squeeze()
    sal_interp = interp1d(x_orig, normalize_saliency_0_100(saliency), kind="linear")
    return sal_interp(x_dense)


def build_qualitative_figure(dataset: str, idx: int, y_true, ts: np.ndarray, method_saliencies: list[tuple[str, np.ndarray]], output_path: Path):
    ''' Main fucntion for actually plotting the qaulitative comparison, 
            1 row per method
            1 bottom row overlaying them all
    '''
    n_methods = len(method_saliencies)

    x_orig, x_dense, y_dense = interpolate_series(ts)
    dense_saliencies = {label: interpolate_saliency(raw, x_orig, x_dense) for label, raw in method_saliencies}
    smooth_saliencies = {label: smooth_curve(dense, window=101) for label, dense in dense_saliencies.items()}

    fig = plt.figure(figsize=(8.8, 2.0 * n_methods + 2.2), constrained_layout=True)
    gs = fig.add_gridspec(n_methods + 1, 2, width_ratios=[30, 1], height_ratios=[1] * n_methods + [0.9])

    axes = []
    for i in range(n_methods):
        ax = fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
    ax_profile = fig.add_subplot(gs[n_methods, 0], sharex=axes[0])
    cax = fig.add_subplot(gs[0:n_methods, 1])

    last_sc = None
    for i, (ax, (label, _)) in enumerate(zip(axes, method_saliencies)):
        last_sc = ax.scatter(x_dense, y_dense, c=dense_saliencies[label], cmap="jet", marker=".", s=1.5, vmin=0, vmax=100)
        ax.set_title(f"{dataset} — {label}")
        ax.set_ylabel("Signal")
        if i == 0:
            ax.text(0.01, 0.95, f"True class: {y_true}", transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", alpha=0.15))

    cbar = fig.colorbar(last_sc, cax=cax)
    cbar.set_label("Normalised saliency")

    for label, _ in method_saliencies:
        ax_profile.plot(x_dense, smooth_saliencies[label], label=label, linewidth=1.8)
    ax_profile.set_title(f"{dataset} — saliency profile comparison")
    ax_profile.set_xlabel("Time step")
    ax_profile.set_ylabel("Saliency")
    ax_profile.set_ylim(0, 150)  # 15% headroom above the 0-100 saliency scale, for the legend box
    ax_profile.legend(frameon=False, fontsize=8, ncol=2, loc='upper right')

    for ax in axes + [ax_profile]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{dataset}: HYDRA vs TSHAP saliency comparison (sample {idx})", fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    ''' 
    '''
    parser = argparse.ArgumentParser(description="Qualitative HYDRA vs TSHAP saliency overlay.")
    parser.add_argument("--dataset", default="GunPoint")
    parser.add_argument("--sample-index", type=int, default=0, help="Index among correctly-classified test samples (0-based).")
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/tshap_qualitative/example.png"))
    parser.add_argument("--tshap-window-fraction", type=float, default=0.10)
    parser.add_argument("--tshap-max-stride", type=int, default=5)
    parser.add_argument("--tshap-train-background-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    '''
    '''
    args = parse_args()
    X_train, y_train, X_test, y_test, _ = load_dataset(args.dataset)
    hydra_model = HydraModelExplainable(input_dim=X_train.shape[-1])
    hydra_model.fit(X_train, y_train)
    adapter = HydraTsCaptumAdapter(hydra_model)

    preds = hydra_model.predict(X_test)
    correct_idx = np.where(preds == y_test)[0]
    if len(correct_idx) == 0:
        raise RuntimeError(f"No correctly classified test samples for {args.dataset}.")
    if args.sample_index >= len(correct_idx):
        raise ValueError(
            f"--sample-index {args.sample_index} out of range "
            f"({len(correct_idx)} correctly classified samples available)."
        )
    idx = correct_idx[args.sample_index]
    x = np.asarray(X_test[idx], dtype=np.float32)
    pred_label = int(preds[idx])

    print(f"{args.dataset}: sample {idx} (correct-sample #{args.sample_index}), "
          f"predicted class {pred_label}")

    # NOTE - Minimal CaptumComparison instance to reuse its hydra_importance()
    comparison = CaptumComparison(
        datasets=[args.dataset],
        include_mrsqm=False,
        compute_deletion_curves=False,
        include_tshap=False,
    )
    hydra_imp = np.abs(comparison.hydra_importance(hydra_model, x, pred_label))

    # NOTE - TSHAP
    TSHAPExplainer = import_tshap_explainer()
    window_length, stride = tshap_window_stride(X_train.shape[-1], args.tshap_window_fraction, args.tshap_max_stride)
    print(f"TSHAP window_length={window_length}, stride={stride}")
    tshap_explainer = TSHAPExplainer(window_length=window_length, stride=stride, interpolation=True, roi=False)

    # Generate backgrounds
    backgrounds = tshap_backgrounds(
        X_train, include_train=True,
        n_train_background_samples=args.tshap_train_background_samples,
        seed=args.seed,
    )

    # Get importances under all background conditions 
    tshap_imps = {name: explain_one_sample_tshap(tshap_explainer, adapter, x, pred_label, baseline) for name, baseline in backgrounds.items()}

    # HYDRA + three TSHAP conditions, stacked, matching the original scatter/colorbar/profile layout
    methods = [("HYDRA", hydra_imp)] + [(f"TSHAP ({name.replace('tshap_', '')})", imp) for name, imp in tshap_imps.items()]


    # NOTE - Plot the figure comparing them all
    build_qualitative_figure(
        dataset=args.dataset,
        idx=int(idx),
        y_true=int(y_test[idx]),
        ts=x,
        method_saliencies=methods,
        output_path=args.output,
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()