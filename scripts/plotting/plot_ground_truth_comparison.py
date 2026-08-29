'''

Qualitative saliency-vs-Phi comparison plots

Usage:
    python3 scripts/plotting/plot_ground_truth_comparison.py
    python3 scripts/plotting/plot_ground_truth_comparison.py --cluster spiky --n-examples 6
    python3 scripts/plotting/plot_ground_truth_comparison.py --cluster high_frequency --background centroid
    python3 scripts/plotting/plot_ground_truth_comparison.py --methods ehydra tshap_roi
'''
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "synthetic_dataset_generation"))
sys.path.insert(0, REPO_ROOT)
from utils.globals_config import DEFAULT_EVAL_DIR, CLUSTER_KEYS

DEFAULT_CALIB_DIR = os.path.join(REPO_ROOT, "scripts", "synthetic_dataset_generation", "calibrated_params")
PLOTS_OUT_DIR = os.path.join(REPO_ROOT, "outputs", "plots", "ground_truth_comparison")  # not DEFAULT_OUT_DIR, that name means something else elsewhere

# separate from plot_config's canonical METHOD_COLOURS/LABELS -- only 3 methods here, with LaTeX phi labels
GT_METHOD_COLORS = {
    "ehydra": "tab:red",
    "tshap_window": "tab:blue",
    "tshap_roi": "tab:green",
}
GT_METHOD_LABELS = {
    "ehydra": r"$\phi$ eHYDRA",
    "tshap_window": r"$\phi$ TSHAP-Window",
    "tshap_roi": r"$\phi$ TSHAP-ROI",
}


def normalize_for_overlay(arr):
    ''' Scales to [-1, 1] by its own max abs value for visual overlay only
    '''
    m = np.max(np.abs(arr))
    return arr / m if m > 1e-12 else arr


def load_arrays(cluster_key, eval_dir, background, methods):
    ''' Loads the combined TSHAP arrays if present, else falls back to the eHYDRA-only file
    '''
    tshap_path = os.path.join(eval_dir, f"{cluster_key}_tshap_arrays.npz")
    ehydra_path = os.path.join(eval_dir, f"{cluster_key}_arrays.npz")

    if os.path.exists(tshap_path):
        data = np.load(tshap_path)
        X, Phi, y = data["X"], data["Phi"], data["y"]
        curves = {}
        for method in methods:
            # eHYDRA is background-independent, everything else is keyed by background condition
            key = "phi_ehydra" if method == "ehydra" else f"phi_{method}_{background}"
            if key in data:
                curves[method] = data[key]
            else:
                print(f"  [warn] {key} not found in {tshap_path} -- "
                      f"was --background {background} used for that run?")
        return X, Phi, y, curves

    if os.path.exists(ehydra_path):
        print(f"  [{cluster_key}] {tshap_path} not found, falling back to eHYDRA-only "
              f"{ehydra_path} (run evaluate_tshap_comparison.py --save-arrays for the full comparison)")
        data = np.load(ehydra_path)
        X, Phi, y = data["X"], data["Phi"], data["y"]
        curves = {"ehydra": data["phi"]} if "ehydra" in methods else {}
        return X, Phi, y, curves

    return None, None, None, None


def plot_one_sample(ax_top, ax_bottom, x, curves, Phi, y_label, cluster_key, calib_params=None):
    ''' Function used to plot one individual sample
    '''
    t = np.arange(len(x))
    ax_top.plot(t, x, color="tab:blue", linewidth=0.8)  # raw series

    nz = np.nonzero(Phi)[0]
    if len(nz) > 0:
        window_start, window_end = nz.min(), nz.max() + 1
        if cluster_key == "spiky" and calib_params is not None:
            # split the window into an approximate burst/plateau shading -- not an exact per-sample boundary
            onset = calib_params["onset"]
            approx_duration = calib_params["max_duration"] * 0.5
            burst_end = int(onset + approx_duration)
            ax_top.axvspan(window_start, burst_end, color="tab:orange", alpha=0.15, label="burst (approx)")
            ax_top.axvspan(burst_end, window_end, color="tab:red", alpha=0.15, label="plateau (approx)")
        else:
            ax_top.axvspan(window_start, window_end, color="tab:red", alpha=0.15)  # ground-truth region

    ax_top.set_title(f"{cluster_key} -- true={y_label}", fontsize=9)
    ax_top.set_ylabel("x(t)", fontsize=8)
    ax_top.tick_params(labelsize=7)

    # ground truth plus every available phi curve, normalised so they're visually comparable
    ax_bottom.plot(t, normalize_for_overlay(Phi), color="black", linewidth=1.4, label=r"$\Phi$ (ground truth)", zorder=10)
    for method, arr in curves.items():
        ax_bottom.plot(t, normalize_for_overlay(arr), color=GT_METHOD_COLORS.get(method, "gray"), linewidth=0.9, alpha=0.85, label=GT_METHOD_LABELS.get(method, method))
    ax_bottom.axhline(0, color="gray", linewidth=0.5)
    ax_bottom.set_ylabel("normalised", fontsize=8)
    ax_bottom.set_xlabel("timestep", fontsize=8)
    ax_bottom.tick_params(labelsize=7)
    ax_bottom.legend(fontsize=6, frameon=False, loc="upper right")


def plot_one_cluster(cluster_key, eval_dir, calib_dir, out_dir, n_examples, background, methods):
    ''' Main function for plotting one cluster wirth of representative samples
    '''
    X, Phi_all, y, curves = load_arrays(cluster_key, eval_dir, background, methods)
    if X is None:
        print(f"[skip] {cluster_key}: no arrays file found -- rerun "
              f"evaluate_tshap_comparison.py or evaluate_synthetic_ground_truth.py with --save-arrays")
        return None
    if not curves:
        print(f"[skip] {cluster_key}: none of {methods} available in the arrays file")
        return None

    calib_params = None
    calib_path = os.path.join(calib_dir, f"{cluster_key}.json")
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            calib_params = json.load(f)

    n_examples = min(n_examples, X.shape[0])
    fig, axes = plt.subplots(2, n_examples, figsize=(4 * n_examples, 6))
    if n_examples == 1:
        axes = axes.reshape(2, 1)

    rng = np.random.RandomState(0)
    idx = rng.choice(X.shape[0], size=n_examples, replace=False)  # fixed seed, same examples every run

    for col, i in enumerate(idx):
        sample_curves = {method: arr[i] for method, arr in curves.items()}
        plot_one_sample(axes[0, col], axes[1, col], X[i], sample_curves, Phi_all[i], y[i],
                         cluster_key, calib_params)

    method_str = "+".join(curves.keys())
    fig.suptitle(f"{cluster_key}: {method_str} vs ground truth "
                 f"(background={background}, {n_examples} random test examples)", fontsize=12)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cluster_key}_ground_truth_comparison.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[{cluster_key}] saved -> {out_path}")
    return out_path


def main():
    ''' 
    '''
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster", choices=CLUSTER_KEYS + ["all"], default="all")
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    parser.add_argument("--calib-dir", default=DEFAULT_CALIB_DIR)
    parser.add_argument("--out-dir", default=PLOTS_OUT_DIR)
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument("--background", choices=["threshold", "centroid"], default="threshold", 
                        help="which TSHAP background condition to plot (eHYDRA is "
                              "background-independent and always shown regardless)")
    parser.add_argument("--methods", nargs="+", default=["ehydra", "tshap_window", "tshap_roi"], choices=["ehydra", "tshap_window", "tshap_roi"])
    args = parser.parse_args()

    clusters = CLUSTER_KEYS if args.cluster == "all" else [args.cluster]
    for cluster_key in clusters:
        plot_one_cluster(cluster_key, args.eval_dir, args.calib_dir, args.out_dir,
                          args.n_examples, args.background, args.methods)


if __name__ == "__main__":
    main()