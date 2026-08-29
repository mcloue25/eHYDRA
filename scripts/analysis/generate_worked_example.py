'''
Used for the worked example subsection in the methodology validation chapter
Trains a HydraModelExplainable with the default hyperparams then does a single slice of get_saliency_map()'s computation to 
isolate one max feature and one min feature at one convolution position

Usage:
    python3 scripts/analysis/generate_worked_example.py
    python3 scripts/analysis/generate_worked_example.py --series-length 16 --seed 42
'''
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, REPO_ROOT)

from classes.models.hydra_explainable import HydraModelExplainable

DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "outputs", "plots", "methodology")


def build_toy_dataset(series_length, n_train=60, seed=42):
    '''
    '''
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 0.3, size=(n_train, series_length)).astype(np.float32)
    y = rng.randint(0, 2, size=n_train)

    bump_start = series_length // 3
    bump_len = max(2, series_length // 4)
    for i in range(n_train):
        if y[i] == 1:
            X[i, bump_start:bump_start + bump_len] += 1.5
    return X, y


def instrument_one_slice(model, x_single, class_index, dilation_index=0, diff_index=0, group=0):
    ''' Reads the tensors get_saliency_map() computes internally for one (dilation_index, diff_index, group) slice and returns every value needed 
    '''
    transform = model.transform  # UpdatedHydra instance
    ridge = model.classifier
    scaler = model.scaler

    x_t = model._to_tensor(x_single[None, :])  # (1, 1, L)
    L = x_t.shape[-1]

    if ridge.coef_.ndim == 1:
        w_ridge = ridge.coef_
    elif ridge.coef_.shape[0] == 1:
        w_ridge = ridge.coef_[0]
    else:
        w_ridge = ridge.coef_[class_index]

    sigma = scaler.sigma.detach().cpu().numpy()
    w_adjusted = w_ridge / sigma

    if transform.divisor > 1:
        diff_x_t = torch.diff(x_t)

    d = int(transform.dilations[dilation_index].item())
    p = int(transform.paddings[dilation_index].item())

    import torch.nn.functional as F
    with torch.inference_mode():
        _Z = F.conv1d(
            x_t if diff_index == 0 else diff_x_t,
            transform.W[dilation_index, diff_index],
            dilation=d, padding=p,
        ).view(1, transform.h, transform.k, -1)

        max_values, max_indices = _Z.max(2)
        min_values, min_indices = _Z.min(2)

    max_vals_np = max_values.squeeze(0).cpu().numpy()   # (h, T_out)
    max_idx_np = max_indices.squeeze(0).cpu().numpy()   # (h, T_out)
    min_vals_np = min_values.squeeze(0).cpu().numpy()
    min_idx_np = min_indices.squeeze(0).cpu().numpy()

    # feature offset for this (dilation_index, diff_index)
    block_size = transform.h * transform.k
    n_blocks_before = 0
    for di in range(dilation_index):
        n_blocks_before += transform.divisor * 2  # max + min block, per diff_index
    n_blocks_before += diff_index * 2
    feature_offset = n_blocks_before * block_size

    weights_max = w_adjusted[feature_offset: feature_offset + block_size].reshape(transform.h, transform.k)
    feature_offset += block_size
    weights_min = w_adjusted[feature_offset: feature_offset + block_size].reshape(transform.h, transform.k)

    T_out = max_vals_np.shape[1]

    # pick the position with the largest positive max activation, purely so the worked example is non-trivial
    candidate_u = int(np.argmax(np.where(max_vals_np[group] > 0, max_vals_np[group], -np.inf)))

    kernel_max = int(max_idx_np[group, candidate_u])
    activation_max = float(max_vals_np[group, candidate_u])
    weight_max = float(weights_max[group, kernel_max])
    max_contrib_value = activation_max * weight_max if activation_max > 0 else 0.0

    kernel_min = int(min_idx_np[group, candidate_u])
    weight_min = float(weights_min[group, kernel_min])
    min_contrib_value = weight_min  # unscaled, as in the real implementation

    receptive_field = sorted({candidate_u - p + m * d for m in range(9)})
    receptive_field = [t for t in receptive_field if 0 <= t < L]

    return {
        "L": L, "dilation": d, "padding": p, "group": group,
        "u": candidate_u, "T_out": T_out,
        "kernel_max": kernel_max, "activation_max": activation_max,
        "weight_max": weight_max, "max_contrib_value": max_contrib_value,
        "kernel_min": kernel_min, "weight_min": weight_min,
        "min_contrib_value": min_contrib_value,
        "receptive_field": receptive_field,
    }


def make_figure(x_single, slice_info, out_path):
    ''' Two-panel figure: the series with its receptive field shaded and per-timestep contribution bars
    '''
    L = slice_info["L"]
    rf = slice_info["receptive_field"]
    max_contrib = np.zeros(L)
    min_contrib = np.zeros(L)
    for t in rf:
        max_contrib[t] = slice_info["max_contrib_value"]
        min_contrib[t] = slice_info["min_contrib_value"]
    total = max_contrib + min_contrib

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

    axes[0].plot(x_single, color="black", linewidth=1.2)
    axes[0].axvspan(min(rf) - 0.5, max(rf) + 0.5, color="tab:orange", alpha=0.2)
    axes[0].set_title(f"Toy series $x$ (L={L}), receptive field $R(u={slice_info['u']})$ shaded")
    axes[0].set_ylabel("Value")

    width = 0.28
    t_range = np.arange(L)
    axes[1].bar(t_range - width, max_contrib, width=width, label="Max feature contribution", color="tab:blue")
    axes[1].bar(t_range, min_contrib, width=width, label="Min feature contribution", color="tab:red")
    axes[1].bar(t_range + width, total, width=width, label="Sum ($S_t^{(c)}$ from this slice)", color="tab:green")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Timestep $t$")
    axes[1].set_ylabel("Contribution")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Per-timestep contribution of the isolated max/min feature pair")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"\nFigure : {out_path}")
    plt.close(fig)


def print_latex_values(slice_info, class_index, pred_label):
    '''Prints every value needed to fill in the LaTeX worked-example table/prose.'''
    L = slice_info["L"]
    rf = slice_info["receptive_field"]
    print("\n" + "=" * 70)
    print("VALUES FOR THE LATEX WORKED EXAMPLE (replace every [bracket])")
    print("=" * 70)
    print(f"L (series length) = {L}")
    print(f"predicted class_index used = {class_index}  (predicted label: {pred_label})")
    print(f"dilation d  = {slice_info['dilation']}")
    print(f"padding p = {slice_info['padding']}")
    print(f"group index (informational) = {slice_info['group']}")
    print(f"convolution position u = {slice_info['u']}")
    print(f"receptive field R(u) = {rf}  (length {len(rf)})")
    print()
    print(f"winning MAX kernel index = {slice_info['kernel_max']}")
    print(f"activation (max, a) = {slice_info['activation_max']:.4f}")
    print(f"adjusted weight (max, w~_max) = {slice_info['weight_max']:+.4f}")
    print(f"max feature contribution = {slice_info['max_contrib_value']:+.4f}  "
          f"(= activation * weight, per t in R(u))")
    print()
    print(f"  winning MIN kernel index = {slice_info['kernel_min']}")
    print(f"  adjusted weight (min, w~_min) = {slice_info['weight_min']:+.4f}")
    print(f"  min feature contribution = {slice_info['min_contrib_value']:+.4f}  "
          f"(= weight alone, per t in R(u))")
    print()
    total = slice_info["max_contrib_value"] + slice_info["min_contrib_value"]
    print(f"  sum (per t in R(u)) = {total:+.4f}")
    print("=" * 70)
    print("\nTable~\\ref{tab:worked_example} row values (per timestep, 0 outside R(u)):")
    print(f"Max feature row : {slice_info['max_contrib_value']:+.3f} at t in {rf}, else 0")
    print(f"Min feature row : {slice_info['min_contrib_value']:+.3f} at t in {rf}, else 0")
    print(f"Sum row : {total:+.3f} at t in {rf}, else 0")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--series-length", type=int, default=16,
                         help="Example series length L (default 16 -- L=8 risks every "
                              "position's receptive field running off both boundaries "
                              "given the 9-point kernel span; 16 keeps at least one "
                              "interior position with a full unclipped receptive field)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-index", type=int, default=0, help="which toy-dataset sample to explain")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    # Build the example dataset
    X, y = build_toy_dataset(args.series_length, seed=args.seed)
    model = HydraModelExplainable(input_dim=args.series_length, seed=args.seed)
    model.fit(X, y)

    x_single = X[args.sample_index]
    pred = model.predict(x_single[None, :])[0]
    if len(model.classifier.classes_) > 2:
        class_index = int(np.where(model.classifier.classes_ == pred)[0][0])
    else:
        class_index = 0

    full_saliency = model.explain(x_single, class_index=None, verbose=False)  # real, unmodified path -- used as a consistency check below

    slice_info = instrument_one_slice(model, x_single, class_index,
                                       dilation_index=0, diff_index=0, group=0)

    # sanity check: the isolated slice is one component among many summed together,
    # so it should be a plausible fraction of the real total, not larger than it
    rf = slice_info["receptive_field"]
    isolated_total = slice_info["max_contrib_value"] + slice_info["min_contrib_value"]
    real_vals_at_rf = full_saliency[rf]
    print(f"\n[CHECK] isolated slice contribution per t in R(u): {isolated_total:+.4f}")
    print(f"[CHECK] real full saliency map at same positions:   {np.round(real_vals_at_rf, 4)}")
    print(f"[CHECK] (isolated value is one of many summed components -- "
          f"should be a plausible fraction of the real total, not identical to it, "
          f"since the real map sums this slice plus every other dilation/branch/group)")

    print_latex_values(slice_info, class_index, pred)
    make_figure(x_single, slice_info, os.path.join(args.out_dir, "saliency_worked_example.png"))


if __name__ == "__main__":
    main()