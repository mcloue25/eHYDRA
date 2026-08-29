'''
Score eHYDRA saliency against synthetic ground-truth Phi per cluster using the trained models from train_hydra_on_synthetic.py.

Usage:
    python3 scripts/analysis/evaluate_synthetic_ground_truth.py
    python3 scripts/analysis/evaluate_synthetic_ground_truth.py --cluster high_frequency
    python3 scripts/analysis/evaluate_synthetic_ground_truth.py --cluster spiky --spiky-saliency-mode predicted
'''
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SYNTH_DIR = os.path.join(REPO_ROOT, "scripts", "synthetic_dataset_generation")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, SYNTH_DIR)
sys.path.insert(0, REPO_ROOT)

import dataset_io  # from scripts/synthetic_dataset_generation
from saliency_ground_truth_metrics import evaluate_one_sample
from utils.data_utils import adapt_hydra_input_shape
from classes.models.hydra_margin_explainable import HydraMarginExplainable
from utils.globals_config import (DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, GROUND_TRUTH_EVAL_DIR, CLUSTER_KEYS, TOPK_FRACTIONS)


def spiky_ordinal_margin_saliency(margin_model, base_model, x):
    ''' Margin saliency (S^pred - S^runnerup), 
        sign-corrected onto Phi's fixed "positive = higher duration" convention using Spiky's ordinal class indices
    '''
    x_batched = x[None, :]
    decision = np.asarray(base_model.decision_function(x_batched))
    scores = decision[0] if decision.ndim == 2 else decision
    order = np.argsort(scores)[::-1]
    pred_index, runnerup_index = int(order[0]), int(order[1])

    margin = margin_model.explain(x, verbose=False)
    sign_correction = 1.0 if pred_index > runnerup_index else -1.0
    return sign_correction * margin



def evaluate_one_cluster(cluster_key, data_dir, model_dir, n_eval, magnitude_tol, spiky_saliency_mode, save_arrays, out_dir):
    ''' Loads a trained cluster's model & test set, computes eHYDRA saliency per test sample and scores it against Phi. 
    Returns: 
        A per-sample metrics DataFrame
    '''
    model_path = os.path.join(model_dir, f"{cluster_key}_hydra_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test, y_test, attribs_test, meta = dataset_io.load_dataset(data_dir, cluster_key, "test")
    X_test_2d = adapt_hydra_input_shape(X_test)
    Phi_test = attribs_test[:, 0, :]  # (N, T)

    n_eval = min(n_eval, X_test_2d.shape[0]) if n_eval else X_test_2d.shape[0]

    margin_model = None
    if cluster_key == "spiky":
        print(f"  [spiky] saliency mode = {spiky_saliency_mode}")
        if spiky_saliency_mode == "margin":
            margin_model = HydraMarginExplainable(model)

    rows = []
    phi_all = [] if save_arrays else None
    for i in range(n_eval):
        x = X_test_2d[i]
        Phi = Phi_test[i]

        if cluster_key == "spiky" and spiky_saliency_mode == "margin":
            phi = spiky_ordinal_margin_saliency(margin_model, model, x)
        else:
            phi = model.explain(x, class_index=None, verbose=False)

        if save_arrays:
            phi_all.append(phi)

        m = evaluate_one_sample(phi, Phi, topk_fractions=TOPK_FRACTIONS, magnitude_tol=magnitude_tol)
        m["sample"] = i
        m["true_label"] = int(y_test[i])
        rows.append(m)
        if (i + 1) % 25 == 0:
            print(f"... {i + 1}/{n_eval}")

    df = pd.DataFrame(rows)

    if save_arrays:
        arrays_path = os.path.join(out_dir, f"{cluster_key}_arrays.npz")
        np.savez_compressed(arrays_path, X=X_test_2d[:n_eval], phi=np.array(phi_all), Phi=Phi_test[:n_eval], y=y_test[:n_eval])
        print(f"  raw arrays -> {arrays_path}")

    return df


def summarise(df):
    ''' Calculate aggregate summaries of all non label cols
    '''
    numeric_cols = [c for c in df.columns if c not in ("sample", "true_label") and pd.api.types.is_numeric_dtype(df[c])]
    return df[numeric_cols].agg(["mean", "median", "std"]).T


def main():
    ''' Main function for evaluating how eHYDRA performs on the synthetic datasets 
    '''
    # Init arg parser and add commands
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster", choices=CLUSTER_KEYS + ["all"], default="all")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", default=GROUND_TRUTH_EVAL_DIR)
    parser.add_argument("--n-eval", type=int, default=None, help="evaluate only the first N test samples per cluster " "(default: all -- eHYDRA saliency is per-sample, this can be slow)")
    parser.add_argument("--magnitude-tol", type=float, default=0.05,
                         help="fraction of max|phi| above which a point counts as "
                              "'phi-relevant' for the confusion-matrix metric (see "
                              "saliency_ground_truth_metrics.py docstring)")
    parser.add_argument("--spiky-saliency-mode", choices=["margin", "predicted"], default="predicted",
                         help="'predicted' (default): plain predicted-class saliency, "
                              "outperforms margin on every sign-based metric -- see module "
                              "docstring. 'margin': ordinal-corrected S^pred - S^runnerup, "
                              "kept for revisiting but not currently recommended.")
    parser.add_argument("--save-arrays", action="store_true", help="also save raw X/phi/Phi/y arrays per cluster to "
                              "<out-dir>/<cluster>_arrays.npz, needed by "
                              "scripts/plotting/plot_ground_truth_comparison.py")
    args = parser.parse_args()

    clusters = CLUSTER_KEYS if args.cluster == "all" else [args.cluster]
    os.makedirs(args.out_dir, exist_ok=True)

    # NOTE - Iterate through clusters and eval results
    all_summaries = {}
    for cluster_key in clusters:
        print(f"[{cluster_key}]")
        df = evaluate_one_cluster(cluster_key, args.data_dir, args.model_dir, args.n_eval, args.magnitude_tol, args.spiky_saliency_mode, args.save_arrays, args.out_dir)
        raw_path = os.path.join(args.out_dir, f"{cluster_key}_per_sample.csv")
        df.to_csv(raw_path, index=False)

        # Summarise & save results
        summary = summarise(df)
        summary_path = os.path.join(args.out_dir, f"{cluster_key}_summary.csv")
        summary.to_csv(summary_path)
        print(summary[["mean", "median"]].round(4).to_string())
        print(f"per-sample: {raw_path}")
        print(f"summary: {summary_path}\n")
        all_summaries[cluster_key] = summary["mean"].to_dict()

    combined_path = os.path.join(args.out_dir, "all_clusters_summary.json")
    existing = {}
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            existing = json.load(f)
    existing.update(all_summaries)
    
    with open(combined_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"combined summary ({len(existing)}/{len(CLUSTER_KEYS)} clusters present) -> {combined_path}")


if __name__ == "__main__":
    main()