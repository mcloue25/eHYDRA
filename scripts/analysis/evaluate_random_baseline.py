'''
Random-attribution baseline (TSHAP paper's Table 3 "Random" row): uniform [-1, 1] per timestep, the minimum bar every real method should beat. 
Needs no trained model, only Phi, so it runs standalone. 
Also gives an empirical chance floor for confmat metrics, which have no closed-form null like top-k does.

TODO: only eHYDRA and random are wired in so far. TSHAP, Shapley Value
Sampling, and Feature Ablation still need adding -- random_saliency() is
the template for any method that only needs X/Phi.

Usage:
    python3 scripts/analysis/evaluate_random_baseline.py [--cluster NAME] [--n-draws N]
'''
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SYNTH_DIR = os.path.join(REPO_ROOT, "scripts", "synthetic_dataset_generation")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, SYNTH_DIR)
sys.path.insert(0, REPO_ROOT)

import dataset_io
from saliency_ground_truth_metrics import evaluate_one_sample
from utils.globals_config import (DEFAULT_DATA_DIR, GROUND_TRUTH_EVAL_DIR, CLUSTER_KEYS, TOPK_FRACTIONS)


def random_saliency(T, rng):
    ''' TSHAP paper's own Random baseline: uniform in [-1, 1] per timestep, no relationship to Phi
    '''
    return rng.uniform(-1, 1, size=T)


def evaluate_one_cluster(cluster_key, data_dir, n_draws_per_sample, magnitude_tol, seed):
    ''' 
    '''
    _, _, attribs_test, meta = dataset_io.load_dataset(data_dir, cluster_key, "test")
    Phi_test = attribs_test[:, 0, :]  # (N, T)
    T = Phi_test.shape[-1]

    rng = np.random.RandomState(seed)
    rows = []
    for i in range(Phi_test.shape[0]):
        Phi = Phi_test[i]
        for draw in range(n_draws_per_sample):
            phi = random_saliency(T, rng)
            m = evaluate_one_sample(phi, Phi, topk_fractions=TOPK_FRACTIONS, magnitude_tol=magnitude_tol)
            m["sample"] = i
            m["draw"] = draw
            rows.append(m)

    return pd.DataFrame(rows)


def main():
    ''' 
    '''
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster", choices=CLUSTER_KEYS + ["all"], default="all")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", default=GROUND_TRUTH_EVAL_DIR)
    parser.add_argument("--n-draws", type=int, default=5,
                         help="independent random draws per test sample, averaged for a "
                              "stable per-cluster chance estimate (default 5; validating "
                              "against the analytic value used 500 draws on a fixed Phi)")
    parser.add_argument("--magnitude-tol", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    clusters = CLUSTER_KEYS if args.cluster == "all" else [args.cluster]
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for cluster_key in clusters:
        print(f"[{cluster_key}] {args.n_draws} random draws/sample")
        df = evaluate_one_cluster(cluster_key, args.data_dir, args.n_draws, args.magnitude_tol, args.seed)
        raw_path = os.path.join(args.out_dir, f"{cluster_key}_random_baseline.csv")
        df.to_csv(raw_path, index=False)

        numeric_cols = [c for c in df.columns if c not in ("sample", "draw") and pd.api.types.is_numeric_dtype(df[c])]
        means = df[numeric_cols].mean()
        print(means[["cosine_similarity", "confmat_precision", "confmat_f1", "topk5_precision", "topk10_precision", "topk20_precision"]].round(4).to_string())
        print(f"-> {raw_path}\n")

        summary[cluster_key] = means.to_dict()

    combined_path = os.path.join(args.out_dir, "random_baseline_summary.json")
    existing = {}
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            existing = json.load(f)
    existing.update(summary)
    with open(combined_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"combined -> {combined_path}")


if __name__ == "__main__":
    main()