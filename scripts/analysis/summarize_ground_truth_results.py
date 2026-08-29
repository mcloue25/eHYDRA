'''
Cross-cluster summary for ground truth results

Usage:
    python3 scripts/analysis/summarize_ground_truth_results.py
'''
import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, REPO_ROOT)
from utils.globals_config import DEFAULT_EVAL_DIR, CLUSTER_KEYS

KEY_METRICS = [
    "cosine_similarity", "confmat_precision", "confmat_recall", "confmat_f1",
    "topk5_precision", "topk10_precision", "topk20_precision",
    "topk5_iou", "topk10_iou", "topk20_iou",
]


def load_available(eval_dir):
    found = {}
    for cluster_key in CLUSTER_KEYS:
        path = os.path.join(eval_dir, f"{cluster_key}_per_sample.csv")
        if os.path.exists(path):
            found[cluster_key] = pd.read_csv(path)
        else:
            print(f"[skip] {cluster_key}: {path} not found")
    return found


def build_comparison_table(per_cluster_dfs):
    rows = []
    for cluster_key, df in per_cluster_dfs.items():
        row = {"cluster": cluster_key, "n_samples": len(df)}
        for metric in KEY_METRICS:
            if metric in df.columns:
                row[metric] = df[metric].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("cluster")


def flag_magnitude_sign_gap(table, threshold=0.3):
    ''' Flags cluster where localisation looks fine but sign agreement doesn't 
    '''
    flags = {}
    for cluster_key in table.index:
        topk10_p = table.loc[cluster_key, "topk10_precision"]
        confmat_p = table.loc[cluster_key, "confmat_precision"]
        gap = topk10_p - confmat_p
        flags[cluster_key] = {
            "topk10_precision": topk10_p,
            "confmat_precision": confmat_p,
            "gap": gap,
            "flagged": bool(gap > threshold),
        }
    return flags




def main():
    ''' 
    '''
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    parser.add_argument("--gap-threshold", type=float, default=0.3, help="flag a cluster if topk10_precision - confmat_precision " "exceeds this (default 0.3)")
    args = parser.parse_args()

    per_cluster_dfs = load_available(args.eval_dir)
    if not per_cluster_dfs:
        print(f"no per-sample CSVs found in {args.eval_dir} -- run "
              "evaluate_synthetic_ground_truth.py first")
        return

    table = build_comparison_table(per_cluster_dfs)
    print()
    print("CROSS-CLUSTER SUMMARY:")
    print(table.round(4).to_string())
    out_path = os.path.join(args.eval_dir, "cross_cluster_comparison.csv")
    table.to_csv(out_path)
    print(f"\nsaved : {out_path}")

    print()
    print()
    print("MAGNITUDE-VS-SIGN CONSISTENCY CHECK:")
    print("(topk10_precision - confmat_precision (large gap = localisation)")
    flags = flag_magnitude_sign_gap(table, args.gap_threshold)
    for cluster_key, info in flags.items():
        status = "FLAGGED -- check qualitative plots" if info["flagged"] else "ok"
        print(f"{cluster_key:16s} topk10_precision={info['topk10_precision']:.3f}  "
              f"confmat_precision={info['confmat_precision']:.3f}  "
              f"gap={info['gap']:.3f}  [{status}]")


if __name__ == "__main__":
    main()
