''' 
    Paired significance testing: eHYDRA vs each TSHAP method+background, on the same test samples per cluster and per metric.
'''
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, REPO_ROOT)
from utils.globals_config import DEFAULT_EVAL_DIR, CLUSTER_KEYS

DEFAULT_METRICS = ["cosine_similarity", "confmat_f1", "topk10_precision", "topk10_iou"]


def paired_test(ehydra_vals, other_vals):
    ''' Wilcoxon signed-rank test, ehydra vs other, on paired same-sample values.
    '''
    ehydra_vals = np.asarray(ehydra_vals, dtype=float)
    other_vals = np.asarray(other_vals, dtype=float)

    valid = ~(np.isnan(ehydra_vals) | np.isnan(other_vals))
    n_nan_dropped = int((~valid).sum())
    ehydra_vals, other_vals = ehydra_vals[valid], other_vals[valid]

    diffs = ehydra_vals - other_vals
    n_wins = int((diffs > 0).sum())
    n_ties = int((diffs == 0).sum())
    n_compared = len(diffs) - n_ties

    if n_compared == 0:
        return {"mean_diff": np.nan, "ehydra_wins": n_wins, "n_compared": n_compared, "p_value": np.nan, "n_nan_dropped": n_nan_dropped}

    try:
        _, p = stats.wilcoxon(ehydra_vals, other_vals)
    except ValueError:
        p = np.nan  # e.g. all remaining differences are zero

    return {
        "mean_diff": float(np.mean(diffs)),
        "ehydra_wins": n_wins,
        "n_compared": n_compared,
        "p_value": p,
        "n_nan_dropped": n_nan_dropped,
    }


def compare_one_cluster(cluster_key, eval_dir, metrics):
    ''' Function for comparing results for one cluster
    '''
    path = os.path.join(eval_dir, f"{cluster_key}_tshap_comparison.csv")
    if not os.path.exists(path):
        print(f"[skip] {cluster_key}: {path} not found")
        return None

    df = pd.read_csv(path)
    ehydra = df[df.method == "ehydra"].sort_values("sample").reset_index(drop=True)
    other_conditions = df[df.method != "ehydra"][["method", "background"]].drop_duplicates()

    rows = []
    for _, cond in other_conditions.iterrows():
        method, background = cond["method"], cond["background"]
        other = df[(df.method == method) & (df.background == background)].sort_values("sample").reset_index(drop=True)

        merged = ehydra.merge(other, on="sample", suffixes=("_ehydra", "_other"))  # aligning on sample id, not row order
        if len(merged) == 0:
            continue

        for metric in metrics:
            col_e, col_o = f"{metric}_ehydra", f"{metric}_other"
            if col_e not in merged.columns or col_o not in merged.columns:
                continue
            result = paired_test(merged[col_e].values, merged[col_o].values)
            rows.append({
                "cluster": cluster_key, "metric": metric,
                "comparison": f"ehydra vs {method}({background})",
                **result,
            })

    return pd.DataFrame(rows)


def main():
    ''' Main function to compare eHYDRA against other explainer methods 
    '''
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster", choices=CLUSTER_KEYS + ["all"], default="all")
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    clusters = CLUSTER_KEYS if args.cluster == "all" else [args.cluster]
    all_results = []
    for cluster_key in clusters:
        df = compare_one_cluster(cluster_key, args.eval_dir, args.metrics)
        if df is None or len(df) == 0:
            continue
        all_results.append(df)

    if not all_results:
        print("no results found -- run evaluate_tshap_comparison.py first")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined["significant"] = combined["p_value"] < args.alpha
    combined["n_samples_per_side"] = combined["n_compared"]

    pd.set_option("display.width", 160)
    for cluster_key in combined.cluster.unique():
        print(f"\n{cluster_key}:")
        sub = combined[combined.cluster == cluster_key]
        print(sub[["metric", "comparison", "mean_diff", "ehydra_wins", "n_compared", "n_nan_dropped", "p_value", "significant"]].round(4).to_string(index=False))

    out_path = os.path.join(args.eval_dir, "method_significance_tests.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nsaved : {out_path}")


if __name__ == "__main__":
    main()