'''
USed to analyse the features of the signlas that mkae up the spiky / Mulit class clsuter
Usage:
    python3 scripts/synthetic_dataset_generation/analyze_real_spiky_features.py
    python3 scripts/synthetic_dataset_generation/analyze_real_spiky_features.py --datasets WordSynonyms ScreenType
'''
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, REPO_ROOT)

from utils.data_utils import load_dataset

CSV_PATH = os.path.join(os.path.dirname(__file__), "ucr_dataset_clusters_k4_with_types.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "smoke_test_outputs")
FEATURE_NAMES = ["level_shift", "max_jump_magnitude", "max_jump_location", "burst_density", "burst_count"]


def get_spiky_cluster_datasets():
    ''' Returns the dataset names in the Spiky/multi-class cluster
    '''
    df = pd.read_csv(CSV_PATH)
    return df[df.cluster_name == "Spiky / multi-class"]["dataset"].tolist()


def znorm(x):
    std = x.std()
    return (x - x.mean()) / std if std > 1e-8 else x - x.mean()


def extract_features_one_sample(x):
    ''' Computes the five candidate separator features for one z-normalised series
    '''
    x = znorm(np.asarray(x, dtype=np.float64))
    T = len(x)
    half = T // 2
    level_shift = x[half:].mean() - x[:half].mean()

    diffs = np.diff(x)
    jump_idx = int(np.argmax(np.abs(diffs)))
    max_jump_magnitude = float(diffs[jump_idx])
    max_jump_location = jump_idx / T

    burst_mask = np.abs(x - np.median(x)) > 2.0
    burst_density = float(burst_mask.mean())
    peaks, _ = find_peaks(x, height=x.mean() + 2.0 * x.std() if x.std() > 1e-8 else None)
    burst_count = float(len(peaks))

    return {
        "level_shift": level_shift,
        "max_jump_magnitude": max_jump_magnitude,
        "max_jump_location": max_jump_location,
        "burst_density": burst_density,
        "burst_count": burst_count,
    }


def analyse_one_dataset(dataset_name):
    ''' Extracts features for every training sample and returns each feature's one-way ANOVA F-statistic across the true class labels
    '''
    X_train, y_train, _, _, _ = load_dataset(dataset_name)

    rows = [extract_features_one_sample(x) for x in X_train]
    feat_df = pd.DataFrame(rows)
    feat_df["y"] = y_train

    f_stats = {}
    for feat in FEATURE_NAMES:
        groups = [feat_df.loc[feat_df.y == c, feat].values for c in np.unique(y_train)]
        groups = [g for g in groups if len(g) > 1]  # f_oneway needs n>1 per group
        if len(groups) < 2:
            f_stats[feat] = np.nan
            continue
        try:
            f_stat, _ = stats.f_oneway(*groups)
            f_stats[feat] = float(f_stat) if np.isfinite(f_stat) else np.nan
        except Exception:
            f_stats[feat] = np.nan

    return f_stats, len(np.unique(y_train)), len(X_train)


def main():
    ''' 
    '''
    # Read argparser commands
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=None, help="override the dataset list (default: full 41-dataset Spiky cluster)")
    args = parser.parse_args()


    dataset_names = args.datasets or get_spiky_cluster_datasets()
    print(f"analyzing {len(dataset_names)} datasets...\n")

    # Analyse datasets
    results = []
    for name in dataset_names:
        print(f"[{name}]", end=" ")
        try:
            f_stats, n_classes, n_samples = analyse_one_dataset(name)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        # rank features by F-stat within this dataset
        valid = {k: v for k, v in f_stats.items() if not np.isnan(v)}
        ranked = sorted(valid.items(), key=lambda kv: -kv[1])
        winner = ranked[0][0] if ranked else None
        print(f"n_classes={n_classes} n={n_samples} winner={winner}")
        row = {"dataset": name, "n_classes": n_classes, "n_samples": n_samples, "winner": winner}
        row.update({f"fstat_{k}": v for k, v in f_stats.items()})
        for rank, (feat, _) in enumerate(ranked, start=1):
            row[f"rank_{feat}"] = rank
        results.append(row)

    # Collate results
    results_df = pd.DataFrame(results)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "real_spiky_feature_analysis.csv")
    results_df.to_csv(out_path, index=False)

    print()
    print("SUMMARY: how often each feature is the TOP separator")
    print()
    win_counts = results_df["winner"].value_counts()
    for feat in FEATURE_NAMES:
        n_wins = int(win_counts.get(feat, 0))
        print(f"  {feat:22s} won {n_wins}/{len(results_df)} datasets")

    print()
    print("\nmean rank per feature (lower = wins more often, best possible = 1.0):")
    for feat in FEATURE_NAMES:
        col = f"rank_{feat}"
        if col in results_df.columns:
            print(f"  {feat:22s} mean rank = {results_df[col].mean():.2f}")

    print(f"\nfull per-dataset results saved to: {out_path}")


if __name__ == "__main__":
    main()