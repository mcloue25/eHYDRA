'''

Script to double check how TSHAP's window_length/stride sales before I do a full run for full UCR archive

Usage
    python scripts/tscaptum/check_tshap_scaling.py \
        --datasets Chinatown ItalyPowerDemand GunPoint CinCECGTorso \
        --window-fraction 0.10 --max-stride 5
'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.captum_comparison import tshap_window_stride
from utils.data_utils import load_dataset


def plot_scaling_check(df: pd.DataFrame, output_fig: Path, flag_threshold: int = 5):
    ''' Series length vs. n_window_positions, with the flag threshold marked shows the scaling formula holds across the archive rather than just
        reporting a table
    '''
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(df["series_length"], df["n_window_positions"], color="#4C72B0", edgecolor="black", linewidth=0.4, s=40, zorder=3)
    ax.axhline(flag_threshold, color="#C44E52", linestyle="--", linewidth=1.2, label=f"Flag threshold ({flag_threshold} positions)")
    ax.set_xlabel("Series length")
    ax.set_ylabel("TSHAP window positions")
    ax.set_title("TSHAP window/stride scaling check, full 128-dataset archive")
    ax.legend(fontsize=9)
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(output_fig, dpi=200)
    plt.close(fig)
    print(f"Figure saved to {output_fig}")


def n_window_positions(series_length: int, window_length: int, stride: int):
    ''' Mirror TSHAPExplainer._explain_instance's window-position count
    '''
    upper = min(series_length, series_length - window_length + stride)
    return len(range(0, max(upper, 0), stride))


def parse_args():
    ''' Func for parsing args passed from command line
    '''
    parser = argparse.ArgumentParser(description="Preview TSHAP window/stride scaling.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--all-datasets", action="store_true", help="Check every dataset in --cluster-csv, not just a representative sample.")
    parser.add_argument("--cluster-csv", default="outputs/clustering/csv/ucr_dataset_clusters_k4_with_types.csv", help="Used to auto-select representative datasets when --datasets is omitted.")
    parser.add_argument("--per-cluster", type=int, default=3, help="Datasets per cluster to sample when auto-selecting.")
    parser.add_argument("--window-fraction", type=float, default=0.10)
    parser.add_argument("--max-stride", type=int, default=5)
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional path to save the full results table (recommended for --all-datasets).")
    parser.add_argument("--output-fig", type=Path, default=None, help="Optional path to save a series-length vs. n_window_positions scatter figure.")
    return parser.parse_args()


def select_representative_datasets(cluster_csv: Path, per_cluster: int):
    ''' Pick a spread of datasets per cluster 
    '''
    clusters = pd.read_csv(cluster_csv)
    picks = []
    for cluster_id, group in clusters.groupby("cluster"):
        if "log_series_length" in group.columns:
            group = group.sort_values("log_series_length")
            idx = sorted(set([0, len(group) // 2, len(group) - 1][:min(per_cluster, len(group))]))
            picks.extend(group.iloc[idx]["dataset"].tolist())
        else:
            picks.extend(group["dataset"].head(per_cluster).tolist())
    return list(dict.fromkeys(picks))   # de-dupe, keep order


def main():
    '''  Main function for souble checking that scaling is working ok and not going to rbeak anything 
    '''
    args = parse_args()

    if args.datasets:
        datasets = args.datasets
    elif args.all_datasets:
        clusters = pd.read_csv(PROJECT_ROOT / args.cluster_csv)
        datasets = clusters["dataset"].dropna().unique().tolist()
        print(f"Checking all {len(datasets)} datasets in {args.cluster_csv}.\n")
    else:
        datasets = select_representative_datasets(PROJECT_ROOT / args.cluster_csv, args.per_cluster)
        print(f"No --datasets given; auto-selected {len(datasets)} representative datasets.\n")

    rows = []
    for name in datasets:
        try:
            X_train, _, _, _, _ = load_dataset(name)
        except Exception as e:
            print(f"[WARN] Could not load {name}: {e}")
            continue

        T = X_train.shape[-1]
        window_length, stride = tshap_window_stride(T, window_fraction=args.window_fraction, max_stride=args.max_stride)
        n_pos = n_window_positions(T, window_length, stride)

        rows.append({
            "dataset": name,
            "n_train": X_train.shape[0],
            "series_length": T,
            "window_length": window_length,
            "stride": stride,
            "n_window_positions": n_pos,
        })

    df = pd.DataFrame(rows).sort_values("series_length")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Full results written to {args.output_csv}\n")

    # NOTE - Show summary stats plus the tail end 
    if args.all_datasets and len(df) > 20:
        print(df["n_window_positions"].describe().to_string())
        print("\nShortest 15 series (most likely to have few window positions):")
        print(df.head(15).to_string(index=False))
    else:
        print(df.to_string(index=False))

    # Find at risk datasets and print them if theye xist
    thin = df[df["n_window_positions"] < 5]
    if not thin.empty:
        print("\n[NOTE] These datasets get fewer than 5 window positions — " "the resulting TSHAP attribution will be very coarse:")
        print(thin[["dataset", "series_length", "n_window_positions"]].to_string(index=False))
    else:
        print(f"\n[OK] All {len(df)} checked datasets get 5+ window positions.")

    if args.output_fig is not None:
        args.output_fig.parent.mkdir(parents=True, exist_ok=True)
        plot_scaling_check(df, args.output_fig)


if __name__ == "__main__":
    main()
