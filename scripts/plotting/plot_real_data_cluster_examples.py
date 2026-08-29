'''
Generates "example real data per morphology cluster" figures for the Data Considerations chapter: for each of the four clusters, 
plots example series for each of the clusters from the dataset closest to that cluster's centroid 

Usage:
    python3 scripts/plotting/plot_real_data_cluster_examples.py
    python3 scripts/plotting/plot_real_data_cluster_examples.py --n-per-class 2 --multi
'''
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SYNTH_DIR = os.path.join(REPO_ROOT, "scripts", "synthetic_dataset_generation")
sys.path.insert(0, REPO_ROOT)

from utils.data_utils import load_dataset

DEFAULT_CLUSTER_CSV = os.path.join(SYNTH_DIR, "ucr_dataset_clusters_k4_with_types.csv")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "outputs", "plots")


CLUSTERS = [
    ("High-frequency / high-curvature", "high_frequency"),
    ("Smooth / low-complexity", "smooth"),
    ("Short / moderately rough", "short_rough"),
    ("Spiky / multi-class", "spiky"),
]


def closest_datasets(cluster_csv, cluster_name, n=1):
    ''' Get the clostest datasets to the centroid  
    '''
    df = pd.read_csv(cluster_csv)
    sub = df[df["cluster_name"] == cluster_name].sort_values("distance_to_centroid")
    return sub["dataset"].head(n).tolist()


def plot_examples_on_axis(ax, dataset_name, n_per_class, seed=42):
    '''Plots up to n_per_class example series per class label from dataset_name, colour-coded, no legend
    '''
    rng = np.random.RandomState(seed)
    x_train, y_train, x_test, y_test, le = load_dataset(dataset_name)

    X = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    classes = np.unique(y)

    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(classes):
        idx = np.where(y == cls)[0]
        chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        for sample_idx in chosen:
            ax.plot(X[sample_idx], color=cmap(i % 10), alpha=0.8, linewidth=1.0)

    ax.set_title(f"{dataset_name}  (N={X.shape[0]}, T={X.shape[-1]}, " f"{len(classes)} classes)", fontsize=11)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    return len(classes)


def make_single_dataset_figures(cluster_csv, out_dir, n_per_class):
    '''
    '''
    single_dir = os.path.join(out_dir, "real_data_example")
    os.makedirs(single_dir, exist_ok=True)

    for cluster_name, tag in CLUSTERS:
        dataset_name = closest_datasets(cluster_csv, cluster_name, n=1)[0]
        print(f"[{cluster_name}] plotting {dataset_name}")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_examples_on_axis(ax, dataset_name, n_per_class)
        fig.suptitle(cluster_name, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        out_path = os.path.join(single_dir, f"{tag}.png")
        fig.savefig(out_path, dpi=200)
        print(f"  -> {out_path}")
        plt.close(fig)


def make_multi_dataset_figures(cluster_csv, out_dir, n_per_class):
    ''' Closest + 2nd-closest dataset per cluster, 8 figures total 
    Returns:
        written to real_data_example_multi/<tag>_<rank>.png
    '''
    multi_dir = os.path.join(out_dir, "real_data_example_multi")
    os.makedirs(multi_dir, exist_ok=True)

    for cluster_name, tag in CLUSTERS:
        dataset_names = closest_datasets(cluster_csv, cluster_name, n=2)
        for rank, dataset_name in enumerate(dataset_names, start=1):
            print(f"[{cluster_name}] plotting {dataset_name} (rank {rank})")

            fig, ax = plt.subplots(figsize=(7, 4.5))
            plot_examples_on_axis(ax, dataset_name, n_per_class)
            fig.suptitle(f"{cluster_name} (dataset {rank} of 2)", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.94])

            out_path = os.path.join(multi_dir, f"{tag}_{rank}.png")
            fig.savefig(out_path, dpi=200)
            print(f"Saved to : {out_path}")
            plt.close(fig)


def main():
    '''
    '''
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster-csv", default=DEFAULT_CLUSTER_CSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-per-class", type=int, default=3, help="example series plotted per class label (default: 3)")
    parser.add_argument("--multi", action="store_true", help="also generate the 2-datasets-per-cluster variant " "(8 figures instead of 4)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    make_single_dataset_figures(args.cluster_csv, args.out_dir, args.n_per_class)
    if args.multi:
        make_multi_dataset_figures(args.cluster_csv, args.out_dir, args.n_per_class)


if __name__ == "__main__":
    main()