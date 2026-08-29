'''
pulls the 8 Spiky-cluster datasets closest to the cluster centroid bydistance_to_centroid in ucr_dataset_clusters_k4_with_types.csv 
and plots a few examples from different classes side by side per dataset

Usage:
    python3 scripts/synthetic_dataset_generation/inspect_real_spiky_examples.py
'''
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, REPO_ROOT)

from utils.data_utils import load_dataset


CENTROID_CLOSEST_DATASETS = [
    "WordSynonyms", # distance_to_centroid=1.529, n_classes=25
    "ScreenType", # distance_to_centroid=1.659, n_classes=3
    "GestureMidAirD3", # distance_to_centroid=1.669, n_classes=26
    "GestureMidAirD2", # distance_to_centroid=1.726, n_classes=26
    "ToeSegmentation1", # distance_to_centroid=1.831, n_classes=2
    "ToeSegmentation2", # distance_to_centroid=1.869, n_classes=2
    "EOGVerticalSignal", # distance_to_centroid=1.957, n_classes=12
    "Computers", # distance_to_centroid=2.019, n_classes=2
]

MAX_CLASSES_TO_SHOW = 4 
SAMPLES_PER_CLASS = 2
OUT_DIR = os.path.join(os.path.dirname(__file__), "smoke_test_outputs", "Real spiky centroid examples")


def plot_one_dataset(ax, dataset_name):
    ''' Plots examples for one dataset
    '''
    try:
        X_train, y_train, X_test, y_test, label_encoder = load_dataset(dataset_name)
    except Exception as e:
        ax.set_title(f"{dataset_name} (FAILED TO LOAD)", fontsize=11, color="red")
        ax.text(0.5, 0.5, str(e), ha="center", va="center", fontsize=8, wrap=True, transform=ax.transAxes)
        return

    classes = np.unique(y_train)[:MAX_CLASSES_TO_SHOW]
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    for c, color in zip(classes, colors):
        idx = np.where(y_train == c)[0][:SAMPLES_PER_CLASS]
        for j, i in enumerate(idx):
            label = f"class {c}" if j == 0 else None
            ax.plot(X_train[i], color=color, alpha=0.8, linewidth=0.9, label=label)

    n_classes_total = len(np.unique(y_train))
    shown_note = f" (showing {len(classes)}/{n_classes_total} classes)" if n_classes_total > MAX_CLASSES_TO_SHOW else ""
    ax.set_title(f"{dataset_name}{shown_note}", fontsize=12)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.tick_params(labelsize=8)


def main():
    ''' Main function for plotting examples from the spiky cluster to see how they actually look
    '''
    os.makedirs(OUT_DIR, exist_ok=True)
    for dataset_name in CENTROID_CLOSEST_DATASETS:
        print(f"loading {dataset_name}...")

        # Create figure and add plots
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_one_dataset(ax, dataset_name)
        fig.tight_layout()
        # Save figure
        out_path = os.path.join(OUT_DIR, f"{dataset_name}.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
