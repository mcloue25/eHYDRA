from pathlib import Path
import argparse
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.saliency_ablation import HydraSaliencyAblation
from utils.cli import select_datasets_from_closest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--closest-csv", default="outputs/clustering/csv/closest_20_datasets_per_cluster.csv")
    parser.add_argument("--output-dir", default="outputs/saliency/ablation")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--datasets-per-cluster",
        type=int,
        default=None,
        help="When --datasets is omitted, take the first N closest datasets per cluster. Use 10 to reproduce the 38-dataset report subset.",
    )
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--max-samples-per-dataset", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    ''' 
    '''
    args = parse_args()

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = select_datasets_from_closest(PROJECT_ROOT / args.closest_csv, datasets_per_cluster=args.datasets_per_cluster)

    print(f"Datasets selected: {len(datasets)}")

    # Init Saliency Ablation class
    ablation = HydraSaliencyAblation(
        datasets=datasets,
        output_dir=PROJECT_ROOT / args.output_dir,
        fraction=args.fraction,
        max_samples_per_dataset=args.max_samples_per_dataset,
        seed=args.seed,
    )

    # NOTE - Run & print results
    results = ablation.run()
    print("\nAblation summary:")
    print(results["summary"].round(3))


if __name__ == "__main__":
    main()
