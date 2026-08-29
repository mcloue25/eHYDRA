from pathlib import Path
import argparse
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.sanity_checks import SanityCheckEvaluator
from utils.cli import select_datasets_from_closest


def parse_args():
    parser = argparse.ArgumentParser(description=("Run HYDRA saliency control tests: seed stability, label permutation and weight-permutation"))
    parser.add_argument("--closest-csv", default="outputs/clustering/csv/closest_20_datasets_per_cluster.csv")
    parser.add_argument("--output-dir", default="outputs/saliency/sanity_checks")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--datasets-per-cluster", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--max-samples-per-dataset", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    ''' Main functionf or running the saliency evaluation control tests 
    '''
    args = parse_args()

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = select_datasets_from_closest(
            PROJECT_ROOT / args.closest_csv,
            datasets_per_cluster=args.datasets_per_cluster,
        )

    print(f"Datasets selected: {len(datasets)}")

    evaluator = SanityCheckEvaluator(
        datasets=datasets,
        output_dir=PROJECT_ROOT / args.output_dir,
        n_seeds=args.n_seeds,
        fraction=args.fraction,
        max_samples_per_dataset=args.max_samples_per_dataset,
        base_seed=args.seed,
    )

    results = evaluator.run()
    print("\nSeed-stability summary (head):")
    print(results["seed_stability"]["summary"].round(3).head(10))
    print()
    print()
    print()
    print()
    print("\nLabel/weight permutation summary:")
    print(results["permutation_checks"]["summary"].round(3))


if __name__ == "__main__":
    main()
