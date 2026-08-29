from pathlib import Path
import argparse
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.margin_saliency_evaluation import MarginSaliencyEvaluator
from utils.cli import select_datasets_from_closest


def parse_args():
    parser = argparse.ArgumentParser(description="Compare predicted class saliency against margin saliency under perturbation.")
    parser.add_argument("--closest-csv", default="outputs/clustering/csv/closest_20_datasets_per_cluster.csv")
    parser.add_argument("--output-dir", default="outputs/saliency/margin_saliency")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--datasets-per-cluster", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--max-samples-per-dataset", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    ''' Main function for testing predicted class VS marginal saliency 
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

    # NOTE - Init Marginal saliency evaluator
    evaluator = MarginSaliencyEvaluator(
        datasets=datasets,
        output_dir=PROJECT_ROOT / args.output_dir,
        fraction=args.fraction,
        random_repeats=args.random_repeats,
        max_samples_per_dataset=args.max_samples_per_dataset,
        seed=args.seed,
    )

    # Run & display results
    results = evaluator.run()
    print("\nPredicted-class vs. margin saliency summary:")
    print(results["summary"].round(3))


if __name__ == "__main__":
    main()
