from pathlib import Path
import argparse
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.saliency_evaluator import SaliencyEvaluator
from utils.cli import select_datasets_from_closest


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rerun a small dataset subset with only_correct=False so correctness stratification (classes/confidence_stratification.py correctness_table) "
            "has both correct and incorrect predictions available"
        )
    )
    parser.add_argument("--closest-csv", default="outputs/clustering/csv/closest_20_datasets_per_cluster.csv")
    parser.add_argument("--output-dir", default="outputs/saliency/correctness_supplement")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--datasets-per-cluster", type=int, default=10)
    parser.add_argument("--models", nargs="*", default=["hydra"], choices=["lr", "hydra", "mrsqm"])
    parser.add_argument("--fractions", default="0.10")
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    ''' Main function for running the correctness stratification testing pipeline'''
    args = parse_args()

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = select_datasets_from_closest(PROJECT_ROOT / args.closest_csv, datasets_per_cluster=args.datasets_per_cluster)

    fractions = tuple(float(f) for f in args.fractions.split(","))

    print(f"Datasets selected: {len(datasets)}")
    print(
        "NOTE: only_correct=False, so base_accuracy and sample composition are not "
        "directly comparable to masking_original_report. Use this output only for "
        "correctness stratification, not as a substitute for the headline tables."
    )

    # NOTE - Initialise the Saliency Evaluator
    evaluator = SaliencyEvaluator(
        datasets=datasets,
        output_dir=PROJECT_ROOT / args.output_dir,
        fractions=fractions,
        random_repeats=args.random_repeats,
        only_correct=False,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    # Run & generate results
    results = evaluator.run(model_names=tuple(args.models))

    # Print results
    for model_name, result in results.items():
        print(f"\n{model_name} summary (head):")
        print(result["summary"].round(3).head(10))


if __name__ == "__main__":
    main()
