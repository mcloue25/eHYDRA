from pathlib import Path
import argparse
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.perturbation_robustness import PerturbationRobustnessEvaluator
from utils.cli import select_datasets_from_closest
from utils.perturbation import CORE_PERTURBATIONS, PERTURBATIONS


def load_full_archive(summary_csv=PROJECT_ROOT / "data" / "summary.csv"):
    summary = pd.read_csv(summary_csv)
    return sorted(summary["dataset"].dropna().unique())


def parse_args():
    ''' Define argparser 
    '''
    parser = argparse.ArgumentParser(description=("Run HYDRA top/random/bottom masking under multiple perturbation operators (default: global_mean, local_mean, linear_interpolation)."))
    parser.add_argument("--closest-csv", default="outputs/clustering/csv/closest_20_datasets_per_cluster.csv")
    parser.add_argument("--output-dir", default="outputs/saliency/perturbation_operators")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--datasets-per-cluster", type=int, default=None, help="Take the first N closest datasets per cluster. Use 10 for the 38-dataset report subset.")
    parser.add_argument("--full-archive", action="store_true", help="Use the full 128-dataset archive (data/summary.csv) instead of the stratified subset.")
    parser.add_argument("--operators", nargs="*", default=list(CORE_PERTURBATIONS), choices=list(PERTURBATIONS.keys()), help=f"Perturbation operators to compare. Default: {list(CORE_PERTURBATIONS)}")
    parser.add_argument("--fractions", default="0.05,0.10,0.20")
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--max-samples-per-dataset", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    ''' Main function for running the perturbation operator analysis pipeline
    '''
    args = parse_args()

    if args.datasets:
        datasets = args.datasets
    elif args.full_archive:
        datasets = load_full_archive()
    else:
        datasets = select_datasets_from_closest(PROJECT_ROOT / args.closest_csv, datasets_per_cluster=args.datasets_per_cluster)

    fractions = tuple(float(f) for f in args.fractions.split(","))
    print(f"Datasets selected: {len(datasets)}")
    print(f"Operators: {args.operators}")
    print(f"Fractions: {fractions}")


    # NOTE - Init the perturbation operator evaluator class
    evaluator = PerturbationRobustnessEvaluator(
        datasets=datasets,
        output_dir=PROJECT_ROOT / args.output_dir,
        operators=tuple(args.operators),
        fractions=fractions,
        random_repeats=args.random_repeats,
        max_samples=args.max_samples_per_dataset,
        seed=args.seed,
    )

    # NOTE - Run & print results
    results = evaluator.run()
    print("\nSummary (head):")
    print(results["summary"].round(3).head(20))

if __name__ == "__main__":
    main()
