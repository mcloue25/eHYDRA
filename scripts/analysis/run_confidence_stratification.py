from pathlib import Path
import argparse
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.confidence_stratification import ConfidenceStratificationAnalysis


def load_combined_samples(saliency_output_dir):
    '''
    '''
    saliency_output_dir = Path(saliency_output_dir)
    sample_files = sorted(saliency_output_dir.glob("*_samples.csv"))
    if not sample_files:
        raise FileNotFoundError(f"No *_samples.csv files found in {saliency_output_dir}")
    return pd.concat([pd.read_csv(p) for p in sample_files], ignore_index=True)


def parse_args():
    ''' Init arg parser 
    '''
    parser = argparse.ArgumentParser(description=("Confidence-stratified and correctness stratified faithfulness analysis"))
    parser.add_argument("--saliency-output-dir", default="outputs/saliency/masking_original_report", help="Directory containing *_samples.csv files (e.g. hydra_samples.csv).")
    parser.add_argument("--output-dir", default="outputs/saliency/confidence_stratification")
    parser.add_argument("--model", default="HYDRA", choices=["HYDRA", "LR", "MrSQM"])
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--n-bins", type=int, default=3)
    return parser.parse_args()


def main():
    ''' Main function for running the confidence stratification and correctness analysis 
    '''
    args = parse_args()

    samples = load_combined_samples(PROJECT_ROOT / args.saliency_output_dir)
    print(f"Loaded {len(samples)} sample rows from {args.saliency_output_dir}")

    analysis = ConfidenceStratificationAnalysis(
        samples=samples,
        output_dir=PROJECT_ROOT / args.output_dir,
        n_bins=args.n_bins,
    )

    results = analysis.run(model=args.model, fraction=args.fraction)

    print(f"\nFlip-rate by confidence bin (%), {args.model}:")
    print(results["flip_by_confidence"].round(2))

    print(f"\nBounded score-drop by confidence bin, {args.model}:")
    print(results["score_by_confidence"].round(3))

    print("\nPaired tests by confidence bin:")
    print(results["paired_tests_by_confidence"].to_string(index=False, float_format="{:.3e}".format))

    if results["correctness"] is not None:
        print("\nFlip-rate by correctness (%):")
        print(results["correctness"].round(2))
    else:
        print(
            "\nCorrectness stratification not available from the samples file "
            "Run this first: scripts/robustness_testing/run_correctness_supplement.py."
        )


if __name__ == "__main__":
    main()
