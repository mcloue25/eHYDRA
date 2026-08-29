from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.perturbation_analysis import PerturbationOperatorAnalysis


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse perturbation-operator robustness results produced by run_perturbation_operators.py.")
    parser.add_argument("--samples-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/saliency/perturbation_operators/analysis")
    parser.add_argument("--fraction", type=float, default=0.10)
    return parser.parse_args()


def main():
    ''' Main function to run the perturbation operator analysis 
    '''
    args = parse_args()
    analysis = PerturbationOperatorAnalysis(samples_csv=PROJECT_ROOT / args.samples_csv, output_dir=PROJECT_ROOT / args.output_dir)

    results = analysis.run(fraction=args.fraction)
    print("\nFlip-rate table (%) by operator:")
    print(results["flip_table"].round(2))

    print("\nBounded relative score-drop table by operator:")
    print(results["bounded_table"].round(3))

    print("\nPaired tests by operator:")
    print(results["paired_tests"].to_string(index=False, float_format="{:.3e}".format))

    print("\nDataset-level consistency by operator:")
    print(results["consistency"].to_string(index=False))

    print("\n Main Hypothesis: Is the top > random > bottom ordering preserved under every operator?")
    print(results["ordering_summary"][["perturbation_label", "flip_ordering_preserved", "score_drop_ordering_preserved"]].to_string(index=False))


if __name__ == "__main__":
    main()
