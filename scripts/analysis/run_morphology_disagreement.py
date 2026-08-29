from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.morphology_disagreement import MorphologyDisagreementAnalysis


def parse_args():
    ''' Init argparser
    '''
    parser = argparse.ArgumentParser(
        description=(
            "Correlate HYDRA-vs-WindowSHAP agreement with continuous signal-morphology "
            "features, and extract the strongest agreement/disagreement samples for "
            "qualitative follow-up. Requires scripts/robustness_testing/run_windowshap_comparison.py to "
            "have already been run (it produces hydra_windowshap_samples_with_clusters.csv)."
        )
    )
    parser.add_argument("--windowshap-samples", default="outputs/saliency/windowshap/hydra_windowshap_samples_with_clusters.csv")
    parser.add_argument("--feature-csv", default="outputs/clustering/csv/ucr_dataset_clusters_k4.csv")
    parser.add_argument("--output-dir", default="outputs/saliency/morphology_disagreement")
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--top-n-cases", type=int, default=10)
    return parser.parse_args()


def main():
    ''' Main function for analysing the morphology disagreement
    '''
    args = parse_args()
    
    analysis = MorphologyDisagreementAnalysis(
        windowshap_samples_csv=PROJECT_ROOT / args.windowshap_samples,
        feature_csv=PROJECT_ROOT / args.feature_csv,
        output_dir=PROJECT_ROOT / args.output_dir,
    )

    results = analysis.run(fraction=args.fraction, top_n_cases=args.top_n_cases)

    print("\n\n\nStrongest feature agreement correlations (ranked by |rho|):")
    print(results["correlation_ranked"].round(3).to_string(index=False))

    print(f"\n\n\nTop {args.top_n_cases} highest agreement samples (highest IoU):")
    print(results["most_agreement_cases"].round(3).to_string(index=False))

    print(f"\n\n\nTop {args.top_n_cases} highestdisagreement samples (lowest IoU):")
    print(results["most_disagreement_cases"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
