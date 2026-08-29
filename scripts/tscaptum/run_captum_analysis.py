'''
Runner script for CaptumAnalysis and CaptumComparator.

Usage examples:
  # Analyse one run:
  python scripts/tscaptum/run_captum_analysis.py \
      --run-a outputs/saliency/captum_comparison \
      --label-a "38-dataset subset"

  # Compare two runs:
  python scripts/tscaptum/run_captum_analysis.py \
      --run-a outputs/saliency/captum_comparison \
      --label-a "38-dataset subset" \
      --run-b outputs/saliency/captum_comparison_full128 \
      --label-b "Full 128 datasets" \
      --cluster-csv outputs/clustering/csv/ucr_dataset_clusters_k4_with_types.csv
'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.captum_analysis import CaptumAnalysis, CaptumComparator


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse and compare CaptumComparison experiment results.")
    parser.add_argument("--run-a", required=True, help="Path to first (or only) CaptumComparison output dir.")
    parser.add_argument("--label-a", default="Run A")
    parser.add_argument("--run-b", default=None, help="Path to second run for comparison (optional).")
    parser.add_argument("--label-b", default="Run B")
    parser.add_argument("--cluster-csv", default="outputs/clustering/csv/ucr_dataset_clusters_k4_with_types.csv", help="Cluster CSV with distance_to_centroid column for centroid-distance plots.")
    return parser.parse_args()


def main():
    ''' Main function used to run the tscaptum analysis and comparison to eHYDRA pipelien 
    '''
    args = parse_args()
    cluster_csv = PROJECT_ROOT / args.cluster_csv

    # NOTE - Run the Captum analysis & generate the report
    run_a = CaptumAnalysis(run_dir=PROJECT_ROOT / args.run_a, label=args.label_a, cluster_csv=cluster_csv,)
    run_a.report()

    # NOTE - Compare runs if more than one run passed
    if args.run_b:
        run_b = CaptumAnalysis(run_dir=PROJECT_ROOT / args.run_b, label=args.label_b, cluster_csv=cluster_csv)
        run_b.report()

        comparator = CaptumComparator(run_a=run_a, run_b=run_b)
        comparator.compare()


if __name__ == "__main__":
    main()
