from pathlib import Path
import argparse

import pandas as pd


def parse_args():
    ''' 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/saliency/per_dataset")
    parser.add_argument("--output-dir", default="outputs/saliency/masking")
    parser.add_argument("--models", nargs="+", default=["lr", "hydra", "mrsqm"])
    return parser.parse_args()


def combine_model(input_dir, output_dir, model):
    ''' 
    '''
    sample_files = sorted(input_dir.glob(f"{model}_samples_*.csv"))
    summary_files = sorted(input_dir.glob(f"{model}_summary_*.csv"))

    if not sample_files:
        print(f"[WARN] No sample files found for {model}")
        return

    if not summary_files:
        print(f"[WARN] No summary files found for {model}")
        return

    # NOTE - Create summary and samples DF & save them 
    samples = pd.concat([pd.read_csv(path) for path in sample_files], ignore_index=True)
    summary = pd.concat([pd.read_csv(path) for path in summary_files], ignore_index=True,)
    samples_path = output_dir / f"{model}_samples.csv"
    summary_path = output_dir / f"{model}_summary.csv"
    samples.to_csv(samples_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"[OK] {model}")
    print(f"samples: {samples.shape} saved to: {samples_path}")
    print(f"summary: {summary.shape} saved to: {summary_path}")


def main():
    ''' 
    '''
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        combine_model(input_dir, output_dir, model)


if __name__ == "__main__":
    main()