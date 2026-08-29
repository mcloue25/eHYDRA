from pathlib import Path
import argparse
import subprocess
import sys
import time

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", nargs="+", default=["lr", "hydra", "mrsqm"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--summary-csv", default="data/summary.csv")
    parser.add_argument("--output-dir", default="outputs/saliency/per_dataset")
    parser.add_argument("--fractions", default="0.05,0.10,0.20")
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def load_datasets(args):
    if args.datasets:
        return args.datasets

    summary = pd.read_csv(PROJECT_ROOT / args.summary_csv)
    return sorted(summary["dataset"].dropna().unique())


def main():
    ''' 
    '''
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(args)
    script = PROJECT_ROOT / "scripts" / "pipeline" / "run_saliency_dataset.py"

    failed = []

    for model in args.models:
        for i, dataset in enumerate(datasets, start=1):
            samples_path = output_dir / f"{model}_samples_{dataset}.csv"
            summary_path = output_dir / f"{model}_summary_{dataset}.csv"

            if not args.overwrite and samples_path.exists() and summary_path.exists():
                print(f"[SKIP] {model} on {dataset} already completed.")
                continue

            print("=" * 80)
            print(f"[{i}/{len(datasets)}] {model} on {dataset}")
            print("=" * 80)

            cmd = [
                sys.executable,
                "-u",
                str(script),
                "--model",
                model,
                "--dataset",
                dataset,
                "--output-dir",
                str(output_dir),
                "--fractions",
                args.fractions,
                "--random-repeats",
                str(args.random_repeats),
                "--seed",
                str(args.seed),
            ]

            if args.max_samples is not None:
                cmd.extend(["--max-samples", str(args.max_samples)])

            if args.overwrite:
                cmd.append("--overwrite")

            log_path = log_dir / f"{model}_{dataset}.log"

            with open(log_path, "w") as log_file:
                result = subprocess.run(
                    cmd,
                    cwd=PROJECT_ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )

            if result.returncode == 0:
                print(f"[OK] {model} on {dataset}")
            else:
                print(f"[FAILED] {model} on {dataset}. See {log_path}")
                failed.append((model, dataset))

            time.sleep(3)

    if failed:
        failed_path = log_dir / "failed_runs.txt"
        with open(failed_path, "w") as f:
            for model, dataset in failed:
                f.write(f"{model},{dataset}\n")

        print(f"\nFailed runs written to: {failed_path}")
        raise SystemExit(1)

    print("\nAll requested runs completed.")


if __name__ == "__main__":
    main()