from pathlib import Path
import argparse
import gc
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.saliency_evaluator import SaliencyEvaluator


def parse_fractions(value):
    return tuple(float(x) for x in value.split(","))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["lr", "hydra", "mrsqm"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="outputs/saliency/per_dataset")
    parser.add_argument("--fractions", default="0.05,0.10,0.20")
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():
    ''' Main function for running the actual saliency evaluation framework'''
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_path = output_dir / f"{args.model}_samples_{args.dataset}.csv"
    summary_path = output_dir / f"{args.model}_summary_{args.dataset}.csv"

    if not args.overwrite and samples_path.exists() and summary_path.exists():
        print(f"[SKIP] {args.model} on {args.dataset} already completed.")
        return

    t0 = time.time()

    print()
    print(f"Running {args.model} on {args.dataset}")
    print()

    evaluator = SaliencyEvaluator(
        datasets=[args.dataset],
        output_dir=output_dir,
        fractions=parse_fractions(args.fractions),
        random_repeats=args.random_repeats,
        max_samples=args.max_samples,
        only_correct=True,
        seed=args.seed,
    )

    result = evaluator.run(model_names=(args.model,))

    combined_samples_path = output_dir / f"{args.model}_samples.csv"
    combined_summary_path = output_dir / f"{args.model}_summary.csv"
    combined_samples_path.rename(samples_path)
    combined_summary_path.rename(summary_path)

    del evaluator
    del result
    gc.collect()

    print(f"Saved samples: {samples_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()