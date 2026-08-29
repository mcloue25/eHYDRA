from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.qualitative_plots import QualitativeSaliencyPlotter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="GunPoint")
    parser.add_argument("--output-dir", default="outputs/saliency/imgs")
    parser.add_argument("--n-segments", type=int, default=100)
    parser.add_argument("--shap-nsamples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    plotter = QualitativeSaliencyPlotter(
        output_dir=PROJECT_ROOT / args.output_dir,
        n_segments=args.n_segments,
        shap_nsamples=args.shap_nsamples,
        seed=args.seed,
    )

    info = plotter.plot_hydra_mrsqm_windowshap(dataset=args.dataset)
    print(info)


if __name__ == "__main__":
    main()
