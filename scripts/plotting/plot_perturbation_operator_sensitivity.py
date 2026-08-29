'''
Generate the perturbation-operator sensitivity figures for Sect. 'Perturbation Operator Sensitivity'
'''

from pathlib import Path
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OPERATORS = ["global_mean", "local_mean", "linear_interpolation", "blur"]
LABELS = {
    "global_mean": "Global mean",
    "local_mean": "Local mean",
    "linear_interpolation": "Linear interpolation",
    "blur": "Blur",
}
COLORS = {
    "global_mean": "#1b6ca8",
    "local_mean": "#e07b39",
    "linear_interpolation": "#3f9142",
    "blur": "#c0392b",
}
MARKERS = {"global_mean": "o", "local_mean": "s", "linear_interpolation": "^", "blur": "D"}
BAR_WIDTH = 0.2
BAR_OFFSETS = [-1.5 * BAR_WIDTH, -0.5 * BAR_WIDTH, 0.5 * BAR_WIDTH, 1.5 * BAR_WIDTH]


def set_style():
    matplotlib.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def style_axis_line(ax):
    ''' Light horizontal grid for line plots
    '''
    ax.grid(True, axis="y", linestyle="--", alpha=0.30)
    ax.set_axisbelow(True)


def style_axis_bar(ax):
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.spines["left"].set_visible(False)


def save_figure(fig, output_dir: Path, name: str):
    ''' Save figure as png
    '''
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
    # fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / name}.png")


def fraction_dir(root: Path, frac_pct: int) -> Path:
    ''' Map an integer percentage (5, 10, 20) to the analysis_0.05 / analysis_0.10 / analysis_0.20 subdirs
    '''
    return root / f"analysis_{frac_pct / 100:.2f}"


def load_gap_data(root_dir: Path, fractions: list[int]):
    ''' Compute the top-random gap for flip rate and bounded score drop per operator at each masking fraction.
    '''
    flip_gap = {op: [] for op in OPERATORS}
    score_gap = {op: [] for op in OPERATORS}

    for frac in fractions:
        input_dir = fraction_dir(root_dir, frac)
        fr = pd.read_csv(input_dir / f"flip_rate_{frac}_by_operator.csv").set_index("perturbation")
        sd = pd.read_csv(input_dir / f"mean_bounded_relative_score_drop_{frac}_by_operator.csv").set_index("perturbation")
        for op in OPERATORS:
            flip_gap[op].append(fr.loc[op, "top"] - fr.loc[op, "random"])
            score_gap[op].append(sd.loc[op, "top"] - sd.loc[op, "random"])

    return flip_gap, score_gap



def load_consistency_data(root_dir: Path, fractions: list[int]):
    ''' Get the proportion of datasets where top-saliency masking beats random masking on flip rate per operator at each masking fraction
    '''
    consistency = {op: [] for op in OPERATORS}
    for frac in fractions:
        input_dir = fraction_dir(root_dir, frac)
        c = pd.read_csv(input_dir / f"consistency_by_operator_{frac}.csv")
        c = c[c["comparison"] == "top_gt_random_pct"].set_index("perturbation")
        for op in OPERATORS:
            consistency[op].append(c.loc[op, "rate_pct"])

    return consistency




def plot_gap_stability_line(flip_gap, score_gap, fractions, output_dir: Path):
    ''' Main function for plotting the gap stability between perturbation operators 
    '''
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    for op in OPERATORS:
        axes[0].plot(fractions, flip_gap[op], marker=MARKERS[op], color=COLORS[op], label=LABELS[op], linewidth=2.5, markersize=8)
        axes[1].plot(fractions, score_gap[op], marker=MARKERS[op], color=COLORS[op], label=LABELS[op], linewidth=2.5, markersize=8)

    axes[0].set_title("Flip-rate gap (Top $-$ Random)")
    axes[0].set_ylabel("Gap (percentage points)")
    axes[1].set_title("Bounded score-drop gap (Top $-$ Random)")
    axes[1].set_ylabel("Gap")

    for ax in axes:
        ax.set_xlabel("Masked fraction of time steps")
        ax.set_xticks(fractions)
        ax.set_xticklabels([f"{f}%" for f in fractions])
        style_axis_line(ax)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle("Faithfulness gap by perturbation operator across masking fractions", y=1.03)
    save_figure(fig, output_dir, "operator_gap_stability_line")




def plot_gap_stability_bar(flip_gap, score_gap, fractions, output_dir: Path):
    ''' Function for plotting an idnividual bar 
    '''
    x = np.arange(len(fractions))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for ax, data, title, is_score in [(axes[0], flip_gap, "Flip-rate gap (Top $-$ Random)", False), (axes[1], score_gap, "Bounded score-drop gap (Top $-$ Random)", True),]:
        for offset, op in zip(BAR_OFFSETS, OPERATORS):
            y = data[op]
            bars = ax.bar(x + offset, y, width=BAR_WIDTH, color=COLORS[op], alpha=0.88, label=LABELS[op])
            for bar, val in zip(bars, y):
                label = f"{val:.2f}" if is_score else f"{val:.1f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label,ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Masked fraction of time steps")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{f}%" for f in fractions])
        style_axis_bar(ax)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax * 1.18)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle("Faithfulness gap by perturbation operator across masking fractions", y=1.03)

    save_figure(fig, output_dir, "operator_gap_stability_bar")




def plot_consistency_line(consistency, fractions, output_dir: Path):
    ''' Main function for plotting dataset level consistency 
    '''
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axhline(50, color="grey", linestyle="--", linewidth=1.5, label="Chance (50%)")

    for op in OPERATORS:
        ax.plot(fractions, consistency[op], marker=MARKERS[op], color=COLORS[op],
                label=LABELS[op], linewidth=2.5, markersize=8)

    ax.set_xlabel("Masked fraction of time steps")
    ax.set_ylabel("Datasets with Top $>$ Random flip rate (%)")
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{f}%" for f in fractions])
    ax.set_ylim(20, 80)
    ax.set_title("Dataset-level ordering consistency by operator")
    style_axis_line(ax)
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, output_dir, "operator_dataset_consistency_line")




def plot_consistency_bar(consistency, fractions, output_dir: Path):
    ''' Plot an indivdual bar of the dataset level consistency plot
    '''
    x = np.arange(len(fractions))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    for offset, op in zip(BAR_OFFSETS, OPERATORS):
        y = consistency[op]
        bars = ax.bar(x + offset, y, width=BAR_WIDTH, color=COLORS[op], alpha=0.88, label=LABELS[op])
        for bar, val in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.0f}", ha="center", va="bottom", fontsize=8)


    ax.axhline(50, color="grey", linestyle="--", linewidth=1.5, label="Chance (50%)", zorder=0)
    ax.set_xlabel("Masked fraction of time steps")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}%" for f in fractions])
    ax.set_ylim(0, 85)
    ax.set_title("Dataset-level ordering consistency by operator")
    style_axis_bar(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    save_figure(fig, output_dir, "operator_dataset_consistency_bar")




def parse_args():
    ''' Init the arg parser
    '''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="outputs/saliency/perturbation_operators_full",
        help=("Parent directory containing the per-fraction analysis_<fraction> subdirectories written by analyse_perturbation_operators.py (analysis_0.05, analysis_0.10, analysis_0.20)."),
    )
    parser.add_argument("--output-dir", default="outputs/plots/perturbation_operators", help="Directory to write the generated figures to.")
    parser.add_argument("--fractions", nargs="+", type=int, default=[5, 10, 20], help="Masking fractions to plot, as integer percentages (must match the CSV filename suffixes).")
    parser.add_argument("--style", choices=["line", "bar", "both"], default="both", help="Which chart style(s) to generate (default: both).")
    return parser.parse_args()


def main():
    '''  Main function for plotting operator based sensitivity
    '''
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    set_style()

    flip_gap, score_gap = load_gap_data(input_dir, args.fractions)
    consistency = load_consistency_data(input_dir, args.fractions)

    if args.style in ("line", "both"):
        plot_gap_stability_line(flip_gap, score_gap, args.fractions, output_dir)
        plot_consistency_line(consistency, args.fractions, output_dir)

    if args.style in ("bar", "both"):
        plot_gap_stability_bar(flip_gap, score_gap, args.fractions, output_dir)
        plot_consistency_bar(consistency, args.fractions, output_dir)


if __name__ == "__main__":
    main()