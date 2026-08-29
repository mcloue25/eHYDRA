'''
    Confidence stratification : Does the top > random > bottom ordering hold equally well for high-confidence and low-confidence predictions
    Correctness stratification : Does the saliency map behave differently when the model's prediction was wrong?
'''

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from classes.stats_utils import paired_wilcoxon
from utils.globals_config import MODE_ORDER



@dataclass
class ConfidenceStratificationAnalysis:
    '''Stratify existing saliency masking results by prediction confidence and correctness
    '''
    samples_csv: Path | str | None = None
    samples: pd.DataFrame | None = None
    output_dir: Path | str = Path("outputs/saliency/confidence_stratification")
    n_bins: int = 3

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.samples_csv is not None:
            self.samples_csv = Path(self.samples_csv)
        if self.samples is None and self.samples_csv is not None:
            self.samples = pd.read_csv(self.samples_csv)
        if self.samples is None:
            raise ValueError("Provide either `samples` (a DataFrame) or `samples_csv` (a path).")

        required = {"dataset", "model", "mode", "fraction", "score_before", "flipped", "base_pred", "true_label"}
        missing = required - set(self.samples.columns)
        if missing:
            raise ValueError(f"samples is missing expected columns: {missing}")

        if "bounded_relative_score_drop" not in self.samples.columns:
            eps = 1e-6
            self.samples = self.samples.copy()
            self.samples["bounded_relative_score_drop"] = np.clip(self.samples["score_drop"] / (np.abs(self.samples["score_before"]) + eps), -1.0, 1.0)

        self.samples = self.samples.copy()
        self.samples["correct"] = self.samples["base_pred"] == self.samples["true_label"]


    def add_confidence_bins(self, model=None):
        ''' Bin samples into n_bins quantile bins of |score_before|, computed per model
            Binning is done per model because decision-score scale differs across LR/HYDRA/MrSQM
        '''
        df = self.samples if model is None else self.samples[self.samples["model"] == model].copy()

        bin_labels = self.bin_labels()

        def bin_one_model(group):
            margin = group["score_before"].abs()
            try:
                group = group.copy()
                group["confidence_bin"] = pd.qcut(margin, q=self.n_bins, labels=bin_labels, duplicates="drop")
            except ValueError:
                group = group.copy()
                group["confidence_bin"] = bin_labels[0]
            return group
        df = df.groupby("model", group_keys=False).apply(bin_one_model)
        return df



    def bin_labels(self):
        ''' Bin labels based on confidence
        '''
        if self.n_bins == 3:
            return ["low_confidence", "medium_confidence", "high_confidence"]
        return [f"bin_{i + 1}_of_{self.n_bins}" for i in range(self.n_bins)]



    def confidence_table(self, model="HYDRA", metric="flipped", fraction=0.10, as_percent=False):
        ''' Confidence_bin x mode table for one model/metric/fraction
        '''
        df = self.add_confidence_bins(model=model)
        df = df[df["fraction"] == fraction]
        table = (df.groupby(["confidence_bin", "mode"], observed=True)[metric].mean().unstack("mode").reindex(columns=MODE_ORDER))
        if as_percent:
            table = table * 100
        filename = f"{model.lower()}_{metric}_{int(fraction * 100)}_by_confidence.csv"
        table.to_csv(self.output_dir / filename)
        return table

    

    def paired_tests_by_confidence(self, model="HYDRA", fraction=0.10):
        ''' Top>random / top>bottom Wilcoxon tests run separately within each confidence bin
            Pairing unit is (dataset, confidence_bin): within a bin, per-dataset mean flip rate / bounded score drop is calculayted first &
            then compared across modes
        '''
        df = self.add_confidence_bins(model=model)
        df = df[df["fraction"] == fraction]

        dataset_level = (
            df.groupby(["dataset", "confidence_bin", "mode"], as_index=False, observed=True)
            .agg(
                flip_rate=("flipped", "mean"),
                mean_bounded_relative_score_drop=("bounded_relative_score_drop", "mean"),
            )
        )

        metrics = [("flip_rate", "Flip rate"), ("mean_bounded_relative_score_drop", "Bounded score drop")]
        comparisons = [("random", "Top > Random"), ("bottom", "Top > Bottom")]
        rows = []

        for confidence_bin, bin_group in dataset_level.groupby("confidence_bin", observed=True):
            for metric_col, metric_name in metrics:
                for baseline_mode, comparison_name in comparisons:
                    # NOTE - Running paried wilcoxon tests
                    result = paired_wilcoxon(
                        bin_group,
                        group_col="dataset",
                        condition_col="mode",
                        value_col=metric_col,
                        candidate="top",
                        baseline=baseline_mode,
                    )
                    rows.append({
                        "model": model,
                        "confidence_bin": confidence_bin,
                        "fraction": fraction,
                        "metric": metric_name,
                        "comparison": comparison_name,
                        "mean_diff": result["mean_diff"],
                        "top_greater_count": result["candidate_greater_count"],
                        "wilcoxon_p": result["wilcoxon_p"],
                    })

        tests_df = pd.DataFrame(rows)
        tests_df.to_csv(self.output_dir / f"{model.lower()}_paired_tests_by_confidence_{int(fraction * 100)}.csv", index=False)
        return tests_df

    

    def correctness_table(self, model="HYDRA", metric="flipped", fraction=0.10, as_percent=False):
        ''' Correct/incorrect x mode table. 
        '''
        df = self.samples[(self.samples["model"] == model) & (self.samples["fraction"] == fraction)]

        n_groups = df["correct"].nunique()
        if n_groups < 2:
            only_value = bool(df["correct"].iloc[0]) if len(df) else None
            print(
                f"[correctness_table] Only one correctness group present for {model} "
                f"(correct={only_value}, n={len(df)}). This samples file was almost "
                "certainly generated with only_correct=True. Run "
                "scripts/run_correctness_supplement.py to get both groups."
            )
            return None

        table = (
            df.groupby(["correct", "mode"])[metric]
            .mean()
            .unstack("mode")
            .reindex(columns=MODE_ORDER)
        )
        if as_percent:
            table = table * 100
        filename = f"{model.lower()}_{metric}_{int(fraction * 100)}_by_correctness.csv"
        table.to_csv(self.output_dir / filename)
        return table



    def run(self, model="HYDRA", fraction=0.10):
        ''' Main function for running both tests above 
        '''
        # Generate confidence based table for flip rate 
        flip_by_confidence = self.confidence_table(model=model, metric="flipped", fraction=fraction, as_percent=True)
        # Generate confidence based table for score drop
        score_by_confidence = self.confidence_table(model=model, metric="bounded_relative_score_drop", fraction=fraction, as_percent=False)
        paired_tests = self.paired_tests_by_confidence(model=model, fraction=fraction)
        correctness = self.correctness_table(model=model, metric="flipped", fraction=fraction, as_percent=True)
        return {
            "flip_by_confidence": flip_by_confidence,
            "score_by_confidence": score_by_confidence,
            "paired_tests_by_confidence": paired_tests,
            "correctness": correctness,
        }
