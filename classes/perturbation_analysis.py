"""
    Methodology validation testing: Checks whether the top > random > bottom faithfulness ordering holds under:
        * local-mean masking
        * linear-interpolation masking
    So that the initial set of results werent unintentionally caused by global mean masking
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from classes.stats_utils import paired_wilcoxon, consistency_rate
from utils.perturbation import PERTURBATION_LABELS
from utils.globals_config import MODE_ORDER



@dataclass
class PerturbationOperatorAnalysis:
    '''Aggregate and test perturbation operator robustness results
    '''
    samples_csv: Path | str
    output_dir: Path | str = Path("outputs/saliency/perturbation_operators/analysis")

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_csv = Path(self.samples_csv)
        self.samples = None

    def load(self):
        self.samples = pd.read_csv(self.samples_csv)
        missing = {"dataset", "mode", "fraction", "perturbation", "flipped"} - set(self.samples.columns)
        if missing:
            raise ValueError(f"samples CSV is missing expected columns: {missing}")
        return self.samples


    def dataset_level_table(self, fraction=0.10):
        '''Collapse per-sample rows to one row per (dataset, operator, mode) for paired tests
        '''
        if self.samples is None:
            self.load()
        data = self.samples[self.samples["fraction"] == fraction]
        table = (
            data.groupby(["dataset", "perturbation", "mode"], as_index=False)
            .agg(
                flip_rate=("flipped", "mean"),
                mean_score_drop=("score_drop", "mean"),
                mean_bounded_relative_score_drop=("bounded_relative_score_drop", "mean"),
            )
        )

        return table

    def operator_mode_table(self, metric="flip_rate", fraction=0.10, as_percent=False):
        '''operator x mode table for one metric, matching the layout of the original
           model x mode tables in SaliencyResultsAnalysis
        '''
        dataset_table = self.dataset_level_table(fraction=fraction)

        table = (
            dataset_table.groupby(["perturbation", "mode"])[metric]
            .mean()
            .unstack("mode")
            .reindex(columns=MODE_ORDER)
        )
        if as_percent:
            table = table * 100
        filename = f"{metric}_{int(fraction * 100)}_by_operator.csv"
        table.to_csv(self.output_dir / filename)
        return table



    def paired_tests_by_operator(self, fraction=0.10):
        '''Run top>random and top>bottom Wilcoxon tests separately for each operator
        '''
        dataset_table = self.dataset_level_table(fraction=fraction)
        metrics = [("flip_rate", "Flip rate"), ("mean_bounded_relative_score_drop", "Bounded score drop")]
        comparisons = [("random", "Top > Random"), ("bottom", "Top > Bottom")]

        rows = []
        for operator, op_group in dataset_table.groupby("perturbation"):
            for metric_col, metric_name in metrics:
                for baseline_mode, comparison_name in comparisons:
                    # Generate all wilcoxon tests needed
                    result = paired_wilcoxon(
                        op_group,
                        group_col="dataset",
                        condition_col="mode",
                        value_col=metric_col,
                        candidate="top",
                        baseline=baseline_mode,
                    )
                    rows.append({
                        "perturbation": operator,
                        "perturbation_label": PERTURBATION_LABELS.get(operator, operator),
                        "fraction": fraction,
                        "metric": metric_name,
                        "comparison": comparison_name,
                        "mean_diff": result["mean_diff"],
                        "top_greater_count": result["candidate_greater_count"],
                        "wilcoxon_p": result["wilcoxon_p"],
                    })
        tests_df = pd.DataFrame(rows)
        tests_df.to_csv(self.output_dir / f"paired_tests_by_operator_{int(fraction * 100)}.csv", index=False)
        return tests_df

    

    def consistency_by_operator(self, fraction=0.10):
        ''' Dataset level consistency rate (top>random, top>bottom) for each operator
        '''
        dataset_table = self.dataset_level_table(fraction=fraction)
        rows = []
        for operator, op_group in dataset_table.groupby("perturbation"):
            for baseline_mode, label in [("random", "top_gt_random_pct"), ("bottom", "top_gt_bottom_pct")]:
                # Calculate consistency rate for each operator
                rate, n = consistency_rate(
                    op_group,
                    group_col="dataset",
                    condition_col="mode",
                    value_col="flip_rate",
                    candidate="top",
                    baseline=baseline_mode,
                )
                rows.append({
                    "perturbation": operator,
                    "perturbation_label": PERTURBATION_LABELS.get(operator, operator),
                    "comparison": label,
                    "rate_pct": np.nan if np.isnan(rate) else round(rate * 100, 1),
                    "n_datasets": n,
                })
        consistency_df = pd.DataFrame(rows)
        consistency_df.to_csv(self.output_dir / f"consistency_by_operator_{int(fraction * 100)}.csv", index=False)
        return consistency_df



    def ordering_preserved_summary(self, fraction=0.10):
        ''' Creates a 1 row per operator summary answering the robustness question (top > random > bottom)
        '''
        flip_table = self.operator_mode_table(metric="flip_rate", fraction=fraction, as_percent=True)
        score_table = self.operator_mode_table(metric="mean_bounded_relative_score_drop", fraction=fraction, as_percent=False)

        rows = []
        for operator in flip_table.index:
            flip_row = flip_table.loc[operator]
            score_row = score_table.loc[operator]
            flip_ordering_ok = bool(flip_row["top"] > flip_row["random"] > flip_row["bottom"])
            score_ordering_ok = bool(score_row["top"] > score_row["random"] > score_row["bottom"])

            rows.append({
                "perturbation": operator,
                "perturbation_label": PERTURBATION_LABELS.get(operator, operator),
                "flip_top": flip_row["top"],
                "flip_random": flip_row["random"],
                "flip_bottom": flip_row["bottom"],
                "flip_ordering_preserved": flip_ordering_ok,
                "bounded_score_drop_top": score_row["top"],
                "bounded_score_drop_random": score_row["random"],
                "bounded_score_drop_bottom": score_row["bottom"],
                "score_drop_ordering_preserved": score_ordering_ok,
            })
        summary = pd.DataFrame(rows)
        summary.to_csv(self.output_dir / f"ordering_preserved_summary_{int(fraction * 100)}.csv", index=False)
        return summary



    def run(self, fraction=0.10):
        ''' Main function for running the perturbation operator analysis
        '''
        self.load()

        flip_table = self.operator_mode_table(metric="flip_rate", fraction=fraction, as_percent=True)
        bounded_table = self.operator_mode_table(metric="mean_bounded_relative_score_drop", fraction=fraction, as_percent=False)
        paired_tests = self.paired_tests_by_operator(fraction=fraction)
        consistency = self.consistency_by_operator(fraction=fraction)
        ordering_summary = self.ordering_preserved_summary(fraction=fraction)
        return {
            "flip_table": flip_table,
            "bounded_table": bounded_table,
            "paired_tests": paired_tests,
            "consistency": consistency,
            "ordering_summary": ordering_summary,
        }
