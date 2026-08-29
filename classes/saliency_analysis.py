from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, wilcoxon

from utils.globals_config import MODE_ORDER

# Stable ordering used in report tables.
MODEL_ORDER = ["HYDRA", "LR", "MrSQM"]


@dataclass
class SaliencyResultsAnalysis:
    '''Aggregate and analyse saliency masking outputs across models/datasets
    '''

    saliency_output_dir: Path | str
    output_dir: Path | str = Path("outputs/saliency/analysis")
    cluster_csv_path: Path | str | None = None

    def __post_init__(self):
        ''' Prepare paths and initialise result containers
        '''
        self.saliency_output_dir = Path(self.saliency_output_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cluster_csv_path = Path(self.cluster_csv_path) if self.cluster_csv_path else None
        self.samples = None
        self.summary = None
        self.clustered_summary = None



    def load_results(self):
        ''' Load and combine per-model sample/summary CSV files
        '''
        sample_files = sorted(self.saliency_output_dir.glob("*_samples.csv"))
        summary_files = sorted(self.saliency_output_dir.glob("*_summary.csv"))
        if not sample_files:
            raise FileNotFoundError(f"No sample files found in {self.saliency_output_dir}")
        if not summary_files:
            raise FileNotFoundError(f"No summary files found in {self.saliency_output_dir}")

        self.samples = pd.concat([pd.read_csv(path) for path in sample_files], ignore_index=True)
        self.summary = pd.concat([pd.read_csv(path) for path in summary_files], ignore_index=True)

        # Add bounded score drop cols when loading older result files.
        self.samples = self.ensure_bounded_score_drop(self.samples)
        self.summary = self.ensure_summary_bounded_score_drop(self.summary)
        self.samples.to_csv(self.output_dir / "combined_saliency_samples.csv", index=False)
        self.summary.to_csv(self.output_dir / "combined_saliency_summary.csv", index=False)
        return self.samples, self.summary



    def ensure_bounded_score_drop(self, df, eps=1e-6):
        '''Compute bounded relative score drop if it didnt already exist for that result
        '''
        df = df.copy()
        if "bounded_relative_score_drop" not in df.columns:
            df["bounded_relative_score_drop"] = np.clip(df["score_drop"] / (np.abs(df["score_before"]) + eps), -1.0, 1.0)
        return df



    def ensure_summary_bounded_score_drop(self, df):
        '''Add mean bounded score-drop to summary rows if absent
        '''
        df = df.copy()

        if "mean_bounded_relative_score_drop" not in df.columns:
            if self.samples is None:
                raise ValueError("Cannot compute bounded summary before samples are loaded.")

            bounded_summary = (
                self.samples
                .groupby(["dataset", "model", "fraction", "mode"], as_index=False)
                .agg(
                    mean_bounded_relative_score_drop=(
                        "bounded_relative_score_drop",
                        "mean",
                    )
                )
            )
            df = df.merge(bounded_summary, on=["dataset", "model", "fraction", "mode"], how="left")

        return df

    def add_clusters(self):
        ''' Attach signal-morphology cluster labels to the saliency summary
        '''
        if self.cluster_csv_path is None:
            return None

        if self.summary is None:
            self.load_results()

        clusters = pd.read_csv(self.cluster_csv_path)

        keep_cols = ["dataset", "cluster"]
        if "cluster_name" in clusters.columns:
            keep_cols.append("cluster_name")

        self.clustered_summary = self.summary.merge(clusters[keep_cols], on="dataset", how="left")
        self.clustered_summary.to_csv(self.output_dir / "combined_saliency_summary_with_clusters.csv", index=False)
        return self.clustered_summary



    def table_at_fraction(self, metric, fraction=0.10, as_percent=False, filename=None):
        ''' Create a model-by-masking-mode table for one metric and masking fraction
        '''
        if self.summary is None:
            self.load_results()
        table = (
            self.summary[self.summary["fraction"] == fraction]
            .groupby(["model", "mode"])[metric]
            .mean()
            .unstack("mode")
            .reindex(index=MODEL_ORDER, columns=MODE_ORDER)
        )
        if as_percent:
            table = table * 100
        if filename:
            table.to_csv(self.output_dir / filename)
        return table



    def flip_rate_table(self, fraction=0.10):
        ''' Generate the 10% masking prediction flip-rate table
        '''
        return self.table_at_fraction(
            metric="flip_rate",
            fraction=fraction,
            as_percent=True,
            filename=f"flip_rate_{int(fraction * 100)}.csv",
        )



    def bounded_score_drop_table(self, fraction=0.10):
        ''' Generate the bounded relative score-drop table
        '''
        return self.table_at_fraction(
            metric="mean_bounded_relative_score_drop",
            fraction=fraction,
            as_percent=False,
            filename=f"bounded_relative_score_drop_{int(fraction * 100)}.csv",
        )



    def paired_tests(self, model="HYDRA", fraction=0.10):
        '''Run paired Wilcoxon tests comparing top masking with baselines
        '''
        if self.summary is None:
            self.load_results()

        rows = []
        tests = [("flip_rate", "Flip rate"), ("mean_bounded_relative_score_drop", "Bounded score drop")]
        comparisons = [("random", "Top > Random"), ("bottom", "Top > Bottom")]
        data = self.summary[(self.summary["model"] == model) & (self.summary["fraction"] == fraction)]

        for metric_col, metric_name in tests:
            # Pivot to dataset-level paired observations.
            pivot = data.pivot_table(index="dataset", columns="mode", values=metric_col, aggfunc="mean").dropna(subset=["top"])
            for baseline_mode, comparison_name in comparisons:
                valid = pivot.dropna(subset=[baseline_mode])
                diff = valid["top"] - valid[baseline_mode]

                # Wilcoxon ignores tied zero diff pairs
                non_tied = diff[diff != 0]

                if len(non_tied) == 0:
                    p_value = np.nan
                    top_greater = 0
                    total = 0
                else:
                    _, p_value = wilcoxon(non_tied, alternative="greater", zero_method="wilcox")
                    top_greater = int((non_tied > 0).sum())
                    total = int(len(non_tied))

                rows.append({
                    "model": model,
                    "fraction": fraction,
                    "metric": metric_name,
                    "comparison": comparison_name,
                    "mean_diff": float(diff.mean()),
                    "top_greater": top_greater,
                    "n_non_tied": total,
                    "top_greater_count": f"{top_greater}/{total}",
                    "wilcoxon_p": float(p_value) if not np.isnan(p_value) else np.nan,
                })

        tests_df = pd.DataFrame(rows)
        tests_df.to_csv(self.output_dir / f"{model.lower()}_paired_tests_{int(fraction * 100)}.csv", index=False)
        return tests_df



    def dataset_consistency(self, fraction=0.10, metric="mean_score_drop"):
        ''' Measure how often top masking is more disruptive per dataset
        '''
        if self.summary is None:
            self.load_results()

        rows = []
        data = self.summary[self.summary["fraction"] == fraction]
        for model, group in data.groupby("model"):
            pivot = group.pivot_table(index="dataset", columns="mode", values=metric, aggfunc="mean")
            valid_random = pivot.dropna(subset=["top", "random"])
            valid_bottom = pivot.dropna(subset=["top", "bottom"])
            rows.append({
                "model": model,
                "fraction": fraction,
                "metric": metric,
                "top_greater_random_percent": float((valid_random["top"] > valid_random["random"]).mean() * 100),
                "top_greater_bottom_percent": float((valid_bottom["top"] > valid_bottom["bottom"]).mean() * 100),
                "n_random": int(len(valid_random)),
                "n_bottom": int(len(valid_bottom)),
            })
        # Save consistency DF
        consistency = pd.DataFrame(rows)
        consistency.to_csv(self.output_dir / f"dataset_consistency_{int(fraction * 100)}.csv", index=False)
        return consistency



    def nonflip_score_effect(self, model="HYDRA", fraction=0.10):
        ''' Check whether top masking reduces score even when prediction does not flip
        '''
        if self.samples is None:
            self.load_results()

        data = self.samples[(self.samples["model"] == model) & (self.samples["fraction"] == fraction) & (self.samples["mode"] == "top")].copy()
        data["group"] = np.where(data["flipped"].astype(float) > 0, "Flipped", "Not flipped")
        data["positive_drop"] = data["score_drop"] > 0
        table = (
            data.groupby("group", as_index=False)
            .agg(
                samples=("sample_idx", "count"),
                positive_drop_percent=("positive_drop", lambda x: float(x.mean() * 100)),
            )
        )
        order = pd.CategoricalDtype(["Flipped", "Not flipped"], ordered=True)
        table["group"] = table["group"].astype(order)
        table = table.sort_values("group")
        table.to_csv(self.output_dir / f"{model.lower()}_nonflip_score_effect_{int(fraction * 100)}.csv", index=False)
        return table



    def cluster_gap_table(self, model="HYDRA", fraction=0.10):
        '''Compute top-minus-random perturbation gaps within each cluster
        '''
        if self.clustered_summary is None:
            self.add_clusters()
        if self.clustered_summary is None:
            raise ValueError("cluster_csv_path is required for cluster gap analysis.")

        data = self.clustered_summary[(self.clustered_summary["model"] == model) & (self.clustered_summary["fraction"] == fraction)]
        rows = []
        cluster_cols = ["cluster"]
        if "cluster_name" in data.columns:
            cluster_cols.append("cluster_name")
        for cluster_values, group in data.groupby(cluster_cols):
            if not isinstance(cluster_values, tuple):
                cluster_values = (cluster_values,)

            pivot = group.pivot_table(index="dataset", columns="mode", values=["flip_rate", "mean_score_drop"], aggfunc="mean")
            row = {
                "cluster": cluster_values[0],
                "flip_rate_gap": float((pivot[("flip_rate", "top")] - pivot[("flip_rate", "random")]).mean()),
                "score_drop_gap": float((pivot[("mean_score_drop", "top")] - pivot[("mean_score_drop", "random")]).mean()),
                "n_datasets": int(group["dataset"].nunique()),
            }
            if len(cluster_values) > 1:
                row["cluster_name"] = cluster_values[1]
            rows.append(row)

        gap_table = pd.DataFrame(rows).sort_values("cluster")
        gap_table.to_csv(self.output_dir / f"{model.lower()}_cluster_gap_{int(fraction * 100)}.csv", index=False)
        return gap_table
    


    def within_cluster_tests(self, model="HYDRA", fraction=0.10):
        ''' Run paired Wilcoxon tests separately inside each cluster
        '''
        if self.clustered_summary is None:
            self.add_clusters()
        if self.clustered_summary is None:
            raise ValueError("cluster_csv_path is required for within-cluster tests.")

        rows = []
        data = self.clustered_summary[(self.clustered_summary["model"] == model) & (self.clustered_summary["fraction"] == fraction)]
        metrics = [("flip_rate", "Flip rate"), ("mean_score_drop", "Score drop"), ("mean_bounded_relative_score_drop", "Bounded score drop")]
        comparisons = [("random", "Top > Random"), ("bottom", "Top > Bottom")]
        for cluster, cluster_group in data.groupby("cluster"):
            cluster_name = (cluster_group["cluster_name"].iloc[0] if "cluster_name" in cluster_group.columns else f"Cluster {cluster}")

            for metric_col, metric_name in metrics:
                pivot = cluster_group.pivot_table(index="dataset", columns="mode", values=metric_col, aggfunc="mean").dropna(subset=["top"])
                for baseline_mode, comparison_name in comparisons:
                    valid = pivot.dropna(subset=[baseline_mode])
                    diff = valid["top"] - valid[baseline_mode]
                    non_tied = diff[diff != 0]

                    if len(non_tied) == 0:
                        p_value = np.nan
                        top_greater = 0
                        total = 0
                    else:
                        _, p_value = wilcoxon(non_tied, alternative="greater", zero_method="wilcox")
                        top_greater = int((non_tied > 0).sum())
                        total = int(len(non_tied))

                    rows.append({
                        "model": model,
                        "fraction": fraction,
                        "cluster": int(cluster),
                        "cluster_name": cluster_name,
                        "metric": metric_name,
                        "comparison": comparison_name,
                        "mean_diff": float(diff.mean()),
                        "top_greater_count": f"{top_greater}/{total}",
                        "wilcoxon_p": float(p_value) if not np.isnan(p_value) else np.nan,
                        "n_non_tied": total,
                    })
        tests = pd.DataFrame(rows)
        tests.to_csv(self.output_dir / f"{model.lower()}_within_cluster_tests_{int(fraction * 100)}.csv", index=False,)
        return tests

    

    def kruskal_cluster_gap_tests(self, model="HYDRA", fraction=0.10):
        ''' Test whether top-minus-random gaps differ across clusters
        '''
        if self.clustered_summary is None:
            self.add_clusters()
        if self.clustered_summary is None:
            raise ValueError("cluster_csv_path is required for Kruskal-Wallis tests.")
        data = self.clustered_summary[(self.clustered_summary["model"] == model) & (self.clustered_summary["fraction"] == fraction)]
        pivot = data.pivot_table(index=["dataset", "cluster"], columns="mode", values=["flip_rate", "mean_score_drop", "mean_bounded_relative_score_drop"], aggfunc="mean").reset_index()

        rows = []
        gap_defs = [
            ("flip_rate", "flip_rate_top_minus_random"),
            ("mean_score_drop", "score_drop_top_minus_random"),
            ("mean_bounded_relative_score_drop", "bounded_score_drop_top_minus_random"),
        ]
        for metric_col, gap_name in gap_defs:
            # Convert per-mode values into a per-dataset top-minus-random gap.
            pivot[gap_name] = pivot[(metric_col, "top")] - pivot[(metric_col, "random")]
            groups = [group[gap_name].dropna().values for _, group in pivot.groupby(("cluster", ""))]
            groups = [g for g in groups if len(g) > 0]
            stat, p_value = kruskal(*groups)
            rows.append({
                "model": model,
                "fraction": fraction,
                "gap": gap_name,
                "kruskal_H": float(stat),
                "p_value": float(p_value),
            })

        tests = pd.DataFrame(rows)
        tests.to_csv(self.output_dir / f"{model.lower()}_kruskal_cluster_gap_tests_{int(fraction * 100)}.csv", index=False)
        return tests



    def run(self, fraction=0.10):
        ''' Run the complete saliency analysis workflow
        '''
        self.load_results()
        # NOTE - Generate ore report tables.
        flip_table = self.flip_rate_table(fraction=fraction)
        bounded_table = self.bounded_score_drop_table(fraction=fraction)
        paired_tests = self.paired_tests(model="HYDRA", fraction=fraction)
        consistency = self.dataset_consistency(fraction=fraction)
        nonflip_effect = self.nonflip_score_effect(model="HYDRA", fraction=fraction)
        # save results
        results = {
            "samples": self.samples,
            "summary": self.summary,
            "flip_table": flip_table,
            "bounded_table": bounded_table,
            "paired_tests": paired_tests,
            "dataset_consistency": consistency,
            "nonflip_score_effect": nonflip_effect,
        }

        # Cluster aware analysis if cluster assignments are provided.
        if self.cluster_csv_path is not None:
            self.add_clusters()
            cluster_gap = self.cluster_gap_table(model="HYDRA", fraction=fraction)
            within_cluster_tests = self.within_cluster_tests(model="HYDRA", fraction=fraction)
            kruskal_tests = self.kruskal_cluster_gap_tests(model="HYDRA", fraction=fraction)
            results.update({
                "clustered_summary": self.clustered_summary,
                "cluster_gap": cluster_gap,
                "within_cluster_tests": within_cluster_tests,
                "kruskal_cluster_gap_tests": kruskal_tests,
            })
        return results
