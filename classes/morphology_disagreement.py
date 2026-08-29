''' HYDRA-vs-WindowSHAP disagreement at the cluster level
    Needs scripts/run_windowshap_comparison.py to have already been run
    It calls cluster_level_analysis which generates hydra_windowshap_samples_with_clusters.csv.
'''

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Continuous morphology features available in ucr_dataset_clusters_k4.csv
# (see classes/clustering.py: series_features / build_feature_table).
MORPHOLOGY_FEATURES = [
    "series_length",
    "n_instances",
    "n_classes",
    "mean_abs_diff",
    "mean_abs_second_diff",
    "spectral_entropy",
    "spectral_centroid",
    "low_freq_energy_ratio",
    "spike_ratio",
    "zero_crossing_rate",
    "class_centroid_distance",
    "normalized_centroid_separation",
]

AGREEMENT_METRICS = [
    "iou",
    "overlap_fraction",
    "shap_minus_hydra_score_drop",
    "shap_minus_hydra_flip",
]


@dataclass
class MorphologyDisagreementAnalysis:
    '''Continuous-feature correlation and case extraction for HYDRA-vs-WindowSHAP disagreement
    '''
    windowshap_samples_csv: Path | str
    feature_csv: Path | str
    output_dir: Path | str = Path("outputs/saliency/morphology_disagreement")


    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.windowshap_samples_csv = Path(self.windowshap_samples_csv)
        self.feature_csv = Path(self.feature_csv)
        self.samples = None
        self.dataset_level = None


    def load(self):
        ''' Load samples from CSV 
        '''
        samples = pd.read_csv(self.windowshap_samples_csv)
        features = pd.read_csv(self.feature_csv)

        missing_feature_cols = set(MORPHOLOGY_FEATURES) - set(features.columns)
        if missing_feature_cols:
            raise ValueError(f"feature_csv is missing expected columns: {missing_feature_cols}")

        missing_sample_cols = {"dataset", "iou", "overlap_fraction"} - set(samples.columns)
        if missing_sample_cols:
            raise ValueError(f"windowshap_samples_csv is missing expected columns: {missing_sample_cols}")

        if "shap_minus_hydra_score_drop" not in samples.columns:
            samples = samples.copy()
            samples["shap_minus_hydra_score_drop"] = samples["shap_score_drop"] - samples["hydra_score_drop"]

        if "shap_minus_hydra_flip" not in samples.columns:
            samples = samples.copy()
            samples["shap_minus_hydra_flip"] = samples["shap_flipped"] - samples["hydra_flipped"]

        feature_cols = ["dataset"] + MORPHOLOGY_FEATURES
        self.samples = samples.merge(features[feature_cols].drop_duplicates("dataset"), on="dataset", how="left")
        return self.samples



    def dataset_level_table(self, fraction=0.10):
        ''' One row per dataset: mean agreement metrics + morphology features
            Aggregating to dataset level before correlating is the right unit of
            analysis here, since the morphology features are properties of the
            dataset, not of individual samples.
        '''
        if self.samples is None:
            self.load()
        data = self.samples[self.samples["fraction"] == fraction]
        agg = data.groupby("dataset", as_index=False)[AGREEMENT_METRICS].mean()
        feature_cols = ["dataset"] + MORPHOLOGY_FEATURES
        features_unique = data[feature_cols].drop_duplicates("dataset")
        self.dataset_level = agg.merge(features_unique, on="dataset", how="left")
        return self.dataset_level



    def feature_correlation_table(self, fraction=0.10):
        '''Spearman correlation between each morphology feature and each agreement metric
        '''
        dataset_level = self.dataset_level_table(fraction=fraction)
        rows = []

        for feature in MORPHOLOGY_FEATURES:
            for metric in AGREEMENT_METRICS:
                valid = dataset_level[[feature, metric]].dropna()
                if len(valid) < 3:
                    rho, p_value = np.nan, np.nan
                else:
                    rho, p_value = spearmanr(valid[feature], valid[metric])

                rows.append({
                    "feature": feature,
                    "metric": metric,
                    "fraction": fraction,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "n_datasets": len(valid),
                })
        table = pd.DataFrame(rows)
        table.to_csv(self.output_dir / f"feature_agreement_correlation_{int(fraction * 100)}.csv", index=False)
        return table
    

    def extract_cases(self, fraction=0.10, top_n=10):
        '''Extract the strongest agreement and strongest disagreement individual samples
        '''
        if self.samples is None:
            self.load()
        data = self.samples[self.samples["fraction"] == fraction].copy()
        most_agreement = data.sort_values("iou", ascending=False).head(top_n)
        most_disagreement = data.sort_values("iou", ascending=True).head(top_n)
        # Cols to keep subset
        keep_cols = [c for c in [
                "dataset", "sample_idx", "cluster", "cluster_name", "iou", "overlap_fraction",
                "hydra_score_drop", "shap_score_drop", "shap_minus_hydra_score_drop",
                "hydra_flipped", "shap_flipped",
            ] if c in data.columns]

        most_agreement = most_agreement[keep_cols]
        most_disagreement = most_disagreement[keep_cols]
        most_agreement.to_csv(self.output_dir / f"most_agreement_cases_{int(fraction * 100)}.csv", index=False)
        most_disagreement.to_csv(self.output_dir / f"most_disagreement_cases_{int(fraction * 100)}.csv", index=False)

        return {"most_agreement": most_agreement, "most_disagreement": most_disagreement}

    def run(self, fraction=0.10, top_n_cases=10):
        ''' Main function'''
        self.load()

        correlation_table = self.feature_correlation_table(fraction=fraction)
        cases = self.extract_cases(fraction=fraction, top_n=top_n_cases)
        dataset_level = self.dataset_level_table(fraction=fraction)
        dataset_level.to_csv(self.output_dir / f"dataset_level_agreement_features_{int(fraction * 100)}.csv", index=False)

        # Convenience view: only the statistically suggestive correlations,
        # sorted by strength, for a quick read in a meeting note.
        notable = (
            correlation_table.dropna(subset=["p_value"])
            .sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
        )
        notable.to_csv(self.output_dir / f"feature_agreement_correlation_ranked_{int(fraction * 100)}.csv", index=False)

        return {
            "dataset_level": dataset_level,
            "correlation_table": correlation_table,
            "correlation_ranked": notable,
            "most_agreement_cases": cases["most_agreement"],
            "most_disagreement_cases": cases["most_disagreement"],
        }
