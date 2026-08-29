from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, kruskal, skew
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)

from utils.data_utils import load_dataset
from utils.globals_config import CLUSTER_NAMES


def clean_name(value):
    ''' Normalise dataset/type names for robust matching
    '''
    return str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def cluster_purity(y_true, y_cluster):
    ''' Compute cluster purity against an external label assignment
    '''
    table = pd.crosstab(y_cluster, y_true)
    return table.max(axis=1).sum() / table.values.sum()


# Full feature set used for clustering.
FEATURE_COLS = [
    "log_series_length",
    "log_n_instances",
    "n_classes",
    "mean_abs_diff",
    "mean_abs_second_diff",
    "spectral_entropy",
    "spectral_centroid",
    "low_freq_energy_ratio",
    "spike_ratio",
    "zero_crossing_rate",
    "acf_lag1",
    "acf_lag5",
    "acf_lag10",
    "skewness",
    "kurtosis",
    "normalized_centroid_separation",
]


# Smaller feature sets used for reporting and profiling
REPORT_FEATURE_COLS = [
    "mean_abs_diff",
    "mean_abs_second_diff",
    "spike_ratio",
    "series_length",
    "spectral_entropy",
    "n_classes",
    "class_centroid_distance",
]

PROFILE_FEATURE_COLS = [
    "series_length",
    "n_classes",
    "mean_abs_diff",
    "mean_abs_second_diff",
    "spectral_entropy",
    "spike_ratio",
    "class_centroid_distance",
]


def z_norm(x, eps=1e-8):
    ''' Z-normalise a single time series
    '''
    x = np.asarray(x, dtype=np.float32).squeeze()
    mean = float(np.mean(x))
    std = float(np.std(x))
    return x - mean if std < eps else (x - mean) / std


def spectral_entropy(x):
    ''' Measure how spread out the frequency-domain energy is
    '''
    x = z_norm(x)
    power = np.abs(np.fft.rfft(x)) ** 2
    p = power / (power.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def spectral_centroid(x):
    ''' Compute the frequency-domain centre of mass
    '''
    x = z_norm(x)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x))
    return float((freqs * power).sum() / (power.sum() + 1e-12))


def low_freq_energy_ratio(x, cutoff=0.1):
    '''Measure the proportion of energy in low frequencies
    '''
    x = z_norm(x)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x))
    return float(power[freqs <= cutoff].sum() / (power.sum() + 1e-12))


def spike_ratio(x, thresh=2.5):
    '''Fraction of points with unusually large z-normalised magnitude
    '''
    return float(np.mean(np.abs(z_norm(x)) > thresh))


def zero_crossing_rate(x):
    '''Measure how often the signal changes sign
    '''
    x = z_norm(x)

    if len(x) < 2:
        return 0.0

    signs = np.sign(x)
    return float(np.mean(signs[1:] != signs[:-1]))


def autocorr_lag(x, lag):
    ''' Autocorrelation at a fixed lag
    '''
    x = z_norm(x)

    if len(x) <= lag:
        return 0.0

    value = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    return 0.0 if np.isnan(value) else float(value)


def series_features(x):
    '''Extract morphology features from a single time series
    '''
    x = z_norm(x)
    d1 = np.diff(x)
    d2 = np.diff(x, n=2)

    x_skew = skew(x)
    x_kurtosis = kurtosis(x)

    return {
        "variance": float(np.var(x)),
        "mean_abs_diff": float(np.mean(np.abs(d1))) if len(d1) else 0.0,
        "mean_abs_second_diff": float(np.mean(np.abs(d2))) if len(d2) else 0.0,
        "spectral_entropy": spectral_entropy(x),
        "spectral_centroid": spectral_centroid(x),
        "low_freq_energy_ratio": low_freq_energy_ratio(x),
        "spike_ratio": spike_ratio(x),
        "zero_crossing_rate": zero_crossing_rate(x),
        "acf_lag1": autocorr_lag(x, 1),
        "acf_lag5": autocorr_lag(x, 5),
        "acf_lag10": autocorr_lag(x, 10),
        "skewness": 0.0 if np.isnan(x_skew) else float(x_skew),
        "kurtosis": 0.0 if np.isnan(x_kurtosis) else float(x_kurtosis),
    }


def class_centroid_distance(X, y):
    '''Average pairwise distance between class centroids
    '''
    centroids = [X[y == c].mean(axis=0) for c in np.unique(y)]

    if len(centroids) < 2:
        return 0.0

    distances = [np.linalg.norm(centroids[i] - centroids[j]) for i in range(len(centroids)) for j in range(i + 1, len(centroids))]
    return float(np.mean(distances))


def normalized_centroid_separation(X, y):
    '''Class-separation score normalised by within-class spread
    '''
    classes = np.unique(y)

    if len(classes) < 2:
        return 0.0

    centroids = []
    within_spreads = []

    for c in classes:
        Xc = X[y == c]
        centroid = Xc.mean(axis=0)
        centroids.append(centroid)

        if len(Xc) > 1:
            within_spreads.append(np.mean(np.linalg.norm(Xc - centroid, axis=1)))

    between_distances = [np.linalg.norm(centroids[i] - centroids[j]) for i in range(len(centroids)) for j in range(i + 1, len(centroids))]

    mean_between = np.mean(between_distances) if between_distances else 0.0
    mean_within = np.mean(within_spreads) if within_spreads else 0.0
    return float(mean_between / (mean_within + 1e-12))




@dataclass
class SignalMorphologyClusterer:
    '''Cluster UCR datasets using global signal-morphology features
    '''
    datasets: Sequence[str]
    output_dir: Path | str = Path("outputs/clustering")
    n_clusters: int = 4
    seed: int = 42
    feature_cols: Sequence[str] = field(default_factory=lambda: FEATURE_COLS)
    cluster_names: dict[int, str] = field(default_factory=lambda: CLUSTER_NAMES.copy())


    def __post_init__(self):
        '''Prepare output folders and reusable preprocessing objects
        '''
        self.output_dir = Path(self.output_dir)
        self.csv_output_path = self.output_dir / "csv"
        self.img_output_path = self.output_dir / "imgs"

        self.csv_output_path.mkdir(parents=True, exist_ok=True)
        self.img_output_path.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=self.seed)

        self.dataset_features = None
        self.feature_matrix = None
        self.clustered_df = None



    def dataset_feature_row(self, dataset):
        '''Build one feature row for a single UCR dataset
        '''
        X_train, y_train, X_test, y_test, _ = load_dataset(dataset)

        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)

        # Average per-series morphology features across the dataset.
        per_series = pd.DataFrame([series_features(x) for x in X])

        row = {
            "dataset": dataset,
            "series_length": int(X.shape[-1]),
            "n_instances": int(X.shape[0]),
            "n_classes": int(len(np.unique(y))),
            "log_series_length": float(np.log1p(X.shape[-1])),
            "log_n_instances": float(np.log1p(X.shape[0])),
            "class_centroid_distance": class_centroid_distance(X, y),
            "normalized_centroid_separation": normalized_centroid_separation(X, y),
        }

        row.update({k: float(v) for k, v in per_series.mean(numeric_only=True).to_dict().items()})
        return row



    def build_feature_table(self):
        ''' Extract morphology features for all requested datasets
        '''
        rows = []
        failed = []

        for i, dataset in enumerate(self.datasets, start=1):
            print(f"[{i}/{len(self.datasets)}] extracting features: {dataset}")
            try:
                rows.append(self.dataset_feature_row(dataset))
            except Exception as e:
                print(f"  [WARN] {dataset} failed, skipping: {e}")
                failed.append({"dataset": dataset, "error": str(e)})

        if failed:
            failed_path = self.csv_output_path / "clustering_failed_datasets.csv"
            pd.DataFrame(failed).to_csv(failed_path, index=False)
            print(f"\n[WARN] {len(failed)} datasets failed — see {failed_path}")

        self.dataset_features = pd.DataFrame(rows)
        return self.dataset_features


    def fit(self):
        ''' Scale features, cluster datasets, and compute PCA coordinates
        '''
        self.build_feature_table()
        self.feature_matrix = self.scaler.fit_transform(self.dataset_features[list(self.feature_cols)])
        labels = AgglomerativeClustering(n_clusters=self.n_clusters).fit_predict(self.feature_matrix)

        pca_coords = self.pca.fit_transform(self.feature_matrix)
        self.clustered_df = self.dataset_features.copy()
        self.clustered_df["cluster"] = labels
        self.clustered_df["cluster_name"] = self.clustered_df["cluster"].map(self.cluster_names)
        self.clustered_df["pca1"] = pca_coords[:, 0]
        self.clustered_df["pca2"] = pca_coords[:, 1]

        # NOTE - Compute each dataset's Euclidean distance to its cluster centroid in the standardised feature space so we can get most representative samples
        cluster_centroids = (
            pd.DataFrame(self.feature_matrix)
            .assign(cluster=labels)
            .groupby("cluster")
            .mean()
            .values  # shape (n_clusters, n_features)
        )
        distances = np.array([np.linalg.norm(self.feature_matrix[i] - cluster_centroids[labels[i]]) for i in range(len(labels))])
        self.clustered_df["distance_to_centroid"] = distances
        return self.clustered_df



    def save(self):
        ''' Write feature and cluster assignment CSV files
        '''
        self.dataset_features.to_csv(self.csv_output_path / "ucr_dataset_feature_table.csv", index=False)
        self.clustered_df.to_csv(self.csv_output_path / f"ucr_dataset_clusters_k{self.n_clusters}.csv", index=False)
        self.clustered_df.to_csv(self.csv_output_path / "ucr_dataset_clusters_final.csv", index=False)



    def run(self):
        '''Run the full clustering pipeline
        '''
        print(f"Writing outputs to {self.output_dir.resolve()}")
        self.fit()
        self.save()
        print("\nCluster sizes:")
        print(self.clustered_df["cluster"].value_counts().sort_index())
        return self.clustered_df