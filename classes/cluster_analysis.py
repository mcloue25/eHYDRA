from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    pairwise_distances_argmin_min,
    silhouette_score,
)

from classes.clustering import (
    FEATURE_COLS,
    PROFILE_FEATURE_COLS,
    clean_name,
    cluster_purity,
    REPORT_FEATURE_COLS,
    SignalMorphologyClusterer,
)


@dataclass
class ClusterAnalysis:
    clusterer: SignalMorphologyClusterer

    def __post_init__(self):
        self.output_dir = self.clusterer.output_dir
        self.csv_output_path = self.output_dir / "csv"
        self.img_output_path = self.output_dir / "imgs"
        self.csv_output_path.mkdir(parents=True, exist_ok=True)
        self.img_output_path.mkdir(parents=True, exist_ok=True)
        self.n_clusters = self.clusterer.n_clusters
        self.dataset_features = self.clusterer.dataset_features
        self.clustered_df = self.clusterer.clustered_df
        self.feature_matrix = self.clusterer.feature_matrix
        self.pca = self.clusterer.pca


    def feature_profile(self, feature_cols=FEATURE_COLS):
        ''' Mean feature value per cluster, z-scored against the full dataset population
        '''
        profile = self.clustered_df.groupby("cluster")[list(feature_cols)].mean()
        profile_z = (profile - self.dataset_features[list(feature_cols)].mean()) / self.dataset_features[list(feature_cols)].std()
        profile_z.round(3).to_csv(self.csv_output_path / f"cluster_feature_profiles_k{self.n_clusters}.csv")
        return profile_z

    def report_feature_profile(self, feature_cols=PROFILE_FEATURE_COLS):
        ''' same as feature profile but used for the report
        '''
        profile = self.clustered_df.groupby("cluster")[list(feature_cols)].mean()
        profile_z = (profile - self.clustered_df[list(feature_cols)].mean()) / self.clustered_df[list(feature_cols)].std()
        profile_z.round(3).to_csv(self.csv_output_path / f"report_feature_profile_k{self.n_clusters}.csv")
        return profile_z

    def kruskal_tests(self, feature_cols=REPORT_FEATURE_COLS):
        '''Kruskal-Wallis test per feature: does it actually differ across clusters
        '''
        rows = []
        for col in feature_cols:
            groups = [self.clustered_df.loc[self.clustered_df["cluster"] == c, col].dropna() for c in sorted(self.clustered_df["cluster"].unique())]
            stat, p_value = kruskal(*groups)
            rows.append({"feature": col, "kruskal_H": float(stat), "p_value": float(p_value)})

        # sort by p-value so the most cluster-discriminative features are on top
        tests = pd.DataFrame(rows).sort_values("p_value")
        tests.to_csv(self.csv_output_path / "cluster_feature_tests.csv", index=False)
        return tests



    def evaluate_k_range(self, k_values=range(3, 20)):
        ''' Refit agglomerative clustering at each k & records cluster quality + size diagnostics to double chekc the choice of k=4
        '''
        rows = []
        for k in k_values:
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(self.feature_matrix)
            counts = pd.Series(labels).value_counts()
            rows.append({
                "k": k,
                "silhouette": silhouette_score(self.feature_matrix, labels),
                "davies_bouldin": davies_bouldin_score(self.feature_matrix, labels),
                "calinski_harabasz": calinski_harabasz_score(self.feature_matrix, labels),
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
                "n_singletons": int((counts == 1).sum()),
                "n_clusters_lt_5": int((counts < 5).sum()),
            })
        k_eval = pd.DataFrame(rows)
        k_eval.to_csv(self.csv_output_path / f"cluster_count_sensitivity_k{min(k_values)}_to_k{max(k_values)}.csv", index=False)
        return k_eval

    

    def attach_ucr_types(self, ucr_types):
        ''' Left-joins the official UCR archive 'Type' label onto each dataset
        '''
        type_df = ucr_types.copy()
        clustered = self.clustered_df.copy()
        type_df["dataset_key"] = type_df["Dataset"].map(clean_name)
        clustered["dataset_key"] = clustered["dataset"].map(clean_name)

        merged = clustered.merge(
            type_df[["dataset_key", "Type"]],
            on="dataset_key",
            how="left",
        ).rename(columns={"Type": "ucr_type"})
        merged.to_csv(self.csv_output_path / f"ucr_dataset_clusters_k{self.n_clusters}_with_types.csv", index=False)
        return merged



    def compare_to_ucr_types(self, ucr_types):
        ''' Checks our clusters against UCR's own type labels as an external validity check
        '''
        merged = self.attach_ucr_types(ucr_types)
        valid = merged.dropna(subset=["ucr_type"])  # drop datasets UCR didn't type
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(valid["ucr_type"], valid["cluster"])
        metrics = pd.DataFrame([
            {
                "ARI": adjusted_rand_score(valid["ucr_type"], valid["cluster"]),
                "NMI": normalized_mutual_info_score(valid["ucr_type"], valid["cluster"]),
                "Homogeneity": homogeneity,
                "Completeness": completeness,
                "V-measure": v_measure,
                "Purity": cluster_purity(valid["ucr_type"], valid["cluster"]),
                "n_matched": int(len(valid)),
            }
        ])
        metrics.to_csv(self.csv_output_path / "external_ucr_type_metrics.csv", index=False)
        return metrics



    def type_contingency(self, ucr_types):
        ''' Cross-tab of cluster vs UCR type, raw counts and row-normalised
        '''
        merged = self.attach_ucr_types(ucr_types)
        contingency = pd.crosstab(merged["cluster"], merged["ucr_type"], dropna=False)
        contingency_norm = contingency.div(contingency.sum(axis=1), axis=0)  # row-normalise so each cluster sums to 1
        contingency.to_csv(self.csv_output_path / "cluster_type_contingency.csv")
        contingency_norm.to_csv(self.csv_output_path / "cluster_type_contingency_normalised.csv")
        return {
            "contingency": contingency,
            "contingency_norm": contingency_norm,
        }



    def type_summary(self, ucr_types):
        ''' Per-cluster summary of which UCR type dominates and by how much
        Args:
            ucr_types (List) : Lisyt of ucr types
        '''
        merged = self.attach_ucr_types(ucr_types)
        rows = []
        for cluster_id, group in merged.dropna(subset=["ucr_type"]).groupby("cluster"):
            counts = group["ucr_type"].value_counts()
            dominant_type = counts.index[0]
            dominant_count = int(counts.iloc[0])
            total = int(len(group))

            rows.append({
                "cluster": int(cluster_id),
                "n_datasets": total,
                "dominant_type": dominant_type,
                "dominant_type_count": dominant_count,
                "dominant_type_percent": dominant_count / total * 100,
                "type_distribution": counts.to_dict(),
            })
        summary = pd.DataFrame(rows).sort_values("cluster")
        summary.to_csv(self.csv_output_path / "cluster_type_summary.csv", index=False)
        return summary



    def closest_datasets_to_centroids(self, top_n=20):
        ''' Ranks each clusters member datasets by distance to the cluster centroid & give the most 'representative' datasets to use
        '''
        rows = []
        for cluster_id in sorted(self.clustered_df["cluster"].unique()):
            cluster_mask = self.clustered_df["cluster"] == cluster_id
            cluster_indices = np.where(cluster_mask.to_numpy())[0]
            cluster_features = self.feature_matrix[cluster_indices]
            centroid = cluster_features.mean(axis=0, keepdims=True)

            # distance from every point in the cluster to its own centroid
            _, distances = pairwise_distances_argmin_min(cluster_features, centroid)
            # Rank all datasets 
            ranked = (
                self.clustered_df.loc[
                    cluster_mask,
                    ["dataset", "cluster", "cluster_name"],
                ]
                .copy()
                .assign(distance_to_centroid=distances)
                .sort_values("distance_to_centroid")
                .head(top_n)
            )
            ranked["rank_within_cluster"] = range(1, len(ranked) + 1)
            rows.append(ranked)
        closest = pd.concat(rows, ignore_index=True)
        closest.to_csv(self.csv_output_path / f"closest_{top_n}_datasets_per_cluster.csv", index=False)
        return closest

    

    def plot_feature_profile(self, profile_z=None):
        ''' Heatmap of the z-scored feature profile from feature_profile()
        '''
        if profile_z is None:
            profile_z = self.feature_profile()

        fig, ax = plt.subplots(figsize=(13, 5))
        im = ax.imshow(profile_z, aspect="auto")
        fig.colorbar(im, ax=ax, label="Standardised feature value")

        ax.set_xticks(range(len(profile_z.columns)))
        ax.set_xticklabels(profile_z.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(profile_z.index)))
        ax.set_yticklabels(profile_z.index)

        ax.set_xlabel("Feature")
        ax.set_ylabel("Cluster")
        ax.set_title(f"Cluster feature profiles (k={self.n_clusters})")

        # annotate every cell with its actual z-score, not just colour
        for i in range(profile_z.shape[0]):
            for j in range(profile_z.shape[1]):
                ax.text(j, i, f"{profile_z.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(self.img_output_path / f"cluster_feature_profiles_k{self.n_clusters}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)



    def plot_pca_clusters(self):
        ''' 2D PCA scatter of the clusters, coloured by cluster label
        '''
        fig, ax = plt.subplots(figsize=(8, 5))
        variance = self.pca.explained_variance_ratio_ * 100

        for cluster in sorted(self.clustered_df["cluster"].unique()):
            subset = self.clustered_df[self.clustered_df["cluster"] == cluster]
            label = self.clusterer.cluster_names.get(cluster, f"Cluster {cluster}")
            ax.scatter(subset["pca1"], subset["pca2"], label=f"{label} (n={len(subset)})", s=55, alpha=0.8, edgecolor="black", linewidth=0.4)

        ax.set_xlabel(f"PC1 ({variance[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({variance[1]:.1f}% variance)")
        ax.set_title(f"PCA projection of signal-morphology clusters (k={self.n_clusters})")
        ax.legend(frameon=True, fontsize=8, loc="best")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(self.img_output_path / f"signal_morphology_clusters_k{self.n_clusters}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)




    def plot_k_diagnostics(self, k_eval=None):
        ''' Plots silhouette, Davies-Bouldin and cluster-size stability
        '''
        if k_eval is None:
            k_eval = self.evaluate_k_range()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(k_eval["k"], k_eval["silhouette"], marker="o")
        axes[0].set_title("Silhouette")
        axes[0].set_xlabel("k")

        axes[1].plot(k_eval["k"], k_eval["davies_bouldin"], marker="o")
        axes[1].set_title("Davies-Bouldin")
        axes[1].set_xlabel("k")

        axes[2].plot(k_eval["k"], k_eval["min_cluster_size"], marker="o", label="Min cluster size")
        axes[2].plot(k_eval["k"], k_eval["n_singletons"], marker="o", label="Singleton clusters")
        axes[2].set_title("Cluster size diagnostics")
        axes[2].set_xlabel("k")
        axes[2].legend()

        for ax in axes:
            ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.img_output_path / "cluster_count_sensitivity.png", dpi=300, bbox_inches="tight")
        plt.close(fig)




    def run(self):
        ''' Main run function to call all functions above
        '''
        # Build summary tables.
        profile_z = self.feature_profile()
        report_profile_z = self.report_feature_profile()
        tests = self.kruskal_tests()
        k_eval = self.evaluate_k_range()
        closest_datasets = self.closest_datasets_to_centroids(top_n=20)

        # Save diagnostic plots.
        self.plot_feature_profile(profile_z)
        self.plot_pca_clusters()
        self.plot_k_diagnostics(k_eval)

        # Return outputs for inspection in main.py or a notebook.
        return {
            "feature_profile": profile_z,
            "report_feature_profile": report_profile_z,
            "feature_tests": tests,
            "k_eval": k_eval,
            "closest_datasets": closest_datasets,
        }