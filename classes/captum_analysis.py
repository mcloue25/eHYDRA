'''
Loads and plots CaptumComparison results, either for one run or comparing two runs side by side
'''

from __future__ import annotations

import sys
from pathlib import Path

# make utils/ importable regardless of where this class is imported from
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.globals_config import TSCAPTUM_PLOT_DIR as _TSCAPTUM_PLOT_DIR
from utils.plot_config import (
    METHOD_COLOURS,
    METHOD_LABELS,
    CLUSTER_ORDER,
    CLUSTER_SHORT,
    FRACTIONS,
    FRACTION_LABELS,
    apply_thesis_style,
)
apply_thesis_style()


def load_run(run_dir: Path):
    ''' Loads all output CSVs from one CaptumComparison run directory
    '''
    files = {
        "overlap": "captum_overlap_summary.csv",
        "deletion": "captum_deletion_summary.csv",
        "timing": "captum_timing_summary.csv",
        "paired": "captum_paired_tests.csv",
        "cluster_overlap": "captum_cluster_overlap_summary.csv",
        "cluster_deletion": "captum_cluster_deletion_summary.csv",
        "pairwise": "captum_pairwise_samples.csv",
    }
    data = {}
    for key, fname in files.items():
        path = run_dir / fname
        if path.exists() and path.stat().st_size > 0:
            try:
                data[key] = pd.read_csv(path)
            except Exception as e:
                print(f"  [WARN] Could not read {fname}: {e}")
                data[key] = pd.DataFrame()
        else:
            data[key] = pd.DataFrame()
    return data


def method_score_drops(overlap_df: pd.DataFrame):
    ''' Extracts per-method mean score drop and flip rate from the overlap summary
    '''
    rows = []
    for frac in FRACTIONS:
        sub = overlap_df[overlap_df.fraction == frac]

        def get(ma, mb, col):
            r = sub[(sub.method_a == ma) & (sub.method_b == mb)]
            return float(r[col].iloc[0]) if len(r) else np.nan

        hydra_drop = get("hydra", "shapley_sampling", "mean_score_drop_a")
        hydra_flip = get("hydra", "shapley_sampling", "flip_rate_a")
        shap_drop = get("hydra", "shapley_sampling", "mean_score_drop_b")
        shap_flip = get("hydra", "shapley_sampling", "flip_rate_b")
        fa_drop = get("hydra", "feature_ablation", "mean_score_drop_b")
        fa_flip = get("hydra", "feature_ablation", "flip_rate_b")
        mr_drop = get("hydra", "mrsqm", "mean_score_drop_b")
        mr_flip = get("hydra", "mrsqm", "flip_rate_b")

        for method, drop, flip in [
            ("hydra", hydra_drop, hydra_flip),
            ("shapley_sampling", shap_drop, shap_flip),
            ("feature_ablation", fa_drop, fa_flip),
            ("mrsqm", mr_drop, mr_flip),
        ]:
            rows.append({"method": method, "fraction": frac, "score_drop": drop, "flip_rate": flip})
    return pd.DataFrame(rows)




class CaptumAnalysis:
    ''' Loads and analyses one CaptumComparison run.
        cluster_csv (ucr_dataset_clusters_k4_with_types.csv) is only needed for
        the centroid-distance plot -- everything else works without it.
    '''
    def __init__(self, run_dir: str | Path, label: str = "Run", cluster_csv: str | Path | None = None):
        self.run_dir = Path(run_dir)
        self.label = label
        self.cluster_csv = Path(cluster_csv) if cluster_csv else None
        # NOTE - Output dir for all plots
        self.plot_dir = _TSCAPTUM_PLOT_DIR 
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self._data = load_run(self.run_dir)
        self._method_drops = method_score_drops(self._data["overlap"])



    def report(self):
        ''' Prints summary tables and saves every plot for this run
        '''
        print(f"\n{'='*60}")
        print(f"CaptumAnalysis: {self.label}")
        print(f"{'='*60}")
        # NOTE - Geneerate all summary tables and plots
        self.print_summary_tables()
        self.plot_score_drops()
        self.plot_deletion_curves()
        self.plot_iou_heatmap()
        self.plot_cluster_deletion()
        self.plot_cluster_score_drops()
        self.plot_timing()

        if self.cluster_csv and self.cluster_csv.exists():
            self.plot_centroid_distance_faithfulness()
        print(f"\nPlots saved to {self.plot_dir}")



    def print_summary_tables(self):
        '''Prints deletion curve, score drop/flip, timing & paired-test tables
        '''
        d = self._data["deletion"]
        t = self._data["timing"]
        ov = self._data["overlap"]

        print(f"\n--- Deletion curve (AUCS\u0303_top), {d.n_datasets.iloc[0] if len(d) else '?'} datasets ---")
        if not d.empty:
            print(d[["method", "mean_aucs_top", "std_aucs_top", "n_datasets"]].to_string(index=False))

        print("\n--- Score drop and flip rate @ 10% masking ---")
        if not ov.empty:
            drops = self._method_drops[self._method_drops.fraction == 0.10]
            for _, row in drops.iterrows():
                lbl = METHOD_LABELS.get(row.method, row.method)
                print(f"  {lbl:25s}  drop={row.score_drop:.3f}  flip={row.flip_rate:.3f}")

        print("\n--- Timing (median s/sample) ---")
        if not t.empty:
            for _, row in t.sort_values("median_time_s").iterrows():
                lbl = METHOD_LABELS.get(row.method, row.method)
                print(f"  {lbl:25s}  {row.median_time_s:.4f}s")



        # NOTE - Main thesis question: is eHYDRA statistically as faithful as Shapley?
        print("\n--- Paired tests: eHYDRA vs Shapley @ 10% ---")
        pt = self._data["paired"]
        if not pt.empty:
            sub = pt[(pt.method_a == "hydra") & (pt.method_b == "shapley_sampling") & (pt.fraction == 0.10)]
            print(sub[["metric", "mean_diff_b_minus_a", "b_greater_count", "n_non_tied", "wilcoxon_p_b_gt_a"]].to_string(index=False))




    def plot_score_drops(self):
        ''' Plot grouped bars: score drop and flip rate per method, per masking fraction
        '''
        drops = self._method_drops
        methods = ["hydra", "shapley_sampling", "feature_ablation", "mrsqm"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        x = np.arange(len(FRACTIONS))
        width = 0.18
        for i, method in enumerate(methods):
            sub = drops[drops.method == method]
            offset = (i - 1.5) * width
            for ax, metric in zip(axes, ["score_drop", "flip_rate"]):
                vals = [float(sub[sub.fraction == f][metric].iloc[0]) if len(sub[sub.fraction == f]) else np.nan for f in FRACTIONS]
                ax.bar(x + offset, vals, width, label=METHOD_LABELS.get(method, method), color=METHOD_COLOURS.get(method, "#AAAAAA"), alpha=0.85)

        axes[0].set_title(f"Mean score drop — {self.label}")
        axes[0].set_ylabel("Score drop")
        axes[0].set_xticks(x); axes[0].set_xticklabels(FRACTION_LABELS)
        axes[0].set_xlabel("Masking fraction")
        axes[0].legend(loc="upper left", fontsize=8)
        axes[1].set_title(f"Prediction flip rate — {self.label}")
        axes[1].set_ylabel("Flip rate")
        axes[1].set_xticks(x); axes[1].set_xticklabels(FRACTION_LABELS)
        axes[1].set_xlabel("Masking fraction")
        axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        fig.tight_layout()
        path = self.plot_dir / f"score_drops_{self.label.replace(' ', '_')}.png"
        fig.savefig(path); plt.close(fig)
        print(f"Saved: {path.name}")
        return path




    def plot_deletion_curves(self):
        ''' Plot AUCS̃_top bar chart
        '''
        d = self._data["deletion"]
        methods = ["hydra", "shapley_sampling", "feature_ablation", "mrsqm"]
        d_idx = d.set_index("method")

        vals = [d_idx.loc[m, "mean_aucs_top"] if m in d_idx.index else np.nan for m in methods]
        errs = [d_idx.loc[m, "std_aucs_top"] if m in d_idx.index else np.nan for m in methods]
        colours = [METHOD_COLOURS.get(m, "#AAAAAA") for m in methods]
        labels = [METHOD_LABELS.get(m, m) for m in methods]

        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(methods))
        bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colours, alpha=0.85, error_kw={"linewidth": 1.2})
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_ylabel("Mean AUCS\u0303$_{top}$")
        ax.set_title(f"Deletion curve faithfulness — {self.label}")
        valid = [v for v in vals if not np.isnan(v)]
        if valid:
            ax.set_ylim(0, max(valid) * 1.35)

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        path = self.plot_dir / f"deletion_aucs_top_{self.label.replace(' ', '_')}.png"
        fig.savefig(path); plt.close(fig)
        print(f"Saved: {path.name}")
        return path




    def plot_iou_heatmap(self):
        ''' Plot the heatmap of mean IoU across every method pair and masking fraction
        '''
        ov = self._data["overlap"]
        pairs = [
            ("hydra", "shapley_sampling"),
            ("hydra", "feature_ablation"),
            ("hydra", "mrsqm"),
            ("shapley_sampling", "feature_ablation"),
            ("shapley_sampling", "mrsqm"),
            ("feature_ablation", "mrsqm"),
        ]
        pair_labels = [
            "eHYDRA vs Shapley",
            "eHYDRA vs Feat. Abl.",
            "eHYDRA vs MrSQM",
            "Shapley vs Feat. Abl.",
            "Shapley vs MrSQM",
            "Feat. Abl. vs MrSQM",
        ]

        matrix = np.zeros((len(pairs), len(FRACTIONS)))
        for i, (ma, mb) in enumerate(pairs):
            for j, frac in enumerate(FRACTIONS):
                row = ov[(ov.method_a == ma) & (ov.method_b == mb) & (ov.fraction == frac)]
                if not row.empty:
                    matrix[i, j] = float(row["mean_iou"].iloc[0])

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=0.5, aspect="auto")
        ax.set_xticks(range(len(FRACTIONS))); ax.set_xticklabels(FRACTION_LABELS)
        ax.set_yticks(range(len(pairs))); ax.set_yticklabels(pair_labels, fontsize=9)
        ax.set_xlabel("Masking fraction")
        ax.set_title(f"Mean IoU between methods — {self.label}")
        plt.colorbar(im, ax=ax, label="Mean IoU")

        for i in range(len(pairs)):
            for j in range(len(FRACTIONS)):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=9, color="black" if matrix[i, j] < 0.3 else "white")

        fig.tight_layout()
        path = self.plot_dir / f"iou_heatmap_{self.label.replace(' ', '_')}.png"
        fig.savefig(path); plt.close(fig)
        print(f"Saved: {path.name}")
        return path



    def plot_cluster_deletion(self):
        ''' Plot AUCS̃_top grouped by signal-morphology cluster
        '''
        cd = self._data["cluster_deletion"]
        methods = ["hydra", "shapley_sampling", "feature_ablation", "mrsqm"]
        clusters = [c for c in CLUSTER_ORDER if c in cd.cluster_name.values]

        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(clusters))
        width = 0.18

        for i, method in enumerate(methods):
            sub = cd[cd.method == method].set_index("cluster_name")
            vals = [float(sub.loc[c, "mean_aucs_top"]) if c in sub.index else np.nan for c in clusters]
            offset = (i - 1.5) * width
            ax.bar(x + offset, vals, width, label=METHOD_LABELS.get(method, method), color=METHOD_COLOURS.get(method, "#AAAAAA"), alpha=0.85)

        ax.set_xticks(x); ax.set_xticklabels([CLUSTER_SHORT[c] for c in clusters])
        ax.set_ylabel("Mean AUCS\u0303$_{top}$")
        ax.set_title(f"Deletion curve by signal-morphology cluster — {self.label}")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 0.65)
        fig.tight_layout()
        path = self.plot_dir / f"cluster_deletion_{self.label.replace(' ', '_')}.png"
        fig.savefig(path); plt.close(fig)
        print(f"Saved: {path.name}")
        return path


    

    def plot_cluster_score_drops(self):
        ''' Function for plotting the score drop at 10% masking grouped by cluster
        '''
        co = self._data["cluster_overlap"]
        sub = co[co.fraction == 0.10]
        clusters = [c for c in CLUSTER_ORDER if c in sub.cluster_name.values]

        rows = []
        for cluster in clusters:
            cs = sub[sub.cluster_name == cluster]
            h = cs[cs.method_a == "hydra"]
            if not h.empty:
                rows.append({"cluster": cluster, "method": "hydra", "score_drop": float(h.iloc[0].mean_score_drop_a)})
            for method_b in ["shapley_sampling", "feature_ablation", "mrsqm"]:
                r = cs[(cs.method_a == "hydra") & (cs.method_b == method_b)]
                if not r.empty:
                    rows.append({"cluster": cluster, "method": method_b, "score_drop": float(r.iloc[0].mean_score_drop_b)})

        df = pd.DataFrame(rows)
        methods = ["hydra", "shapley_sampling", "feature_ablation", "mrsqm"]

        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(clusters))
        width = 0.18
        for i, method in enumerate(methods):
            vals = [float(df[(df.cluster == c) & (df.method == method)]["score_drop"].iloc[0]) if len(df[(df.cluster == c) & (df.method == method)]) else np.nan for c in clusters]
            offset = (i - 1.5) * width
            ax.bar(x + offset, vals, width, label=METHOD_LABELS.get(method, method), color=METHOD_COLOURS.get(method, "#AAAAAA"), alpha=0.85)

        ax.set_xticks(x); ax.set_xticklabels([CLUSTER_SHORT[c] for c in clusters])
        ax.set_ylabel("Mean score drop")
        ax.set_title(f"Score drop at 10% masking by cluster — {self.label}")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        path = self.plot_dir / f"cluster_score_drops_{self.label.replace(' ', '_')}.png"
        fig.savefig(path); plt.close(fig)
        print(f"Saved: {path.name}")
        return path



    def plot_timing(self):
        ''' Function for plotting the median explanation time per method in log-scaled form
        '''
        t = self._data["timing"]
        if t.empty:
            return None

        methods = ["mrsqm", "feature_ablation", "hydra", "shapley_sampling"]
        t_idx = t.set_index("method")
        vals = [float(t_idx.loc[m, "median_time_s"]) if m in t_idx.index else np.nan for m in methods]
        colours = [METHOD_COLOURS.get(m, "#AAAAAA") for m in methods]

        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(methods))
        bars = ax.bar(x, vals, color=colours, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=12, ha="right")
        ax.set_ylabel("Median explanation time (s)")
        ax.set_title(f"Explanation time per sample — {self.label}")
        ax.set_yscale("log")

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15, f"{val:.3f}s", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        path = self.plot_dir / f"timing_{self.label.replace(' ', '_')}.png"
        fig.savefig(path); plt.close(fig)
        print(f"Saved: {path.name}")
        return path



    def plot_centroid_distance_faithfulness(self) -> Path:
        ''' Two-panel plot: faithfulness and region agreement vs distance from cluster centroid
        '''
        if not self.cluster_csv or not self.cluster_csv.exists():
            print("  [SKIP] centroid distance plot: cluster CSV not found")
            return None

        clusters_df = pd.read_csv(self.cluster_csv)
        if "distance_to_centroid" not in clusters_df.columns:
            print("  [SKIP] centroid distance plot: no distance_to_centroid column")
            return None

        pairwise = self._data["pairwise"]
        deletion_raw_path = self.run_dir / "captum_deletion_samples.csv"
        if not deletion_raw_path.exists():
            print("  [SKIP] centroid distance plot: captum_deletion_samples.csv not found")
            return None

        del_samples = pd.read_csv(deletion_raw_path)
        ds_del = del_samples.groupby(["dataset", "method"], as_index=False).agg(mean_aucs_top=("aucs_top", "mean"))
        hs_iou = (
            pairwise[(pairwise.method_a == "hydra") & (pairwise.method_b == "shapley_sampling") & (pairwise.fraction == 0.10)]
            .groupby("dataset", as_index=False)
            .agg(mean_iou=("iou", "mean"))
        )

        dist_cols = [c for c in ["dataset", "cluster", "cluster_name", "distance_to_centroid"] if c in clusters_df.columns]
        merged_del = ds_del.merge(clusters_df[dist_cols].drop_duplicates(), on="dataset", how="inner")
        merged_iou = hs_iou.merge(clusters_df[dist_cols].drop_duplicates(), on="dataset", how="inner")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # NOTE - panel A: AUCS̃_top vs centroid distance, with a rolling mean per method
        ax = axes[0]
        for method in ["hydra", "shapley_sampling", "feature_ablation"]:
            sub = merged_del[merged_del.method == method].sort_values("distance_to_centroid")
            ax.scatter(sub.distance_to_centroid, sub.mean_aucs_top, color=METHOD_COLOURS.get(method, "#AAAAAA"), alpha=0.5, s=20, label=METHOD_LABELS.get(method, method))
            if len(sub) >= 5:
                window = max(5, len(sub) // 6)
                smoothed = sub.mean_aucs_top.rolling(window, center=True, min_periods=1).mean()
                ax.plot(sub.distance_to_centroid.values, smoothed.values, color=METHOD_COLOURS.get(method, "#AAAAAA"), linewidth=1.5)

        ax.set_xlabel("Distance from cluster centroid")
        ax.set_ylabel("Mean AUCS\u0303$_{top}$")
        ax.set_title("Faithfulness vs centroid distance")
        ax.legend(fontsize=8)

        for method in ["hydra", "shapley_sampling", "feature_ablation"]:
            sub = merged_del[merged_del.method == method].dropna(subset=["distance_to_centroid", "mean_aucs_top"])
            if len(sub) >= 5:
                rho, p = spearmanr(sub.distance_to_centroid, sub.mean_aucs_top)
                print(f"  Centroid dist vs AUCS\u0303_top ({METHOD_LABELS.get(method, method)}): \u03c1={rho:.3f}, p={p:.4f}")



        # NOTE - panel B: IoU vs centroid distance, coloured by cluster
        ax2 = axes[1]
        cluster_colours = {
            "High-frequency / high-curvature": "#2a78d6",
            "Smooth / low-complexity": "#1baf7a",
            "Short / moderately rough": "#eda100",
            "Spiky / multi-class": "#e34948",
        }
        for cluster in CLUSTER_ORDER:
            sub = merged_iou[merged_iou.cluster_name == cluster].sort_values("distance_to_centroid")
            if sub.empty:
                continue
            ax2.scatter(sub.distance_to_centroid, sub.mean_iou, color=cluster_colours.get(cluster, "#AAAAAA"), alpha=0.6, s=25, label=CLUSTER_SHORT.get(cluster, cluster))

        ax2.set_xlabel("Distance from cluster centroid")
        ax2.set_ylabel("Mean IoU (eHYDRA vs Shapley, 10%)")
        ax2.set_title("Region agreement vs centroid distance")
        ax2.legend(fontsize=8, title="Cluster")

        if len(merged_iou) >= 5:
            rho, p = spearmanr(merged_iou.distance_to_centroid, merged_iou.mean_iou)
            ax2.text(0.05, 0.92, f"\u03c1={rho:.3f}, p={p:.3f}", transform=ax2.transAxes, fontsize=9)
            print(f"  Centroid dist vs IoU (eHYDRA vs Shapley): \u03c1={rho:.3f}, p={p:.4f}")

        fig.suptitle(f"Effect of centroid distance — {self.label}", fontsize=11)
        fig.tight_layout()
        path = self.plot_dir / f"centroid_distance_{self.label.replace(' ', '_')}.png"
        fig.savefig(path); plt.close(fig)
        print(f"  Saved: {path.name}")
        return path







# Seperate class for comparing two seperate captum analysis runs side by side


class CaptumComparator:
    '''Compares two CaptumAnalysis runs side by side (e.g. 38 dataset subset vs full 128)
    '''
    def __init__(self, run_a: CaptumAnalysis, run_b: CaptumAnalysis):
        self.run_a = run_a
        self.run_b = run_b
        self.plot_dir = _TSCAPTUM_PLOT_DIR / "comparison"
        self.plot_dir.mkdir(parents=True, exist_ok=True)


    def compare(self):
        '''Runs every comparative plot and prints the delta tables
        '''
        self.print_delta_table()
        self.plot_score_drop_comparison()
        self.plot_deletion_comparison()
        self.plot_iou_comparison()
        self.plot_cluster_deletion_comparison()
        print(f"\nComparison plots saved to {self.plot_dir}")


    def print_delta_table(self):
        '''Prints run_a -> run_b deltas for AUCS̃_top, score drop @ 10%, and IoU
        '''
        print(f"\n{'='*60}")
        print(f"Comparison: {self.run_a.label}  \u2192  {self.run_b.label}")
        print(f"{'='*60}")
        d_a = self.run_a._data["deletion"].set_index("method")
        d_b = self.run_b._data["deletion"].set_index("method")
        methods = ["hydra", "shapley_sampling", "feature_ablation", "mrsqm"]

        print("\n--- AUCS\u0303_top delta ---")
        print(f"{'Method':25s}  {self.run_a.label:>14s}  {self.run_b.label:>14s}  {'\u0394':>8s}")
        for m in methods:
            a = float(d_a.loc[m, "mean_aucs_top"]) if m in d_a.index else np.nan
            b = float(d_b.loc[m, "mean_aucs_top"]) if m in d_b.index else np.nan
            delta = b - a if not (np.isnan(a) or np.isnan(b)) else np.nan
            sign = "+" if (not np.isnan(delta) and delta >= 0) else ""
            print(f"  {METHOD_LABELS.get(m, m):23s}  {a:14.3f}  {b:14.3f}  {sign}{delta:.3f}")

        print("\n--- Score drop @ 10% delta ---")
        da10 = self.run_a._method_drops[self.run_a._method_drops.fraction == 0.10].set_index("method")
        db10 = self.run_b._method_drops[self.run_b._method_drops.fraction == 0.10].set_index("method")
        print(f"{'Method':25s}  {self.run_a.label:>14s}  {self.run_b.label:>14s}  {'\u0394':>8s}")
        for m in methods:
            a = float(da10.loc[m, "score_drop"]) if m in da10.index else np.nan
            b = float(db10.loc[m, "score_drop"]) if m in db10.index else np.nan
            delta = b - a if not (np.isnan(a) or np.isnan(b)) else np.nan
            sign = "+" if (not np.isnan(delta) and delta >= 0) else ""
            print(f"  {METHOD_LABELS.get(m, m):23s}  {a:14.3f}  {b:14.3f}  {sign}{delta:.3f}")

        print("\n--- IoU (eHYDRA vs Shapley @ 10%) ---")
        ov_a = self.run_a._data["overlap"]
        ov_b = self.run_b._data["overlap"]
        for frac, flabel in zip(FRACTIONS, FRACTION_LABELS):
            def iou(ov, ma, mb, f):
                ''' Calculate the iou between runs '''
                r = ov[(ov.method_a == ma) & (ov.method_b == mb) & (ov.fraction == f)]
                return float(r.mean_iou.iloc[0]) if len(r) else np.nan
            a = iou(ov_a, "hydra", "shapley_sampling", frac)
            b = iou(ov_b, "hydra", "shapley_sampling", frac)
            delta = b - a if not (np.isnan(a) or np.isnan(b)) else np.nan
            sign = "+" if (not np.isnan(delta) and delta >= 0) else ""
            print(f"  {flabel}: {self.run_a.label}={a:.3f}  {self.run_b.label}={b:.3f}  \u0394={sign}{delta:.3f}")



    def compare_bar(self, ax, vals_a, vals_b, labels, title, ylabel, ylim=None):
        ''' Shared grouped-bar helper used by all the comparison plots below
        '''
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, vals_a, width, label=self.run_a.label, color=METHOD_COLOURS.get("ehydra", "#4C72B0"), alpha=0.8)
        ax.bar(x + width / 2, vals_b, width, label=self.run_b.label, color=METHOD_COLOURS.get("shapley_sampling", "#DD8452"), alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)



    def plot_score_drop_comparison(self):
        ''' Function for plotting the score dorp comparison between both runs
        '''
        methods = ["hydra", "shapley_sampling", "feature_ablation", "mrsqm"]
        method_labels = [METHOD_LABELS.get(m, m) for m in methods]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, frac, flabel in zip(axes, FRACTIONS, FRACTION_LABELS):
            da = self.run_a._method_drops[self.run_a._method_drops.fraction == frac].set_index("method")
            db = self.run_b._method_drops[self.run_b._method_drops.fraction == frac].set_index("method")
            vals_a = [float(da.loc[m, "score_drop"]) if m in da.index else np.nan for m in methods]
            vals_b = [float(db.loc[m, "score_drop"]) if m in db.index else np.nan for m in methods]
            self.compare_bar(ax, vals_a, vals_b, method_labels, f"Score drop — {flabel} masking", "Score drop")

        fig.suptitle(f"Score drop: {self.run_a.label} vs {self.run_b.label}", fontsize=11)
        fig.tight_layout()
        path = self.plot_dir / "comparison_score_drops.png"
        fig.savefig(path); plt.close(fig); print(f"  Saved: {path.name}")
        return path



    def plot_deletion_comparison(self):
        ''' Function for plotting the deletion curve results between both runs
        '''
        methods = ["hydra", "shapley_sampling", "feature_ablation", "mrsqm"]
        da = self.run_a._data["deletion"].set_index("method")
        db = self.run_b._data["deletion"].set_index("method")
        vals_a = [float(da.loc[m, "mean_aucs_top"]) if m in da.index else np.nan for m in methods]
        vals_b = [float(db.loc[m, "mean_aucs_top"]) if m in db.index else np.nan for m in methods]
        errs_a = [float(da.loc[m, "std_aucs_top"]) if m in da.index else np.nan for m in methods]
        errs_b = [float(db.loc[m, "std_aucs_top"]) if m in db.index else np.nan for m in methods]

        x = np.arange(len(methods))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - width / 2, vals_a, width, yerr=errs_a, capsize=3, label=self.run_a.label, color=METHOD_COLOURS.get("ehydra", "#4C72B0"), alpha=0.8)
        ax.bar(x + width / 2, vals_b, width, yerr=errs_b, capsize=3, label=self.run_b.label, color=METHOD_COLOURS.get("shapley_sampling", "#DD8452"), alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=12, ha="right")
        ax.set_ylabel("Mean AUCS\u0303$_{top}$")
        ax.set_title(f"Deletion curve: {self.run_a.label} vs {self.run_b.label}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = self.plot_dir / "comparison_deletion.png"
        fig.savefig(path); plt.close(fig); print(f"  Saved: {path.name}")
        return path



    def plot_iou_comparison(self):
        ''' Plot the ious between both runs
        '''
        pairs = [
            ("hydra", "shapley_sampling", "eHYDRA vs Shapley"),
            ("hydra", "feature_ablation", "eHYDRA vs Feat. Abl."),
            ("shapley_sampling", "feature_ablation", "Shapley vs Feat. Abl."),
            ("hydra", "mrsqm", "eHYDRA vs MrSQM"),
        ]
        fig, axes = plt.subplots(1, len(FRACTIONS), figsize=(13, 4))

        for ax, frac, flabel in zip(axes, FRACTIONS, FRACTION_LABELS):
            ov_a = self.run_a._data["overlap"]
            ov_b = self.run_b._data["overlap"]
            labels = [p[2] for p in pairs]
            vals_a, vals_b = [], []
            for ma, mb, _ in pairs:
                ra = ov_a[(ov_a.method_a == ma) & (ov_a.method_b == mb) & (ov_a.fraction == frac)]
                rb = ov_b[(ov_b.method_a == ma) & (ov_b.method_b == mb) & (ov_b.fraction == frac)]
                vals_a.append(float(ra.mean_iou.iloc[0]) if len(ra) else np.nan)
                vals_b.append(float(rb.mean_iou.iloc[0]) if len(rb) else np.nan)
            self.compare_bar(ax, vals_a, vals_b, labels, f"IoU — {flabel}", "Mean IoU", ylim=(0, 0.55))

        fig.suptitle(f"Region agreement (IoU): {self.run_a.label} vs {self.run_b.label}", fontsize=11)
        fig.tight_layout()
        path = self.plot_dir / "comparison_iou.png"
        fig.savefig(path); plt.close(fig); print(f"  Saved: {path.name}")
        return path



    def plot_cluster_deletion_comparison(self):
        '''
        '''
        cd_a = self.run_a._data["cluster_deletion"]
        cd_b = self.run_b._data["cluster_deletion"]
        methods = ["hydra", "shapley_sampling", "feature_ablation"]
        clusters = CLUSTER_ORDER

        fig, axes = plt.subplots(1, len(methods), figsize=(14, 4), sharey=True)

        for ax, method in zip(axes, methods):
            vals_a, vals_b = [], []
            for cluster in clusters:
                ra = cd_a[(cd_a.method == method) & (cd_a.cluster_name == cluster)]
                rb = cd_b[(cd_b.method == method) & (cd_b.cluster_name == cluster)]
                vals_a.append(float(ra.mean_aucs_top.iloc[0]) if len(ra) else np.nan)
                vals_b.append(float(rb.mean_aucs_top.iloc[0]) if len(rb) else np.nan)

            x = np.arange(len(clusters))
            width = 0.35
            ax.bar(x - width / 2, vals_a, width, label=self.run_a.label, color=METHOD_COLOURS.get("ehydra", "#4C72B0"), alpha=0.8)
            ax.bar(x + width / 2, vals_b, width, label=self.run_b.label, color=METHOD_COLOURS.get("shapley_sampling", "#DD8452"), alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([CLUSTER_SHORT.get(c, c) for c in clusters], rotation=14, ha="right")
            ax.set_title(METHOD_LABELS.get(method, method), fontsize=10)
            ax.set_ylim(0, 0.65)
            if ax is axes[0]:
                ax.set_ylabel("Mean AUCS\u0303$_{top}$")
            ax.legend(fontsize=7)

        fig.suptitle(f"Cluster deletion curve: {self.run_a.label} vs {self.run_b.label}", fontsize=11)
        fig.tight_layout()
        path = self.plot_dir / "comparison_cluster_deletion.png"
        fig.savefig(path); plt.close(fig); print(f"  Saved: {path.name}")
        return path