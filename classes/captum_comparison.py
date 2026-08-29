'''
Compares HYDRA saliency against tsCaptum/TSHAP/MrSQM explainers on the same model and samples
'''

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from classes.models.hydra_explainable import HydraModelExplainable
from classes.models.mrsqm_explainable import MrSQMExplainableModel
from utils.data_utils import load_dataset
from utils.explainability import apply_mask, pred_to_hydra_class_index, select_contiguous_window
from utils.mask_utils import compare_masks
from utils.globals_config import CLUSTER_NAMES
from classes.windowshap import get_predicted_class_score


class HydraTsCaptumAdapter:
    ''' Wraps HydraModelExplainable so tsCaptum can call it like a sklearn classifier.
        tsCaptum passes (N, channels, T) arrays into predict_proba/predict. HYDRA is
        univariate, so we drop the channel dim, run it through HYDRA's own
        transform -> scale -> ridge pipeline, and hand back probabilities.
    '''
    def __init__(self, hydra_model: HydraModelExplainable):
        if not hydra_model.is_fitted:
            raise RuntimeError("HydraModelExplainable must be fitted first.")
        self._hydra = hydra_model
        self.classes_ = hydra_model.classifier.classes_  # tsCaptum inspects this directly


    def squeeze_and_transform(self, X: np.ndarray) -> np.ndarray:
        # accepts (N, 1, T) or (N, T), returns HYDRA-scaled features
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            X = X[:, 0, :]
        return self._hydra._transform_and_scale(X)


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Platt-scaled sigmoid on decision scores -- monotone, preserves ranking
        Z = self.squeeze_and_transform(X)
        return self._hydra.classifier._predict_proba_lr(Z)


    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            X = X[:, 0, :]
        return self._hydra.predict(X)



def make_tscaptum_explainer(method_name: str, adapter: HydraTsCaptumAdapter):
    ''' Builds the tsCaptum explainer
    '''
    try:
        from tsCaptum.explainers import Shapley_Value_Sampling, Feature_Ablation
    except ImportError as exc:
        raise ImportError("Install tsCaptum first:  pip install tsCaptum") from exc

    if method_name == "shapley_sampling":
        return Shapley_Value_Sampling(adapter)
    elif method_name == "feature_ablation":
        return Feature_Ablation(adapter)
    else:
        raise ValueError(f"Unknown tsCaptum method: {method_name}")


def explain_one_sample_tscaptum(explainer, x: np.ndarray, pred_label: int, n_segments: int):
    ''' Runs a tsCaptum explainer on one sample, returns |importance| of shape (T,)
    '''
    x_3d = x[np.newaxis, np.newaxis, :].astype(np.float32)  # (1, 1, T)
    labels = np.array([pred_label])

    raw = explainer.explain(
        samples=x_3d,
        labels=labels,
        batch_size=1,
        n_segments=n_segments,
        normalise=False,
        baseline=float(x.mean()),
    )
    return np.abs(raw[0, 0, :]).astype(np.float32)




def import_tshap_explainer():
    ''' Importing TSHAP (Le Nguyen & Ifrim 2025)
        Link to paper - https://www.researchgate.net/publication/396003966_TSHAP_Fast_and_Exact_SHAP_for_Explaining_Time_Series_Classification_and_Regression
        Github repo was pulled and stored locally for my implementation 
    '''
    try:
        from tshap.tshap import TSHAPExplainer
    except ImportError as exc:
        raise ImportError(
            "Install tshap first: pip install tshap (or add the mlgig/tshap repo to PYTHONPATH)"
        ) from exc
    return TSHAPExplainer




def tshap_window_stride(series_length: int, window_fraction: float = 0.10, max_stride: int = 5):
    ''' Scales TSHAP's window_length/stride to series length.
        Library defaults (20, 5) are tuned to the paper's T=200 synthetic series and
        give almost no window positions on short UCR series (e.g. Chinatown, T=24 -> 2 positions at defaults). 
        window_length here follows the paper's own 10% experimental setting rather than the library default. Stride is capped and shrinks for short 
        series so there are enough positions to interpolate.
    '''
    window_length = max(2, round(window_fraction * series_length))
    stride = max(1, min(max_stride, series_length // 20))
    return window_length, stride



def tshap_backgrounds(X_train: np.ndarray, include_train: bool, n_train_background_samples: int, seed: int):
    ''' Builds the TSHAP background conditions for one dataset.
        Each background is (n_baselines, 1, T)
    '''
    T = X_train.shape[-1]
    backgrounds = {
        "tshap_centroid": X_train.mean(axis=0).reshape(1, 1, T).astype(np.float32),
        "tshap_zero": np.zeros((1, 1, T), dtype=np.float32),
    }

    if include_train:
        rng = np.random.default_rng(seed)
        n = min(n_train_background_samples, X_train.shape[0])
        idx = rng.choice(X_train.shape[0], size=n, replace=False)
        backgrounds["tshap_train"] = X_train[idx][:, np.newaxis, :].astype(np.float32)  # (n, T) -> (n, 1, T)
    return backgrounds



def explain_one_sample_tshap(tshap_explainer, adapter: HydraTsCaptumAdapter, x: np.ndarray, pred_label: int, baseline: np.ndarray):
    ''' Runs TSHAP-Window on one sample against one background condition.
        clf_targets passed explicitly rather than using predict_proba(X)[:, 1] because it assumes column 1 is the positive class
        roi is disabled on the shared explainer (see run_dataset), so only
        window_exp is populated here; roi_exp is ignored.
    '''
    x_3d = x[np.newaxis, np.newaxis, :].astype(np.float32)  # (1, 1, T)
    window_exp, _roi_exp = tshap_explainer.explain(
        x_3d,
        baselines=baseline,
        model=adapter,
        clf_targets=np.array([pred_label]),
    )
    return np.abs(window_exp[0, 0, :]).astype(np.float32)



# NOTE - Where I actually calculate the deletion curve
def turbe_deletion_curve(hydra_model: HydraModelExplainable, x: np.ndarray, pred_label: int, importance: np.ndarray, score_before: float, n_steps: int = 20):
    ''' Computes the Turbé et al. deletion curve and AUCS̃_top / F1S̃ for one (sample, method).
        Progressively replaces timesteps in descending importance order (topdeletion) and ascending order (bottom deletion), 
        Tracks normalised score drop at each step. 
        Denominator is the drop when everything is replaced, matching eq. 5 in the paper.
    '''
    x = np.asarray(x, dtype=np.float32)
    T = len(x)
    importance = np.asarray(importance, dtype=np.float32)
    fill = float(x.mean())

    rank_top = np.argsort(importance)[::-1]
    rank_bottom = np.argsort(importance)

    # fully-masked score, used to normalise the drop curve
    x_all = np.full_like(x, fill)
    score_all = get_predicted_class_score(hydra_model, x_all[None, :], pred_label)
    denom = abs(score_before - score_all) + 1e-8

    fractions = np.linspace(0, 1, n_steps + 1)
    drops_top, drops_bottom = [], []

    for frac in fractions:
        n_mask = int(round(frac * T))

        x_top = x.copy()
        if n_mask > 0:
            x_top[rank_top[:n_mask]] = fill

        x_bot = x.copy()
        if n_mask > 0:
            x_bot[rank_bottom[:n_mask]] = fill

        s_top = get_predicted_class_score(hydra_model, x_top[None, :], pred_label)
        s_bot = get_predicted_class_score(hydra_model, x_bot[None, :], pred_label)

        drops_top.append(float(np.clip((score_before - s_top) / denom, -1, 1)))
        drops_bottom.append(float(np.clip((score_before - s_bot) / denom, -1, 1)))

    aucs_top = float(np.trapezoid(drops_top, fractions))
    aucs_bottom = float(np.trapezoid(drops_bottom, fractions))

    # Turbé F1S̃: harmonic mean of top-deletion AUC and (1 - bottom-deletion AUC)
    a, b = aucs_top, 1.0 - aucs_bottom
    f1s = (2 * a * b / (a + b)) if (a + b) > 0 else 0.0

    return {
        "fractions_removed": fractions.tolist(),
        "score_drops_top": drops_top,
        "score_drops_bottom": drops_bottom,
        "aucs_top": aucs_top,
        "aucs_bottom": aucs_bottom,
        "f1s": f1s,
    }




@dataclass
class CaptumComparison:
    ''' Runs HYDRA projection, tsCaptum (Shapley Sampling + Feature Ablation), TSHAP-Window and optionally MrSQM across a list of UCR datasets and compares them pairwise.
    '''

    datasets: list[str]
    output_dir: Path | str = Path("outputs/saliency/captum_comparison")
    fractions: tuple[float, ...] = (0.05, 0.10, 0.20)
    n_segments: int = 20# -1 = point-wise (slow, exact), 20 = segment-level (fast)
    max_samples_per_dataset: int | None = 20
    include_mrsqm: bool = True
    compute_deletion_curves: bool = True  # adds ~2x model calls
    deletion_n_steps: int = 20
    include_tshap: bool = True
    include_tshap_train_background: bool = False # Faithful to the paper's setup, but expensive
    tshap_train_background_samples: int = 20
    tshap_window_fraction: float = 0.10
    tshap_max_stride: int = 5
    force: bool = False
    seed: int = 42
    device: str | torch.device | None = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Loading my GPU
        if self.device is None:
            self.device = torch.device("cpu")
        elif not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)



    def hydra_importance(self, hydra_model: HydraModelExplainable, x: np.ndarray, pred_label: int):
        ''' Returns |HYDRA saliency| for one sample
        '''
        class_index = pred_to_hydra_class_index(hydra_model, pred_label)
        x_t = hydra_model._to_tensor(x[None, :])
        saliency = np.asarray(
            hydra_model.transform.get_saliency_map(
                x_t, hydra_model.classifier, hydra_model.scaler, class_index=class_index
            ),
            dtype=np.float32,
        )
        return np.abs(saliency)



    def explain_all_methods(self, hydra_model, adapter, tscaptum_explainers, tshap_conditions, mrsqm_model, x, pred_label):
        ''' Runs every active method on one sample and times each. 
        Returns
            method_name: {"importance": array, "time_s": float}}
        '''
        results = {}
        t0 = time.perf_counter()
        results["hydra"] = {
            "importance": self.hydra_importance(hydra_model, x, pred_label),
            "time_s": time.perf_counter() - t0,
        }

        for name, explainer in tscaptum_explainers.items():
            t0 = time.perf_counter()
            try:
                imp = explain_one_sample_tscaptum(explainer, x, pred_label, self.n_segments)
                results[name] = {"importance": imp, "time_s": time.perf_counter() - t0}
            except Exception as e:
                print(f"[WARN] {name} failed: {e}")
                results[name] = {"importance": np.zeros_like(x, dtype=np.float32), "time_s": time.perf_counter() - t0}

        for name, (tshap_explainer, baseline) in tshap_conditions.items():
            t0 = time.perf_counter()
            try:
                imp = explain_one_sample_tshap(tshap_explainer, adapter, x, pred_label, baseline)
                results[name] = {"importance": imp, "time_s": time.perf_counter() - t0}
            except Exception as e:
                print(f"[WARN] {name} failed: {e}")
                results[name] = {"importance": np.zeros_like(x, dtype=np.float32), "time_s": time.perf_counter() - t0}

        if mrsqm_model is not None:
            t0 = time.perf_counter()
            try:
                imp = np.abs(mrsqm_model.explain(x))
                results["mrsqm"] = {"importance": imp, "time_s": time.perf_counter() - t0}
            except Exception as e:
                print(f"[WARN] mrsqm explain failed: {e}")
                results["mrsqm"] = {"importance": np.zeros_like(x, dtype=np.float32), "time_s": time.perf_counter() - t0}

        return results




    def compare_sample(self, hydra_model, adapter, tscaptum_explainers, tshap_conditions, mrsqm_model, x, y_true):
        ''' Compares all method pairs for one correctly classified sample
        Returns:
            (pairwise_rows, deletion_rows, timing_rows)
        '''
        x = np.asarray(x, dtype=np.float32)
        pred_label = int(hydra_model.predict(x[None, :])[0])
        score_before = get_predicted_class_score(hydra_model, x[None, :], pred_label)

        # Get all explainer results for comparison
        method_results = self.explain_all_methods(hydra_model, adapter, tscaptum_explainers, tshap_conditions, mrsqm_model, x, pred_label)

        # score drop + flip rate per method per fraction, always scored on HYDRA
        score_drops: dict[str, dict[float, float]] = {}
        flip_results: dict[str, dict[float, int]] = {}

        for method_name, res in method_results.items():
            importance = res["importance"]
            score_drops[method_name] = {}
            flip_results[method_name] = {}

            for frac in self.fractions:
                mask = select_contiguous_window(importance, fraction=frac, mode="top")
                x_masked = apply_mask(x, mask)
                score_after = get_predicted_class_score(hydra_model, x_masked[None, :], pred_label)
                pred_after = int(hydra_model.predict(x_masked[None, :])[0])
                score_drops[method_name][frac] = float(score_before - score_after)
                flip_results[method_name][frac] = int(pred_after != pred_label)

        # region-agreement rows, every unique unordered method pair
        pairwise_rows = []
        method_names = list(method_results.keys())

        for i, ma in enumerate(method_names):
            for mb in method_names[i + 1:]:
                imp_a = method_results[ma]["importance"]
                imp_b = method_results[mb]["importance"]

                for frac in self.fractions:
                    mask_a = select_contiguous_window(imp_a, fraction=frac, mode="top")
                    mask_b = select_contiguous_window(imp_b, fraction=frac, mode="top")
                    overlap = compare_masks(mask_a, mask_b)

                    pairwise_rows.append({
                        "true_label": int(y_true),
                        "pred_label": pred_label,
                        "fraction": frac,
                        "method_a": ma,
                        "method_b": mb,
                        "iou": overlap["iou"],
                        "overlap_fraction": overlap["overlap_fraction"],
                        "normalised_centre_distance": overlap["normalised_centre_distance"],
                        "score_drop_a": score_drops[ma][frac],
                        "score_drop_b": score_drops[mb][frac],
                        "flipped_a": flip_results[ma][frac],
                        "flipped_b": flip_results[mb][frac],
                        "explain_time_a_s": method_results[ma]["time_s"],
                        "explain_time_b_s": method_results[mb]["time_s"],
                    })

        # NOTE - Store timing results
        timing_rows = [
            {
                "method": m,
                "true_label": int(y_true),
                "pred_label": pred_label,
                "explain_time_s": method_results[m]["time_s"],
            }
            for m in method_names
        ]

        # NOTE - Store deletion curve results
        deletion_rows = []
        if self.compute_deletion_curves:
            for method_name, res in method_results.items():
                curve = turbe_deletion_curve(
                    hydra_model=hydra_model,
                    x=x,
                    pred_label=pred_label,
                    importance=res["importance"],
                    score_before=score_before,
                    n_steps=self.deletion_n_steps,
                )
                deletion_rows.append({
                    "true_label": int(y_true),
                    "pred_label": pred_label,
                    "method": method_name,
                    "aucs_top": curve["aucs_top"],
                    "aucs_bottom": curve["aucs_bottom"],
                    "f1s": curve["f1s"],
                })
        return pairwise_rows, deletion_rows, timing_rows



    def run_dataset(self, dataset: str):
        ''' Main function for running one dataset
        '''
        pairwise_path = self.output_dir / f"captum_{dataset}_pairwise.csv"
        deletion_path = self.output_dir / f"captum_{dataset}_deletion.csv"
        timing_path = self.output_dir / f"captum_{dataset}_timing.csv"

        # Skipping previously computed results
        if not self.force and pairwise_path.exists() and deletion_path.exists():
            print(f"[SKIP] {dataset} already done.")
            try:
                deletion_df = pd.read_csv(deletion_path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                deletion_df = pd.DataFrame()
            return (
                pd.read_csv(pairwise_path),
                deletion_df,
                pd.read_csv(timing_path) if timing_path.exists() else pd.DataFrame(),
            )

        print(f"\nCaptumComparison: {dataset}")
        X_train, y_train, X_test, y_test, _ = load_dataset(dataset)

        # NOTE - Fit eHYDRA
        hydra_model = HydraModelExplainable(input_dim=X_train.shape[-1], seed=self.seed, device=self.device)
        hydra_model.fit(X_train, y_train)

        # NOTE - adapter + explainers built once per dataset, reused across samples
        adapter = HydraTsCaptumAdapter(hydra_model)
        tscaptum_explainers = {}
        for method_name in ("shapley_sampling", "feature_ablation"):
            try:
                tscaptum_explainers[method_name] = make_tscaptum_explainer(method_name, adapter)
            except Exception as e:
                print(f"  [WARN] Could not build {method_name}: {e}")

        # NOTE - TSHAP explainer + backgrounds are also dataset-level (window sizing depends only on series length, backgrounds only on X_train)
        tshap_conditions = {}
        if self.include_tshap:
            try:
                TSHAPExplainer = import_tshap_explainer()
                window_length, stride = tshap_window_stride(X_train.shape[-1], self.tshap_window_fraction, self.tshap_max_stride)
                tshap_explainer = TSHAPExplainer(
                    window_length=window_length,
                    stride=stride,
                    interpolation=True,
                    roi=False,  # ROI deferred to the synthetic ground-truth phase
                )

                # NOTE - Generate bacvkgrounds for dataset
                backgrounds = tshap_backgrounds(
                    X_train,
                    include_train=self.include_tshap_train_background,
                    n_train_background_samples=self.tshap_train_background_samples,
                    seed=self.seed,
                )
                tshap_conditions = {name: (tshap_explainer, baseline) for name, baseline in backgrounds.items()}
                print(f"  TSHAP: window_length={window_length}, stride={stride}, backgrounds={list(backgrounds.keys())}")
            except Exception as e:
                print(f"  [WARN] Could not build TSHAP explainer, skipping: {e}")

        mrsqm_model = None
        if self.include_mrsqm:
            try:
                # Fit MrSQM
                mrsqm_model = MrSQMExplainableModel()
                mrsqm_model.fit(X_train, y_train)
                print("  MrSQM fitted.")
            except Exception as e:
                print(f"  [WARN] MrSQM fit failed, skipping: {e}")

        preds = hydra_model.predict(X_test)
        correct_indices = np.where(preds == y_test)[0]

        if self.max_samples_per_dataset is not None:
            rng = np.random.default_rng(self.seed)
            if len(correct_indices) > self.max_samples_per_dataset:
                correct_indices = rng.choice(correct_indices, size=self.max_samples_per_dataset, replace=False)

        all_pairwise, all_deletion, all_timing = [], [], []

        for count, idx in enumerate(correct_indices, start=1):
            print(f"  Sample {count}/{len(correct_indices)} | test_idx={idx}")
            # NOTE - Comapre sample results between explainers
            pw, dc, tm = self.compare_sample(
                hydra_model=hydra_model,
                adapter=adapter,
                tscaptum_explainers=tscaptum_explainers,
                tshap_conditions=tshap_conditions,
                mrsqm_model=mrsqm_model,
                x=X_test[idx],
                y_true=y_test[idx],
            )

            for row in pw:
                row["dataset"] = dataset
                row["sample_idx"] = int(idx)
            for row in dc:
                row["dataset"] = dataset
                row["sample_idx"] = int(idx)
            for row in tm:
                row["dataset"] = dataset
                row["sample_idx"] = int(idx)

            all_pairwise.extend(pw)
            all_deletion.extend(dc)
            all_timing.extend(tm)

        df_pairwise = pd.DataFrame(all_pairwise)
        df_deletion = pd.DataFrame(all_deletion)
        df_timing = pd.DataFrame(all_timing)
        df_pairwise.to_csv(pairwise_path, index=False)
        df_deletion.to_csv(deletion_path, index=False)
        df_timing.to_csv(timing_path, index=False)

        # Clean up for memory 
        del hydra_model, adapter, tscaptum_explainers, tshap_conditions, mrsqm_model
        gc.collect()
        return df_pairwise, df_deletion, df_timing



    def run(self):
        '''  Main run function to iterate over N datasets depending on CLI args
        '''
        all_pairwise, all_deletion, all_timing = [], [], []
        failed_datasets = []

        for i, dataset in enumerate(self.datasets, start=1):
            print(f"\n[{i}/{len(self.datasets)}] {dataset}")
            try:
                pw, dc, tm = self.run_dataset(dataset)
                all_pairwise.append(pw)
                all_deletion.append(dc)
                all_timing.append(tm)
            except Exception as e:
                print(f"  [ERROR] {dataset} failed and will be skipped: {e}")
                failed_datasets.append({"dataset": dataset, "error": str(e)})
                continue

        if failed_datasets:
            pd.DataFrame(failed_datasets).to_csv(self.output_dir / "captum_failed_datasets.csv", index=False)
            print(f"\n[WARN] {len(failed_datasets)} datasets failed — see captum_failed_datasets.csv")

        # NOTE - Generate pair, deletion and timing results & save 
        pairwise = pd.concat(all_pairwise, ignore_index=True)
        deletion = pd.concat(all_deletion, ignore_index=True)
        timing = pd.concat(all_timing, ignore_index=True)
        pairwise.to_csv(self.output_dir / "captum_pairwise_samples.csv", index=False)
        deletion.to_csv(self.output_dir / "captum_deletion_samples.csv", index=False)
        timing.to_csv(self.output_dir / "captum_timing_samples.csv", index=False)

        # NOTE - Generate overlap, deletion and timing summaries & save 
        overlap_summary = self.overlap_summary(pairwise)
        deletion_summary = self.deletion_curve_summary(deletion)
        timing_summary = self.timing_summary(timing)
        paired_tests = self.paired_tests(pairwise)
        overlap_summary.to_csv(self.output_dir / "captum_overlap_summary.csv", index=False)
        deletion_summary.to_csv(self.output_dir / "captum_deletion_summary.csv", index=False)
        timing_summary.to_csv(self.output_dir / "captum_timing_summary.csv", index=False)
        paired_tests.to_csv(self.output_dir / "captum_paired_tests.csv", index=False)

        print("\nOverlap summary:")
        print(overlap_summary.to_string(index=False))
        print("\nDeletion curve summary (AUCS̃_top / F1S̃):")
        print(deletion_summary.to_string(index=False))
        print("\nTiming summary:")
        print(timing_summary.to_string(index=False))

        return {
            "pairwise": pairwise,
            "deletion": deletion,
            "timing": timing,
            "overlap_summary": overlap_summary,
            "deletion_summary": deletion_summary,
            "timing_summary": timing_summary,
            "paired_tests": paired_tests,
        }



    def overlap_summary(self, df: pd.DataFrame):
        ''' Mean pairwise overlap/perturbation metrics by (method_a, method_b, fraction)
        '''
        return df.groupby(["method_a", "method_b", "fraction"], as_index=False).agg(
            mean_iou=("iou", "mean"),
            median_iou=("iou", "median"),
            mean_overlap_fraction=("overlap_fraction", "mean"),
            median_overlap_fraction=("overlap_fraction", "median"),
            mean_centre_distance=("normalised_centre_distance", "mean"),
            mean_score_drop_a=("score_drop_a", "mean"),
            mean_score_drop_b=("score_drop_b", "mean"),
            flip_rate_a=("flipped_a", "mean"),
            flip_rate_b=("flipped_b", "mean"),
            n_samples=("sample_idx", "count"),
            n_datasets=("dataset", "nunique"),
        )

    def deletion_curve_summary(self, df: pd.DataFrame):
        ''' Mean AUCS̃_top / F1S̃ per method, averaged per dataset then across datasets
        '''
        if df.empty:
            return pd.DataFrame()

        dataset_level = df.groupby(["method", "dataset"], as_index=False).agg(
            mean_aucs_top=("aucs_top", "mean"),
            mean_aucs_bottom=("aucs_bottom", "mean"),
            mean_f1s=("f1s", "mean"),
        )
        return dataset_level.groupby("method", as_index=False).agg(
            mean_aucs_top=("mean_aucs_top", "mean"),
            std_aucs_top=("mean_aucs_top", "std"),
            mean_aucs_bottom=("mean_aucs_bottom", "mean"),
            mean_f1s=("mean_f1s", "mean"),
            std_f1s=("mean_f1s", "std"),
            n_datasets=("dataset", "nunique"),
        ).sort_values("mean_aucs_top", ascending=False)



    def timing_summary(self, df: pd.DataFrame):
        ''' Mean/median explanation time per method
        '''
        if df.empty:
            return pd.DataFrame()
        return df.groupby("method", as_index=False).agg(
            mean_time_s=("explain_time_s", "mean"),
            median_time_s=("explain_time_s", "median"),
            std_time_s=("explain_time_s", "std"),
            n_samples=("sample_idx", "count"),
        ).sort_values("median_time_s")


    
    def paired_tests(self, df: pd.DataFrame):
        ''' Wilcoxon paired tests: does method_b produce larger score drops than method_a?
            In my case "hydra vs shapley_sampling " = is the model-agnostic method statistically more faithful than the projection?
        '''
        rows = []
        for (ma, mb, frac), group in df.groupby(["method_a", "method_b", "fraction"]):
            for label, diff in [("score_drop", group["score_drop_b"] - group["score_drop_a"]), ("flip_rate", group["flipped_b"] - group["flipped_a"])]:
                nonzero = diff[diff != 0]
                p = wilcoxon(nonzero, alternative="greater").pvalue if len(nonzero) >= 10 else np.nan
                rows.append({
                    "method_a": ma,
                    "method_b": mb,
                    "fraction": frac,
                    "metric": label,
                    "mean_diff_b_minus_a": float(diff.mean()),
                    "median_diff_b_minus_a": float(diff.median()),
                    "b_greater_count": int((diff > 0).sum()),
                    "n_non_tied": int(len(nonzero)),
                    "wilcoxon_p_b_gt_a": float(p) if not np.isnan(p) else np.nan,
                })
        return pd.DataFrame(rows)



    def cluster_level_analysis(self, cluster_csv_path: str | Path) -> dict:
        ''' Attaches morphology cluster labels and produces cluster-stratified summaries
        '''
        pairwise_path = self.output_dir / "captum_pairwise_samples.csv"
        deletion_path = self.output_dir / "captum_deletion_samples.csv"
        if not pairwise_path.exists():
            raise FileNotFoundError(f"Run run() first. Missing: {pairwise_path}")

        pairwise = pd.read_csv(pairwise_path)
        deletion = pd.DataFrame()
        if deletion_path.exists():
            try:
                deletion = pd.read_csv(deletion_path)
            except pd.errors.EmptyDataError:
                print(f"[INFO] {deletion_path} has no data (deletion curves likely disabled) — skipping cluster deletion summary.")

        clusters = pd.read_csv(cluster_csv_path)

        cluster_cols = ["dataset", "cluster"]
        if "cluster_name" in clusters.columns:
            cluster_cols.append("cluster_name")

        pairwise = pairwise.merge(clusters[cluster_cols].drop_duplicates(), on="dataset", how="left")
        if "cluster_name" not in pairwise.columns:
            pairwise["cluster_name"] = pairwise["cluster"].map(CLUSTER_NAMES)

        cluster_overlap = pairwise.groupby(
            ["cluster", "cluster_name", "method_a", "method_b", "fraction"], as_index=False
        ).agg(
            mean_iou=("iou", "mean"),
            median_iou=("iou", "median"),
            mean_overlap_fraction=("overlap_fraction", "mean"),
            mean_score_drop_a=("score_drop_a", "mean"),
            mean_score_drop_b=("score_drop_b", "mean"),
            n_samples=("sample_idx", "count"),
            n_datasets=("dataset", "nunique"),
        )
        cluster_overlap.to_csv(self.output_dir / "captum_cluster_overlap_summary.csv", index=False)

        if not deletion.empty:
            deletion = deletion.merge(clusters[cluster_cols].drop_duplicates(), on="dataset", how="left")
            if "cluster_name" not in deletion.columns:
                deletion["cluster_name"] = deletion["cluster"].map(CLUSTER_NAMES)

            cluster_deletion = deletion.groupby(["cluster", "cluster_name", "method"], as_index=False).agg(
                mean_aucs_top=("aucs_top", "mean"),
                mean_f1s=("f1s", "mean"),
                n_datasets=("dataset", "nunique"),
            ).sort_values(["cluster", "mean_aucs_top"], ascending=[True, False])
            cluster_deletion.to_csv(self.output_dir / "captum_cluster_deletion_summary.csv", index=False)
        else:
            print("[INFO] Deletion curves not available — skipping cluster deletion summary.")
            cluster_deletion = pd.DataFrame()

        return {"cluster_overlap": cluster_overlap, "cluster_deletion": cluster_deletion}