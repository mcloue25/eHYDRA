''' Control tests for eHYDRA saliency: seed stability, label permutation and weight permutation
'''
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import copy
import gc
import types

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeClassifierCV

from classes.models.hydra_explainable import HydraModelExplainable
from utils.data_utils import load_dataset
from utils.explainability import evaluate_masking_dataset
from utils.saliency_stability import pearson_correlation, spearman_correlation, top_window_iou


class PermutedWeightSaliencyModel:
    ''' Wraps a fitted model so predict/decision_function stay real but explain() uses feature permuted classifier weights
    '''
    def __init__(self, base_model: HydraModelExplainable, seed: int):
        self.base_model = base_model
        self.classifier = base_model.classifier  # exposed for code that inspects .classifier directly

        rng = np.random.default_rng(seed)
        coef = np.array(base_model.classifier.coef_, copy=True)

        # binary case: coef_ is 1D and ulticlass: permute along the feature axis
        if coef.ndim == 1:
            perm = rng.permutation(coef.shape[0])
            permuted_coef = coef[perm]
        else:
            perm = rng.permutation(coef.shape[1])
            permuted_coef = coef[:, perm]

        self._dummy_classifier = types.SimpleNamespace(coef_=permuted_coef)


    def predict(self, X):
        return self.base_model.predict(X)


    def decision_function(self, X):
        return self.base_model.decision_function(X)


    def explain(self, x_single, class_index=None, verbose=False):
        x_single = np.asarray(x_single, dtype=np.float32)
        x_batched = x_single[None, :]
        x_single_t = self.base_model._to_tensor(x_batched)

        if class_index is None:
            pred = self.base_model.predict(x_batched)[0]
            if len(self.base_model.classifier.classes_) > 2:
                class_index = int(np.where(self.base_model.classifier.classes_ == pred)[0][0])
            else:
                class_index = 0

        with torch.inference_mode():
            # saliency computed against the permuted weights, prediction untouched above
            saliency = self.base_model.transform.get_saliency_map(
                x_single_t,
                self._dummy_classifier,
                self.base_model.scaler,
                class_index=class_index,
            )

        saliency = np.asarray(saliency, dtype=np.float32)
        if saliency.ndim == 2 and saliency.shape[0] == 1:
            saliency = saliency[0]
        return saliency


def refit_classifier_on_labels(base_model: HydraModelExplainable, X_train, y_train_new, alphas=None):
    ''' Refits only the ridge classifier on new labels and reuses the already fitted HYDRA transform/scaler
    '''
    if alphas is None:
        alphas = np.logspace(-3, 3, 10)
    # transform doesn't depend on labels so it's reused as is
    z_train_np = base_model._transform_and_scale(X_train)

    classifier = RidgeClassifierCV(alphas=alphas)
    classifier.fit(z_train_np, y_train_new)
    shuffled_model = copy.copy(base_model)
    shuffled_model.classifier = classifier
    return shuffled_model


@dataclass
class SanityCheckEvaluator:
    ''' Runs seed-stability and label/weight-permutation sanity checks for HYDRA
    '''
    datasets: Sequence[str]
    output_dir: Path | str = Path("outputs/saliency/sanity_checks")
    n_seeds: int = 5
    fraction: float = 0.10
    max_samples_per_dataset: int | None = 20
    base_seed: int = 42
    device: str | None = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def seed_stability_dataset(self, dataset):
        '''Refits HYDRA under n_seeds different seeds and correlates saliency maps against the first seed.'''
        out_path = self.output_dir / f"seed_stability_{dataset}.csv"
        if out_path.exists():
            print(f"[SKIP] Loading existing seed-stability results for {dataset}")
            return pd.read_csv(out_path)


        print(f"\nSeed stability: {dataset}:")
        X_train, y_train, X_test, y_test, _ = load_dataset(dataset)
        seeds = [self.base_seed + i for i in range(self.n_seeds)]
        models = []
        for seed in seeds:
            model = HydraModelExplainable(input_dim=X_train.shape[-1], seed=seed, device=self.device)
            model.fit(X_train, y_train)
            models.append(model)

        # fix the sample set to what's correct under the reference (first-seed) model, for an apples-to-apples comparison
        ref_preds = models[0].predict(X_test)
        correct_indices = np.where(ref_preds == y_test)[0]
        if self.max_samples_per_dataset is not None:
            correct_indices = correct_indices[: self.max_samples_per_dataset]

        rows = []
        for count, idx in enumerate(correct_indices, start=1):
            x = X_test[idx]
            print(f"Sample {count}/{len(correct_indices)} | idx={idx}")
            saliencies = [np.asarray(m.explain(x), dtype=np.float32) for m in models]
            reference = saliencies[0]
            for seed, saliency in zip(seeds[1:], saliencies[1:]):
                rows.append({
                    "dataset": dataset,
                    "sample_idx": int(idx),
                    "reference_seed": seeds[0],
                    "compared_seed": seed,
                    "pearson": pearson_correlation(reference, saliency),
                    "spearman": spearman_correlation(reference, saliency),
                    "top_window_iou": top_window_iou(reference, saliency, fraction=self.fraction),
                })

        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        del models, X_train, y_train, X_test, y_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return df



    def run_seed_stability(self):
        ''' Main function for running the seed stability control test
        '''
        frames = []
        for i, dataset in enumerate(self.datasets, start=1):
            print(f"\n[{i}/{len(self.datasets)}] Seed stability on {dataset}")
            frames.append(self.seed_stability_dataset(dataset))

        samples = pd.concat(frames, ignore_index=True)
        samples_path = self.output_dir / "seed_stability_samples.csv"
        samples.to_csv(samples_path, index=False)

        summary = samples.groupby("dataset", as_index=False).agg(
            mean_pearson=("pearson", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_top_window_iou=("top_window_iou", "mean"),
            n_pairs=("sample_idx", "count"),
        )
        summary_path = self.output_dir / "seed_stability_summary.csv"
        summary.to_csv(summary_path, index=False)

        print(f"\nSaved samples: {samples_path}")
        print(f"Saved summary: {summary_path}")

        return {"samples": samples, "summary": summary}



    def perturbation_flip_gap(self, model, X_test, y_test):
        ''' Runs top/random/bottom masking and returns the (top-random, top-bottom) flip-rate gaps
        '''
        sample_df, _ = evaluate_masking_dataset(
            model=model,
            X_test=X_test,
            y_test=y_test,
            fractions=(self.fraction,),
            only_correct=True,
            random_repeats=3,
            seed=self.base_seed,
            max_samples=self.max_samples_per_dataset,
        )
        if sample_df.empty:
            return np.nan, np.nan, 0
        by_mode = sample_df.groupby("mode")["flipped"].mean()
        top = by_mode.get("top", np.nan)
        random_ = by_mode.get("random", np.nan)
        bottom = by_mode.get("bottom", np.nan)
        return float(top - random_), float(top - bottom), int(len(sample_df))



    def permutation_checks_dataset(self, dataset):
        ''' Compares the flip rate gap for the real model against shuffled label and permuted weight variants
        '''
        out_path = self.output_dir / f"permutation_checks_{dataset}.csv"
        if out_path.exists():
            print(f"[SKIP] Loading existing permutation-check results for {dataset}")
            return pd.read_csv(out_path)

        # NOTE - Train regular eHYDRA model
        print(f"\nLabel/weight permutation sanity checks: {dataset}:")
        X_train, y_train, X_test, y_test, _ = load_dataset(dataset)
        real_model = HydraModelExplainable(input_dim=X_train.shape[-1], seed=self.base_seed, device=self.device)
        real_model.fit(X_train, y_train)

        # Now refit the classifier on labels
        rng = np.random.default_rng(self.base_seed)
        y_train_shuffled = rng.permutation(np.asarray(y_train))
        shuffled_label_model = refit_classifier_on_labels(real_model, X_train, y_train_shuffled)

        # Now generate permuted weight model
        permuted_weight_model = PermutedWeightSaliencyModel(real_model, seed=self.base_seed)

        # Store results
        rows = []
        for variant, model in [("real", real_model), ("shuffled_labels", shuffled_label_model), ("permuted_weights", permuted_weight_model)]:
            top_minus_random, top_minus_bottom, n_rows = self.perturbation_flip_gap(model, X_test, y_test)
            rows.append({
                "dataset": dataset,
                "variant": variant,
                "flip_rate_top_minus_random": top_minus_random,
                "flip_rate_top_minus_bottom": top_minus_bottom,
                "n_rows": n_rows,
            })

        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)

        del real_model, shuffled_label_model, permuted_weight_model, X_train, y_train, X_test, y_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return df



    def run_permutation_checks(self):
        ''' 
        '''
        frames = []
        for i, dataset in enumerate(self.datasets, start=1):
            print(f"\n[{i}/{len(self.datasets)}] Permutation checks on {dataset}")
            frames.append(self.permutation_checks_dataset(dataset))

        samples = pd.concat(frames, ignore_index=True)
        samples_path = self.output_dir / "permutation_checks_samples.csv"
        samples.to_csv(samples_path, index=False)

        summary = samples.groupby("variant", as_index=False).agg(
            mean_flip_rate_top_minus_random=("flip_rate_top_minus_random", "mean"),
            mean_flip_rate_top_minus_bottom=("flip_rate_top_minus_bottom", "mean"),
            n_datasets=("dataset", "nunique"),
        )
        summary_path = self.output_dir / "permutation_checks_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved samples: {samples_path}")
        print(f"Saved summary: {summary_path}")
        return {"samples": samples, "summary": summary}



    def run(self):
        ''' Mauin function for running the seed tability and the permutation test
        '''
        seed_results = self.run_seed_stability()
        permutation_results = self.run_permutation_checks()
        return {
            "seed_stability": seed_results,
            "permutation_checks": permutation_results,
        }