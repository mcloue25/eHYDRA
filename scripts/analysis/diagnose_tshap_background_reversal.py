'''
Diagnostic: isolates the cause of the TSHAP background-ordering reversal in thesis Section 5.7.4 (centroid beats threshold on 3/4 synthetic 
generators, reversing Le Nguyen & Ifrim's own DoubleFreqTest ablation).

Runs the same threshold-vs-centroid TSHAP-Window comparison under three conditions:
    A: paper-exact params
    B: A + the production noise_std, 
    C: The full production config (calibrated_params/high_frequency.json) to isolate whether noise, recalibration or our own TSHAP wiring drives the
    reversal.

Usage:
    python3 scripts/analysis/diagnose_tshap_background_reversal.py [--n-eval N]
'''


import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SYNTH_DIR = os.path.join(REPO_ROOT, "scripts", "synthetic_dataset_generation")
sys.path.insert(0, SYNTH_DIR)
sys.path.insert(0, REPO_ROOT)

from generators.frequency_burst import FrequencyBurstGenerator, FrequencyBurstConfig
from saliency_ground_truth_metrics import evaluate_one_sample
from utils.data_utils import adapt_hydra_input_shape
from classes.models.hydra_explainable import HydraModelExplainable
from classes.captum_comparison import (
    HydraTsCaptumAdapter,
    import_tshap_explainer,
    tshap_window_stride,
)

import generate_datasets


def build_paper_exact_config(noise_std: float, tag: str) -> FrequencyBurstConfig:
    ''' Reconstructs the ORIGINAL DoubleFreqTest exactly, varying only noise_std.
        legacy_rng stays False since this cares about data distribution and background construction, 
        bit-exact RNG was done in validate_reproduction.py
    '''
    return FrequencyBurstConfig(
        cluster_key=tag,
        cluster_name=tag,
        tslength=200,
        splength=40,
        sp_idx=[30, 130],
        min_f=10.0,
        max_f=50.0,
        f_base=0.0,
        wave="sine",
        noise_std=noise_std,
        clf_threshold=60.0,
        legacy_rng=False,
    )


def build_calibrated_config() -> FrequencyBurstConfig:
    ''' The actual production config from calibrated_params/high_frequency.json exactly what generated the data behind Table results_tshap_full
    '''
    generator, params = generate_datasets.build_generator("high_frequency")
    return generator.config


def run_one_condition(condition_name, config, n_samples, test_fraction, n_eval, window_fraction, max_stride, seed=42):
    ''' Generates one condition's synthetic data, trains HYDRA, runs TSHAP under both backgrounds, and scores against ground-truth attribution
    '''
    print(f"\n{'=' * 70}")
    print(f"CONDITION: {condition_name}")
    print(f"tslength={config.tslength}  splength={config.splength}  "
          f"sp_idx={config.sp_idx}  min_f={config.min_f}  max_f={config.max_f}  "
          f"noise_std={config.noise_std}  clf_threshold={config.clf_threshold}")
    print(f"{'=' * 70}")

    # NOTE - Init the FrequencyBurstGenerator
    generator = FrequencyBurstGenerator(config)
    X, y, attribs = generator.generate_data_and_attribs(n_samples, seed=seed)

    n_test = int(round(n_samples * test_fraction))
    rng = np.random.RandomState(seed + 1000)
    perm = rng.permutation(n_samples)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test, Phi_test = X[test_idx], y[test_idx], attribs[test_idx][:, 0, :]

    X_train_2d = adapt_hydra_input_shape(X_train)
    X_test_2d = adapt_hydra_input_shape(X_test)

    # Fit eHYDRA instance
    model = HydraModelExplainable(input_dim=X_train_2d.shape[-1])
    model.fit(X_train_2d, y_train)
    preds = model.predict(X_test_2d)
    accuracy = float(np.mean(preds == y_test))
    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    print(f"  HYDRA test accuracy: {accuracy:.4f}  macro F1: {macro_f1:.4f}")
    if accuracy < 0.85:
        print(f"  [WARN] accuracy below 0.85 -- interpret this condition's "
              f"TSHAP comparison with caution, the model may not have "
              f"learned the injected signal reliably under these params.")

    T = X_train_2d.shape[-1]
    backgrounds = {
        "threshold": generator.generate_background_sample().astype(np.float32),  # generator's own background
        "centroid": X_train_2d.mean(axis=0).reshape(1, 1, T).astype(np.float32),  # train mean
    }

    # NOTE - Import the TSHAP explainer & init inst
    TSHAPExplainer = import_tshap_explainer()
    window_length, stride = tshap_window_stride(T, window_fraction, max_stride)
    tshap_explainer = TSHAPExplainer(
        window_length=window_length, stride=stride,
        interpolation=True, roi=False,  # window only (This diagnostic is about background choice, not window vs ROI)
    )
    print(f"TSHAP: window_length={window_length}, stride={stride}")

    adapter = HydraTsCaptumAdapter(model)
    n_classes = len(model.classifier.classes_)
    assert n_classes == 2, "diagnostic assumes binary FrequencyBurstGenerator output"
    fixed_target = int(model.classifier.classes_[1])

    n_eval = min(n_eval, X_test_2d.shape[0])
    rows = []
    for i in range(n_eval):
        x = X_test_2d[i]
        Phi = Phi_test[i]
        for bg_name, baseline in backgrounds.items():
            x_3d = x[np.newaxis, np.newaxis, :].astype(np.float32)
            try:
                window_exp, _ = tshap_explainer.explain(x_3d, baselines=baseline, model=adapter, clf_targets=np.array([fixed_target]))
            except Exception as e:
                print(f"  [WARN] TSHAP failed sample={i} background={bg_name}: {e}")
                continue
            phi = np.asarray(window_exp[0, 0, :], dtype=np.float32)
            m = evaluate_one_sample(phi, Phi, topk_fractions=(0.05, 0.10, 0.20))
            m.update({"condition": condition_name, "sample": i, "background": bg_name})
            rows.append(m)
        if (i + 1) % 10 == 0:
            print(f"... {i + 1}/{n_eval}")

    df = pd.DataFrame(rows)
    df["hydra_test_accuracy"] = accuracy
    return df


def main():
    ''' Main function for looking intot he change in background conditions 
    '''
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=300, help="total samples generated before train/test split, " "matches generate_datasets.py default")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--n-eval", type=int, default=30, help="test samples explained per condition per background " "(matches evaluate_tshap_comparison.py default)")
    parser.add_argument("--window-fraction", type=float, default=0.10)
    parser.add_argument("--max-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", default=os.path.join(REPO_ROOT, "data", "synthetic_dataset", "ground_truth_evaluation", "tshap_background_reversal_diagnostic.csv"))
    args = parser.parse_args()

    conditions = {
        "A_paper_exact": build_paper_exact_config(noise_std=0.0, tag="diag_A"),
        "B_paper_plus_noise": build_paper_exact_config(noise_std=0.15, tag="diag_B"),
        "C_calibrated": build_calibrated_config(),
    }

    all_dfs = []
    for name, config in conditions.items():
        df = run_one_condition(name, config, args.n_samples, args.test_fraction, args.n_eval, args.window_fraction, args.max_stride, args.seed)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    combined.to_csv(args.out_csv, index=False)
    print(f"\nRaw per-sample results -> {args.out_csv}")

    # mean metric per condition x background, plus the threshold-minus-centroid gap
    summary = (
        combined.groupby(["condition", "background"])
        [["cosine_similarity", "confmat_f1", "topk10_precision", "hydra_test_accuracy"]]
        .mean()
        .round(4)
    )
    print("\n" + "=" * 70)
    print("SUMMARY: mean metric by condition x background")
    print("=" * 70)
    print(summary.to_string())

    for cond in conditions:
        try:
            thr = summary.loc[(cond, "threshold")]
            cen = summary.loc[(cond, "centroid")]
        except KeyError:
            print(f"  {cond}: incomplete data, skipping")
            continue
        gap_cos = thr["cosine_similarity"] - cen["cosine_similarity"]
        gap_f1 = thr["confmat_f1"] - cen["confmat_f1"]
        gap_topk = thr["topk10_precision"] - cen["topk10_precision"]
        print(f"  {cond:22s}  cosine: {gap_cos:+.4f}   confmat_f1: {gap_f1:+.4f}   "
              f"topk10_precision: {gap_topk:+.4f}")

if __name__ == "__main__":
    main()