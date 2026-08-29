'''
Compare eHYDRA vs TSHAP (Window and ROI) against synthetic ground truth.

Usage:
    python3 scripts/analysis/evaluate_tshap_comparison.py
    python3 scripts/analysis/evaluate_tshap_comparison.py --cluster high_frequency --n-eval 30
    python3 scripts/analysis/evaluate_tshap_comparison.py --background threshold
'''
import argparse
import json
import os
import sys
import pickle

import numpy as np
import pandas as pd

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SYNTH_DIR = os.path.join(REPO_ROOT, "scripts", "synthetic_dataset_generation")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, SYNTH_DIR)
sys.path.insert(0, REPO_ROOT)

import dataset_io
import generate_datasets  # for build_generator() -- reuses the exact calibrated generator
from saliency_ground_truth_metrics import evaluate_one_sample
from utils.data_utils import adapt_hydra_input_shape
from classes.captum_comparison import (
    HydraTsCaptumAdapter,
    import_tshap_explainer,
    tshap_window_stride,
)
from utils.globals_config import (
    DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, GROUND_TRUTH_EVAL_DIR, CLUSTER_KEYS, TOPK_FRACTIONS,
)


def ehydra_signed_saliency(model, x):
    ''' Generate signed eHYDRA saliency
    '''
    return model.explain(x, class_index=None, verbose=False)


def tshap_signed_explain(tshap_explainer, adapter, x, pred_label, baseline):
    '''Generate signed TSHAP-Window and TSHAP-ROI saliency for one sample
    '''
    x_3d = x[np.newaxis, np.newaxis, :].astype(np.float32)
    window_exp, roi_exp = tshap_explainer.explain(
        x_3d,
        baselines=baseline,
        model=adapter,
        clf_targets=np.array([pred_label]),
    )
    window_signed = np.asarray(window_exp[0, 0, :], dtype=np.float32)
    roi_signed = np.asarray(roi_exp[0, 0, :], dtype=np.float32) if roi_exp is not None else None
    return window_signed, roi_signed


def build_backgrounds(cluster_key, X_train, background_choice):
    ''' Returns {background_name: (1, 1, T) array}
    '''
    T = X_train.shape[-1]
    backgrounds = {}

    if background_choice in ("threshold", "both"):
        generator, _ = generate_datasets.build_generator(cluster_key)
        backgrounds["threshold"] = generator.generate_background_sample().astype(np.float32)

    if background_choice in ("centroid", "both"):
        backgrounds["centroid"] = X_train.mean(axis=0).reshape(1, 1, T).astype(np.float32)

    return backgrounds



def evaluate_one_cluster(cluster_key, data_dir, model_dir, n_eval, background_choice, window_fraction, max_stride, out_dir, save_arrays=False, tshap_target_mode="auto"):
    ''' Main function for comparing the results between eHYDRA and TSHAP within one morphology cluster 
    '''
    # Load model
    model_path = os.path.join(model_dir, f"{cluster_key}_hydra_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load dataset & fix input shapes
    X_train, y_train, _, _ = dataset_io.load_dataset(data_dir, cluster_key, "train")
    X_train_2d = adapt_hydra_input_shape(X_train)
    X_test, y_test, attribs_test, _ = dataset_io.load_dataset(data_dir, cluster_key, "test")
    X_test_2d = adapt_hydra_input_shape(X_test)
    Phi_test = attribs_test[:, 0, :]
    T = X_test_2d.shape[-1]

    n_eval = min(n_eval, X_test_2d.shape[0]) if n_eval else X_test_2d.shape[0]

    # Build backgrounds
    backgrounds = build_backgrounds(cluster_key, X_train_2d, background_choice)
    print(f"  backgrounds: {list(backgrounds.keys())}")

    TSHAPExplainer = import_tshap_explainer()
    window_length, stride = tshap_window_stride(X_train_2d.shape[-1], window_fraction, max_stride)
    tshap_explainer = TSHAPExplainer(window_length=window_length, stride=stride, interpolation=True, roi=True)
    print(f"TSHAP: window_length={window_length}, stride={stride}, roi=True")

    adapter = HydraTsCaptumAdapter(model)

    n_classes = len(model.classifier.classes_)
    if tshap_target_mode == "auto":
        use_fixed = (n_classes == 2)
    else:
        use_fixed = (tshap_target_mode == "fixed")

    if use_fixed:
        if n_classes != 2:
            raise ValueError(f"--tshap-target-mode fixed requested but {cluster_key} has " f"{n_classes} classes -- 'fixed' only has a defined meaning for binary")
        fixed_target = int(model.classifier.classes_[1])
        print(f"TSHAP target: FIXED = classes_[1] = {fixed_target} (matches eHYDRA's fixed direction)")
    else:
        fixed_target = None
        print(f"TSHAP target: per-sample PREDICTED class ({n_classes}-class)")

    arrays = None
    if save_arrays:
        arrays = {"phi_ehydra": np.zeros((n_eval, T), dtype=np.float32)}
        for bg_name in backgrounds:
            arrays[f"phi_tshap_window_{bg_name}"] = np.zeros((n_eval, T), dtype=np.float32)
            arrays[f"phi_tshap_roi_{bg_name}"] = np.zeros((n_eval, T), dtype=np.float32)

    rows = []
    for i in range(n_eval):
        x = X_test_2d[i]
        Phi = Phi_test[i]
        pred_label = fixed_target if fixed_target is not None else int(model.predict(x[None, :])[0])

        # Get ehydra signed saliency   
        phi_ehydra = ehydra_signed_saliency(model, x)
        if save_arrays:
            arrays["phi_ehydra"][i] = phi_ehydra
        m = evaluate_one_sample(phi_ehydra, Phi, topk_fractions=TOPK_FRACTIONS)
        m.update({"sample": i, "true_label": int(y_test[i]), "method": "ehydra", "background": "n/a"})
        rows.append(m)

        for bg_name, baseline in backgrounds.items():
            try:
                phi_window, phi_roi = tshap_signed_explain(tshap_explainer, adapter, x, pred_label, baseline)
            except Exception as e:
                print(f"[WARN] TSHAP failed on sample {i}, background={bg_name}: {e}")
                continue

            if save_arrays:
                arrays[f"phi_tshap_window_{bg_name}"][i] = phi_window
                if phi_roi is not None:
                    arrays[f"phi_tshap_roi_{bg_name}"][i] = phi_roi

            m_window = evaluate_one_sample(phi_window, Phi, topk_fractions=TOPK_FRACTIONS)
            m_window.update({"sample": i, "true_label": int(y_test[i]), "method": "tshap_window", "background": bg_name})
            rows.append(m_window)

            if phi_roi is not None:
                m_roi = evaluate_one_sample(phi_roi, Phi, topk_fractions=TOPK_FRACTIONS)
                m_roi.update({"sample": i, "true_label": int(y_test[i]), "method": "tshap_roi", "background": bg_name})
                rows.append(m_roi)

        if (i + 1) % 10 == 0:
            print(f"... {i + 1}/{n_eval}")

    df = pd.DataFrame(rows)
    raw_path = os.path.join(out_dir, f"{cluster_key}_tshap_comparison.csv")
    df.to_csv(raw_path, index=False)
    print(f"RAW PATH : {raw_path}")

    if save_arrays:
        arrays_path = os.path.join(out_dir, f"{cluster_key}_tshap_arrays.npz")
        np.savez_compressed(arrays_path, X=X_test_2d[:n_eval], Phi=Phi_test[:n_eval], y=y_test[:n_eval], **arrays)
        print(f"RAW ARRAYS : {arrays_path}")

    return df


def summarise(df):
    numeric_cols = [c for c in df.columns if c not in ("sample", "true_label", "method", "background")
                    and pd.api.types.is_numeric_dtype(df[c])]
    return df.groupby(["method", "background"])[numeric_cols].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster", choices=CLUSTER_KEYS + ["all"], default="all")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", default=GROUND_TRUTH_EVAL_DIR)
    parser.add_argument("--n-eval", type=int, default=30, help="TSHAP is more expensive per sample than eHYDRA, default caps at 30 test samples/cluster")
    parser.add_argument("--background", choices=["threshold", "centroid", "both"], default="threshold")

    parser.add_argument("--window-fraction", type=float, default=0.10, help="TSHAP window_length as a fraction of series length (paper uses 0.10)")
    parser.add_argument("--max-stride", type=int, default=5)

    parser.add_argument("--save-arrays", action="store_true",
                         help="Save raw signed phi curves (eHYDRA + TSHAP-Window + TSHAP-ROI, per background) to <cluster>_tshap_arrays.npz, "
                              "needed by scripts/plotting/plot_ground_truth_comparison.py")
    parser.add_argument("--tshap-target-mode", choices=["auto", "fixed", "predicted"], default="auto",
                         help="'auto' (default, recommended): fixed classes_[1] target for binary "
                              "clusters (matches eHYDRA's fixed sign convention), per-sample "
                              "predicted-class target for spiky. 'fixed'/'predicted' force that "
                              "choice for ALL clusters -- 'predicted' reproduces the original "
                              "(buggy, for binary) behaviour, kept only for an A/B check.")
    args = parser.parse_args()

    clusters = CLUSTER_KEYS if args.cluster == "all" else [args.cluster]
    os.makedirs(args.out_dir, exist_ok=True)

    all_summaries = {}
    for cluster_key in clusters:
        print(f"[{cluster_key}]")
        # Pass CLI args to cluster evaluation 
        df = evaluate_one_cluster(cluster_key, args.data_dir, args.model_dir, args.n_eval,
                                   args.background, args.window_fraction, args.max_stride,
                                   args.out_dir, args.save_arrays, args.tshap_target_mode)
        summary = summarise(df)
        print(summary[["cosine_similarity", "confmat_precision", "confmat_f1", "topk10_precision", "topk10_iou"]].round(4).to_string())
        print()
        all_summaries[cluster_key] = summary.reset_index().to_dict(orient="records")

    combined_path = os.path.join(args.out_dir, "tshap_comparison_summary.json")
    existing = {}
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            existing = json.load(f)
    existing.update(all_summaries)
    with open(combined_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"combined : {combined_path}")


if __name__ == "__main__":
    main()