'''
HYDRA-trainability check on the syntheitc data Ive created

Usage:
    python3 scripts/pipeline/train_hydra_on_synthetic.py
    python3 scripts/pipeline/train_hydra_on_synthetic.py --cluster spiky
    python3 scripts/pipeline/train_hydra_on_synthetic.py --min-accuracy 0.90 --save-models
'''
import argparse
import json
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import f1_score

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SYNTH_DIR = os.path.join(REPO_ROOT, "scripts", "synthetic_dataset_generation")
sys.path.insert(0, SYNTH_DIR)
sys.path.insert(0, REPO_ROOT)

import dataset_io  # from scripts/synthetic_dataset_generation
from classes.models.hydra_explainable import HydraModelExplainable
from utils.data_utils import adapt_hydra_input_shape
from utils.globals_config import DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, CLUSTER_KEYS

# spiky gets a lower bar: 6-way localisation from one sparse impulse is harder
# for a fixed random-kernel transform than the binary tasks, with only ~35
# train samples per class (~210 total split across 6) rather than 2
DEFAULT_MIN_ACCURACY = {
    "high_frequency": 0.90,
    "short_rough": 0.90,
    "smooth": 0.90,
    "spiky": 0.80,
}


def train_and_gate_one_cluster(cluster_key, data_dir, min_accuracy, save_models, model_out_dir):
    '''
    '''
    X_train, y_train, _, meta_train = dataset_io.load_dataset(data_dir, cluster_key, "train")
    X_test, y_test, _, meta_test = dataset_io.load_dataset(data_dir, cluster_key, "test")

    X_train = adapt_hydra_input_shape(X_train)
    X_test = adapt_hydra_input_shape(X_test)

    model = HydraModelExplainable(input_dim=X_train.shape[-1])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = float(np.mean(preds == y_test))
    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    n_classes = len(np.unique(y_train))

    gate = min_accuracy[cluster_key]
    passed = accuracy >= gate

    result = {
        "cluster_key": cluster_key,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "tslength": int(X_train.shape[-1]),
        "n_classes": int(n_classes),
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "min_accuracy_gate": gate,
        "passed": passed,
    }

    print(f"[{cluster_key}] n_train={result['n_train']} n_test={result['n_test']}" f"tslength={result['tslength']} n_classes={n_classes}")
    print(f"Test accuracy: {accuracy:.4f}  (gate: >= {gate})")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(f"RESULT: {'PASS' if passed else 'FAIL -- do not trust saliency-vs-Phi on this cluster'}")

    if save_models:
        os.makedirs(model_out_dir, exist_ok=True)
        model_path = os.path.join(model_out_dir, f"{cluster_key}_hydra_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        result["saved_model_path"] = model_path
        print(f"  saved trained model -> {model_path}")

    print()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster", choices=CLUSTER_KEYS + ["all"], default="all")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--min-accuracy", type=float, default=None, help="override the per-cluster default gate for ALL clusters (otherwise uses the per-cluster defaults in DEFAULT_MIN_ACCURACY)")
    parser.add_argument("--save-models", action="store_true", help=f"pickle each trained eHYDRA model to {DEFAULT_MODEL_DIR}/<cluster>_hydra_model.pkl for reuse in the saliency-vs-Phi comparison")
    parser.add_argument("--model-out-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-out", default=os.path.join(DEFAULT_DATA_DIR, "hydra_trainability_report.json"))
    args = parser.parse_args()

    gates = dict(DEFAULT_MIN_ACCURACY)
    if args.min_accuracy is not None:
        gates = {k: args.min_accuracy for k in CLUSTER_KEYS}

    clusters = CLUSTER_KEYS if args.cluster == "all" else [args.cluster]
    results = []
    for cluster_key in clusters:
        results.append(train_and_gate_one_cluster(
            cluster_key, args.data_dir, gates, args.save_models, args.model_out_dir
        ))

    all_passed = all(r["passed"] for r in results)

    print(f"\n\n\nSUMMARY: {sum(r['passed'] for r in results)}/{len(results)} clusters passed")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['cluster_key']:16s} acc={r['test_accuracy']:.4f}  f1={r['test_macro_f1']:.4f}  {status}")

    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"report written to: {args.report_out}")

    if not all_passed:
        print("Not all clusters passed the ACC threshold. Don't proceed to the saliency-vs-Phi comparison for the failing cluster(s) until this is resolved")
        sys.exit(1)


if __name__ == "__main__":
    main()