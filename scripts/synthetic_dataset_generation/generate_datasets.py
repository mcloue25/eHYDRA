'''
Entry point for generating synthetic ground-truth datasets. 

Usage:
    # everything, defaults
    python generate_datasets.py --cluster all

    # just the high-frequency cluster, bigger sample, custom seed
    python generate_datasets.py --cluster high_frequency \\
        --n-samples 400 --seed 7 --out-dir outputs/saliency/synthetic_evaluation

Reads calibrated_params/<cluster_key>.json (run calibration.py first if missing) and writes <cluster_key>_train.npz/.json and 
<cluster_key>_test.npz/.json through dataset_io.save_dataset
'''
import argparse
import json
import os
import sys

import numpy as np

from generators import (
    FrequencyBurstGenerator, FrequencyBurstConfig,
    LevelShiftGenerator, LevelShiftConfig,
    SpikeMultiClassGenerator, SpikeMultiClassConfig,
)
import dataset_io
import calibration

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _REPO_ROOT)
from utils.globals_config import DEFAULT_DATA_DIR, CLUSTER_KEYS

# Same path as DEFAULT_DATA_DIR above -- kept under this name too since
# --out-dir in this script's CLI means "where generated datasets go", which
# happens to be DEFAULT_DATA_DIR, but the two names read differently to a
# caller of this script.
DEFAULT_OUT_DIR = DEFAULT_DATA_DIR
CALIB_DIR = calibration.DEFAULT_CALIB_DIR


def load_params(cluster_key):
    ''' Load calibrated paramters based on calibration.py
    '''
    path = os.path.join(CALIB_DIR, f"{cluster_key}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run calibration.py first "
            f"(python calibration.py) to derive calibrated_params/*.json "
            f"from the clustering CSV."
        )
    with open(path) as f:
        return json.load(f)


def strip_private_keys(params):
    ''' Drops the '_source_*' provenance keys before passing into a dataclass
    '''
    return {k: v for k, v in params.items() if not k.startswith("_")}


def build_generator(cluster_key):
    ''' Main function for building each generator that will be used to generate the synthetic dataset
    '''
    params = load_params(cluster_key)
    clean = strip_private_keys(params)

    if cluster_key == "high_frequency" or cluster_key == "short_rough":
        config = FrequencyBurstConfig(**clean)
        return FrequencyBurstGenerator(config), params
    
    elif cluster_key == "smooth":
        config = LevelShiftConfig(**clean)
        return LevelShiftGenerator(config), params
    
    elif cluster_key == "spiky":
        config = SpikeMultiClassConfig(**clean)
        return SpikeMultiClassGenerator(config), params
    else:
        raise ValueError(f"Unknown cluster_key={cluster_key!r}")


def generate_one_cluster(cluster_key, n_samples, seed, test_fraction, out_dir):
    ''' Create one clusters dataset based on passed params
    '''
    # NOTE - Load the generator 
    generator, calib_params = build_generator(cluster_key)

    # USe the calibration params to generate synthetic data
    X, y, attribs = generator.generate_data_and_attribs(n_samples, seed=seed)

    n_test = int(round(n_samples * test_fraction))
    rng = np.random.RandomState(seed + 1000)  # independent split RNG
    perm = rng.permutation(n_samples)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    # Create splits based on train / test passed
    splits = {
        "train": (X[train_idx], y[train_idx], attribs[train_idx]),
        "test": (X[test_idx], y[test_idx], attribs[test_idx]),
    }

    metadata_base = generator.metadata()
    metadata_base["calibration_source"] = calib_params
    metadata_base["n_samples_total"] = n_samples
    metadata_base["seed"] = seed
    metadata_base["test_fraction"] = test_fraction

    written = []
    for split_name, (Xs, ys, attribs_s) in splits.items():
        meta = dict(metadata_base)
        meta["split"] = split_name
        meta["n_samples_split"] = int(Xs.shape[0])
        # NOTE - Save the dataset 
        npz_path, json_path = dataset_io.save_dataset(out_dir, cluster_key, split_name, Xs, ys, attribs_s, meta)
        written.append(npz_path)
        print(f"  {split_name}: {Xs.shape[0]} samples -> {npz_path}")

    return written


def main():
    ''' Main function for generating the synthetic data
    '''
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster", choices=CLUSTER_KEYS + ["all"], required=True)
    parser.add_argument("--n-samples", type=int, default=300, help="total samples generated before train/test split (default: 300)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    # NOTE - Create datasets based on each clusters calculated features
    clusters = CLUSTER_KEYS if args.cluster == "all" else [args.cluster]
    for cluster_key in clusters:
        print(f"[{cluster_key}]")
        generate_one_cluster(cluster_key, args.n_samples, args.seed, args.test_fraction, args.out_dir)


if __name__ == "__main__":
    main()
