'''
Standardising the format for generated synthetic datasets.
File layout for one generated dataset:
    <out_dir>/<cluster_key>_<split>.npz     X, y, attribs
    <out_dir>/<cluster_key>_<split>.json    metadata
'''
import json
import os
import numpy as np


def save_dataset(out_dir, cluster_key, split, X, y, attribs, metadata):
    '''split: 'train' | 'test' | 'all' | any other free-text tag
    '''
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{cluster_key}_{split}"
    npz_path = os.path.join(out_dir, f"{stem}.npz")
    json_path = os.path.join(out_dir, f"{stem}.json")

    np.savez_compressed(npz_path, X=X, y=y, attribs=attribs)
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return npz_path, json_path


def load_dataset(out_dir, cluster_key, split):
    ''' Loads a dataset
    '''
    stem = f"{cluster_key}_{split}"
    npz_path = os.path.join(out_dir, f"{stem}.npz")
    json_path = os.path.join(out_dir, f"{stem}.json")

    data = np.load(npz_path)
    with open(json_path) as f:
        metadata = json.load(f)

    return data["X"], data["y"], data["attribs"], metadata


def list_available(out_dir):
    '''Returns {(cluster_key, split): npz_path} for everything in out_dir
    '''
    found = {}
    if not os.path.isdir(out_dir):
        return found
    for fname in os.listdir(out_dir):
        if fname.endswith(".npz"):
            stem = fname[:-4]
            cluster_key, _, split = stem.rpartition("_")
            found[(cluster_key, split)] = os.path.join(out_dir, fname)
    return found
