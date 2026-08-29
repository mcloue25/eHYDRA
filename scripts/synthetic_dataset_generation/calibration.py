'''
Calculates per cluster generator parameters from ucr_dataset_clusters_k4_with_types.csv and writes them to
calibrated_params/<cluster_key>.json.
'''
import argparse
import json
import os
import pandas as pd


DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "ucr_dataset_clusters_k4_with_types.csv")
DEFAULT_CALIB_DIR = os.path.join(os.path.dirname(__file__), "calibrated_params")

CLUSTER_NAME_MAP = {
    "high_frequency": "High-frequency / high-curvature",
    "short_rough": "Short / moderately rough",
    "smooth": "Smooth / low-complexity",
    "spiky": "Spiky / multi-class",
}


def quantiles(df, cluster_name, col, qs=(0.25, 0.5, 0.75)):
    ''' Used to calculate quantiles for a given DF cluster column 
    '''
    sub = df[df.cluster_name == cluster_name]
    if len(sub) == 0:
        raise ValueError(f"No rows found for cluster_name={cluster_name!r}")
    return sub[col].quantile(list(qs)).to_dict()


def calibrate_high_frequency(df, tslength=250, splength=50):
    ''' Main fucntion for claibration of the high frequency cluster 
    '''
    q = quantiles(df, CLUSTER_NAME_MAP["high_frequency"], "spectral_centroid")
    return {
        "cluster_key": "high_frequency",
        "cluster_name": CLUSTER_NAME_MAP["high_frequency"],
        "tslength": tslength,
        "splength": splength,
        "min_f": round(q[0.25] * tslength, 1),
        "max_f": round(q[0.75] * tslength, 1),
        "f_base": 0.0,
        "wave": "sine",
        "noise_std": 0.15,
        "_source_spectral_centroidquantiles": q,
    }


def calibrate_short_rough(df, tslength=250, splength=50):
    ''' Main function for calibration the shoutr & rough cluster
    '''
    q = quantiles(df, CLUSTER_NAME_MAP["short_rough"], "spectral_centroid")
    mad = quantiles(df, CLUSTER_NAME_MAP["short_rough"], "mean_abs_diff")
    return {
        "cluster_key": "short_rough",
        "cluster_name": CLUSTER_NAME_MAP["short_rough"],
        "tslength": tslength,
        "splength": splength,
        "min_f": round(q[0.25] * tslength, 1),
        "max_f": round(q[0.75] * tslength, 1),
        "f_base": 0.0,
        "wave": "sine",
        "noise_std": round(mad[0.5] * 0.3, 3),
        "_source_spectral_centroidquantiles": q,
        "_source_mean_abs_diff_median": mad[0.5],
        "_window_to_series_ratio": round(2 * splength / tslength, 3),
    }


def calibrate_smooth(df, tslength=500, splength=100):
    ''' Main function for calibrating the smooth cluster
    '''
    q = quantiles(df, CLUSTER_NAME_MAP["smooth"], "spectral_centroid")
    mad_q = quantiles(df, CLUSTER_NAME_MAP["smooth"], "mean_abs_diff")
    local_wander = mad_q[0.5] * (splength ** 0.5)
    return {
        "cluster_key": "smooth",
        "cluster_name": CLUSTER_NAME_MAP["smooth"],
        "tslength": tslength,
        "splength": splength,
        "min_level": round(local_wander * 2.0, 3),
        "max_level": round(local_wander * 8.0, 3),
        "drift_freq": round(q[0.5] * tslength, 2),
        "drift_amplitude": 0.5,
        "_source_spectral_centroidquantiles": q,
        "_source_mean_abs_diff_median": mad_q[0.5],
        "_source_local_wander_estimate": local_wander,
    }


def calibrate_spiky(df, tslength=500):
    ''' Main function for calibrating the spiky / multiclass cluster
    '''
    nclasses_q = quantiles(df, CLUSTER_NAME_MAP["spiky"], "n_classes")
    mad_q = quantiles(df, CLUSTER_NAME_MAP["spiky"], "mean_abs_diff")
    mad2_q = quantiles(df, CLUSTER_NAME_MAP["spiky"], "mean_abs_second_diff")
    n_tiers = int(min(max(round(nclasses_q[0.5] ** 0.5), 3), 6))  # sqrt-compressed, same spirit as v1's cap
    burst_noise_std = round(max(mad2_q[0.5] * 3.0, 0.2), 3)
    drift_rate = round(max(mad_q[0.5] * 0.4, 0.01), 3)
    max_duration = 175
    min_duration = 10
    return {
        "cluster_key": "spiky",
        "cluster_name": CLUSTER_NAME_MAP["spiky"],
        "tslength": tslength,
        "onset": 100,
        "n_tiers": n_tiers,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "drift_rate": drift_rate,
        "burst_noise_std": burst_noise_std,
        "bg_noise_std": 0.15,
        "duration_readout_len": 15,
        "readout_margin": 10,
        "_source_n_classes_median": nclasses_q[0.5],
        "_source_mean_abs_diff_median": mad_q[0.5],
        "_source_mean_abs_second_diff_median": mad2_q[0.5],
    }


CALIBRATORS = {
    "high_frequency": calibrate_high_frequency,
    "short_rough": calibrate_short_rough,
    "smooth": calibrate_smooth,
    "spiky": calibrate_spiky,
}


def run_calibration(csv_path=DEFAULT_CSV, out_dir=DEFAULT_CALIB_DIR):
    ''' Main function for actually running the claibration for each cluster
    '''
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for key, fn in CALIBRATORS.items():
        params = fn(df)
        out_path = os.path.join(out_dir, f"{key}.json")
        with open(out_path, "w") as f:
            json.dump(params, f, indent=2)
        results[key] = params
        print(f"wrote {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="path to ucr_dataset_clusters_k4_with_types.csv")
    parser.add_argument("--out-dir", default=DEFAULT_CALIB_DIR, help="directory to write calibrated_params/*.json")
    args = parser.parse_args()
    run_calibration(args.csv, args.out_dir)
