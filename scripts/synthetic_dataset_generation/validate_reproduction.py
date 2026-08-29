'''
BIT-EXACT REGRESSION against reference/original_tshap_synthetic.py
'''
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reference"))

from generators import FrequencyBurstGenerator, FrequencyBurstConfig
from generators import LevelShiftGenerator, LevelShiftConfig
from generators import SpikeMultiClassGenerator, SpikeMultiClassConfig
from generators.frequency_burst import estimate_noise_robustness
from generators.level_shift import estimate_baseline_bias
import calibration

import original_tshap_synthetic as orig


def check_bitexact_reproduction(n_samples=30, seed=42, verbose=True):
    ''' Check 1: FrequencyBurstGenerator(legacy_rng=True) vs the original
        DoubleFreqTest class, same defaults, same seed
    '''
    original = orig.DoubleFreqTest(clf_threshold=60)
    X_orig, y_orig, attribs_orig = original.generate_classification_data_and_attribs(n_samples=n_samples, random_seed=seed)

    config = FrequencyBurstConfig(
        cluster_key="legacy_reproduction",
        cluster_name="legacy_reproduction",
        tslength=200, splength=40, sp_idx=[30, 130],
        min_f=10.0, max_f=50.0, f_base=0.0, wave="sine",
        noise_std=0.0, clf_threshold=60.0, legacy_rng=True,
    )
    ours = FrequencyBurstGenerator(config)
    X_ours, y_ours, attribs_ours = ours.generate_data_and_attribs(n_samples, seed=seed)

    checks = {
        "X_allclose": bool(np.allclose(X_orig, X_ours)),
        "y_array_equal": bool(np.array_equal(y_orig, y_ours)),
        "attribs_allclose": bool(np.allclose(attribs_orig, attribs_ours)),
        "X_max_abs_diff": float(np.max(np.abs(X_orig - X_ours))),
        "attribs_max_abs_diff": float(np.max(np.abs(attribs_orig - attribs_ours))),
    }

    # background sample check too
    bg_orig = original.generate_classification_background_sample()
    bg_ours = ours.generate_background_sample()
    checks["background_allclose"] = bool(np.allclose(bg_orig, bg_ours))

    # also check the hypothetical model itself agrees on a batch of samples
    model_orig = original.get_classification_model()
    model_ours = ours.get_classifier_model()
    checks["predict_proba_allclose"] = bool(np.allclose(model_orig.predict_proba(X_orig), model_ours.predict_proba(X_ours)))

    passed = all(v for k, v in checks.items() if isinstance(v, bool))
    if verbose:
        print("Test 1: bit exact reproduction vs reference / original_tshap_synthetic.py")
        for k, v in checks.items():
            print(f"{k}: {v}")
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")
        print()
    return passed, checks


def check_recovered_parameters(n_samples=200, seed=123, verbose=True):
    ''' Test 2: Does the hypothetical model recover the injected parameter on noiseless data with high
        fidelity for each calibrated cluster config

        Calculates the correlation & Mean Absolute Error between original and my implementation
    '''
    results = {}
    calib_dir = calibration.DEFAULT_CALIB_DIR

    for key in ["high_frequency", "short_rough"]:
        path = os.path.join(calib_dir, f"{key}.json")
        if not os.path.exists(path):
            if verbose:
                print(f"  [skip] {key}: {path} not found, run calibration.py first")
            continue
        with open(path) as f:
            params = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

        config = FrequencyBurstConfig(**params)
        gen = FrequencyBurstGenerator(config)

        noiseless_config = FrequencyBurstConfig(**{**params, "noise_std": 0.0})
        X, y, attribs = FrequencyBurstGenerator(noiseless_config).generate_data_and_attribs(n_samples, seed=seed)
        model = FrequencyBurstGenerator(noiseless_config).get_regression_model()
        _, y_reg, _ = FrequencyBurstGenerator(noiseless_config).generate_regression_data_and_attribs(n_samples, seed=seed)
        pred = model.predict(X)
        corr = float(np.corrcoef(pred, y_reg)[0, 1])
        mae = float(np.mean(np.abs(pred - y_reg)))
        results[key] = {"recovery_corr": corr, "recovery_mae": mae}

        if config.noise_std > 0:
            results[key]["noise_robustness"] = estimate_noise_robustness(config, n_samples=n_samples, seed=seed)

        if verbose:
            print(f"Test 2: parameter recovery: {key}")
            print(f"correlation(recovered, true): {corr:.4f}")
            print(f"MAE(recovered, true): {mae:.3f}")
            if "noise_robustness" in results[key]:
                nr = results[key]["noise_robustness"]
                print(f"noise robustness: clean_corr={nr['corr_clean_vs_true']:.4f} "
                      f"noisy_corr={nr['corr_noisy_vs_true']:.4f} "
                      f"clean_mae={nr['mae_clean_vs_true']:.3f} "
                      f"noisy_mae={nr['mae_noisy_vs_true']:.3f}")
            print()

    path = os.path.join(calib_dir, "smooth.json")
    if os.path.exists(path):
        with open(path) as f:
            params = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        config = LevelShiftConfig(**params)
        gen = LevelShiftGenerator(config)
        X, y, attribs = gen.generate_data_and_attribs(n_samples, seed=seed)
        model = gen.get_regression_model()
        _, y_reg, _ = gen.generate_regression_data_and_attribs(n_samples, seed=seed)
        pred = model.predict(X)
        corr = float(np.corrcoef(pred, y_reg)[0, 1])
        mae = float(np.mean(np.abs(pred - y_reg)))
        results["smooth"] = {"recovery_corr": corr, "recovery_mae": mae}
        results["smooth"]["baseline_bias"] = estimate_baseline_bias(config, n_samples=n_samples, seed=seed)
        if verbose:
            print("Test 2: parameter recovery for smooth cluster:")
            print(f"correlation(recovered, true): {corr:.4f}")
            print(f"MAE(recovered, true): {mae:.3f}")
            bb = results["smooth"]["baseline_bias"]
            print(f"baseline bias (no-injection estimator output): "
                  f"mean={bb['bias_mean']:.3f} std={bb['bias_std']:.3f} "
                  f"vs typical injected level scale={bb['typical_level_scale']:.2f}")
            print()

    path = os.path.join(calib_dir, "spiky.json")
    if os.path.exists(path):
        with open(path) as f:
            params = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        config = SpikeMultiClassConfig(**params)
        gen = SpikeMultiClassGenerator(config)
        X, y, attribs = gen.generate_data_and_attribs(n_samples, seed=seed)
        model = gen.get_classifier_model()
        pred = model.predict(X)
        acc = float(np.mean(pred == y))
        results["spiky"] = {"slot_classification_accuracy": acc}
        if verbose:
            print("Test 2: parameter recovery for the spiky cluster data")
            print(f"tier classification accuracy: {acc:.4f}")
            print()

    return results


if __name__ == "__main__":
    ok, _ = check_bitexact_reproduction()
    check_recovered_parameters()
    if not ok:
        sys.exit(1)
