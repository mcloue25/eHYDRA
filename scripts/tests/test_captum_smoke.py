'''
Smoke test for classes/captum_comparison.py using tsCaptum.

Usage:
    python scripts/tests/test_captum_smoke.py
'''

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classes.models.hydra_explainable import HydraModelExplainable
from classes.captum_comparison import (
    HydraTsCaptumAdapter,
    make_tscaptum_explainer,
    explain_one_sample_tscaptum,
    turbe_deletion_curve,
    CaptumComparison,
)
from classes.windowshap import get_predicted_class_score



def make_data(n_train=60, n_test=20, T=80, n_classes=2, seed=0):
    ''' Generate random data for testing 
    '''
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, T)).astype(np.float32)
    y_train = rng.integers(0, n_classes, size=n_train)
    X_test = rng.standard_normal((n_test, T)).astype(np.float32)
    y_test = rng.integers(0, n_classes, size=n_test)
    return X_train, y_train, X_test, y_test


def fitted_hydra(T=80):
    ''' Fit an eHYDRA model
    '''
    X_train, y_train, X_test, y_test = make_data(T=T)
    model = HydraModelExplainable(input_dim=T, device=torch.device("cpu"))
    model.fit(X_train, y_train)
    return model, X_test, y_test



def test_adapter_predict_proba():
    '''  Test 1: adapter predict_proba shape
    '''
    print("test_adapter_predict_proba")
    T = 80
    hydra_model, X_test, _ = fitted_hydra(T)

    adapter = HydraTsCaptumAdapter(hydra_model)

    # tsCaptum passes (N, 1, T) — verify our squeeze handles it
    X_3d = X_test[:5, np.newaxis, :].astype(np.float32)     # (5, 1, T)
    proba = adapter.predict_proba(X_3d)
    assert proba.shape == (5, 2), f"Expected (5, 2), got {proba.shape}"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), "Probabilities don't sum to 1"
    print(f"  predict_proba shape: {proba.shape}  row sums: {proba.sum(axis=1)[:3]}")
    print("  PASSED")



def test_tscaptum_explainers():
    ''' Test 2 & 3: tsCaptum Feature Ablation and Shapley Sampling
    '''
    print("test_tscaptum_explainers")
    T = 80
    hydra_model, X_test, y_test = fitted_hydra(T)
    adapter = HydraTsCaptumAdapter(hydra_model)

    x = X_test[0]
    pred_label = int(hydra_model.predict(x[None, :])[0])

    for method_name in ("feature_ablation", "shapley_sampling"):
        explainer = make_tscaptum_explainer(method_name, adapter)
        imp = explain_one_sample_tscaptum(explainer, x, pred_label, n_segments=10)
        assert imp.shape == (T,), f"{method_name}: expected ({T},), got {imp.shape}"
        assert np.all(imp >= 0), f"{method_name}: importance has negative values"
        assert imp.max() > 0, f"{method_name}: all-zero importance (suspicious)"
        print(f"  {method_name}: shape={imp.shape}  max={imp.max():.4f}  min={imp.min():.4f}")

    print("  PASSED")


def test_deletion_curve():
    ''' Test 4: deletion curve
    '''
    print("test_deletion_curve:")
    T = 80
    hydra_model, X_test, y_test = fitted_hydra(T)

    x = X_test[0]
    pred_label = int(hydra_model.predict(x[None, :])[0])
    score_before = get_predicted_class_score(hydra_model, x[None, :], pred_label)

    rng = np.random.default_rng(0)
    importance = rng.standard_normal(T).astype(np.float32)
    importance = np.abs(importance)

    curve = turbe_deletion_curve(
        hydra_model=hydra_model,
        x=x,
        pred_label=pred_label,
        importance=importance,
        score_before=score_before,
        n_steps=10,
    )

    assert "aucs_top" in curve and "f1s" in curve
    assert len(curve["score_drops_top"]) == 11   # n_steps + 1
    assert len(curve["score_drops_bottom"]) == 11
    assert 0.0 <= curve["f1s"] <= 1.0 or curve["f1s"] < 0, "f1s out of expected range"
    print(f"aucs_top={curve['aucs_top']:.4f}  f1s={curve['f1s']:.4f}")
    print("PASSED")



def test_compare_sample():
    ''' Test 5: _compare_sample row counts and column keys
    '''
    print("test_compare_sample")
    T = 80
    hydra_model, X_test, y_test = fitted_hydra(T)
    adapter = HydraTsCaptumAdapter(hydra_model)

    tscaptum_explainers = {}
    for name in ("feature_ablation", "shapley_sampling"):
        tscaptum_explainers[name] = make_tscaptum_explainer(name, adapter)

    comparison = CaptumComparison(
        datasets=[],              # not running full pipeline here
        n_segments=10,
        include_mrsqm=False,
        compute_deletion_curves=True,
        deletion_n_steps=5,
        device="cpu",
    )

    x = X_test[0]
    y_true = int(y_test[0])

    pairwise_rows, deletion_rows, timing_rows = comparison._compare_sample(
        hydra_model=hydra_model,
        adapter=adapter,
        tscaptum_explainers=tscaptum_explainers,
        mrsqm_model=None,
        x=x,
        y_true=y_true,
    )

    # 3 methods → 3 pairs; 3 fractions → 9 pairwise rows
    n_methods = 3    # hydra + shapley_sampling + feature_ablation
    n_pairs = n_methods * (n_methods - 1) // 2
    n_fractions = len(comparison.fractions)
    expected_pairwise = n_pairs * n_fractions

    assert len(pairwise_rows) == expected_pairwise, (f"Expected {expected_pairwise} pairwise rows, got {len(pairwise_rows)}")
    assert len(deletion_rows) == n_methods, (f"Expected {n_methods} deletion rows, got {len(deletion_rows)}")
    assert len(timing_rows) == n_methods

    for row in pairwise_rows:
        for key in ("iou", "overlap_fraction", "score_drop_a", "score_drop_b", "flipped_a", "flipped_b", "explain_time_a_s", "explain_time_b_s"):
            assert key in row, f"Missing key '{key}' in pairwise row"

    for row in deletion_rows:
        for key in ("method", "aucs_top", "aucs_bottom", "f1s"):
            assert key in row, f"Missing key '{key}' in deletion row"

    # Verify all method names appear in deletion rows
    deletion_methods = {r["method"] for r in deletion_rows}
    assert "hydra" in deletion_methods
    assert "shapley_sampling" in deletion_methods
    assert "feature_ablation" in deletion_methods

    print(f"pairwise rows: {len(pairwise_rows)} (expected {expected_pairwise})")
    print(f"deletion rows: {len(deletion_rows)}")
    print(f"timing rows: {len(timing_rows)}")
    print(f"deletion methods: {sorted(deletion_methods)}")
    print("PASSED")




def testing_main():
    ''' Run all of the tests to make sure everything functioning ok
    '''
    test_adapter_predict_proba()
    test_tscaptum_explainers()
    test_deletion_curve()
    test_compare_sample()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    testing_main()