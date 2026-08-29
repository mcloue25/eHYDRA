import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference"))

from generators import FrequencyBurstGenerator, FrequencyBurstConfig
from generators.frequency_burst import estimate_noise_robustness


def default_config(**overrides):
    base = dict(cluster_key="test", cluster_name="test", tslength=200, splength=40, sp_idx=[30, 130], min_f=10.0, max_f=50.0, f_base=0.0, wave="sine", noise_std=0.0, clf_threshold=60.0)
    base.update(overrides)
    return FrequencyBurstConfig(**base)


def test_output_shapes():
    ''' Tests that generate_data_and_attribs returns the expected shapes and dtype
    '''
    gen = FrequencyBurstGenerator(default_config())
    X, y, attribs = gen.generate_data_and_attribs(n_samples=25, seed=1)
    assert X.shape == (25, 1, 200)
    assert y.shape == (25,)
    assert attribs.shape == (25, 1, 200)
    assert y.dtype == bool


def test_attribs_zero_outside_windows():
    ''' Tests that attribs are zero everywhere outside the two injected windows
    '''
    gen = FrequencyBurstGenerator(default_config())
    X, y, attribs = gen.generate_data_and_attribs(n_samples=10, seed=2)
    mask = np.ones(200, dtype=bool)
    mask[30:70] = False
    mask[130:170] = False
    assert np.all(attribs[:, 0, mask] == 0)


def test_attribs_nonzero_inside_windows():
    ''' Tests that every sample has a nonzero attribution somewhere inside each window
    '''
    gen = FrequencyBurstGenerator(default_config())
    X, y, attribs = gen.generate_data_and_attribs(n_samples=10, seed=3)
    assert np.any(attribs[:, 0, 30:70] != 0, axis=1).all()
    assert np.any(attribs[:, 0, 130:170] != 0, axis=1).all()


def test_background_sample_near_decision_boundary():
    ''' Tests that the background sample sits near, but not exactly at, the decision boundary
    '''
    gen = FrequencyBurstGenerator(default_config())
    bg = gen.generate_background_sample()
    assert bg.shape == (1, 1, 200)
    model = gen.get_classifier_model()
    proba = model.predict_proba(bg)
    # not exactly 0.5 due to the zero-crossing counter's discretisation (paper's own background gives 0.616)
    assert abs(proba[0, 0] - 0.5) < 0.2


def test_recovers_true_frequency_sum_noiseless():
    ''' Tests that the regression model recovers the true frequency sum with high correlation when noiseless
    '''
    gen = FrequencyBurstGenerator(default_config(noise_std=0.0))
    X, y, attribs = gen.generate_regression_data_and_attribs(n_samples=100, seed=4)
    model = gen.get_regression_model()
    pred = model.predict(X)
    corr = np.corrcoef(pred, y)[0, 1]
    assert corr > 0.95, f"expected high recovery correlation, got {corr}"


def test_square_wave_variant_runs():
    ''' Tests that the square-wave variant generates without shape errors or NaNs
    '''
    gen = FrequencyBurstGenerator(default_config(wave="square"))
    X, y, attribs = gen.generate_data_and_attribs(n_samples=5, seed=5)
    assert X.shape == (5, 1, 200)
    assert not np.any(np.isnan(X))


def test_nonzero_f_base_corrupts_recovery():
    ''' Tests that a nonzero f_base measurably degrades frequency-recovery correlation
    '''
    clean = FrequencyBurstGenerator(default_config(f_base=0.0))
    corrupted = FrequencyBurstGenerator(default_config(f_base=20.0))

    X_clean, y_clean, _ = clean.generate_regression_data_and_attribs(100, seed=6)
    X_corrupt, y_corrupt, _ = corrupted.generate_regression_data_and_attribs(100, seed=6)

    corr_clean = np.corrcoef(clean.get_regression_model().predict(X_clean), y_clean)[0, 1]
    corr_corrupt = np.corrcoef(corrupted.get_regression_model().predict(X_corrupt), y_corrupt)[0, 1]

    assert corr_clean > 0.95
    assert corr_corrupt < corr_clean - 0.1, (
        "expected nonzero f_base to measurably degrade recovery -- if this "
        "fails, generate_feature's superposition behaviour may have changed"
    )


def test_noise_robustness_helper_runs():
    ''' Tests that estimate_noise_robustness runs and returns the expected keys
    '''
    config = default_config(noise_std=0.5)
    result = estimate_noise_robustness(config, n_samples=50, seed=7)
    assert "corr_clean_vs_true" in result
    assert "corr_noisy_vs_true" in result
    assert result["corr_clean_vs_true"] > result["corr_noisy_vs_true"] - 1e-9


if __name__ == "__main__":
    import inspect
    tests = [f for name, f in globals().items() if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tests passed")