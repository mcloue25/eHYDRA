import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generators import LevelShiftGenerator, LevelShiftConfig
from generators.level_shift import window_level, estimate_baseline_bias


def default_config(**overrides):
    base = dict(cluster_key="test", cluster_name="test", tslength=500, splength=100, sp_idx=[100, 300], min_level=1.0, max_level=4.0, drift_freq=4.0, drift_amplitude=0.5)
    base.update(overrides)
    return LevelShiftConfig(**base)


def test_output_shapes():
    ''' Tests that generate_data_and_attribs returns the expected shapes
    '''
    gen = LevelShiftGenerator(default_config())
    X, y, attribs = gen.generate_data_and_attribs(n_samples=25, seed=1)
    assert X.shape == (25, 1, 500)
    assert y.shape == (25,)
    assert attribs.shape == (25, 1, 500)


def test_window_level_returns_1d():
    ''' Tests that window_level returns shape (n,), not (n,1)
    '''
    x = np.random.RandomState(0).randn(5, 1, 500)
    out = window_level(x, 100, 100, 100)
    assert out.shape == (5,), f"expected shape (5,), got {out.shape}"


def test_attribs_zero_outside_windows():
    ''' Tests that attribs are zero everywhere outside the two injected windows
    '''
    gen = LevelShiftGenerator(default_config())
    X, y, attribs = gen.generate_data_and_attribs(n_samples=10, seed=2)
    mask = np.ones(500, dtype=bool)
    mask[100:200] = False
    mask[300:400] = False
    assert np.all(attribs[:, 0, mask] == 0)


def test_recovers_true_level_sum():
    ''' Tests that the regression model recovers the true level sum with high correlation
    '''
    gen = LevelShiftGenerator(default_config())
    X, y, attribs = gen.generate_regression_data_and_attribs(n_samples=150, seed=3)
    model = gen.get_regression_model()
    pred = model.predict(X)
    corr = np.corrcoef(pred, y)[0, 1]
    assert corr > 0.9, f"expected high recovery correlation, got {corr}"


def test_background_sample_near_decision_boundary():
    ''' Tests that the background sample sits near the decision boundary
    '''
    gen = LevelShiftGenerator(default_config())
    bg = gen.generate_background_sample()
    assert bg.shape == (1, 1, 500)
    model = gen.get_classifier_model()
    proba = model.predict_proba(bg)
    assert abs(proba[0, 0] - 0.5) < 0.1


def test_baseline_bias_small_relative_to_signal():
    ''' Tests that drift-only baseline bias is small relative to the typical injected level scale
    '''
    config = default_config()
    result = estimate_baseline_bias(config, n_samples=100, seed=4)
    typical_scale = (config.min_level + config.max_level) / 2
    assert result["bias_std"] < 0.2 * typical_scale, (
        f"baseline drift is leaking into the level estimator: "
        f"bias_std={result['bias_std']:.3f} vs typical_scale={typical_scale:.3f}"
    )


if __name__ == "__main__":
    tests = [f for name, f in globals().items() if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tests passed")