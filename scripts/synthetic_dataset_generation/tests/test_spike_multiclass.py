import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generators import SpikeMultiClassGenerator, SpikeMultiClassConfig


def default_config(**overrides):
    base = dict(cluster_key="test", cluster_name="test", tslength=500, onset=100,
                n_tiers=3, min_duration=10, max_duration=80, drift_rate=0.03,
                burst_noise_std=0.2, bg_noise_std=0.15, duration_readout_len=15,
                readout_margin=10)
    base.update(overrides)
    return SpikeMultiClassConfig(**base)


def test_output_shapes():
    '''Tests that generate_data_and_attribs returns the expected shapes and label range.'''
    gen = SpikeMultiClassGenerator(default_config())
    X, y, attribs = gen.generate_data_and_attribs(n_samples=25, seed=1)
    assert X.shape == (25, 1, 500)
    assert y.shape == (25,)
    assert y.min() >= 0
    assert y.max() < 3
    assert attribs.shape == (25, 1, 500)


def test_attribs_zero_outside_window():
    '''Tests that attribs are zero everywhere outside the single injected window.'''
    gen = SpikeMultiClassGenerator(default_config())
    config = gen.config
    X, y, attribs = gen.generate_data_and_attribs(n_samples=20, seed=2)
    window_end = config.onset + config.window_length
    mask = np.ones(500, dtype=bool)
    mask[config.onset:window_end] = False
    assert np.all(attribs[:, 0, mask] == 0)


def test_attribs_uniform_nonzero_inside_window():
    '''Tests that attribs are uniform and nonzero across the whole window for every sample.'''
    gen = SpikeMultiClassGenerator(default_config())
    config = gen.config
    X, y, attribs = gen.generate_data_and_attribs(n_samples=10, seed=3)
    window_end = config.onset + config.window_length
    for i in range(10):
        window_vals = attribs[i, 0, config.onset:window_end]
        assert np.all(window_vals == window_vals[0])
        assert window_vals[0] != 0


def test_tier_bounds_are_contiguous_and_nonoverlapping():
    '''Tests that tier_bounds forms a contiguous, non-overlapping partition of [min_duration, max_duration].'''
    config = default_config(n_tiers=4)
    bounds = config.tier_bounds
    assert len(bounds) == 4
    assert bounds[0][0] == config.min_duration
    assert bounds[-1][1] == config.max_duration
    for i in range(len(bounds) - 1):
        assert bounds[i][1] == bounds[i + 1][0]


def test_classifier_recovers_correct_tier_noiseless():
    '''Tests that the classifier recovers the correct tier with high accuracy at low noise.'''
    gen = SpikeMultiClassGenerator(default_config(burst_noise_std=0.05, bg_noise_std=0.02))
    X, y, attribs = gen.generate_data_and_attribs(n_samples=200, seed=4)
    model = gen.get_classifier_model()
    preds = model.predict(X)
    acc = np.mean(preds == y)
    assert acc > 0.9, f"expected high tier-recovery accuracy at low noise, got {acc}"


def test_config_raises_on_window_too_small_for_max_duration():
    '''Tests that a window_length too small for max_duration + readout raises ValueError.'''
    try:
        default_config(window_length=50, max_duration=80, duration_readout_len=15)
        raised = False
    except ValueError:
        raised = True
    assert raised, "expected ValueError when window_length can't fit max_duration + readout"


def test_config_raises_on_onset_too_close_to_start():
    '''Tests that an onset too close to t=0 for the baseline region raises ValueError.'''
    try:
        default_config(onset=5, duration_readout_len=15)
        raised = False
    except ValueError:
        raised = True
    assert raised, "expected ValueError when onset leaves no room for the baseline region"


def test_background_sample_is_mid_tier_not_uniform():
    '''Tests that the background sample is confidently mid-tier, not near-uniform.'''
    gen = SpikeMultiClassGenerator(default_config(n_tiers=3))
    bg = gen.generate_background_sample()
    model = gen.get_classifier_model()
    proba = model.predict_proba(bg)
    assert proba[0, 1] > 0.5, "expected the background sample to be confidently mid-tier"


if __name__ == "__main__":
    tests = [f for name, f in globals().items() if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tests passed")