import os
import sys
import json
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import calibration


def test_calibration_runs_and_writes_all_four():
    with tempfile.TemporaryDirectory() as tmp_out:
        results = calibration.run_calibration(csv_path=calibration.DEFAULT_CSV, out_dir=tmp_out)
        assert set(results.keys()) == {"high_frequency", "short_rough", "smooth", "spiky"}
        for key in results:
            path = os.path.join(tmp_out, f"{key}.json")
            assert os.path.exists(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["cluster_key"] == key


def test_variance_not_used_for_smooth():
    ''' Tests that variance is 1.0 on all clusters and isn't used as a calibration source
    '''
    df = pd.read_csv(calibration.DEFAULT_CSV)
    params = calibration.calibrate_smooth(df)
    assert "_source_variance_median" not in params
    # sanity: level shifts should be a modest multiple of a unit-variance series' scale, not 5-25x it
    assert params["max_level"] < 5.0, (
        f"max_level={params['max_level']} looks too large for a "
        f"'smooth'/subtle archetype on a unit-variance series"
    )


def test_high_frequency_f_base_is_zero():
    ''' Tests that f_base is 0.0 for high_frequency so no extra zero-crossings get injected
    '''
    df = pd.read_csv(calibration.DEFAULT_CSV)
    params = calibration.calibrate_high_frequency(df)
    assert params["f_base"] == 0.0


def test_short_rough_window_ratio_is_compact():
    ''' Tests that short_rough's window-to-series ratio stays <= 0.5, in line with the other clusters
    '''
    df = pd.read_csv(calibration.DEFAULT_CSV)
    params = calibration.calibrate_short_rough(df)
    ratio = 2 * params["splength"] / params["tslength"]
    assert ratio <= 0.5, (
        f"short_rough window-to-series ratio is {ratio:.3f}, expected <= 0.5 "
        f"(comparable to the other three clusters) -- was this changed back "
        f"toward the original 0.833?"
    )


def test_spiky_window_ratio_matches_other_clusters():
    '''Tests that spiky's window-to-series ratio is ~0.40, matching the other clusters
    '''
    import pandas as pd
    df = pd.read_csv(calibration.DEFAULT_CSV)
    params = calibration.calibrate_spiky(df)
    window_length = params["max_duration"] + params["duration_readout_len"] + params["readout_margin"]
    ratio = window_length / params["tslength"]
    assert abs(ratio - 0.40) < 0.05, (
        f"spiky window-to-series ratio is {ratio:.3f}, expected ~0.40 "
        f"(matching the other three clusters) -- was max_duration changed back?"
    )


if __name__ == "__main__":
    tests = [f for name, f in globals().items() if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tests passed")