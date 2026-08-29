'''
Testing the synthetic_dataset_generation package.
'''
import io
import os
import sys
import contextlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "reference"))

import calibration
import generate_datasets as gendata
import dataset_io
import validate_reproduction as vr

OUT_DIR = os.path.join(HERE, "smoke_test_outputs")
DATA_DIR = os.path.join(OUT_DIR, "generated_datasets")
SMOKE_N_SAMPLES = 40
SMOKE_SEED = 42


def section(title):
    bar = " " * 30
    return f"\n{bar}\n{title}\n{bar}\n"


def run_calibration_step(report):
    report.write(section("Testing: calibration.py"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        calibration.run_calibration()
    # Write results to report
    report.write(buf.getvalue())
    report.write("STATUS: OK\n")


def run_validation_step(report):
    ''' Tests the validation step
    '''
    report.write(section("Testing: validate_reproduction.py (Tier A)"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok, check1 = vr.check_bitexact_reproduction()
        check2 = vr.check_recovered_parameters()

    # Write results to report
    report.write(buf.getvalue())
    report.write(f"STATUS: {'OK' if ok else 'FAIL -- bit-exact reproduction broken, stop here'}\n")
    return ok, check1, check2


def run_generation_step(report):
    report.write(section(f"Testing: generate_datasets.py --cluster all " f"--n-samples {SMOKE_N_SAMPLES} --seed {SMOKE_SEED}"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for cluster_key in gendata.CLUSTER_KEYS:
            print(f"[{cluster_key}]")
            gendata.generate_one_cluster(cluster_key, SMOKE_N_SAMPLES, SMOKE_SEED, test_fraction=0.3, out_dir=DATA_DIR)
    # Write results to report
    report.write(buf.getvalue())
    report.write("STATUS: OK\n")


def run_roundtrip_check(report):
    ''' Test that the dataset_io is functioning how I want it to 
    '''
    report.write(section("Testing: dataset_io check"))
    available = dataset_io.list_available(DATA_DIR)
    all_ok = True
    for (cluster_key, split), path in sorted(available.items()):
        X, y, attribs, meta = dataset_io.load_dataset(DATA_DIR, cluster_key, split)
        n_expected = meta["n_samples_split"]
        # Passing conditions
        ok = (X.shape[0] == n_expected and y.shape[0] == n_expected and attribs.shape[0] == n_expected and not np.any(np.isnan(X)))
        all_ok = all_ok and ok
        report.write(
            f"  {cluster_key:16s} {split:6s}  X={X.shape}  y={y.shape}  "
            f"attribs={attribs.shape}  nan_free={not np.any(np.isnan(X))}  "
            f"{'OK' if ok else 'FAIL'}\n"
        )
    # Save results to report
    report.write(f"STATUS: {'OK' if all_ok else 'FAIL'}\n")
    return all_ok


def plot_frequency_or_level_cluster(ax_top, ax_bottom, X, attribs, title, xlabel_extra=""):
    t = np.arange(X.shape[-1])
    ax_top.plot(t, X[0, 0], color="tab:blue", linewidth=0.9)
    nz = np.nonzero(attribs[0, 0])[0]
    if len(nz) > 0:
        ax_top.axvspan(nz.min(), nz.max(), color="tab:red", alpha=0.15)
    ax_top.set_title(title)
    ax_top.set_ylabel("x(t)")

    ax_bottom.plot(t, attribs[0, 0], color="tab:red", linewidth=0.9)
    ax_bottom.axhline(0, color="black", linewidth=0.5)
    ax_bottom.set_ylabel(r"$\Phi$ (ground truth)")
    ax_bottom.set_xlabel(f"timestep{xlabel_extra}")


def plot_spiky_cluster(ax_top, ax_bottom, X, attribs, y, config, title):
    t = np.arange(X.shape[-1])
    ax_top.plot(t, X[0, 0], color="tab:blue", linewidth=0.9)
    window_end = config.onset + config.window_length
    ax_top.axvspan(config.onset, window_end, color="tab:red", alpha=0.15)
    ax_top.set_title(f"{title}  (true tier = {y[0]})")
    ax_top.set_ylabel("x(t)")

    ax_bottom.plot(t, attribs[0, 0], color="tab:red", linewidth=0.9)
    ax_bottom.axhline(0, color="black", linewidth=0.5)
    ax_bottom.set_ylabel(r"$\Phi$ (ground truth)")
    ax_bottom.set_xlabel("timestep")


def run_qualitative_plots(report):
    ''' 
    '''
    report.write(section("Testing: qualitative plots"))

    fig, axes = plt.subplots(2, 4, figsize=(20, 6), sharex=False)
    cluster_titles = {
        "high_frequency": "High-frequency / high-curvature",
        "short_rough": "Short / moderately rough",
        "smooth": "Smooth / low-complexity",
        "spiky": "Spiky / multi-class",
    }

    for col, cluster_key in enumerate(gendata.CLUSTER_KEYS):
        X, y, attribs, meta = dataset_io.load_dataset(DATA_DIR, cluster_key, "train")
        title = cluster_titles[cluster_key]

        if cluster_key == "spiky":
            generator, _ = gendata.build_generator(cluster_key)
            plot_spiky_cluster(axes[0, col], axes[1, col], X, attribs, y, generator.config, title)
        else:
            plot_frequency_or_level_cluster(axes[0, col], axes[1, col], X, attribs, title)

        report.write(f"  {cluster_key}: plotted example 0 of {X.shape[0]} train samples " f"(tslength={X.shape[-1]})\n")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "qualitative_all_clusters.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    report.write(f"  saved -> {out_path}\n")
    report.write("STATUS: OK (inspect the PNG -- this is the check nothing numeric replaces)\n")
    return out_path


def main():
    ''' Mainf function for running all tests
    ToDo:
        Add in qualitative plot testing
        wrap everything up together  
    '''
    os.makedirs(OUT_DIR, exist_ok=True)
    report = io.StringIO()
    report.write(section("TEST: synthetic_dataset_generation"))
    report.write(f"n_samples={SMOKE_N_SAMPLES}  seed={SMOKE_SEED}\n")

    run_calibration_step(report)
    ok, check1, check2 = run_validation_step(report)
    if not ok:
        report_path = os.path.join(OUT_DIR, "smoke_test_report.txt")
        with open(report_path, "w") as f:
            f.write(report.getvalue())
        print(report.getvalue())
        print(f"\nSTOPPED EARLY -- bit-exact reproduction failed. Report: {report_path}")
        sys.exit(1)

    run_generation_step(report)
    roundtrip_ok = run_roundtrip_check(report)
    plot_path = run_qualitative_plots(report)

    report.write(section("SUMMARY"))
    report.write(f"Tier A bit-exact reproduction: {'PASS' if ok else 'FAIL'}\n")
    report.write(f"dataset_io round-trip: {'PASS' if roundtrip_ok else 'FAIL'}\n")
    report.write(f"qualitative plot: {plot_path}\n")
    report.write("\nParameter recovery (noiseless, from Step 2):\n")
    for key, vals in check2.items():
        if "recovery_corr" in vals:
            report.write(f"{key:16s} recovery_corr={vals['recovery_corr']:.4f}\n")
        elif "slot_classification_accuracy" in vals:
            report.write(f"{key:16s} slot_accuracy={vals['slot_classification_accuracy']:.4f}\n")

    report_path = os.path.join(OUT_DIR, "smoke_test_report.txt")
    with open(report_path, "w") as f:
        f.write(report.getvalue())

    print(report.getvalue())
    print(f"Full report written to {report_path}")


if __name__ == "__main__":
    main()
