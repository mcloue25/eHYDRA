from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d

from classes.models.hydra_explainable import HydraModelExplainable
from classes.models.mrsqm_explainable import MrSQMExplainableModel
from classes.windowshap import WindowSHAPExplainer
from utils.data_utils import load_dataset


def clean_saliency(s):
    ''' Convert saliency to a finite absolute-valued 1D array
    '''
    s = np.asarray(s, dtype=float).squeeze()
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    return np.abs(s)


def normalize_saliency_0_100(s):
    ''' Normalise saliency values onto a 0-100 scale for plotting
    '''
    s = clean_saliency(s)
    s_min = np.min(s)
    s_max = np.max(s)

    if np.isclose(s_max, s_min):
        return np.zeros_like(s)

    return 100.0 * (s - s_min) / (s_max - s_min)


def interpolate_signal_and_saliency(ts, saliency, interp_points=5000):
    ''' Interpolate the signal and saliency onto a dense grid for smoother plots
    '''
    ts = np.asarray(ts, dtype=float).squeeze()
    saliency = np.asarray(saliency, dtype=float).squeeze()

    x_orig = np.linspace(0, len(ts) - 1, num=len(ts))
    x_dense = np.linspace(0, len(ts) - 1, num=interp_points)

    ts_interp = interp1d(x_orig, ts, kind="linear")
    sal_interp = interp1d(x_orig, normalize_saliency_0_100(saliency), kind="linear")
    return x_dense, ts_interp(x_dense), sal_interp(x_dense)


def smooth_curve(y, window=101):
    ''' Apply a simple moving average smoother to saliency profiles
    '''
    y = np.asarray(y, dtype=float)
    window = max(1, int(window))

    if window <= 1:
        return y

    # Use an odd window so the smoothing kernel is centred.
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")




@dataclass
class QualitativeSaliencyPlotter:
    ''' Generate qualitative saliency comparison plots for one dataset/sample
    '''
    output_dir: Path | str = Path("outputs/saliency/imgs")
    n_segments: int = 100
    shap_nsamples: int = 500
    interp_points: int = 5000
    smooth_window: int = 101
    seed: int = 42
    device: str | torch.device | None = None


    def __post_init__(self):
        ''' Prepare output directory, device, and WindowSHAP explainer
        '''
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use GPU when available unless a device is explicitly provided.
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)

        self.windowshap = WindowSHAPExplainer(n_segments=self.n_segments, nsamples=self.shap_nsamples)



    def choose_shared_correct_example(self, hydra_model, mrsqm_model, X_test, y_test):
        ''' Select a test sample that both HYDRA and MrSQM classify correctly
        '''
        hydra_preds = hydra_model.predict(X_test)
        mrsqm_preds = mrsqm_model.predict(X_test)
        idxs = np.where((hydra_preds == y_test) & (mrsqm_preds == y_test))[0]
        # Fall back to the first test sample if no shared correct example exists.
        return int(idxs[0]) if len(idxs) > 0 else 0



    def plot_hydra_mrsqm_windowshap(self, dataset="GunPoint"):
        ''' Fit models, compute saliency maps and save the comparison figure
        '''
        print(f"\nPlotting {dataset} saliency comparison")
        X_train, y_train, X_test, y_test, _ = load_dataset(dataset)

        hydra = HydraModelExplainable(input_dim=X_train.shape[-1], seed=self.seed, device=self.device,)
        mrsqm = MrSQMExplainableModel(nsax=5, nsfa=1)
        print("fitting HYDRA...")
        hydra.fit(X_train, y_train)
        print("fitting MrSQM...")
        mrsqm.fit(X_train, y_train)

        # Use the same example for all saliency methods.
        idx = self.choose_shared_correct_example(hydra, mrsqm, X_test, y_test)
        ts = np.asarray(X_test[idx], dtype=np.float32)
        y_true = y_test[idx]

        hydra_pred = int(hydra.predict(ts[None, :])[0])
        mrsqm_pred = int(mrsqm.predict(ts[None, :])[0])

        print(f"selected sample index: {idx}")
        print(f"true label: {y_true}, HYDRA pred: {hydra_pred}, MrSQM pred: {mrsqm_pred}")

        # NOTE  - Compute eHYDRA saliency
        print("computing HYDRA saliency...")
        hydra_sal = hydra.explain(ts)

        # NOTE - Compute MrSQM saliency
        print("computing MrSQM saliency...")
        mrsqm_sal = mrsqm.explain(ts)

        print("  computing WindowSHAP saliency...")
        shap_sal = self.windowshap.explain(model=hydra, x=ts, pred_label=hydra_pred)

        # Interpolate all saliency maps onto the same dense time grid.
        x_dense, y_dense, hydra_dense = interpolate_signal_and_saliency(ts, hydra_sal, interp_points=self.interp_points)
        _, _, mrsqm_dense = interpolate_signal_and_saliency(ts, mrsqm_sal, interp_points=self.interp_points)
        _, _, shap_dense = interpolate_signal_and_saliency(ts, shap_sal, interp_points=self.interp_points)

        # Smooth the line-only comparison panel.
        hydra_smooth = smooth_curve(hydra_dense, window=self.smooth_window)
        mrsqm_smooth = smooth_curve(mrsqm_dense, window=self.smooth_window)
        shap_smooth = smooth_curve(shap_dense, window=self.smooth_window)

        # Three saliency-over-signal panels plus one profile comparison panel.
        fig = plt.figure(figsize=(9.5, 10.5), constrained_layout=True)
        grid = fig.add_gridspec(4, 2, width_ratios=[30, 1], height_ratios=[1, 1, 1, 1.35])

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(grid[2, 0], sharex=ax1)
        ax4 = fig.add_subplot(grid[3, 0], sharex=ax1)
        cax = fig.add_subplot(grid[0:3, 1])

        # Colour each point in the signal by its saliency value.
        ax1.scatter(x_dense, y_dense, c=hydra_dense, cmap="jet", marker=".", s=1.5, vmin=0, vmax=100)
        ax1.set_title(f"{dataset} — HYDRA")
        ax1.set_ylabel("Signal")
        ax1.text(0.01, 0.95, f"True class: {y_true}", transform=ax1.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", alpha=0.15))

        ax2.scatter(x_dense, y_dense, c=mrsqm_dense, cmap="jet", marker=".", s=1.5, vmin=0, vmax=100)
        ax2.set_title(f"{dataset} — MrSQM")
        ax2.set_ylabel("Signal")

        sc3 = ax3.scatter(x_dense, y_dense, c=shap_dense, cmap="jet", marker=".", s=1.5, vmin=0, vmax=100)
        ax3.set_title(f"{dataset} — WindowSHAP")
        ax3.set_ylabel("Signal")

        colorbar = fig.colorbar(sc3, cax=cax)
        colorbar.set_label("Normalised saliency")

        # Line plots to make the relative saliency profiles easier to compare.
        ax4.plot(x_dense, hydra_smooth, label="HYDRA", linewidth=1.8)
        ax4.plot(x_dense, mrsqm_smooth, label="MrSQM", linewidth=1.8, linestyle="--")
        ax4.plot(x_dense, shap_smooth, label="WindowSHAP", linewidth=1.8, linestyle=":")
        ax4.set_title(f"{dataset} — saliency profile comparison")
        ax4.set_xlabel("Time step")
        ax4.set_ylabel("Saliency")
        ax4.set_ylim(0, 100)
        ax4.legend(frameon=False, ncol=1, loc="upper center", bbox_to_anchor=(1.0, 1.0))

        for ax in [ax1, ax2, ax3, ax4]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(f"{dataset}: HYDRA, MrSQM and WindowSHAP saliency comparison", fontsize=13)
        # File naming issue between versions fix later
        save_path = self.output_dir / f"{dataset}_hydra_mrsqm_windowshap.png"
        report_path = self.output_dir / f"{dataset}_saliency_comparison_{self.n_segments}_{self.shap_nsamples}.png"
        legacy_report_name = "Gunpoint" if dataset == "GunPoint" else dataset
        legacy_report_path = self.output_dir / f"{legacy_report_name}_saliency_comparison_{self.n_segments}_{self.shap_nsamples}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        fig.savefig(report_path, dpi=300, bbox_inches="tight")
        if legacy_report_path != report_path:
            fig.savefig(legacy_report_path, dpi=300, bbox_inches="tight")

        plt.close(fig)
        print(f"Saved: {save_path}")
        print(f"Saved: {report_path}")
        if legacy_report_path != report_path:
            print(f"Saved: {legacy_report_path}")
        return {
            "dataset": dataset,
            "sample_idx": idx,
            "true_label": y_true,
            "hydra_pred": hydra_pred,
            "mrsqm_pred": mrsqm_pred,
            "save_path": save_path,
        }