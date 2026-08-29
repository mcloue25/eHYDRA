'''
Shared plotting configuration for the eHYDRA thesis figures.

Mix : https://www.youtube.com/watch?v=83PMyBk8hh4&list=RD83PMyBk8hh4&start_radio=1&t=1624s
'''

from __future__ import annotations
import matplotlib.pyplot as plt


MODE_COLOURS = {
    "top": "#1f77b4",
    "random": "#7f7f7f",
    "bottom": "#d62728",
}

MODE_LABELS = {
    "top": "Top",
    "random": "Random",
    "bottom": "Bottom",
}

METHOD_COLOURS: dict[str, str] = {
    "ehydra": "#4C72B0",
    "shapley_sampling": "#DD8452",
    "feature_ablation": "#55A868",
    "mrsqm": "#C44E52",
    "windowshap": "#8172B2",
    "tshap_centroid": "#937860",
    "tshap_train": "#DA8BC3",
    "tshap_zero": "#8C8C8C",
    "random":  "#BFBFBF",
}

METHOD_LABELS: dict[str, str] = {
    "ehydra": "eHYDRA",
    "shapley_sampling": "Shapley Sampling",
    "feature_ablation": "Feature Ablation",
    "mrsqm": "MrSQM",
    "windowshap": "WindowSHAP",
    "tshap_centroid": "TSHAP (centroid)",
    "tshap_train": "TSHAP (train)",
    "tshap_zero": "TSHAP (zero)",
    "random": "Random",
}

METHOD_ORDER: list[str] = [
    "tshap_train",
    "tshap_centroid",
    "shapley_sampling",
    "tshap_zero",
    "feature_ablation",
    "ehydra",
    "mrsqm",
]

CLUSTER_COLOURS: dict[str, str] = {
    "High-frequency / high-curvature":"#2a78d6",
    "Smooth / low-complexity": "#1baf7a",
    "Short / moderately rough": "#eda100",
    "Spiky / multi-class":"#e34948",
}

CLUSTER_SHORT: dict[str, str] = {
    "High-frequency / high-curvature": "High-freq",
    "Smooth / low-complexity":"Smooth",
    "Short / moderately rough": "Short/rough",
    "Spiky / multi-class": "Spiky",
}

CLUSTER_ORDER: list[str] = [
    "High-frequency / high-curvature",
    "Smooth / low-complexity",
    "Short / moderately rough",
    "Spiky / multi-class",
]

FRACTIONS: list[float] = [0.05, 0.10, 0.20]
FRACTION_LABELS: list[str] = ["5%", "10%", "20%"]

THESIS_RCPARAMS: dict = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.85,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}

def apply_thesis_style():
    plt.rcParams.update(THESIS_RCPARAMS)


# Shared base for viva slide figures and rest of params passed based on how I want them on a per script basis
SLIDE_RCPARAMS: dict = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
}

def apply_slide_style(overrides: dict | None = None):
    plt.rcParams.update({**SLIDE_RCPARAMS, **(overrides or {})})

def apply_slide_style(overrides: dict | None = None):
    plt.rcParams.update({**SLIDE_RCPARAMS, **(overrides or {})})

def method_colour(method_key: str):
    return METHOD_COLOURS.get(method_key, "#AAAAAA")

def method_label(method_key: str):
    return METHOD_LABELS.get(method_key, method_key)

def cluster_colour(cluster_name: str):
    return CLUSTER_COLOURS.get(cluster_name, "#AAAAAA")

def cluster_label(cluster_name: str):
    return CLUSTER_SHORT.get(cluster_name, cluster_name)
